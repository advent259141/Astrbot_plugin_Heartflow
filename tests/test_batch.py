import asyncio
import json
import time
import unittest
import weakref
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from test_main import _Context, _Event, _Response, _config, heartflow


_GROUP = "test:GroupMessage:10001"
_SCORES = json.dumps(
    {name: 0 for name in ("relevance", "willingness", "social", "timing", "continuity")}
)


def _event(
    text="", *, group=_GROUP, sender="alice", sender_id="user-1", images=(), **flags
):
    event = _Event(text, **flags)
    event.unified_msg_origin = group
    event.get_sender_name = lambda: sender
    event.get_sender_id = lambda: sender_id
    event.message_obj = SimpleNamespace(message=list(images))
    return event


def _image(source):
    return SimpleNamespace(
        type="Image",
        file=source,
        convert_to_base64=AsyncMock(return_value=source),
    )


def _batch_contents(event):
    return [entry.message.content for entry in event.get_extra("heartflow_judge_batch")]


class HeartflowBatchTests(unittest.IsolatedAsyncioTestCase):
    def plugin(self, **overrides):
        provider = heartflow.Provider()
        provider.text_chat = AsyncMock(return_value=_Response(_SCORES))
        plugin = heartflow.HeartflowPlugin(_Context(provider), _config(**overrides))
        plugin._get_persona_system_prompt = AsyncMock(return_value="Complete persona")
        if plugin.jev_client is not None:
            plugin.jev_client.evaluate = AsyncMock(
                return_value={
                    name: {"score": 0, "confidence": 1} for name in plugin.weights
                }
            )
        return plugin, provider

    async def test_default_interval_bounds_and_single_message_format(self):
        plugin, _ = self.plugin()
        self.assertEqual(plugin.judge_message_interval, 1)
        for value, expected in [
            (0, 1),
            (1, 1),
            (100, 100),
            (101, 100),
            (2.9, 2),
            ("3", 3),
            (None, 1),
            ("invalid", 1),
            (float("nan"), 1),
            (float("inf"), 1),
        ]:
            with self.subTest(interval=value):
                configured, _ = self.plugin(judge_message_interval=value)
                self.assertIs(type(configured.judge_message_interval), int)
                self.assertEqual(configured.judge_message_interval, expected)

        for backend in ("llm", "jev"):
            with self.subTest(backend=backend):
                plugin, provider = self.plugin(judge_backend=backend)
                for text in ("first", "second"):
                    await plugin.on_group_message(_event(text))
                self.assertEqual(plugin.chat_states[_GROUP].pending_messages, [])
                if backend == "llm":
                    self.assertEqual(provider.text_chat.await_count, 2)
                    self.assertEqual(
                        [
                            call.kwargs["prompt"]
                            for call in provider.text_chat.await_args_list
                        ],
                        ["[alice]: first", "[alice]: second"],
                    )
                    self.assertEqual(
                        provider.text_chat.await_args.kwargs["contexts"],
                        [{"role": "user", "content": "[alice]: first"}],
                    )
                else:
                    self.assertEqual(plugin.jev_client.evaluate.await_count, 2)
                    state = plugin.jev_client.evaluate.await_args.args[0]
                    self.assertNotIn("current_messages", state)
                    self.assertEqual(state["current_message"]["content"], "second")

    async def test_exact_n_and_2n_judgments_are_separate_per_group(self):
        plugin, _ = self.plugin(judge_message_interval=3)
        batches = []

        async def judge(event):
            batches.append(
                (
                    event.unified_msg_origin,
                    _batch_contents(event),
                    len(plugin.chat_states[event.unified_msg_origin].pending_messages),
                )
            )
            return heartflow.JudgeResult(should_reply=True, overall_score=0.9)

        plugin.judge_with_tiny_model = AsyncMock(side_effect=judge)
        counts = {_GROUP: 0, "other-group": 0}
        sequence = [
            (_GROUP, "a1"),
            ("other-group", "b1"),
            (_GROUP, "a2"),
            (_GROUP, "a3"),
            (_GROUP, "a4"),
            ("other-group", "b2"),
            (_GROUP, "a5"),
            ("other-group", "b3"),
            (_GROUP, "a6"),
        ]
        for group, text in sequence:
            event = _event(text, group=group)
            await plugin.on_group_message(event)
            counts[group] += 1
            self.assertEqual(
                plugin.judge_with_tiny_model.await_count,
                sum(count // 3 for count in counts.values()),
            )
            self.assertEqual(
                event.get_extra("heartflow_triggered", False), counts[group] % 3 == 0
            )
            self.assertIsNone(event.get_extra("heartflow_judge_batch"))
        self.assertEqual(
            batches,
            [
                (_GROUP, ["a1", "a2", "a3"], 0),
                ("other-group", ["b1", "b2", "b3"], 0),
                (_GROUP, ["a4", "a5", "a6"], 0),
            ],
        )
        self.assertTrue(
            all(not state.pending_messages for state in plugin.chat_states.values())
        )

    async def test_filters_and_cooldown_do_not_advance_pending_count(self):
        plugin, _ = self.plugin(
            judge_message_interval=2,
            min_reply_interval_seconds=60,
            whitelist_enabled=True,
            chat_whitelist=[_GROUP],
        )
        batches = []

        async def judge(event):
            batches.append(_batch_contents(event))
            return heartflow.JudgeResult()

        plugin.judge_with_tiny_model = AsyncMock(side_effect=judge)
        await plugin.on_group_message(_event("first"))
        state = plugin.chat_states[_GROUP]
        for event in [
            _event("wake", wake=True),
            _event("command", command=True),
            _event("self", sender_id="bot-1"),
            _event("outside", group="not-whitelisted"),
            _event("   "),
        ]:
            await plugin.on_group_message(event)
            self.assertEqual(len(state.pending_messages), 1)
        for field in ("last_reply_time", "last_trigger_time"):
            setattr(state, field, time.time())
            await plugin.on_group_message(_event("cooldown-" + field))
            self.assertEqual(len(state.pending_messages), 1)
            setattr(state, field, 0)
        plugin.judge_with_tiny_model.assert_not_awaited()
        await plugin.on_group_message(_event("second"))
        self.assertEqual(batches, [["first", "second"]])
        self.assertNotIn("not-whitelisted", plugin.chat_states)
        self.assertEqual(
            [message.content for message in plugin._get_raw_buffer(_GROUP)],
            [
                "first",
                "wake",
                "cooldown-last_reply_time",
                "cooldown-last_trigger_time",
                "second",
            ],
        )

    async def test_llm_full_batch_survives_history_truncation_and_repeated_text(self):
        for truncate in (False, True):
            with self.subTest(truncated_buffer=truncate):
                plugin, provider = self.plugin(
                    judge_message_interval=4,
                    judge_context_count=2,
                    context_messages_count=1,
                )
                # Equal-valued history must survive: only batch object identities are excluded.
                prior = [
                    heartflow.RawMessage("alice", "user-1", "repeat", 1000.0),
                    heartflow.RawMessage(
                        "bot", "bot-1", "earlier answer", 1000.0, is_bot=True
                    ),
                ]
                plugin._raw_msg_buffer[_GROUP] = deque(
                    prior, maxlen=2 if truncate else plugin._raw_msg_buffer_size
                )
                texts = ["first new", "repeat", "middle", "repeat"]
                with patch.object(heartflow.time, "time", return_value=1000.0):
                    for text in texts:
                        await plugin.on_group_message(_event(text))
                provider.text_chat.assert_awaited_once()
                request = provider.text_chat.await_args.kwargs
                self.assertEqual(
                    request["prompt"],
                    "\n\n".join(
                        f"[消息 {index} | alice (user-1)]: {text}"
                        for index, text in enumerate(texts, 1)
                    ),
                )
                self.assertEqual(
                    request["contexts"],
                    []
                    if truncate
                    else [
                        {"role": "user", "content": "[alice]: repeat"},
                        {"role": "assistant", "content": "earlier answer"},
                    ],
                )
                self.assertIn("Complete persona", request["system_prompt"])
                self.assertNotIn("first new", request["system_prompt"])
                if truncate:
                    self.assertEqual(
                        [message.content for message in plugin._get_raw_buffer(_GROUP)],
                        ["middle", "repeat"],
                    )

    async def test_llm_batch_images_keep_association_and_are_reused_on_retry(self):
        plugin, provider = self.plugin(judge_message_interval=3)
        provider.text_chat.side_effect = [_Response("[]"), _Response(_SCORES)]
        await plugin.on_group_message(_event("background", wake=True))
        first, second, third = [_image(source) for source in ("A", "B", "C")]
        await plugin.on_group_message(
            _event("", sender="bob", sender_id="2", images=[first])
        )
        await plugin.on_group_message(
            _event(
                "two pictures", sender="carol", sender_id="3", images=[second, third]
            )
        )
        provider.text_chat.assert_not_awaited()
        for picture in (first, second, third):
            picture.convert_to_base64.assert_not_awaited()
        final = _event("last text")
        await plugin.on_group_message(final)
        self.assertEqual(provider.text_chat.await_count, 2)
        requests = [call.kwargs for call in provider.text_chat.await_args_list]
        for request in requests:
            self.assertEqual(
                request["image_urls"], ["base64://A", "base64://B", "base64://C"]
            )
            self.assertEqual(
                request["prompt"],
                "[消息 1 | bob (2)]: [图片消息]\n附图序号: 1\n\n"
                "[消息 2 | carol (3)]: two pictures\n附图序号: 2, 3\n\n"
                "[消息 3 | alice (user-1)]: last text",
            )
            self.assertEqual(
                request["contexts"],
                [{"role": "user", "content": "[alice]: background"}],
            )
        self.assertIs(requests[0]["image_urls"], requests[1]["image_urls"])
        self.assertIn("上一份响应格式无效", requests[1]["system_prompt"])
        for picture in (first, second, third):
            picture.convert_to_base64.assert_awaited_once()
        self.assertIsNone(final.get_extra("heartflow_judge_batch"))
        self.assertEqual(plugin.chat_states[_GROUP].pending_messages, [])

    async def test_disabled_images_skip_pure_images_but_text_counts_for_both_backends(
        self,
    ):
        for backend in ("llm", "jev"):
            with self.subTest(backend=backend):
                plugin, provider = self.plugin(
                    judge_backend=backend,
                    judge_message_interval=2,
                    llm_enable_images=False,
                    jev_enable_images=False,
                )
                judge = (
                    provider.text_chat
                    if backend == "llm"
                    else plugin.jev_client.evaluate
                )
                picture = _image("disabled")
                await plugin.on_group_message(_event(images=[picture]))
                self.assertNotIn(_GROUP, plugin.chat_states)
                await plugin.on_group_message(_event("first", images=[picture]))
                await plugin.on_group_message(_event(images=[picture]))
                judge.assert_not_awaited()
                self.assertEqual(len(plugin.chat_states[_GROUP].pending_messages), 1)
                self.assertEqual(
                    plugin.chat_states[_GROUP].pending_messages[0].images, []
                )
                await plugin.on_group_message(_event("second"))
                judge.assert_awaited_once()
                self.assertEqual(
                    [message.content for message in plugin._get_raw_buffer(_GROUP)],
                    ["first", "second"],
                )
                if backend == "llm":
                    self.assertEqual(judge.await_args.kwargs["image_urls"], [])
                    self.assertEqual(
                        judge.await_args.kwargs["prompt"],
                        "[消息 1 | alice (user-1)]: first\n\n"
                        "[消息 2 | alice (user-1)]: second",
                    )
                else:
                    state = judge.await_args.args[0]
                    self.assertEqual(
                        [message["content"] for message in state["current_messages"]],
                        ["first", "second"],
                    )
                    self.assertEqual(
                        [
                            message["image_indices"]
                            for message in state["current_messages"]
                        ],
                        [[], []],
                    )
                    self.assertNotIn("images", judge.await_args.kwargs)
                picture.convert_to_base64.assert_not_awaited()

    async def test_jev_batch_sends_full_ordered_state_and_flattened_images(self):
        plugin, _ = self.plugin(
            judge_backend="jev", judge_message_interval=3, judge_context_count=2
        )
        persona = "Full persona\n" + "Keep every requirement.\n" * 50
        plugin._get_persona_system_prompt.return_value = persona
        prior = [
            heartflow.RawMessage("bob", "2", "repeat", 999.0),
            heartflow.RawMessage("bot", "bot-1", "earlier answer", 1000.0, is_bot=True),
        ]
        plugin._raw_msg_buffer[_GROUP] = deque(
            prior, maxlen=plugin._raw_msg_buffer_size
        )
        plugin._raw_msg_buffer["other-group"] = deque(
            [heartflow.RawMessage("other", "9", "unrelated", 998.0)]
        )
        pictures = [_image(source) for source in ("A", "B", "C")]
        events = [
            _event("repeat", sender="bob", sender_id="2", images=pictures[:1]),
            _event("", sender="carol", sender_id="3", images=pictures[1:]),
            _event("repeat"),
        ]
        with patch.object(heartflow.time, "time", return_value=1001.0) as clock:
            for index, event in enumerate(events):
                clock.return_value = 1001.0 + index
                await plugin.on_group_message(event)
        plugin.jev_client.evaluate.assert_awaited_once()
        state, questions = plugin.jev_client.evaluate.await_args.args
        self.assertNotIn("current_message", state)
        self.assertEqual(state["persona"], persona)
        self.assertEqual(
            state["history"],
            [
                {
                    "role": "user",
                    "sender": "bob",
                    "sender_id": "2",
                    "content": "repeat",
                    "timestamp": 999.0,
                },
                {
                    "role": "assistant",
                    "sender": "bot",
                    "sender_id": "bot-1",
                    "content": "earlier answer",
                    "timestamp": 1000.0,
                },
            ],
        )
        self.assertEqual(
            state["current_messages"],
            [
                {
                    "role": "user",
                    "sender": "bob",
                    "sender_id": "2",
                    "content": "repeat",
                    "timestamp": 1001.0,
                    "image_indices": [1],
                },
                {
                    "role": "user",
                    "sender": "carol",
                    "sender_id": "3",
                    "content": "[图片消息]",
                    "timestamp": 1002.0,
                    "image_indices": [2, 3],
                },
                {
                    "role": "user",
                    "sender": "alice",
                    "sender_id": "user-1",
                    "content": "repeat",
                    "timestamp": 1003.0,
                    "image_indices": [],
                },
            ],
        )
        self.assertEqual(
            plugin.jev_client.evaluate.await_args.kwargs["images"], pictures
        )
        self.assertEqual(state["last_bot_reply"], "earlier answer")
        self.assertEqual(state["minutes_since_last_reply"], 999)
        self.assertEqual(set(questions), set(plugin.weights))
        self.assertTrue(
            all(
                "current_messages" in question["instructions"]
                for question in questions.values()
            )
        )
        for picture in pictures:
            picture.convert_to_base64.assert_not_awaited()

    async def test_failure_and_cancellation_consume_batch_and_release_references(self):
        for outcome in ("failure", "cancellation"):
            with self.subTest(outcome=outcome):
                plugin, _ = self.plugin(judge_message_interval=2)
                lock = plugin._get_chat_lock(_GROUP)
                entered = asyncio.Event()
                release = asyncio.Event()
                snapshots = []
                pending_refs = []

                async def judge(event):
                    pending_refs.append(
                        weakref.ref(event.get_extra("heartflow_judge_batch")[0])
                    )
                    snapshots.append(
                        (
                            _batch_contents(event),
                            len(plugin.chat_states[_GROUP].pending_messages),
                            lock.locked(),
                        )
                    )
                    entered.set()
                    if outcome == "failure":
                        raise RuntimeError("judge unavailable")
                    await release.wait()
                    return heartflow.JudgeResult()

                plugin.judge_with_tiny_model = judge
                await plugin.on_group_message(_event(images=[_image("pending-image")]))
                final = _event("last")
                task = asyncio.create_task(plugin.on_group_message(final))
                try:
                    await asyncio.wait_for(entered.wait(), 2)
                    if outcome == "cancellation":
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        await asyncio.wait_for(task, 2)
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                self.assertEqual(snapshots, [(["[图片消息]", "last"], 0, True)])
                self.assertEqual(plugin.chat_states[_GROUP].pending_messages, [])
                self.assertIsNone(final.get_extra("heartflow_judge_batch"))
                self.assertIsNone(pending_refs[0]())
                self.assertFalse(lock.locked())

                async def recovered(event):
                    snapshots.append(_batch_contents(event))
                    return heartflow.JudgeResult()

                plugin.judge_with_tiny_model = AsyncMock(side_effect=recovered)
                await plugin.on_group_message(_event("fresh first"))
                plugin.judge_with_tiny_model.assert_not_awaited()
                await plugin.on_group_message(_event("fresh second"))
                plugin.judge_with_tiny_model.assert_awaited_once()
                self.assertEqual(snapshots[-1], ["fresh first", "fresh second"])

    async def test_concurrent_arrivals_judge_once_per_full_batch(self):
        plugin, _ = self.plugin(judge_message_interval=3)
        lock = plugin._get_chat_lock(_GROUP)
        entered = asyncio.Event()
        release = asyncio.Event()
        batches = []

        async def judge(event):
            batches.append(
                (
                    _batch_contents(event),
                    len(plugin.chat_states[_GROUP].pending_messages),
                )
            )
            if len(batches) == 1:
                entered.set()
                await release.wait()
            return heartflow.JudgeResult()

        plugin.judge_with_tiny_model = judge
        events = [_event(f"message-{index}") for index in range(7)]
        tasks = [
            asyncio.create_task(plugin.on_group_message(event)) for event in events[:3]
        ]
        try:
            await asyncio.wait_for(entered.wait(), 2)
            tasks.extend(
                asyncio.create_task(plugin.on_group_message(event))
                for event in events[3:]
            )
            await asyncio.sleep(0)
            self.assertTrue(lock.locked())
            self.assertEqual(len(batches), 1)
            self.assertEqual(plugin.chat_states[_GROUP].pending_messages, [])
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 2)
        self.assertEqual(
            batches,
            [
                (["message-0", "message-1", "message-2"], 0),
                (["message-3", "message-4", "message-5"], 0),
            ],
        )
        self.assertEqual(
            [
                entry.message.content
                for entry in plugin.chat_states[_GROUP].pending_messages
            ],
            ["message-6"],
        )
        self.assertFalse(lock.locked())
        self.assertTrue(
            all(event.get_extra("heartflow_judge_batch") is None for event in events)
        )

    async def test_reset_and_eviction_discard_pending_batches(self):
        for action in ("reset", "evict"):
            with self.subTest(action=action):
                plugin, _ = self.plugin(judge_message_interval=3, max_tracked_chats=1)
                batches = []

                async def judge(event):
                    batches.append(_batch_contents(event))
                    return heartflow.JudgeResult()

                plugin.judge_with_tiny_model = AsyncMock(side_effect=judge)
                await plugin.on_group_message(_event(images=[_image("old-image")]))
                await plugin.on_group_message(_event("old text"))
                pending_ref = weakref.ref(
                    plugin.chat_states[_GROUP].pending_messages[0]
                )
                if action == "reset":
                    command = _event("reset", command=True)
                    command.plain_result = lambda text: text
                    command.set_result = Mock()
                    await plugin.heartflow_reset(command)
                    command.set_result.assert_called_once()
                else:
                    await plugin.on_group_message(_event("other", group="other-group"))
                self.assertNotIn(_GROUP, plugin.chat_states)
                self.assertNotIn(_GROUP, plugin._raw_msg_buffer)
                self.assertIsNone(pending_ref())
                for text in ("fresh first", "fresh second"):
                    await plugin.on_group_message(_event(text))
                    plugin.judge_with_tiny_model.assert_not_awaited()
                await plugin.on_group_message(_event("fresh third"))
                self.assertEqual(
                    batches, [["fresh first", "fresh second", "fresh third"]]
                )

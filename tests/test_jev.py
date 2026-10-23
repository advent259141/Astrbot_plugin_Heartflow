import asyncio
import math
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

from test_main import _Context, _Event, _config, heartflow


class JevJudgeTests(unittest.IsolatedAsyncioTestCase):
    def plugin(self, **config):
        plugin = heartflow.HeartflowPlugin(
            _Context(), _config(judge_backend="jev", judge_provider_name="", **config)
        )
        plugin._get_persona_system_prompt = AsyncMock(return_value="A helpful bot")
        return plugin

    async def test_five_scores_are_rescaled_and_use_existing_weights(self):
        plugin = self.plugin(reply_threshold=0.7)
        plugin.jev_client.evaluate = AsyncMock(
            return_value={
                name: {"score": 4 if name == "relevance" else 2, "confidence": 0.8}
                for name in plugin.weights
            }
        )
        result = await plugin.judge_with_tiny_model(_Event("hello"))
        self.assertEqual(result.relevance, 10)
        self.assertEqual(result.social, 5)
        self.assertAlmostEqual(result.overall_score, 0.625)
        self.assertAlmostEqual(result.confidence, 0.8)
        self.assertFalse(result.should_reply)
        state, questions = plugin.jev_client.evaluate.call_args.args
        self.assertEqual(state["current_message"]["content"], "hello")
        self.assertEqual(state["persona"], "A helpful bot")
        self.assertEqual(set(questions), set(plugin.weights))
        self.assertTrue(all(len(q["criteria"]) == 5 for q in questions.values()))

    async def test_high_scores_trigger_and_maximum_is_one(self):
        plugin = self.plugin()
        plugin.jev_client.evaluate = AsyncMock(
            return_value={
                name: {"score": 4, "confidence": 1} for name in plugin.weights
            }
        )
        result = await plugin.judge_with_tiny_model(_Event())
        self.assertTrue(result.should_reply)
        self.assertAlmostEqual(result.overall_score, 1)

    async def test_verbatim_history_persona_roles_and_timestamps_reach_jev(self):
        plugin = self.plugin(judge_context_count=0)
        persona = "完整人格设定：" + "保持角色性格，不省略要求。" * 120
        plugin._get_persona_system_prompt.return_value = persona
        event = _Event("那这个具体怎么改？")
        prior = [
            heartflow.RawMessage("bob", "2", "我在写一个Python异步爬虫", 1000.0),
            heartflow.RawMessage(
                "bot", "bot", "可以使用aiohttp代替阻塞请求", 1001.0, is_bot=True
            ),
            heartflow.RawMessage("alice", "user-1", "还有三个页面要并发抓取", 1002.0),
        ]
        plugin._raw_msg_buffer[event.unified_msg_origin] = deque(prior)
        plugin._raw_msg_buffer["another-group"] = deque(
            [heartflow.RawMessage("other", "9", "不要混入其他群的消息", 999.0)]
        )
        plugin._record_raw_message(event)
        plugin.jev_client.evaluate = AsyncMock(
            return_value={
                name: {"score": 3, "confidence": 0.9} for name in plugin.weights
            }
        )
        await plugin.judge_with_tiny_model(event)
        state, questions = plugin.jev_client.evaluate.call_args.args
        self.assertEqual(state["persona"], persona)
        self.assertEqual(
            [entry["content"] for entry in state["history"]],
            [message.content for message in prior],
        )
        self.assertEqual(
            [entry["role"] for entry in state["history"]], ["user", "assistant", "user"]
        )
        self.assertEqual(
            [entry["timestamp"] for entry in state["history"]], [1000.0, 1001.0, 1002.0]
        )
        self.assertEqual(state["history"][0]["sender_id"], "2")
        self.assertEqual(state["current_message"]["role"], "user")
        self.assertEqual(state["current_message"]["content"], event.message_str)
        self.assertTrue(
            all(
                "history" in question["instructions"] for question in questions.values()
            )
        )

    async def test_history_count_limits_jev_window_and_excludes_current_image(self):
        plugin = self.plugin(judge_context_count=2)
        event = _Event("")
        event.message_obj = SimpleNamespace(
            message=[SimpleNamespace(type="Image", file="image")]
        )
        for text in ["旧消息", "中间消息", "最近消息"]:
            plugin._record_raw_message(_Event(text))
        plugin._record_raw_message(event)
        plugin.jev_client.evaluate = AsyncMock(
            return_value={
                name: {"score": 3, "confidence": 0.9} for name in plugin.weights
            }
        )
        await plugin.judge_with_tiny_model(event)
        state = plugin.jev_client.evaluate.call_args.args[0]
        self.assertEqual(
            [message["content"] for message in state["history"]],
            ["中间消息", "最近消息"],
        )
        self.assertEqual(state["current_message"]["content"], "[图片消息]")

    async def test_invalid_scores_and_transport_errors_do_not_trigger(self):
        for invalid in [True, "3", math.nan, math.inf, -1, 4.1, None]:
            with self.subTest(value=invalid):
                plugin = self.plugin()
                plugin.jev_client.evaluate = AsyncMock(
                    return_value={
                        name: {"score": invalid, "confidence": 1}
                        for name in plugin.weights
                    }
                )
                self.assertFalse(
                    (await plugin.judge_with_tiny_model(_Event())).should_reply
                )
        plugin.jev_client.evaluate = AsyncMock(side_effect=ValueError("HTTP 401"))
        self.assertFalse((await plugin.judge_with_tiny_model(_Event())).should_reply)

    async def test_timeout_and_cancellation(self):
        plugin = self.plugin()
        plugin.judge_timeout_seconds = 0.01

        async def slow(*_args):
            await asyncio.sleep(10)

        plugin.jev_client.evaluate = slow
        self.assertFalse((await plugin.judge_with_tiny_model(_Event())).should_reply)
        plugin.jev_client.evaluate = AsyncMock(side_effect=asyncio.CancelledError)
        with self.assertRaises(asyncio.CancelledError):
            await plugin.judge_with_tiny_model(_Event())

    async def test_image_only_message_reaches_judge_and_can_trigger(self):
        plugin = self.plugin()
        picture = SimpleNamespace(type="Image", file="test-image")
        event = _Event("")
        event.message_obj = SimpleNamespace(message=[picture])
        plugin.jev_client.evaluate = AsyncMock(
            return_value={
                name: {"score": 4, "confidence": 1} for name in plugin.weights
            }
        )
        await plugin.on_group_message(event)
        self.assertTrue(event.get_extra("heartflow_triggered"))
        self.assertEqual(
            plugin.jev_client.evaluate.call_args.kwargs["images"], [picture]
        )
        self.assertEqual(
            plugin._get_raw_buffer(event.unified_msg_origin)[0].content, "[图片消息]"
        )

    async def test_disabling_vision_skips_image_only_messages(self):
        plugin = self.plugin(jev_enable_images=False)
        event = _Event("")
        event.message_obj = SimpleNamespace(message=[SimpleNamespace(type="Image")])
        plugin.jev_client.evaluate = AsyncMock()
        await plugin.on_group_message(event)
        plugin.jev_client.evaluate.assert_not_awaited()

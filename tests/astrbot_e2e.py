"""End-to-end checks against a real AstrBot source installation.

Run this script through AstrBot's isolated uv environment. It uses AstrBot's
real event, context, provider, response, configuration, and respond-stage
types while replacing only network-facing provider and platform operations.
"""

import argparse
import asyncio
import copy
import functools
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


def _load_plugin(plugin_root: Path):
    """Load Heartflow with the real AstrBot decorators.

    Args:
        plugin_root: Heartflow repository root.

    Returns:
        Loaded Heartflow module.
    """
    module_path = plugin_root / "main.py"
    spec = importlib.util.spec_from_file_location("heartflow_astrbot_e2e", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load plugin module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


async def _run(plugin_root: Path) -> dict:
    """Run Heartflow through real AstrBot runtime types.

    Args:
        plugin_root: Heartflow repository root.

    Returns:
        Scenario summary suitable for JSON output.
    """
    from astrbot.core.config.astrbot_config import AstrBotConfig
    from astrbot.core.config.default import DEFAULT_CONFIG
    from astrbot.core.message.components import Plain
    from astrbot.core.message.message_event_result import (
        MessageChain,
        MessageEventResult,
        ResultContentType,
    )
    from astrbot.core.pipeline.context import PipelineContext
    from astrbot.core.pipeline.context_utils import call_event_hook, call_handler
    from astrbot.core.pipeline.respond.stage import RespondStage
    from astrbot.core.platform.astr_message_event import AstrMessageEvent
    from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
    from astrbot.core.platform.message_type import MessageType
    from astrbot.core.platform.platform_metadata import PlatformMetadata
    from astrbot.core.provider.entities import LLMResponse, ProviderRequest
    from astrbot.core.provider.provider import Provider
    from astrbot.core.star.context import Context
    from astrbot.core.star.star import star_map
    from astrbot.core.star.star_handler import EventType, star_handlers_registry
    from astrbot.core.star.star_manager import PluginManager

    metadata = PluginManager._load_plugin_metadata(str(plugin_root))
    assert metadata is not None
    compatible, error = PluginManager._validate_astrbot_version_specifier(
        metadata.astrbot_version
    )
    assert compatible, error

    schema = json.loads((plugin_root / "_conf_schema.json").read_text(encoding="utf-8"))

    class DeterministicProvider(Provider):
        """Network-free real AstrBot chat provider."""

        def __init__(self) -> None:
            super().__init__(
                {"id": "judge", "type": "heartflow_e2e", "model": "deterministic"},
                {},
            )
            self.calls = []
            self.delay = 0.0

        def get_current_key(self) -> str:
            """Return the provider key used by the integration test."""
            return "e2e"

        def set_key(self, key: str) -> None:
            """Accept a provider key without contacting a service.

            Args:
                key: Provider key supplied by AstrBot.
            """
            del key

        async def get_models(self) -> list[str]:
            """Return the deterministic model name."""
            return ["deterministic"]

        async def text_chat(self, **kwargs) -> LLMResponse:
            """Return a deterministic high-score response.

            Args:
                **kwargs: Real AstrBot Provider.text_chat arguments.

            Returns:
                Successful AstrBot LLM response containing score JSON.
            """
            self.calls.append(kwargs)
            if self.delay:
                await asyncio.sleep(self.delay)
            scores = {
                "relevance": 9,
                "willingness": 9,
                "social": 9,
                "timing": 9,
                "continuity": 9,
                "reasoning": "deterministic e2e score",
            }
            return LLMResponse(
                role="assistant",
                completion_text=json.dumps(scores, ensure_ascii=False),
            )

    class RecordingEvent(AstrMessageEvent):
        """Real AstrBot event with in-memory platform delivery."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.sent_texts = []
            self.streamed_texts = []

        async def send(self, message: MessageChain) -> None:
            """Record a non-streaming platform send.

            Args:
                message: Real AstrBot message chain.
            """
            self._has_send_oper = True
            self.sent_texts.append(message.get_plain_text())

        async def send_streaming(
            self,
            generator,
            use_fallback: bool = False,
        ) -> None:
            """Consume and record a streaming platform send.

            Args:
                generator: AstrBot message-chain async generator.
                use_fallback: Whether platform fallback delivery is requested.
            """
            del use_fallback
            self._has_send_oper = True
            async for chain in generator:
                self.streamed_texts.append(chain.get_plain_text())

    class ConversationManager:
        """Conversation manager exposing the no-current-conversation path."""

        async def get_curr_conversation_id(self, _umo):
            """Return no current conversation for a fresh test group."""
            return None

    class PersonaManager:
        """Persona manager returning a deterministic default persona."""

        async def get_default_persona_v3(self, _umo):
            """Return a default persona compatible with AstrBot 4.27."""
            return {"prompt": "A friendly group-chat bot."}

    def make_event(session_id: str, text: str, *, wake: bool = False):
        message = AstrBotMessage()
        message.type = MessageType.GROUP_MESSAGE
        message.self_id = "bot-1"
        message.session_id = session_id
        message.message_id = f"message-{session_id}-{time.monotonic_ns()}"
        message.group_id = session_id
        message.sender = MessageMember(user_id="user-1", nickname="Alice")
        message.message = [Plain(text)]
        message.message_str = text
        message.raw_message = None
        event = RecordingEvent(
            message_str=text,
            message_obj=message,
            platform_meta=PlatformMetadata(
                name="heartflow_e2e",
                description="Heartflow integration platform",
                id="heartflow_e2e",
                support_streaming_message=True,
            ),
            session_id=session_id,
        )
        event.is_at_or_wake_command = wake
        event.is_wake = wake
        event.set_extra("handlers_parsed_params", {})
        return event

    with tempfile.TemporaryDirectory(prefix="heartflow_e2e_") as temp_dir:
        temp_path = Path(temp_dir)
        plugin_config = AstrBotConfig(
            config_path=str(temp_path / "heartflow.json"),
            schema=schema,
        )
        plugin_config.update(
            {
                "enable_heartflow": True,
                "judge_provider_name": "judge",
                "reply_threshold": 0.6,
                "judge_max_retries": 1,
                "judge_timeout_seconds": 5,
                "min_reply_interval_seconds": 0,
            }
        )

        main_config = copy.deepcopy(DEFAULT_CONFIG)
        main_config["provider_ltm_settings"]["active_reply"]["enable"] = False
        provider = DeterministicProvider()
        provider_manager = SimpleNamespace(inst_map={"judge": provider})
        config_manager = SimpleNamespace(get_conf=lambda _umo: main_config)
        context = Context(
            asyncio.Queue(),
            main_config,
            MagicMock(),
            provider_manager,
            MagicMock(),
            ConversationManager(),
            MagicMock(),
            PersonaManager(),
            config_manager,
            MagicMock(),
            MagicMock(),
        )

        heartflow = _load_plugin(plugin_root)
        plugin = heartflow.HeartflowPlugin(context=context, config=plugin_config)

        module_handlers = star_handlers_registry.get_handlers_by_module_name(
            heartflow.__name__
        )
        handler_types = {handler.event_type for handler in module_handlers}
        assert EventType.AdapterMessageEvent in handler_types
        assert EventType.OnLLMRequestEvent in handler_types
        assert EventType.OnLLMResponseEvent in handler_types
        runtime_metadata = star_map[heartflow.__name__]
        runtime_metadata.name = metadata.name
        runtime_metadata.activated = True

        def bind_plugin(instance) -> None:
            runtime_metadata.star_cls = instance
            for handler in module_handlers:
                raw_handler = (
                    handler.handler.func
                    if isinstance(handler.handler, functools.partial)
                    else handler.handler
                )
                handler.handler = functools.partial(raw_handler, instance)

        async def dispatch_group_message(event) -> None:
            group_handler = next(
                handler
                for handler in module_handlers
                if handler.handler_name == "on_group_message"
            )
            assert all(
                event_filter.filter(event, main_config)
                for event_filter in group_handler.event_filters
            )
            async for _ in call_handler(event, group_handler.handler):
                pass

        pipeline_context = PipelineContext(main_config, MagicMock(), "heartflow-e2e")
        respond_stage = RespondStage()
        await respond_stage.initialize(pipeline_context)

        success_event = make_event("success-group", "Should we discuss this topic?")
        bind_plugin(plugin)
        await dispatch_group_message(success_event)
        assert success_event.get_extra("heartflow_triggered") is True
        request = ProviderRequest(system_prompt="base system prompt")
        await call_event_hook(success_event, EventType.OnLLMRequestEvent, request)
        assert "主动参与群聊" in request.system_prompt
        success_response = LLMResponse(
            role="assistant",
            result_chain=MessageChain().message("A natural proactive reply."),
        )
        await call_event_hook(
            success_event,
            EventType.OnLLMResponseEvent,
            success_response,
        )
        success_state = plugin._get_chat_state(success_event.unified_msg_origin)
        assert success_state.total_replies == 1
        assert success_state.energy < 1.0
        assert plugin._get_last_bot_reply(success_event) == "A natural proactive reply."
        success_event.set_result(
            MessageEventResult(
                chain=list(success_response.result_chain.chain),
                result_content_type=ResultContentType.LLM_RESULT,
            )
        )
        await respond_stage.process(success_event)
        assert success_event.sent_texts == ["A natural proactive reply."]

        streaming_event = make_event(
            "streaming-group", "A direct wake message", wake=True
        )
        await dispatch_group_message(streaming_event)
        streaming_response = LLMResponse(
            role="assistant",
            result_chain=MessageChain().message("Final streamed reply."),
        )
        await call_event_hook(
            streaming_event,
            EventType.OnLLMResponseEvent,
            streaming_response,
        )

        async def stream_chunks():
            yield MessageChain().message("Final streamed reply.")

        streaming_event.set_result(
            MessageEventResult(
                result_content_type=ResultContentType.STREAMING_RESULT,
                async_stream=stream_chunks(),
            )
        )
        await respond_stage.process(streaming_event)
        assert streaming_event.streamed_texts == ["Final streamed reply."]
        assert plugin._get_last_bot_reply(streaming_event) == "Final streamed reply."

        error_event = make_event("error-group", "Trigger then fail")
        await dispatch_group_message(error_event)
        error_state = plugin._get_chat_state(error_event.unified_msg_origin)
        assert error_state.last_trigger_time > 0
        await call_event_hook(
            error_event,
            EventType.OnLLMResponseEvent,
            LLMResponse(role="err", completion_text="provider failed"),
        )
        assert error_state.last_trigger_time == 0
        assert error_state.total_replies == 0

        timeout_config = dict(plugin_config)
        timeout_config["judge_timeout_seconds"] = 5
        timeout_plugin = heartflow.HeartflowPlugin(
            context=context,
            config=timeout_config,
        )
        timeout_plugin.judge_timeout_seconds = 0.05
        provider.delay = 0.2
        timeout_event = make_event("timeout-group", "This judgment should timeout")
        bind_plugin(timeout_plugin)
        timeout_started = time.monotonic()
        await dispatch_group_message(timeout_event)
        timeout_elapsed = time.monotonic() - timeout_started
        timeout_lock = timeout_plugin._get_chat_lock(timeout_event.unified_msg_origin)
        assert timeout_elapsed < 0.15
        assert not timeout_lock.locked()
        assert not timeout_event.get_extra("heartflow_triggered", False)

        concurrency_config = dict(plugin_config)
        concurrency_config["min_reply_interval_seconds"] = 60
        concurrency_plugin = heartflow.HeartflowPlugin(
            context=context,
            config=concurrency_config,
        )
        provider.delay = 0.03
        first = make_event("concurrent-group", "first concurrent message")
        second = make_event("concurrent-group", "second concurrent message")
        bind_plugin(concurrency_plugin)
        await asyncio.gather(
            dispatch_group_message(first),
            dispatch_group_message(second),
        )
        triggered_count = sum(
            bool(event.get_extra("heartflow_triggered", False))
            for event in (first, second)
        )
        assert triggered_count == 1

        await plugin.terminate()
        await timeout_plugin.terminate()
        await concurrency_plugin.terminate()

    return {
        "astrbot_version_compatible": True,
        "schema_loaded_by_astrbot": True,
        "real_handler_registration": True,
        "successful_trigger_and_send": True,
        "streaming_delivery_and_context": True,
        "provider_error_rollback": True,
        "judge_timeout_unlock": True,
        "concurrent_cooldown": True,
    }


def main() -> None:
    """Parse arguments and run the end-to-end suite."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-root", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(_run(args.plugin_root.resolve()))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

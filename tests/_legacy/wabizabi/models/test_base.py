from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic import ValidationError
from wabizabi.history import MessageHistory
from wabizabi.messages import FinishReason, ModelRequest, ModelResponse, TextPart, UserPromptPart
from wabizabi.models import (
    Model,
    ModelProfile,
    ModelResponseCompletedEvent,
    ModelResponsePartEvent,
    ModelResult,
    ModelSettings,
    ModelStreamEvent,
    merge_model_settings,
    model_result_events,
)
from wabizabi.output import OutputConfig
from wabizabi.providers.ollama import OllamaSettings
from wabizabi.tools import ToolDefinition
from wabizabi.usage import RunUsage


class EchoModel:
    @property
    def profile(self) -> ModelProfile:
        return ModelProfile(provider_name="test", model_name="echo")

    async def request(
        self,
        request: ModelRequest,
        *,
        message_history: MessageHistory | None = None,
        settings: ModelSettings | None = None,
        tools: tuple[ToolDefinition, ...] = (),
        output: OutputConfig[object] | None = None,
    ) -> ModelResult:
        del request
        del message_history
        del settings
        del tools
        del output
        return ModelResult(
            response=ModelResponse(parts=(TextPart(text="ok"),), model_name="echo"),
            usage=RunUsage(input_tokens=2, output_tokens=1),
        )

    async def stream_response(
        self,
        request: ModelRequest,
        *,
        message_history: MessageHistory | None = None,
        settings: ModelSettings | None = None,
        tools: tuple[ToolDefinition, ...] = (),
        output: OutputConfig[object] | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        result = await self.request(
            request,
            message_history=message_history,
            settings=settings,
            tools=tools,
            output=output,
        )
        for event in model_result_events(result):
            yield event


class DummySettings(ModelSettings):
    dummy_flag: bool = False


def test_model_profile_defaults_are_sensible() -> None:
    profile = ModelProfile(provider_name="ollama", model_name="qwen-demo")

    assert profile.supports_tools is True
    assert profile.supports_streaming is True


def test_model_settings_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ModelSettings.model_validate({"unknown": "value"})


def test_merge_model_settings_prefers_override_values() -> None:
    base = OllamaSettings(
        ollama_model="qwen-base",
        ollama_temperature=0.1,
    )
    override = OllamaSettings(
        ollama_temperature=0.7,
        ollama_num_predict=256,
    )

    merged = merge_model_settings(base, override)

    assert merged == OllamaSettings(
        ollama_model="qwen-base",
        ollama_temperature=0.7,
        ollama_num_predict=256,
    )


def test_merge_model_settings_rejects_mixed_types() -> None:
    base = OllamaSettings(ollama_model="qwen-base")
    override = DummySettings(dummy_flag=True)

    with pytest.raises(TypeError, match="Cannot merge model settings of different types"):
        merge_model_settings(base, override)


def test_model_result_defaults_usage_to_zero() -> None:
    result = ModelResult(
        response=ModelResponse(parts=(TextPart(text="ok"),), model_name="echo"),
    )

    assert result.usage == RunUsage.zero()


@pytest.mark.asyncio
async def test_model_protocol_can_be_used_by_adapter() -> None:
    model = EchoModel()
    request = ModelRequest(parts=(UserPromptPart(text="Hello"),))

    assert isinstance(model, Model)

    result = await model.request(request)

    assert result.response.model_name == "echo"
    assert result.response.parts == (TextPart(text="ok"),)
    assert result.usage == RunUsage(input_tokens=2, output_tokens=1)


@pytest.mark.asyncio
async def test_model_protocol_can_stream_response_parts() -> None:
    model = EchoModel()
    request = ModelRequest(parts=(UserPromptPart(text="Hello"),))

    events = [event async for event in model.stream_response(request)]

    assert events == [
        ModelResponsePartEvent(part=TextPart(text="ok")),
        ModelResponseCompletedEvent(
            model_name="echo",
            usage=RunUsage(input_tokens=2, output_tokens=1),
        ),
    ]


def test_model_result_events_projects_result_into_canonical_stream_events() -> None:
    result = ModelResult(
        response=ModelResponse(
            parts=(TextPart(text="ok"),),
            model_name="echo",
            finish_reason=FinishReason.STOP,
            metadata={"provider_request_id": "req-1"},
        ),
        usage=RunUsage(input_tokens=2, output_tokens=1),
    )

    assert model_result_events(result) == (
        ModelResponsePartEvent(part=TextPart(text="ok")),
        ModelResponseCompletedEvent(
            model_name="echo",
            usage=RunUsage(input_tokens=2, output_tokens=1),
            finish_reason=FinishReason.STOP,
            metadata={"provider_request_id": "req-1"},
        ),
    )

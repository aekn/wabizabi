"""Public test models for exercising the Wabizabi kernel."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from wabizabi.history import MessageHistory
from wabizabi.messages import ModelRequest
from wabizabi.models import (
    ModelProfile,
    ModelResult,
    ModelSettings,
    ModelStreamEvent,
    model_result_events,
)
from wabizabi.output import OutputConfig
from wabizabi.tools import ToolDefinition


@dataclass(frozen=True, slots=True)
class CapturedModelCall:
    """One captured model call made during a test."""

    request: ModelRequest
    message_history: MessageHistory | None
    settings: ModelSettings | None
    tools: tuple[ToolDefinition, ...]
    output: OutputConfig[object] | None


class ScriptedModel:
    """A simple model that returns pre-scripted one-shot results."""

    def __init__(
        self,
        results: Sequence[ModelResult],
        *,
        provider_name: str = "test",
        model_name: str = "scripted",
        profile: ModelProfile | None = None,
    ) -> None:
        self._results = list(results)
        self._profile = profile or ModelProfile(
            provider_name=provider_name,
            model_name=model_name,
        )
        self.calls: list[CapturedModelCall] = []

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    @property
    def requests(self) -> list[ModelRequest]:
        return [call.request for call in self.calls]

    @property
    def histories(self) -> list[MessageHistory | None]:
        return [call.message_history for call in self.calls]

    @property
    def received_settings(self) -> list[ModelSettings | None]:
        return [call.settings for call in self.calls]

    @property
    def received_tools(self) -> list[tuple[ToolDefinition, ...]]:
        return [call.tools for call in self.calls]

    async def request(
        self,
        request: ModelRequest,
        *,
        message_history: MessageHistory | None = None,
        settings: ModelSettings | None = None,
        tools: tuple[ToolDefinition, ...] = (),
        output: OutputConfig[object] | None = None,
    ) -> ModelResult:
        self.calls.append(
            CapturedModelCall(
                request=request,
                message_history=message_history,
                settings=settings,
                tools=tools,
                output=output,
            )
        )
        if not self._results:
            raise AssertionError("ScriptedModel received more requests than scripted results.")
        return self._results.pop(0)

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


class StreamingScriptedModel:
    """A model that yields pre-scripted stream event sequences."""

    def __init__(
        self,
        scripts: Sequence[tuple[ModelStreamEvent, ...]],
        *,
        provider_name: str = "test",
        model_name: str = "streamed",
    ) -> None:
        self._scripts = list(scripts)
        self._profile = ModelProfile(
            provider_name=provider_name,
            model_name=model_name,
        )
        self.calls: list[CapturedModelCall] = []

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    @property
    def requests(self) -> list[ModelRequest]:
        return [call.request for call in self.calls]

    @property
    def histories(self) -> list[MessageHistory | None]:
        return [call.message_history for call in self.calls]

    @property
    def received_settings(self) -> list[ModelSettings | None]:
        return [call.settings for call in self.calls]

    @property
    def received_tools(self) -> list[tuple[ToolDefinition, ...]]:
        return [call.tools for call in self.calls]

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
        raise AssertionError("StreamingScriptedModel.request() should not be called.")

    async def stream_response(
        self,
        request: ModelRequest,
        *,
        message_history: MessageHistory | None = None,
        settings: ModelSettings | None = None,
        tools: tuple[ToolDefinition, ...] = (),
        output: OutputConfig[object] | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.calls.append(
            CapturedModelCall(
                request=request,
                message_history=message_history,
                settings=settings,
                tools=tools,
                output=output,
            )
        )
        if not self._scripts:
            raise AssertionError(
                "StreamingScriptedModel received more requests than scripted streams."
            )
        for event in self._scripts.pop(0):
            yield event


__all__ = [
    "CapturedModelCall",
    "ScriptedModel",
    "StreamingScriptedModel",
]

"""Ollama chat model adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from wabizabi.history import MessageHistory
from wabizabi.messages import ModelRequest, ModelResponse
from wabizabi.models import (
    ModelProfile,
    ModelReasoningDeltaEvent,
    ModelResponseCompletedEvent,
    ModelResponsePartEvent,
    ModelResult,
    ModelSettings,
    ModelStreamEvent,
    ModelTextDeltaEvent,
)
from wabizabi.output import OutputConfig
from wabizabi.providers.ollama.client import (
    OllamaChatFn,
    OllamaStreamChatFn,
    build_default_chat,
    build_default_stream_chat,
    json_object_from_unknown,
)
from wabizabi.providers.ollama.convert import (
    normalize_ollama_response,
    request_to_ollama_messages,
    tool_definition_to_ollama_tool,
)
from wabizabi.providers.ollama.schemas import (
    OllamaChatResponseSchema,
    OllamaToolCallSchema,
)
from wabizabi.providers.ollama.settings import OllamaSettings
from wabizabi.tools import ToolDefinition
from wabizabi.types import JsonObject, JsonValue
from wabizabi.usage import RunUsage


@dataclass(frozen=True, slots=True)
class _StreamState:
    thinking: str = ""
    content: str = ""
    tool_calls: tuple[JsonObject, ...] = ()


def _json_value_list_from_objects(objects: tuple[JsonObject, ...]) -> list[JsonValue]:
    return list(objects)


def _tool_call_payload(
    tool_call: OllamaToolCallSchema,
    *,
    fallback_index: int,
) -> JsonObject:
    tool_call_index = tool_call.function.index
    if tool_call_index is None:
        tool_call_index = fallback_index

    return {
        "type": "function",
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
            "index": tool_call_index,
        },
    }


def _accumulated_message_payload(state: _StreamState) -> dict[str, JsonValue]:
    message_payload: dict[str, JsonValue] = {"role": "assistant"}
    if state.thinking:
        message_payload["thinking"] = state.thinking
    if state.content:
        message_payload["content"] = state.content
    if state.tool_calls:
        message_payload["tool_calls"] = _json_value_list_from_objects(state.tool_calls)
    return message_payload


def _final_payload_from_chunk(
    response: OllamaChatResponseSchema,
    state: _StreamState,
) -> JsonObject:
    payload = json_object_from_unknown(response)
    payload["message"] = _accumulated_message_payload(state)
    return payload


def _synthetic_final_payload(
    last_payload: JsonObject,
    state: _StreamState,
) -> JsonObject:
    payload: JsonObject = dict(last_payload)
    payload["message"] = _accumulated_message_payload(state)
    payload["done"] = True
    if "done_reason" not in payload:
        payload["done_reason"] = "stop"
    return payload


@dataclass(slots=True)
class OllamaChatModel:
    """Provider adapter for Ollama's chat API."""

    model_name: str
    host: str | None = None
    chat_fn: OllamaChatFn | None = None
    stream_chat_fn: OllamaStreamChatFn | None = None
    _resolved_chat_fn: OllamaChatFn | None = field(default=None, init=False, repr=False)
    _resolved_stream_chat_fn: OllamaStreamChatFn | None = field(
        default=None, init=False, repr=False
    )

    def _resolve_settings(
        self,
        settings: ModelSettings | None,
    ) -> tuple[str, OllamaSettings | None]:
        """Validate and resolve model name and settings for a request."""
        if settings is not None and not isinstance(settings, OllamaSettings):
            raise TypeError("OllamaChatModel requires OllamaSettings or None.")
        model_name = self.model_name
        if settings is not None and settings.ollama_model is not None:
            model_name = settings.ollama_model
        return model_name, settings

    @property
    def profile(self) -> ModelProfile:
        return ModelProfile(
            provider_name="ollama",
            model_name=self.model_name,
            supports_tools=True,
            supports_streaming=True,
        )

    async def request(
        self,
        request: ModelRequest,
        *,
        message_history: MessageHistory | None = None,
        settings: ModelSettings | None = None,
        tools: tuple[ToolDefinition, ...] = (),
        output: OutputConfig[object] | None = None,
    ) -> ModelResult:
        resolved_model_name, ollama_settings = self._resolve_settings(settings)
        messages = request_to_ollama_messages(request, message_history)
        ollama_tools = tuple(tool_definition_to_ollama_tool(tool) for tool in tools)
        payload = await self._get_chat_fn()(
            model=resolved_model_name,
            messages=messages,
            tools=ollama_tools,
            settings=ollama_settings,
            output=output,
        )
        response, usage = normalize_ollama_response(payload, output=output)
        return ModelResult(response=response, usage=usage)

    async def stream_response(
        self,
        request: ModelRequest,
        *,
        message_history: MessageHistory | None = None,
        settings: ModelSettings | None = None,
        tools: tuple[ToolDefinition, ...] = (),
        output: OutputConfig[object] | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        resolved_model_name, ollama_settings = self._resolve_settings(settings)
        messages = request_to_ollama_messages(request, message_history)
        ollama_tools = tuple(tool_definition_to_ollama_tool(tool) for tool in tools)

        state = _StreamState()
        last_payload: JsonObject | None = None
        final_response: ModelResponse | None = None
        final_usage = RunUsage.zero()

        async for payload in self._get_stream_chat_fn()(
            model=resolved_model_name,
            messages=messages,
            tools=ollama_tools,
            settings=ollama_settings,
            output=output,
        ):
            last_payload = payload
            chunk = OllamaChatResponseSchema.model_validate(payload)
            message = chunk.message

            accumulated_tool_calls = state.tool_calls
            if message.tool_calls is not None:
                accumulated_tool_calls = state.tool_calls + tuple(
                    _tool_call_payload(tool_call, fallback_index=len(state.tool_calls) + index)
                    for index, tool_call in enumerate(message.tool_calls)
                )

            if (
                message.thinking
                or message.content
                or accumulated_tool_calls is not state.tool_calls
            ):
                state = _StreamState(
                    thinking=state.thinking + (message.thinking or ""),
                    content=state.content + (message.content or ""),
                    tool_calls=accumulated_tool_calls,
                )

            if message.thinking:
                yield ModelReasoningDeltaEvent(text=message.thinking)
            if message.content:
                yield ModelTextDeltaEvent(text=message.content)

            if chunk.done and final_response is None:
                final_response, final_usage = normalize_ollama_response(
                    _final_payload_from_chunk(chunk, state),
                    output=output,
                )

        if final_response is None:
            if last_payload is None:
                raise RuntimeError("Ollama stream completed without any chunks.")
            final_response, final_usage = normalize_ollama_response(
                _synthetic_final_payload(last_payload, state),
                output=output,
            )

        for part in final_response.parts:
            yield ModelResponsePartEvent(part=part)

        yield ModelResponseCompletedEvent(
            model_name=final_response.model_name,
            usage=final_usage,
            finish_reason=final_response.finish_reason,
            metadata=final_response.metadata,
        )

    def _get_chat_fn(self) -> OllamaChatFn:
        if self._resolved_chat_fn is None:
            if self.chat_fn is not None:
                self._resolved_chat_fn = self.chat_fn
            else:
                self._resolved_chat_fn = build_default_chat(self.host)
        return self._resolved_chat_fn

    def _get_stream_chat_fn(self) -> OllamaStreamChatFn:
        if self._resolved_stream_chat_fn is None:
            if self.stream_chat_fn is not None:
                self._resolved_stream_chat_fn = self.stream_chat_fn
            else:
                self._resolved_stream_chat_fn = build_default_stream_chat(self.host)
        return self._resolved_stream_chat_fn


__all__ = [
    "OllamaChatModel",
]

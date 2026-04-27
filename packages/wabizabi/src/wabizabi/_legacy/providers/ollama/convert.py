"""Canonical message ↔ Ollama payload conversion."""

from __future__ import annotations

import json

from wabizabi.history import MessageHistory
from wabizabi.messages import (
    DocumentPart,
    FinishReason,
    ImagePart,
    ModelRequest,
    ModelResponse,
    NativeOutputPart,
    ReasoningPart,
    RefusalPart,
    RequestPart,
    ResponsePart,
    RetryFeedbackPart,
    SystemInstructionPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from wabizabi.output import OutputConfig
from wabizabi.providers.ollama.schemas import (
    OllamaChatResponseSchema,
    OllamaMessageSchema,
)
from wabizabi.tools import ToolDefinition
from wabizabi.types import JsonObject, JsonValue, json_value_from_unknown
from wabizabi.usage import RunUsage


def serialize_json_value(value: JsonValue) -> str:
    """Compact JSON serialization."""
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _tool_return_content(part: ToolReturnPart) -> str:
    if part.is_error:
        return serialize_json_value({"is_error": True, "content": part.content})
    if isinstance(part.content, str):
        return part.content
    return serialize_json_value(part.content)


def request_part_to_ollama_messages(part: RequestPart) -> tuple[JsonObject, ...]:
    """Convert one canonical request part into Ollama message dicts."""
    if isinstance(part, SystemInstructionPart):
        return ({"role": "system", "content": part.text},)
    if isinstance(part, UserPromptPart):
        return ({"role": "user", "content": part.text},)
    if isinstance(part, RetryFeedbackPart):
        return ({"role": "user", "content": part.message},)
    if isinstance(part, ToolReturnPart):
        return (
            {
                "role": "tool",
                "tool_name": part.tool_name,
                "content": _tool_return_content(part),
            },
        )
    if isinstance(part, ImagePart):
        return ({"role": "user", "content": "", "images": [part.source]},)
    if isinstance(part, DocumentPart):
        if part.source_kind == "text":
            return ({"role": "user", "content": part.source},)
        raise NotImplementedError(
            "OllamaChatModel does not yet support non-text DocumentPart inputs."
        )
    raise TypeError(f"Unsupported request part for Ollama chat: {type(part)!r}")


def response_to_ollama_messages(response: ModelResponse) -> tuple[JsonObject, ...]:
    """Convert a canonical response into Ollama-format assistant messages."""
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[JsonValue] = []

    for index, part in enumerate(response.parts):
        if isinstance(part, TextPart | RefusalPart):
            content_parts.append(part.text)
            continue
        if isinstance(part, ReasoningPart):
            thinking_parts.append(part.text)
            continue
        if isinstance(part, NativeOutputPart):
            content_parts.append(serialize_json_value(part.data))
            continue
        if isinstance(part, ToolCallPart):
            tool_calls.append(
                {
                    "type": "function",
                    "function": {
                        "index": index,
                        "name": part.tool_name,
                        "arguments": part.arguments,
                    },
                }
            )
            continue
        raise TypeError(f"Unsupported response part for Ollama chat: {type(part)!r}")

    message: dict[str, JsonValue] = {"role": "assistant"}
    if content_parts:
        message["content"] = "".join(content_parts)
    if thinking_parts:
        message["thinking"] = "".join(thinking_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return (message,)


def message_to_ollama_messages(message: ModelRequest | ModelResponse) -> tuple[JsonObject, ...]:
    """Convert a canonical message into Ollama message dicts."""
    if isinstance(message, ModelRequest):
        items: list[JsonObject] = []
        for part in message.parts:
            items.extend(request_part_to_ollama_messages(part))
        return tuple(items)
    return response_to_ollama_messages(message)


def request_to_ollama_messages(
    request: ModelRequest,
    message_history: MessageHistory | None,
) -> tuple[JsonObject, ...]:
    """Build the full Ollama messages list from history + current request."""
    messages: list[JsonObject] = []
    if message_history is not None:
        for message in message_history.messages:
            messages.extend(message_to_ollama_messages(message))
    messages.extend(message_to_ollama_messages(request))
    return tuple(messages)


def tool_definition_to_ollama_tool(definition: ToolDefinition) -> JsonObject:
    """Convert a canonical tool definition into Ollama tool format."""
    function: dict[str, JsonValue] = {
        "name": definition.name,
        "parameters": definition.input_schema,
    }
    if definition.description is not None:
        function["description"] = definition.description

    return {
        "type": "function",
        "function": function,
    }


def tool_call_part(
    *,
    name: str,
    arguments: JsonObject,
    index: int,
) -> ToolCallPart:
    """Build a canonical tool call part from Ollama data."""
    return ToolCallPart(
        tool_name=name,
        call_id=f"ollama-call-{index}",
        arguments=arguments,
    )


def _structured_native_part(
    content: str,
) -> NativeOutputPart | None:
    try:
        return NativeOutputPart(data=json_value_from_unknown(json.loads(content)))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def parts_from_message(
    message: OllamaMessageSchema,
    *,
    output: OutputConfig[object] | None = None,
) -> tuple[ResponsePart, ...]:
    """Extract canonical response parts from an Ollama message."""
    parts: list[ResponsePart] = []
    if message.thinking:
        parts.append(ReasoningPart(text=message.thinking))

    content = message.content or ""
    if content:
        native_output: NativeOutputPart | None = None
        if output is not None and output.response_format is not None:
            native_output = _structured_native_part(content)
        if native_output is not None:
            parts.append(native_output)
        else:
            parts.append(TextPart(text=content))

    if message.tool_calls is not None:
        for index, tc in enumerate(message.tool_calls):
            parts.append(
                tool_call_part(
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                    index=tc.function.index if tc.function.index is not None else index,
                )
            )
    return tuple(parts)


def usage_from_response(response: OllamaChatResponseSchema) -> RunUsage:
    """Extract token usage from an Ollama response."""
    return RunUsage(
        input_tokens=response.prompt_eval_count or 0,
        output_tokens=response.eval_count or 0,
    )


def metadata_from_response(response: OllamaChatResponseSchema) -> JsonObject | None:
    """Extract provider metadata from an Ollama response."""
    metadata: dict[str, JsonValue] = {}
    if response.created_at is not None:
        metadata["ollama_created_at"] = response.created_at
    if response.done_reason is not None:
        metadata["ollama_done_reason"] = response.done_reason
    if response.total_duration is not None:
        metadata["ollama_total_duration"] = response.total_duration
    if response.load_duration is not None:
        metadata["ollama_load_duration"] = response.load_duration
    if response.prompt_eval_duration is not None:
        metadata["ollama_prompt_eval_duration"] = response.prompt_eval_duration
    if response.eval_duration is not None:
        metadata["ollama_eval_duration"] = response.eval_duration
    return metadata or None


def finish_reason_from_response(
    response: OllamaChatResponseSchema,
    parts: tuple[ResponsePart, ...],
) -> FinishReason | None:
    """Map an Ollama done_reason to a canonical FinishReason."""
    if any(isinstance(part, ToolCallPart) for part in parts):
        return FinishReason.TOOL_CALLS

    done_reason = response.done_reason
    if done_reason in {"stop", None}:
        return FinishReason.STOP if response.done else None
    if done_reason in {"length", "max_tokens", "max_output_tokens"}:
        return FinishReason.LENGTH
    if done_reason == "content_filter":
        return FinishReason.CONTENT_FILTER
    if done_reason == "error":
        return FinishReason.ERROR
    return FinishReason.STOP if response.done else None


def normalize_ollama_response(
    payload: JsonObject,
    *,
    output: OutputConfig[object] | None = None,
) -> tuple[ModelResponse, RunUsage]:
    """Parse an Ollama chat response payload into canonical types."""
    response = OllamaChatResponseSchema.model_validate(payload)
    parts = parts_from_message(response.message, output=output)
    if not parts:
        raise ValueError("Ollama response did not contain supported output content.")

    model_response = ModelResponse(
        parts=parts,
        model_name=response.model,
        finish_reason=finish_reason_from_response(response, parts),
        metadata=metadata_from_response(response),
    )
    return model_response, usage_from_response(response)


__all__ = [
    "finish_reason_from_response",
    "message_to_ollama_messages",
    "metadata_from_response",
    "normalize_ollama_response",
    "parts_from_message",
    "request_part_to_ollama_messages",
    "request_to_ollama_messages",
    "response_to_ollama_messages",
    "serialize_json_value",
    "tool_call_part",
    "tool_definition_to_ollama_tool",
    "usage_from_response",
]

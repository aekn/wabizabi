"""Small builders for public tests and examples."""

from __future__ import annotations

from wabizabi.messages import (
    FinishReason,
    ModelResponse,
    RefusalPart,
    ResponsePart,
    TextPart,
    ToolCallPart,
)
from wabizabi.models import ModelResult, ModelStreamEvent, model_result_events
from wabizabi.types import JsonObject
from wabizabi.usage import RunUsage


def response_message(
    parts: tuple[ResponsePart, ...],
    *,
    model_name: str | None = "scripted",
    finish_reason: FinishReason | None = None,
    metadata: JsonObject | None = None,
) -> ModelResponse:
    """Build a canonical response from explicit parts."""

    return ModelResponse(
        parts=parts,
        model_name=model_name,
        finish_reason=finish_reason,
        metadata=metadata,
    )


def response_result(
    parts: tuple[ResponsePart, ...],
    *,
    model_name: str | None = "scripted",
    usage: RunUsage | None = None,
    finish_reason: FinishReason | None = None,
    metadata: JsonObject | None = None,
) -> ModelResult:
    """Build a one-shot result from explicit response parts."""

    return ModelResult(
        response=response_message(
            parts,
            model_name=model_name,
            finish_reason=finish_reason,
            metadata=metadata,
        ),
        usage=usage or RunUsage.zero(),
    )


def text_response(
    text: str,
    *,
    model_name: str | None = "scripted",
    finish_reason: FinishReason | None = None,
    metadata: JsonObject | None = None,
) -> ModelResponse:
    """Build a simple text response."""

    return response_message(
        (TextPart(text=text),),
        model_name=model_name,
        finish_reason=finish_reason,
        metadata=metadata,
    )


def text_result(
    text: str,
    *,
    model_name: str | None = "scripted",
    usage: RunUsage | None = None,
    finish_reason: FinishReason | None = None,
    metadata: JsonObject | None = None,
) -> ModelResult:
    """Build a one-shot text model result."""

    return response_result(
        (TextPart(text=text),),
        model_name=model_name,
        usage=usage,
        finish_reason=finish_reason,
        metadata=metadata,
    )


def tool_call_response(
    tool_name: str,
    *,
    call_id: str,
    arguments: JsonObject,
    model_name: str | None = "scripted",
    finish_reason: FinishReason | None = None,
    metadata: JsonObject | None = None,
) -> ModelResponse:
    """Build a single tool-call response."""

    return response_message(
        (
            ToolCallPart(
                tool_name=tool_name,
                call_id=call_id,
                arguments=arguments,
            ),
        ),
        model_name=model_name,
        finish_reason=finish_reason,
        metadata=metadata,
    )


def tool_call_result(
    tool_name: str,
    *,
    call_id: str,
    arguments: JsonObject,
    model_name: str | None = "scripted",
    usage: RunUsage | None = None,
    finish_reason: FinishReason | None = None,
    metadata: JsonObject | None = None,
) -> ModelResult:
    """Build a one-shot tool-call model result."""

    return response_result(
        (
            ToolCallPart(
                tool_name=tool_name,
                call_id=call_id,
                arguments=arguments,
            ),
        ),
        model_name=model_name,
        usage=usage,
        finish_reason=finish_reason,
        metadata=metadata,
    )


def refusal_response(
    text: str,
    *,
    model_name: str | None = "scripted",
    finish_reason: FinishReason | None = None,
    metadata: JsonObject | None = None,
) -> ModelResponse:
    """Build a single refusal response."""

    return response_message(
        (RefusalPart(text=text),),
        model_name=model_name,
        finish_reason=finish_reason,
        metadata=metadata,
    )


def refusal_result(
    text: str,
    *,
    model_name: str | None = "scripted",
    usage: RunUsage | None = None,
    finish_reason: FinishReason | None = None,
    metadata: JsonObject | None = None,
) -> ModelResult:
    """Build a one-shot refusal model result."""

    return response_result(
        (RefusalPart(text=text),),
        model_name=model_name,
        usage=usage,
        finish_reason=finish_reason,
        metadata=metadata,
    )


def stream_script_from_result(result: ModelResult) -> tuple[ModelStreamEvent, ...]:
    """Project a one-shot result into a canonical stream script."""

    return model_result_events(result)


__all__ = [
    "refusal_response",
    "refusal_result",
    "response_message",
    "response_result",
    "stream_script_from_result",
    "text_response",
    "text_result",
    "tool_call_response",
    "tool_call_result",
]

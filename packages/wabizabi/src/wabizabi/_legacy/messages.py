"""Canonical model message and message-part types."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from wabizabi.types import JsonObject, JsonValue


class FinishReason(StrEnum):
    """Why a model response ended."""

    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SystemInstructionPart(_FrozenModel):
    """A system instruction sent to the model."""

    part_kind: Literal["system_instruction"] = "system_instruction"
    text: str = Field(min_length=1)


class UserPromptPart(_FrozenModel):
    """A user-authored text prompt."""

    part_kind: Literal["user_prompt"] = "user_prompt"
    text: str = Field(min_length=1)


class ToolReturnPart(_FrozenModel):
    """The return value from a tool execution."""

    part_kind: Literal["tool_return"] = "tool_return"
    tool_name: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    content: JsonValue
    is_error: bool = False


class RetryFeedbackPart(_FrozenModel):
    """Feedback sent back to the model after validation failure."""

    part_kind: Literal["retry_feedback"] = "retry_feedback"
    message: str = Field(min_length=1)


class ImagePart(_FrozenModel):
    """An image included in a request."""

    part_kind: Literal["image"] = "image"
    source_kind: Literal["url", "base64"] = "url"
    source: str = Field(min_length=1)
    media_type: str = Field(min_length=1)


class DocumentPart(_FrozenModel):
    """A document included in a request."""

    part_kind: Literal["document"] = "document"
    source_kind: Literal["url", "base64", "text"] = "text"
    source: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    name: str | None = None


class TextPart(_FrozenModel):
    """Plain text emitted by the model."""

    part_kind: Literal["text"] = "text"
    text: str = Field(min_length=1)


class ReasoningPart(_FrozenModel):
    """Reasoning or thinking content emitted by the model."""

    part_kind: Literal["reasoning"] = "reasoning"
    text: str = Field(min_length=1)


class ToolCallPart(_FrozenModel):
    """A request from the model to execute a tool."""

    part_kind: Literal["tool_call"] = "tool_call"
    tool_name: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    arguments: JsonObject


class NativeOutputPart(_FrozenModel):
    """Provider-native structured output returned by the model."""

    part_kind: Literal["native_output"] = "native_output"
    data: JsonValue


class RefusalPart(_FrozenModel):
    """A refusal emitted by the model."""

    part_kind: Literal["refusal"] = "refusal"
    text: str = Field(min_length=1)


type RequestPart = Annotated[
    SystemInstructionPart
    | UserPromptPart
    | ToolReturnPart
    | RetryFeedbackPart
    | ImagePart
    | DocumentPart,
    Field(discriminator="part_kind"),
]

type ResponsePart = Annotated[
    TextPart | ReasoningPart | ToolCallPart | NativeOutputPart | RefusalPart,
    Field(discriminator="part_kind"),
]


class ModelRequest(_FrozenModel):
    """A normalized request sent to a model adapter."""

    message_kind: Literal["request"] = "request"
    parts: tuple[RequestPart, ...]
    metadata: JsonObject | None = None

    @model_validator(mode="after")
    def validate_parts_not_empty(self) -> Self:
        if not self.parts:
            raise ValueError("parts must not be empty.")
        return self


class ModelResponse(_FrozenModel):
    """A normalized response returned from a model adapter."""

    message_kind: Literal["response"] = "response"
    parts: tuple[ResponsePart, ...]
    model_name: str | None = None
    finish_reason: FinishReason | None = None
    metadata: JsonObject | None = None

    @model_validator(mode="after")
    def validate_parts_not_empty(self) -> Self:
        if not self.parts:
            raise ValueError("parts must not be empty.")
        return self


type ModelMessage = Annotated[
    ModelRequest | ModelResponse,
    Field(discriminator="message_kind"),
]


class RequestPartModel(RootModel[RequestPart]):
    """Typed validation wrapper for request parts."""


class ResponsePartModel(RootModel[ResponsePart]):
    """Typed validation wrapper for response parts."""


class MessageModel(RootModel[ModelMessage]):
    """Typed validation wrapper for canonical model messages."""


def validate_request_part(value: object) -> RequestPart:
    """Validate a request part from Python data."""

    return RequestPartModel.model_validate(value).root


def validate_response_part(value: object) -> ResponsePart:
    """Validate a response part from Python data."""

    return ResponsePartModel.model_validate(value).root


def validate_message(value: object) -> ModelMessage:
    """Validate a model message from Python data."""

    return MessageModel.model_validate(value).root


__all__ = [
    "DocumentPart",
    "FinishReason",
    "ImagePart",
    "MessageModel",
    "ModelMessage",
    "ModelRequest",
    "ModelResponse",
    "NativeOutputPart",
    "ReasoningPart",
    "RefusalPart",
    "RequestPart",
    "RequestPartModel",
    "ResponsePart",
    "ResponsePartModel",
    "RetryFeedbackPart",
    "SystemInstructionPart",
    "TextPart",
    "ToolCallPart",
    "ToolReturnPart",
    "UserPromptPart",
    "validate_message",
    "validate_request_part",
    "validate_response_part",
]

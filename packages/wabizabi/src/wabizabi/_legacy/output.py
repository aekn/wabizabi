"""Structured output modes, decoders, output configs, and validation pipeline."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, TypeAdapter

from wabizabi._async import resolve
from wabizabi.context import RunContext
from wabizabi.messages import ModelResponse, NativeOutputPart, RefusalPart, TextPart, ToolCallPart
from wabizabi.types import JsonObject, JsonValue

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)
_JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)

type ResponseFormat = Literal["json"] | JsonObject | None


class OutputMode(StrEnum):
    """How an agent should interpret its final output."""

    TEXT = "text"
    TOOL = "tool"
    JSON = "json"
    SCHEMA = "schema"


class OutputValidationError(ValueError):
    """Raised when output validation fails and a retry should be considered."""

    def __init__(self, message: str, *, retry_feedback: str | None = None) -> None:
        super().__init__(message)
        self.retry_feedback = message if retry_feedback is None else retry_feedback


class OutputDecodingError(ValueError):
    """Raised when a model response cannot be decoded into the requested output."""


class OutputDecoder[OutputDataT](Protocol):
    """Decode a normalized model response into typed output."""

    def decode(self, response: ModelResponse) -> OutputDataT: ...


@dataclass(frozen=True, slots=True)
class TextOutputDecoder:
    """Decode text output from a normalized model response."""

    separator: str = ""

    def decode(self, response: ModelResponse) -> str:
        return self.separator.join(_extract_text_parts(response))


@dataclass(frozen=True, slots=True)
class JsonOutputDecoder:
    """Decode JSON output from a structured-response model output."""

    def decode(self, response: ModelResponse) -> JsonValue:
        return _structured_json_value(response)


@dataclass(frozen=True, slots=True)
class SchemaOutputDecoder[OutputDataT: BaseModel]:
    """Decode a Pydantic model from structured JSON output."""

    output_type: type[OutputDataT]

    def decode(self, response: ModelResponse) -> OutputDataT:
        return self.output_type.model_validate(_structured_json_value(response))


@dataclass(frozen=True, slots=True)
class ToolArgumentsOutputDecoder[OutputDataT: BaseModel]:
    """Decode a Pydantic model from tool-call arguments."""

    output_type: type[OutputDataT]
    tool_name: str | None = None

    def decode(self, response: ModelResponse) -> OutputDataT:
        tool_calls: list[ToolCallPart] = []
        for part in response.parts:
            if isinstance(part, ToolCallPart) and (
                self.tool_name is None or part.tool_name == self.tool_name
            ):
                tool_calls.append(part)

        if not tool_calls:
            if self.tool_name is None:
                raise OutputDecodingError("ModelResponse did not contain any tool call parts.")
            raise OutputDecodingError(
                f"ModelResponse did not contain a tool call for {self.tool_name!r}."
            )

        if len(tool_calls) > 1:
            if self.tool_name is None:
                raise OutputDecodingError(
                    "ModelResponse contained multiple tool calls; a tool_name is required."
                )
            raise OutputDecodingError(
                f"ModelResponse contained multiple tool calls for {self.tool_name!r}."
            )

        return self.output_type.model_validate(tool_calls[0].arguments)


@dataclass(frozen=True, slots=True)
class OutputConfig[OutputDataT]:
    """A first-class agent output configuration."""

    mode: OutputMode
    decoder: OutputDecoder[OutputDataT]
    response_format: ResponseFormat = None
    terminal_tool_names: frozenset[str] = field(default_factory=lambda: frozenset[str]())


def _extract_text_parts(response: ModelResponse) -> tuple[str, ...]:
    text_parts = tuple(
        part.text for part in response.parts if isinstance(part, TextPart | RefusalPart)
    )
    if not text_parts:
        raise OutputDecodingError("ModelResponse did not contain any text parts.")
    return text_parts


def _text_content(response: ModelResponse) -> str:
    return "".join(_extract_text_parts(response))


def _structured_json_value(response: ModelResponse) -> JsonValue:
    native_parts: list[NativeOutputPart] = []
    for part in response.parts:
        if isinstance(part, NativeOutputPart):
            native_parts.append(part)

    if native_parts:
        if len(native_parts) > 1:
            raise OutputDecodingError("ModelResponse contained multiple native output parts.")
        return native_parts[0].data

    raw_text = _text_content(response)
    try:
        return _JSON_VALUE_ADAPTER.validate_python(json.loads(raw_text))
    except json.JSONDecodeError as error:
        raise OutputDecodingError(
            f"ModelResponse did not contain valid structured JSON output: {error.msg}."
        ) from error


def infer_output_mode[OutputDataT](decoder: OutputDecoder[OutputDataT]) -> OutputMode:
    """Infer an output mode from a built-in decoder type.

    Raises ``TypeError`` for custom decoders — pass an explicit
    :class:`OutputConfig` via ``output=`` instead of ``decoder=``.
    """

    if isinstance(decoder, TextOutputDecoder):
        return OutputMode.TEXT
    if isinstance(decoder, ToolArgumentsOutputDecoder):
        return OutputMode.TOOL
    if isinstance(decoder, JsonOutputDecoder):
        return OutputMode.JSON
    if isinstance(decoder, SchemaOutputDecoder):
        return OutputMode.SCHEMA
    raise TypeError(
        f"Cannot infer output mode for decoder {type(decoder).__name__!r}. "
        "Pass an explicit OutputConfig via `output=...` instead of `decoder=...`."
    )


def json_output_decoder() -> JsonOutputDecoder:
    return JsonOutputDecoder()


def schema_output_decoder[OutputDataT: BaseModel](
    output_type: type[OutputDataT],
) -> SchemaOutputDecoder[OutputDataT]:
    return SchemaOutputDecoder(output_type=output_type)


def tool_output_decoder[OutputDataT: BaseModel](
    output_type: type[OutputDataT],
    *,
    tool_name: str | None = None,
) -> ToolArgumentsOutputDecoder[OutputDataT]:
    return ToolArgumentsOutputDecoder(output_type=output_type, tool_name=tool_name)


def _schema_response_format[OutputDataT: BaseModel](output_type: type[OutputDataT]) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(output_type.model_json_schema())


def text_output_config(*, separator: str = "") -> OutputConfig[str]:
    return OutputConfig(
        mode=OutputMode.TEXT,
        decoder=TextOutputDecoder(separator=separator),
        response_format=None,
    )


def json_output_config() -> OutputConfig[JsonValue]:
    return OutputConfig(
        mode=OutputMode.JSON,
        decoder=json_output_decoder(),
        response_format="json",
    )


def schema_output_config[OutputDataT: BaseModel](
    output_type: type[OutputDataT],
) -> OutputConfig[OutputDataT]:
    return OutputConfig(
        mode=OutputMode.SCHEMA,
        decoder=schema_output_decoder(output_type),
        response_format=_schema_response_format(output_type),
    )


def tool_output_config[OutputDataT: BaseModel](
    output_type: type[OutputDataT],
    *,
    tool_name: str,
) -> OutputConfig[OutputDataT]:
    return OutputConfig(
        mode=OutputMode.TOOL,
        decoder=tool_output_decoder(output_type, tool_name=tool_name),
        response_format=None,
        terminal_tool_names=frozenset((tool_name,)),
    )


type OutputValidatorLike[AgentDepsT, OutputDataT] = Callable[
    [RunContext[AgentDepsT], OutputDataT],
    OutputDataT | Awaitable[OutputDataT],
]

type OutputValidator[AgentDepsT, OutputDataT] = Callable[
    [RunContext[AgentDepsT], OutputDataT],
    Awaitable[OutputDataT],
]


def normalize_output_validator[AgentDepsT, OutputDataT](
    validator: OutputValidatorLike[AgentDepsT, OutputDataT],
) -> OutputValidator[AgentDepsT, OutputDataT]:
    async def wrapped(
        context: RunContext[AgentDepsT],
        output: OutputDataT,
    ) -> OutputDataT:
        return await resolve(validator(context, output))

    return wrapped


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputPipeline[AgentDepsT, OutputDataT]:
    """Run a sequence of typed output validators."""

    mode: OutputMode = OutputMode.TEXT
    validators: tuple[OutputValidator[AgentDepsT, OutputDataT], ...] = ()

    @classmethod
    def empty(
        cls,
        *,
        mode: OutputMode = OutputMode.TEXT,
    ) -> OutputPipeline[AgentDepsT, OutputDataT]:
        return cls(mode=mode)

    def with_validator(
        self,
        validator: OutputValidatorLike[AgentDepsT, OutputDataT],
    ) -> OutputPipeline[AgentDepsT, OutputDataT]:
        return replace(self, validators=(*self.validators, normalize_output_validator(validator)))

    async def validate(
        self,
        context: RunContext[AgentDepsT],
        output: OutputDataT,
    ) -> OutputDataT:
        current = output
        for validator in self.validators:
            current = await validator(context, current)
        return current


__all__ = [
    "JsonOutputDecoder",
    "OutputConfig",
    "OutputDecoder",
    "OutputDecodingError",
    "OutputMode",
    "OutputPipeline",
    "OutputValidationError",
    "OutputValidatorLike",
    "SchemaOutputDecoder",
    "TextOutputDecoder",
    "ToolArgumentsOutputDecoder",
    "infer_output_mode",
    "json_output_config",
    "json_output_decoder",
    "normalize_output_validator",
    "schema_output_config",
    "schema_output_decoder",
    "text_output_config",
    "tool_output_config",
    "tool_output_decoder",
]

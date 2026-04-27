"""Typed telemetry events for agent runs."""

from __future__ import annotations

from dataclasses import dataclass

from wabizabi.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from wabizabi.state import RunState


@dataclass(frozen=True, slots=True)
class RunStartedEvent:
    """An agent run has started."""

    state: RunState


@dataclass(frozen=True, slots=True)
class RequestRecordedEvent:
    """A model request is about to be sent."""

    state: RunState
    request: ModelRequest


@dataclass(frozen=True, slots=True)
class ResponseRecordedEvent:
    """A model response was received."""

    state: RunState
    response: ModelResponse


@dataclass(frozen=True, slots=True)
class ToolCallRecordedEvent:
    """A tool call was requested by the model."""

    state: RunState
    tool_call: ToolCallPart


@dataclass(frozen=True, slots=True)
class ToolResultRecordedEvent:
    """A tool result was produced."""

    state: RunState
    tool_return: ToolReturnPart


@dataclass(frozen=True, slots=True)
class OutputValidationFailedEvent:
    """Decoded output failed validation and a retry will be attempted or raised."""

    state: RunState
    error_message: str
    retry_feedback: str


@dataclass(frozen=True, slots=True)
class OutputDecodingFailedEvent:
    """Model output shape could not be decoded and a retry will be attempted or raised."""

    state: RunState
    error_message: str
    retry_feedback: str


@dataclass(frozen=True, slots=True)
class OutputRecordedEvent[OutputDataT]:
    """A final validated output was produced."""

    state: RunState
    output: OutputDataT


@dataclass(frozen=True, slots=True)
class HandoffRecordedEvent:
    """An agent run terminated with a handoff."""

    state: RunState
    handoff_name: str
    tool_call: ToolCallPart


@dataclass(frozen=True, slots=True)
class RunFinishedEvent[OutputDataT]:
    """An agent run finished successfully."""

    state: RunState
    output: OutputDataT


@dataclass(frozen=True, slots=True)
class RunFailedEvent:
    """An agent run failed with an exception."""

    state: RunState
    error_message: str


type TelemetryEvent[OutputDataT] = (
    RunStartedEvent
    | RequestRecordedEvent
    | ResponseRecordedEvent
    | ToolCallRecordedEvent
    | ToolResultRecordedEvent
    | OutputValidationFailedEvent
    | OutputDecodingFailedEvent
    | OutputRecordedEvent[OutputDataT]
    | HandoffRecordedEvent
    | RunFinishedEvent[OutputDataT]
    | RunFailedEvent
)

__all__ = [
    "HandoffRecordedEvent",
    "OutputDecodingFailedEvent",
    "OutputRecordedEvent",
    "OutputValidationFailedEvent",
    "RequestRecordedEvent",
    "ResponseRecordedEvent",
    "RunFailedEvent",
    "RunFinishedEvent",
    "RunStartedEvent",
    "TelemetryEvent",
    "ToolCallRecordedEvent",
    "ToolResultRecordedEvent",
]

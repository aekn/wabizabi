"""Streaming run interfaces and event projections."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from wabizabi.messages import (
    ModelRequest,
    ModelResponse,
    ReasoningPart,
    RefusalPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from wabizabi.state import RunState


@dataclass(frozen=True, slots=True)
class RequestEvent:
    """A request is about to be sent to the model."""

    state: RunState
    request: ModelRequest


@dataclass(frozen=True, slots=True)
class ResponseEvent:
    """A response was received from the model."""

    state: RunState
    response: ModelResponse


@dataclass(frozen=True, slots=True)
class ToolCallEvent:
    """The model requested a tool call."""

    state: RunState
    tool_call: ToolCallPart


@dataclass(frozen=True, slots=True)
class ToolResultEvent:
    """A tool result was produced and will be sent back to the model."""

    state: RunState
    tool_return: ToolReturnPart


@dataclass(frozen=True, slots=True)
class TextChunkEvent:
    """A text chunk projected from a model response."""

    state: RunState
    text: str


@dataclass(frozen=True, slots=True)
class ReasoningChunkEvent:
    """A reasoning/thinking chunk projected from a model response."""

    state: RunState
    text: str


@dataclass(frozen=True, slots=True)
class OutputEvent[OutputDataT]:
    """A final validated output was produced for the run."""

    state: RunState
    output: OutputDataT


@dataclass(frozen=True, slots=True)
class HandoffEvent:
    """The agent run terminated with a handoff to another agent."""

    state: RunState
    handoff_name: str
    tool_call: ToolCallPart


type RunEvent[OutputDataT] = (
    RequestEvent
    | ResponseEvent
    | ToolCallEvent
    | ToolResultEvent
    | TextChunkEvent
    | ReasoningChunkEvent
    | OutputEvent[OutputDataT]
    | HandoffEvent
)

type EventIteratorFactory[OutputDataT] = Callable[[], AsyncIterator[RunEvent[OutputDataT]]]


@dataclass(slots=True)
class StreamedRun[OutputDataT]:
    """A single-consumer wrapper over a run event stream."""

    events_factory: EventIteratorFactory[OutputDataT]
    _started: bool = field(default=False, init=False, repr=False)

    def __aiter__(self) -> AsyncIterator[RunEvent[OutputDataT]]:
        return self._take_iterator()

    async def responses(self) -> AsyncIterator[ModelResponse]:
        """Yield model responses from the run."""
        async for event in self._take_iterator():
            if isinstance(event, ResponseEvent):
                yield event.response

    async def outputs(self) -> AsyncIterator[OutputDataT]:
        """Yield validated outputs from the run."""
        async for event in self._take_iterator():
            if isinstance(event, OutputEvent):
                yield event.output

    async def text(self) -> AsyncIterator[str]:
        """Yield projected text chunks from the run."""
        async for event in self._take_iterator():
            if isinstance(event, TextChunkEvent):
                yield event.text

    async def reasoning(self) -> AsyncIterator[str]:
        """Yield projected reasoning/thinking chunks from the run."""
        async for event in self._take_iterator():
            if isinstance(event, ReasoningChunkEvent):
                yield event.text

    def _take_iterator(self) -> AsyncIterator[RunEvent[OutputDataT]]:
        if self._started:
            raise RuntimeError("This StreamedRun has already been consumed.")
        self._started = True
        return self.events_factory()


def text_chunk_events[OutputDataT](
    event: RunEvent[OutputDataT],
) -> tuple[TextChunkEvent, ...]:
    """Project text chunk events from a run event."""

    if not isinstance(event, ResponseEvent):
        return ()

    chunks: list[TextChunkEvent] = []
    for part in event.response.parts:
        if isinstance(part, TextPart | RefusalPart) and part.text:
            chunks.append(TextChunkEvent(state=event.state, text=part.text))
    return tuple(chunks)


def reasoning_chunk_events[OutputDataT](
    event: RunEvent[OutputDataT],
) -> tuple[ReasoningChunkEvent, ...]:
    """Project reasoning chunk events from a run event."""

    if not isinstance(event, ResponseEvent):
        return ()

    chunks: list[ReasoningChunkEvent] = []
    for part in event.response.parts:
        if isinstance(part, ReasoningPart) and part.text:
            chunks.append(ReasoningChunkEvent(state=event.state, text=part.text))
    return tuple(chunks)


__all__ = [
    "HandoffEvent",
    "OutputEvent",
    "ReasoningChunkEvent",
    "RequestEvent",
    "ResponseEvent",
    "RunEvent",
    "StreamedRun",
    "TextChunkEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "reasoning_chunk_events",
    "text_chunk_events",
]

"""Incremental assembly of streamed model responses."""

from __future__ import annotations

from dataclasses import dataclass, field

from wabizabi.messages import FinishReason, ModelResponse, ResponsePart
from wabizabi.models.stream import (
    ModelReasoningDeltaEvent,
    ModelResponseCompletedEvent,
    ModelResponsePartEvent,
    ModelStreamEvent,
    ModelTextDeltaEvent,
)
from wabizabi.types import JsonObject
from wabizabi.usage import RunUsage


def _new_parts() -> list[ResponsePart]:
    return []


@dataclass(frozen=True, slots=True)
class ResponseAccumulatorUpdate:
    """The externally visible result of applying one stream event."""

    text_delta: str | None = None
    reasoning_delta: str | None = None
    finalized_parts: tuple[ResponsePart, ...] = ()
    completed: bool = False


@dataclass(slots=True)
class ResponseAccumulator:
    """Assemble a canonical :class:`ModelResponse` from stream events."""

    _parts: list[ResponsePart] = field(default_factory=_new_parts)
    _model_name: str | None = None
    _usage: RunUsage = field(default_factory=RunUsage.zero)
    _finish_reason: FinishReason | None = None
    _metadata: JsonObject | None = None
    _completed: bool = False

    def add(self, event: ModelStreamEvent) -> ResponseAccumulatorUpdate:
        if isinstance(event, ModelTextDeltaEvent):
            return ResponseAccumulatorUpdate(text_delta=event.text)

        if isinstance(event, ModelReasoningDeltaEvent):
            return ResponseAccumulatorUpdate(reasoning_delta=event.text)

        if isinstance(event, ModelResponsePartEvent):
            self._parts.append(event.part)
            return ResponseAccumulatorUpdate(finalized_parts=(event.part,))

        if isinstance(event, ModelResponseCompletedEvent):
            self._model_name = event.model_name
            self._usage = event.usage
            self._finish_reason = event.finish_reason
            self._metadata = event.metadata
            self._completed = True
            return ResponseAccumulatorUpdate(completed=True)

        return ResponseAccumulatorUpdate()

    @property
    def usage(self) -> RunUsage:
        return self._usage

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @property
    def finish_reason(self) -> FinishReason | None:
        return self._finish_reason

    @property
    def metadata(self) -> JsonObject | None:
        return self._metadata

    @property
    def parts(self) -> tuple[ResponsePart, ...]:
        return tuple(self._parts)

    def build_response(self) -> ModelResponse:
        if not self._completed:
            raise RuntimeError("Cannot build response before completion.")
        if not self._parts:
            raise RuntimeError("Model stream completed without response parts.")

        return ModelResponse(
            parts=tuple(self._parts),
            model_name=self._model_name,
            finish_reason=self._finish_reason,
            metadata=self._metadata,
        )


__all__ = [
    "ResponseAccumulator",
    "ResponseAccumulatorUpdate",
]

"""Provider-neutral model streaming events."""

from __future__ import annotations

from dataclasses import dataclass

from wabizabi.messages import FinishReason, ResponsePart
from wabizabi.models.base import ModelResult
from wabizabi.types import JsonObject
from wabizabi.usage import RunUsage


@dataclass(frozen=True, slots=True)
class ModelTextDeltaEvent:
    """A streamed text delta from a model."""

    text: str


@dataclass(frozen=True, slots=True)
class ModelReasoningDeltaEvent:
    """A streamed reasoning/thinking delta from a model."""

    text: str


@dataclass(frozen=True, slots=True)
class ModelResponsePartEvent:
    """A finalized canonical response part emitted during streaming."""

    part: ResponsePart


@dataclass(frozen=True, slots=True)
class ModelResponseCompletedEvent:
    """A streamed response completed with canonical response metadata."""

    model_name: str | None
    usage: RunUsage
    finish_reason: FinishReason | None = None
    metadata: JsonObject | None = None


type ModelStreamEvent = (
    ModelTextDeltaEvent
    | ModelReasoningDeltaEvent
    | ModelResponsePartEvent
    | ModelResponseCompletedEvent
)


def model_result_events(result: ModelResult) -> tuple[ModelStreamEvent, ...]:
    """Project a one-shot model result into canonical stream events."""

    part_events = tuple(ModelResponsePartEvent(part=part) for part in result.response.parts)
    return (
        *part_events,
        ModelResponseCompletedEvent(
            model_name=result.response.model_name,
            usage=result.usage,
            finish_reason=result.response.finish_reason,
            metadata=result.response.metadata,
        ),
    )


__all__ = [
    "ModelReasoningDeltaEvent",
    "ModelResponseCompletedEvent",
    "ModelResponsePartEvent",
    "ModelStreamEvent",
    "ModelTextDeltaEvent",
    "model_result_events",
]

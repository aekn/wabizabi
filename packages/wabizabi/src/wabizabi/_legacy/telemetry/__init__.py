"""Typed telemetry event surface for Wabizabi."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from wabizabi.telemetry.events import (
    HandoffRecordedEvent,
    OutputDecodingFailedEvent,
    OutputRecordedEvent,
    OutputValidationFailedEvent,
    RequestRecordedEvent,
    ResponseRecordedEvent,
    RunFailedEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TelemetryEvent,
    ToolCallRecordedEvent,
    ToolResultRecordedEvent,
)
from wabizabi.telemetry.recorder import (
    InMemoryTelemetryRecorder,
    NoopTelemetryRecorder,
    TelemetryRecorder,
)

OutputDataT = TypeVar("OutputDataT")

try:
    from wabizabi.telemetry.otel import OpenTelemetryRecorder as OpenTelemetryRecorder
except ImportError as exc:  # pragma: no cover - exercised in public import test
    _otel_import_error = exc

    class OpenTelemetryRecorder(Generic[OutputDataT]):
        """Placeholder raised when the optional OpenTelemetry extra is missing."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError(
                "OpenTelemetryRecorder requires the optional wabizabi[otel] extra."
            ) from _otel_import_error


__all__ = [
    "HandoffRecordedEvent",
    "InMemoryTelemetryRecorder",
    "NoopTelemetryRecorder",
    "OpenTelemetryRecorder",
    "OutputDecodingFailedEvent",
    "OutputRecordedEvent",
    "OutputValidationFailedEvent",
    "RequestRecordedEvent",
    "ResponseRecordedEvent",
    "RunFailedEvent",
    "RunFinishedEvent",
    "RunStartedEvent",
    "TelemetryEvent",
    "TelemetryRecorder",
    "ToolCallRecordedEvent",
    "ToolResultRecordedEvent",
]

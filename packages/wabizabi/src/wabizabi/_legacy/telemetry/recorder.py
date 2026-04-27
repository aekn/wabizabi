"""Telemetry recorder protocols and implementations."""

from __future__ import annotations

from typing import Protocol

from wabizabi.telemetry.events import TelemetryEvent


class TelemetryRecorder[OutputDataT](Protocol):
    """A sink for typed telemetry events.

    Recorders are invoked from within the async runtime loop. Implementations
    must be non-blocking: offload any real I/O to a background task rather
    than awaiting it inline, or the runtime will stall.
    """

    async def record(
        self,
        event: TelemetryEvent[OutputDataT],
    ) -> None:
        """Record one telemetry event."""
        ...


class NoopTelemetryRecorder[OutputDataT]:
    """A telemetry recorder that ignores all events."""

    __slots__ = ()

    async def record(
        self,
        event: TelemetryEvent[OutputDataT],
    ) -> None:
        del event


class InMemoryTelemetryRecorder[OutputDataT]:
    """A recorder that stores telemetry events in memory."""

    __slots__ = ("events",)

    events: list[TelemetryEvent[OutputDataT]]

    def __init__(self) -> None:
        self.events = []

    async def record(
        self,
        event: TelemetryEvent[OutputDataT],
    ) -> None:
        self.events.append(event)


__all__ = [
    "InMemoryTelemetryRecorder",
    "NoopTelemetryRecorder",
    "TelemetryRecorder",
]

"""Model-visible history processors."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from wabizabi._async import resolve
from wabizabi.context import RunContext
from wabizabi.messages import ModelMessage, ModelRequest

type HistoryProcessor[AgentDepsT] = Callable[
    [RunContext[AgentDepsT], tuple[ModelMessage, ...]],
    tuple[ModelMessage, ...] | Awaitable[tuple[ModelMessage, ...]],
]

type NormalizedHistoryProcessor[AgentDepsT] = Callable[
    [RunContext[AgentDepsT], tuple[ModelMessage, ...]],
    Awaitable[tuple[ModelMessage, ...]],
]


def normalize_history_processor[AgentDepsT](
    processor: HistoryProcessor[AgentDepsT],
) -> NormalizedHistoryProcessor[AgentDepsT]:
    async def wrapped(
        ctx: RunContext[AgentDepsT],
        messages: tuple[ModelMessage, ...],
    ) -> tuple[ModelMessage, ...]:
        return await resolve(processor(ctx, messages))

    return wrapped


@dataclass(frozen=True, slots=True)
class TrimHistoryProcessor:
    """Keep only the newest ``max_messages`` model-visible messages."""

    max_messages: int

    def __post_init__(self) -> None:
        if self.max_messages < 0:
            raise ValueError("max_messages must be non-negative.")

    def __call__(
        self,
        ctx: RunContext[object],
        messages: tuple[ModelMessage, ...],
    ) -> tuple[ModelMessage, ...]:
        del ctx
        if self.max_messages == 0:
            return ()
        if len(messages) <= self.max_messages:
            return messages
        return messages[-self.max_messages :]


@dataclass(frozen=True, slots=True)
class TrimOldestRequestsProcessor:
    """Keep only the newest ``max_pairs`` request-led history segments."""

    max_pairs: int

    def __post_init__(self) -> None:
        if self.max_pairs < 0:
            raise ValueError("max_pairs must be non-negative.")

    def __call__(
        self,
        ctx: RunContext[object],
        messages: tuple[ModelMessage, ...],
    ) -> tuple[ModelMessage, ...]:
        del ctx
        if self.max_pairs == 0:
            return ()

        request_indices = [
            index for index, message in enumerate(messages) if isinstance(message, ModelRequest)
        ]
        if len(request_indices) <= self.max_pairs:
            return messages
        return messages[request_indices[-self.max_pairs] :]


async def apply_history_processors[AgentDepsT](
    processors: tuple[NormalizedHistoryProcessor[AgentDepsT], ...],
    ctx: RunContext[AgentDepsT],
    messages: tuple[ModelMessage, ...],
) -> tuple[ModelMessage, ...]:
    """Apply processors in registration order."""
    current = messages
    for processor in processors:
        current = await processor(ctx, current)
    return current


__all__ = [
    "HistoryProcessor",
    "TrimHistoryProcessor",
    "TrimOldestRequestsProcessor",
    "apply_history_processors",
    "normalize_history_processor",
]

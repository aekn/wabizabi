"""Execution state models for agent runs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from wabizabi.context import RunContext, _UsageRecorder
from wabizabi.history import MessageHistory
from wabizabi.messages import ModelMessage
from wabizabi.types import JsonObject
from wabizabi.usage import RunUsage


@dataclass(frozen=True, slots=True, kw_only=True)
class RunState:
    """Immutable execution state carried across an agent run."""

    run_id: str
    message_history: MessageHistory = field(default_factory=MessageHistory.empty)
    usage: RunUsage = field(default_factory=RunUsage.zero)
    retries: int = 0
    run_step: int = 0
    metadata: JsonObject | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty.")
        if self.retries < 0:
            raise ValueError("retries must be non-negative.")
        if self.run_step < 0:
            raise ValueError("run_step must be non-negative.")

    @classmethod
    def create(
        cls,
        run_id: str,
        *,
        metadata: JsonObject | None = None,
    ) -> RunState:
        """Create a new empty run state."""
        return cls(run_id=run_id, metadata=metadata)

    def with_message(self, message: ModelMessage) -> RunState:
        """Return a copy with one message appended."""
        return replace(self, message_history=self.message_history.append(message))

    def with_messages(self, messages: Iterable[ModelMessage]) -> RunState:
        """Return a copy with multiple messages appended."""
        return replace(self, message_history=self.message_history.extend(messages))

    def with_usage(self, usage: RunUsage) -> RunState:
        """Return a copy with usage replaced."""
        return replace(self, usage=usage)

    def add_usage(self, usage: RunUsage) -> RunState:
        """Return a copy with usage accumulated."""
        return replace(self, usage=self.usage + usage)

    def increment_retry(self) -> RunState:
        """Return a copy with retries incremented by one."""
        return replace(self, retries=self.retries + 1)

    def increment_run_step(self) -> RunState:
        """Return a copy with run_step incremented by one."""
        return replace(self, run_step=self.run_step + 1)

    def context_for[AgentDepsT](
        self,
        deps: AgentDepsT,
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        _usage_recorder: _UsageRecorder | None = None,
    ) -> RunContext[AgentDepsT]:
        """Create a typed run context for tools, validators, and hooks."""
        return RunContext(
            deps=deps,
            run_id=self.run_id,
            run_step=self.run_step,
            usage=self.usage,
            metadata=self.metadata,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            _usage_recorder=_usage_recorder,
        )


__all__ = ["RunState"]

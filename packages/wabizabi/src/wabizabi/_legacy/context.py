"""Run-scoped dependency injection context."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

from wabizabi.types import JsonObject
from wabizabi.usage import RunUsage


class _UsageRecorder(Protocol):
    """Internal protocol for recording nested usage during tool execution."""

    def record_usage(self, usage: RunUsage) -> None: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class RunContext[AgentDepsT]:
    """Typed context passed to tools, validators, and hooks."""

    deps: AgentDepsT
    run_id: str
    run_step: int = 0
    usage: RunUsage = field(default_factory=RunUsage.zero)
    metadata: JsonObject | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    _usage_recorder: _UsageRecorder | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty.")
        if self.run_step < 0:
            raise ValueError("run_step must be non-negative.")
        if (self.tool_name is None) != (self.tool_call_id is None):
            raise ValueError("tool_name and tool_call_id must either both be set or both be None.")
        if self.tool_name == "":
            raise ValueError("tool_name must not be empty.")
        if self.tool_call_id == "":
            raise ValueError("tool_call_id must not be empty.")

    def with_run_step(
        self,
        run_step: int,
        *,
        usage: RunUsage | None = None,
    ) -> RunContext[AgentDepsT]:
        return replace(self, run_step=run_step, usage=self.usage if usage is None else usage)

    def record_usage(self, usage: RunUsage) -> None:
        """Record nested usage produced during tool execution."""
        if self._usage_recorder is None:
            return
        self._usage_recorder.record_usage(usage)


__all__ = ["RunContext"]

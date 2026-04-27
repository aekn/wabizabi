"""Agent run result."""

from __future__ import annotations

from dataclasses import dataclass

from wabizabi.handoff import HandoffResult
from wabizabi.messages import ModelMessage, ModelRequest, ModelResponse
from wabizabi.state import RunState
from wabizabi.usage import RunUsage


@dataclass(frozen=True, slots=True, kw_only=True)
class RunResult[OutputDataT]:
    """Terminal result of a completed agent run.

    A run terminates either with a decoded output or with a handoff. Exactly
    one of ``output`` and ``handoff`` is meaningful per result; consumers
    should branch on ``handoff`` first:

        if result.handoff is not None:
            target = result.handoff.handoff.name
        else:
            value = result.output

    Note: when the output type itself can be ``None`` (e.g. ``JsonValue``
    decoded to JSON ``null``), use ``handoff`` — not ``output is None`` — to
    discriminate between a handoff and a normal terminal.
    """

    state: RunState
    new_messages: tuple[ModelMessage, ...] = ()
    output: OutputDataT | None = None
    handoff: HandoffResult | None = None

    def __post_init__(self) -> None:
        if not self.new_messages:
            return

        all_messages = self.state.message_history.messages
        suffix = all_messages[-len(self.new_messages) :]

        if suffix != self.new_messages:
            raise ValueError("new_messages must match the suffix of state.message_history.")

    @property
    def is_handoff(self) -> bool:
        """Return True when this run terminated with a handoff."""
        return self.handoff is not None

    @property
    def run_id(self) -> str:
        """Return the run identifier."""
        return self.state.run_id

    @property
    def all_messages(self) -> tuple[ModelMessage, ...]:
        """Return the full canonical message history."""
        return self.state.message_history.messages

    @property
    def usage(self) -> RunUsage:
        """Return accumulated usage for the run."""
        return self.state.usage

    @property
    def requests(self) -> tuple[ModelRequest, ...]:
        """Return all request messages in the run."""
        return self.state.message_history.requests

    @property
    def responses(self) -> tuple[ModelResponse, ...]:
        """Return all response messages in the run."""
        return self.state.message_history.responses


__all__ = ["RunResult"]

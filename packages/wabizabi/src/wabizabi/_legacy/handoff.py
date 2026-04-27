"""Handoff primitives for multi-agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel
from pydantic import Field as PydanticField

from wabizabi.messages import ToolCallPart
from wabizabi.state import RunState
from wabizabi.tools.base import ToolDefinition
from wabizabi.tools.schema import tool_input_schema


class HandoffInput(BaseModel):
    """Arguments model for a handoff tool call."""

    input: str = PydanticField(description="The input to pass to the target agent.")


_HANDOFF_INPUT_SCHEMA = tool_input_schema(HandoffInput)


def _handoff_tool_name(name: str) -> str:
    """Derive the tool name from the handoff name."""
    return f"handoff_{name}"


@dataclass(frozen=True, slots=True)
class Handoff:
    """A handoff target that terminates the current agent run.

    When the model calls this handoff's tool, the run terminates and yields
    a HandoffEvent. The caller/orchestrator decides whether to continue
    by running the target agent.

    The handoff itself is a pure registration primitive — it does not
    store a reference to the target agent. The orchestrator connects
    handoff names to agents in its own dispatch logic.
    """

    name: str
    description: str | None = None
    tool_name: str = field(init=False)
    tool_definition: ToolDefinition = field(init=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("handoff name must not be empty.")
        object.__setattr__(self, "tool_name", _handoff_tool_name(self.name))
        object.__setattr__(
            self,
            "tool_definition",
            ToolDefinition(
                name=self.tool_name,
                description=self.description,
                input_schema=_HANDOFF_INPUT_SCHEMA,
            ),
        )


@dataclass(frozen=True, slots=True)
class HandoffResult:
    """Terminal outcome when a handoff tool is called.

    Surfaced via :class:`wabizabi.RunResult.handoff` when an agent run
    terminates with a handoff. The orchestrator inspects ``handoff.name``
    to dispatch the next agent.
    """

    handoff: Handoff
    tool_call: ToolCallPart
    state: RunState


__all__ = [
    "Handoff",
    "HandoffInput",
    "HandoffResult",
]

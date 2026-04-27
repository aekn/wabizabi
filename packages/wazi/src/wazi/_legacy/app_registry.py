"""AgentApp: a named registry of agents for Wazi to load and switch between.

This is a shell-facing primitive, not a runtime concept. It exists so users
can export multiple cooperating agents from a single module and have Wazi
present handoffs truthfully — switching the active agent when a handoff
targets a registered name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from wabizabi import Agent


@dataclass(frozen=True, slots=True)
class AgentApp:
    """A named registry of agents with a default entry point.

    Agents are keyed by handoff-friendly names. When an agent in the app
    emits a handoff whose target name is also in ``agents``, Wazi can
    continue the session with the target as the active agent.
    """

    agents: Mapping[str, Agent[object, object]]
    default: str

    def __post_init__(self) -> None:
        if not self.agents:
            raise ValueError("AgentApp must contain at least one agent.")
        if self.default not in self.agents:
            raise ValueError(f"Default agent {self.default!r} is not in the agent registry.")
        object.__setattr__(self, "agents", MappingProxyType(dict(self.agents)))

    def get(self, name: str) -> Agent[object, object]:
        """Return the agent registered under ``name``.

        Raises ``KeyError`` if the name is not registered.
        """
        return self.agents[name]

    def names(self) -> tuple[str, ...]:
        """Return the registered agent names in insertion order."""
        return tuple(self.agents.keys())


__all__ = ["AgentApp"]

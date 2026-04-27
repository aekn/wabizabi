"""Multi-agent incident response system.

coord agent --> {log analyst, metrics inspector, diagnostician}

The coordinator can hand off to a human on-call engineer, which is a
terminal outcome (humans are not registered agents). The ``app`` export
wraps the coordinator as an ``AgentApp`` so Wazi can load it via the
registry entry point: ``wazi agent chat examples.incident_response:app``.
"""

from wabizabi import Agent
from wazi.app_registry import AgentApp

from .agents import agent, diagnostician

_coordinator: Agent[object, object] = agent  # pyright: ignore[reportAssignmentType]
_diagnostician: Agent[object, object] = diagnostician  # pyright: ignore[reportAssignmentType]
app = AgentApp(
    agents={"coordinator": _coordinator, "diagnostician": _diagnostician}, default="coordinator"
)

__all__ = ["agent", "app"]

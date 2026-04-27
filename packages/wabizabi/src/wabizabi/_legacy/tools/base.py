"""Core tool protocols and provider-neutral definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from wabizabi.context import RunContext
from wabizabi.types import JsonObject, JsonValue


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A provider-neutral tool definition exposed to model adapters."""

    name: str
    input_schema: JsonObject
    description: str | None = None
    strict: bool = True


class ToolError(RuntimeError):
    """Base error raised for tool-related failures."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool does not exist."""


class ToolInvocationError(ToolError):
    """Raised when a tool call cannot be executed."""


@runtime_checkable
class Tool[AgentDepsT](Protocol):
    """A typed tool that can validate arguments and execute."""

    @property
    def name(self) -> str:
        """Return the tool name."""
        ...

    @property
    def description(self) -> str | None:
        """Return the optional tool description."""
        ...

    @property
    def definition(self) -> ToolDefinition:
        """Return the provider-neutral tool definition."""
        ...

    async def invoke(
        self,
        context: RunContext[AgentDepsT],
        arguments: JsonObject,
    ) -> JsonValue:
        """Invoke the tool with validated arguments."""
        ...


__all__ = [
    "Tool",
    "ToolDefinition",
    "ToolError",
    "ToolInvocationError",
    "ToolNotFoundError",
]

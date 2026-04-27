"""Reusable toolset composition primitives."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from wabizabi.context import RunContext
from wabizabi.tools.base import Tool, ToolDefinition, ToolNotFoundError
from wabizabi.types import JsonObject, JsonValue


@dataclass(frozen=True, slots=True)
class Toolset[AgentDepsT]:
    """An immutable collection of tools addressable by name."""

    tools: tuple[Tool[AgentDepsT], ...] = ()
    _by_name: dict[str, Tool[AgentDepsT]] = field(default_factory=dict, init=False, repr=False)
    _definitions: tuple[ToolDefinition, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        seen: dict[str, Tool[AgentDepsT]] = {}
        for tool in self.tools:
            if tool.name in seen:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            seen[tool.name] = tool
        object.__setattr__(self, "_by_name", seen)
        object.__setattr__(self, "_definitions", tuple(tool.definition for tool in self.tools))

    @classmethod
    def empty(cls) -> Toolset[AgentDepsT]:
        """Create an empty toolset."""
        return cls()

    def with_tool(self, tool: Tool[AgentDepsT]) -> Toolset[AgentDepsT]:
        """Return a new toolset with one additional tool."""
        return replace(self, tools=(*self.tools, tool))

    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return provider-neutral definitions for all registered tools."""
        return self._definitions

    def get(self, name: str) -> Tool[AgentDepsT] | None:
        """Look up a tool by name."""
        return self._by_name.get(name)

    async def invoke(
        self,
        name: str,
        *,
        context: RunContext[AgentDepsT],
        arguments: JsonObject,
    ) -> JsonValue:
        """Invoke a tool by name."""
        tool = self.get(name)
        if tool is None:
            raise ToolNotFoundError(f"Unknown tool: {name}")
        return await tool.invoke(context, arguments)


__all__ = ["Toolset"]

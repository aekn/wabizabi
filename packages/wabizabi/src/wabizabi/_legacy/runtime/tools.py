"""Helpers for extracting and invoking runtime tool calls."""

from __future__ import annotations

from dataclasses import dataclass

from wabizabi.context import RunContext
from wabizabi.messages import ModelResponse, ToolCallPart, ToolReturnPart
from wabizabi.tools.base import ToolError
from wabizabi.tools.toolset import Toolset
from wabizabi.types import JsonObject


@dataclass(frozen=True, slots=True)
class ToolCallPartition:
    """Partitioned tool calls from one model response."""

    function_calls: tuple[ToolCallPart, ...]
    terminal_calls: tuple[ToolCallPart, ...]


def partition_tool_calls(
    response: ModelResponse,
    *,
    terminal_tool_names: frozenset[str],
) -> ToolCallPartition:
    """Split response tool calls into executable and terminal calls."""
    function_calls: list[ToolCallPart] = []
    terminal_calls: list[ToolCallPart] = []
    for part in response.parts:
        if not isinstance(part, ToolCallPart):
            continue
        if part.tool_name in terminal_tool_names:
            terminal_calls.append(part)
        else:
            function_calls.append(part)
    return ToolCallPartition(
        function_calls=tuple(function_calls),
        terminal_calls=tuple(terminal_calls),
    )


async def invoke_tool[AgentDepsT](
    *,
    toolset: Toolset[AgentDepsT],
    context: RunContext[AgentDepsT],
    tool_name: str,
    call_id: str,
    arguments: JsonObject,
) -> ToolReturnPart:
    """Invoke a tool and normalize its result into a tool return part."""
    try:
        result = await toolset.invoke(
            tool_name,
            context=context,
            arguments=arguments,
        )
    except ToolError as error:
        return ToolReturnPart(
            tool_name=tool_name,
            call_id=call_id,
            content={"error": str(error)},
            is_error=True,
        )

    return ToolReturnPart(
        tool_name=tool_name,
        call_id=call_id,
        content=result,
    )


__all__ = ["ToolCallPartition", "invoke_tool", "partition_tool_calls"]

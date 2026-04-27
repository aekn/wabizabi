"""Agent-as-tool: wrap a child agent as a standard async tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from wabizabi.context import RunContext
from wabizabi.tools.function import AsyncFunctionTool, define_async_function_tool
from wabizabi.types import JsonValue, json_value_from_unknown

if TYPE_CHECKING:
    from wabizabi.agent import Agent


class AgentToolInput(BaseModel):
    """Arguments model for an agent-as-tool invocation."""

    input: str = Field(description="The input prompt to send to the agent.")


def _output_to_json_value(output: object) -> JsonValue:
    """Convert an agent output to a JSON-compatible value."""
    if isinstance(output, BaseModel):
        return json_value_from_unknown(output.model_dump())
    return json_value_from_unknown(output)


def agent_as_tool[AgentDepsT, OutputDataT](
    agent: Agent[AgentDepsT, OutputDataT],
    *,
    name: str,
    description: str | None = None,
) -> AsyncFunctionTool[AgentDepsT, AgentToolInput]:
    """Wrap an agent as an async tool that runs the agent on invocation.

    The parent agent's dependencies are forwarded to the child agent.
    The child agent's output is converted to a JSON value for the tool result.
    """

    async def invoke(
        context: RunContext[AgentDepsT],
        arguments: AgentToolInput,
    ) -> JsonValue:
        result = await agent.run(arguments.input, deps=context.deps)
        context.record_usage(result.state.usage)
        return _output_to_json_value(result.output)

    return define_async_function_tool(
        name=name,
        arguments_type=AgentToolInput,
        func=invoke,
        description=description,
    )


__all__ = ["AgentToolInput", "agent_as_tool"]

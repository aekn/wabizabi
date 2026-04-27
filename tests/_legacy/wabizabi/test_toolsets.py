from __future__ import annotations

import pytest
from pydantic import BaseModel
from wabizabi.context import RunContext
from wabizabi.tools import Toolset, define_function_tool


class EchoArguments(BaseModel):
    value: str


def test_toolset_get_returns_matching_tool() -> None:
    def echo_tool(
        context: RunContext[tuple[str, str]],
        arguments: EchoArguments,
    ) -> str:
        del context
        return arguments.value

    tool = define_function_tool(
        name="echo",
        arguments_type=EchoArguments,
        func=echo_tool,
    )

    toolset = Toolset[tuple[str, str]].empty().with_tool(tool)

    assert toolset.get("echo") is tool
    assert toolset.get("missing") is None


def test_toolset_rejects_duplicate_tool_names() -> None:
    def echo_tool(
        context: RunContext[tuple[str, str]],
        arguments: EchoArguments,
    ) -> str:
        del context
        return arguments.value

    first = define_function_tool(
        name="echo",
        arguments_type=EchoArguments,
        func=echo_tool,
    )
    second = define_function_tool(
        name="echo",
        arguments_type=EchoArguments,
        func=echo_tool,
    )

    with pytest.raises(ValueError, match="Duplicate tool name: echo"):
        Toolset[tuple[str, str]](tools=(first, second))


def test_toolset_definitions_returns_registered_tool_definitions() -> None:
    def echo_tool(
        context: RunContext[tuple[str, str]],
        arguments: EchoArguments,
    ) -> str:
        del context
        return arguments.value

    tool = define_function_tool(
        name="echo",
        arguments_type=EchoArguments,
        func=echo_tool,
        description="Echo a value.",
    )

    toolset = Toolset[tuple[str, str]].empty().with_tool(tool)

    assert toolset.definitions() == (tool.definition,)

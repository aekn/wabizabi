from __future__ import annotations

import pytest
from pydantic import BaseModel, TypeAdapter
from wabizabi.context import RunContext
from wabizabi.tools import (
    ToolInvocationError,
    ToolNotFoundError,
    Toolset,
    define_async_function_tool,
    define_function_tool,
)
from wabizabi.types import JsonObject, JsonValue


class AddArguments(BaseModel):
    left: int
    right: int


_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


def _json_object(value: object) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(value)


def _schema_properties(schema: JsonObject) -> dict[str, JsonObject]:
    properties = schema.get("properties")
    assert isinstance(properties, dict)
    normalized: dict[str, JsonObject] = {}
    for name, value in properties.items():
        normalized[name] = _json_object(value)
    return normalized


def make_context() -> RunContext[tuple[str, str]]:
    return RunContext(
        deps=("svc", "cfg"),
        run_id="run-1",
        run_step=1,
        tool_name="add",
        tool_call_id="call-1",
    )


@pytest.mark.asyncio
async def test_function_tool_validates_arguments_and_executes() -> None:
    def add_tool(context: RunContext[tuple[str, str]], arguments: AddArguments) -> int:
        assert context.tool_name == "add"
        assert arguments.left == 2
        assert arguments.right == 3
        return arguments.left + arguments.right

    tool = define_function_tool(
        name="add",
        arguments_type=AddArguments,
        func=add_tool,
        description="Add two integers.",
    )

    result = await tool.invoke(
        make_context(),
        {"left": 2, "right": 3},
    )

    assert result == 5


@pytest.mark.asyncio
async def test_async_function_tool_executes() -> None:
    async def add_tool(
        context: RunContext[tuple[str, str]],
        arguments: AddArguments,
    ) -> int:
        assert context.tool_call_id == "call-1"
        return arguments.left + arguments.right

    tool = define_async_function_tool(
        name="add",
        arguments_type=AddArguments,
        func=add_tool,
    )

    result = await tool.invoke(
        make_context(),
        {"left": 4, "right": 5},
    )

    assert result == 9


@pytest.mark.asyncio
async def test_toolset_invokes_tool_by_name() -> None:
    def add_tool(context: RunContext[tuple[str, str]], arguments: AddArguments) -> int:
        assert context.run_step == 1
        return arguments.left + arguments.right

    toolset = (
        Toolset[tuple[str, str]]
        .empty()
        .with_tool(
            define_function_tool(
                name="add",
                arguments_type=AddArguments,
                func=add_tool,
            )
        )
    )

    result = await toolset.invoke(
        "add",
        context=make_context(),
        arguments={"left": 1, "right": 6},
    )

    assert result == 7


@pytest.mark.asyncio
async def test_toolset_raises_for_unknown_tool() -> None:
    toolset = Toolset[tuple[str, str]].empty()

    with pytest.raises(ToolNotFoundError, match="Unknown tool: missing"):
        await toolset.invoke(
            "missing",
            context=make_context(),
            arguments={"left": 1, "right": 2},
        )


@pytest.mark.asyncio
async def test_tool_argument_validation_is_enforced() -> None:
    def add_tool(context: RunContext[tuple[str, str]], arguments: AddArguments) -> int:
        del context
        return arguments.left + arguments.right

    tool = define_function_tool(
        name="add",
        arguments_type=AddArguments,
        func=add_tool,
    )

    with pytest.raises(ToolInvocationError, match="validation error"):
        await tool.invoke(
            make_context(),
            {"left": "bad", "right": 2},
        )


def test_function_tool_exposes_provider_neutral_definition() -> None:
    def add_tool(context: RunContext[tuple[str, str]], arguments: AddArguments) -> int:
        del context
        return arguments.left + arguments.right

    tool = define_function_tool(
        name="add",
        arguments_type=AddArguments,
        func=add_tool,
        description="Add two integers.",
    )

    assert tool.definition.name == "add"
    assert tool.definition.description == "Add two integers."
    assert tool.definition.strict is True
    assert tool.definition.input_schema["type"] == "object"
    assert _schema_properties(tool.definition.input_schema) == {
        "left": {"type": "integer"},
        "right": {"type": "integer"},
    }


def _schema_contains_title(value: JsonValue) -> bool:
    if isinstance(value, dict):
        return "title" in value or any(_schema_contains_title(item) for item in value.values())
    if isinstance(value, list):
        return any(_schema_contains_title(item) for item in value)
    return False


def test_tool_definition_schema_strips_titles_recursively() -> None:
    class InnerArguments(BaseModel):
        count: int

    class OuterArguments(BaseModel):
        inner: InnerArguments

    def use_nested(context: RunContext[tuple[str, str]], arguments: OuterArguments) -> int:
        del context
        return arguments.inner.count

    tool = define_function_tool(
        name="use_nested",
        arguments_type=OuterArguments,
        func=use_nested,
    )

    assert _schema_contains_title(tool.definition.input_schema) is False

from __future__ import annotations

from typing import Annotated, cast

import pytest
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from wabizabi.context import RunContext
from wabizabi.tools import AsyncFunctionTool, FunctionTool, ToolInvocationError, tool, tool_plain
from wabizabi.types import JsonObject

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


@pytest.mark.asyncio
async def test_tool_decorator_builds_function_tool_with_docstring_schema() -> None:
    @tool
    def add(ctx: RunContext[tuple[str, str]], a: int, b: int = 10) -> int:
        """Add two integers.

        Args:
            a: The first integer.
            b: The second integer.
        """
        assert ctx.deps == ("svc", "cfg")
        return a + b

    assert isinstance(add, FunctionTool)
    assert add.name == "add"
    assert add.description == "Add two integers."
    assert _schema_properties(add.definition.input_schema) == {
        "a": {
            "description": "The first integer.",
            "type": "integer",
        },
        "b": {
            "default": 10,
            "description": "The second integer.",
            "type": "integer",
        },
    }

    result = await add.invoke(
        RunContext(deps=("svc", "cfg"), run_id="run-1"),
        {"a": 3},
    )

    assert result == 13


def test_tool_decorator_prefers_annotated_field_descriptions() -> None:
    @tool(name="sum_values")
    def add(
        ctx: RunContext[None],
        a: Annotated[int, Field(description="Description from Field")],
        b: int,
    ) -> int:
        """Add values.

        Args:
            a: Description from docstring.
            b: Description from docstring.
        """
        del ctx
        return a + b

    assert isinstance(add, FunctionTool)
    properties = _schema_properties(add.definition.input_schema)
    assert properties["a"]["description"] == "Description from Field"
    assert properties["b"]["description"] == "Description from docstring."
    assert add.definition.name == "sum_values"


@pytest.mark.asyncio
async def test_tool_plain_supports_async_functions_without_context() -> None:
    @tool_plain(description="Fetch a resource.")
    async def fetch(url: str, timeout: int = 5) -> str:
        """Fetch a URL.

        Args:
            url: The target URL.
            timeout: Maximum timeout in seconds.
        """
        return f"{url}:{timeout}"

    assert isinstance(fetch, AsyncFunctionTool)
    typed_fetch = cast(AsyncFunctionTool[object, BaseModel], fetch)
    assert typed_fetch.description == "Fetch a resource."

    result = await typed_fetch.invoke(
        RunContext(deps={"unused": True}, run_id="run-2"),
        {"url": "https://example.test"},
    )

    assert result == "https://example.test:5"


def test_tool_decorator_rejects_missing_run_context_annotation() -> None:
    def add(a: int, b: int) -> int:
        return a + b

    with pytest.raises(TypeError, match="first parameter to be annotated as RunContext"):
        tool(add)  # type: ignore[arg-type]


def test_tool_decorator_rejects_variadic_parameters() -> None:
    def add(*values: int) -> int:
        return sum(values)

    with pytest.raises(TypeError, match="only support positional-or-keyword and keyword-only"):
        tool_plain(add)


def test_tool_decorator_generated_arguments_model_validates_inputs() -> None:
    @tool_plain
    def repeat(text: str, times: int) -> str:
        """Repeat a string."""
        return text * times

    with pytest.raises(ValidationError):
        repeat.arguments_type.model_validate({"text": "x", "times": "bad"})


@pytest.mark.asyncio
async def test_tool_decorators_require_json_serializable_return_values() -> None:
    def invalid_result() -> complex:
        return 1 + 2j

    decorated = tool_plain(invalid_result)  # type: ignore[arg-type, reportUnknownMemberType]  # testing runtime rejection of non-JSON return
    with pytest.raises(ToolInvocationError, match="JSON-serializable"):
        await decorated.invoke(RunContext(deps=None, run_id="run-1"), {})  # type: ignore[reportUnknownMemberType]

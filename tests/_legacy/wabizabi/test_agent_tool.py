"""Tests for Agent.as_tool() — wrapping an agent as a tool for parent agents."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from wabizabi.agent import Agent
from wabizabi.context import RunContext
from wabizabi.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from wabizabi.models import ModelResult
from wabizabi.output import schema_output_config, text_output_config
from wabizabi.stream import OutputEvent, ToolCallEvent, ToolResultEvent
from wabizabi.testing import ScriptedModel
from wabizabi.tools import define_function_tool
from wabizabi.tools.agent import AgentToolInput
from wabizabi.types import JsonValue
from wabizabi.usage import RunUsage


def _text_result(text: str) -> ModelResult:
    return ModelResult(
        response=ModelResponse(parts=(TextPart(text=text),), model_name="scripted"),
        usage=RunUsage.zero(),
    )


def _tool_call_result(tool_name: str, call_id: str, arguments: dict[str, JsonValue]) -> ModelResult:
    return ModelResult(
        response=ModelResponse(
            parts=(ToolCallPart(tool_name=tool_name, call_id=call_id, arguments=arguments),),
            model_name="scripted",
        ),
        usage=RunUsage.zero(),
    )


@pytest.mark.asyncio
async def test_agent_as_tool_returns_async_function_tool() -> None:
    child = Agent[None, str](
        model=ScriptedModel((_text_result("ok"),)), output=text_output_config()
    )
    tool = child.as_tool(name="child_agent", description="A child agent")

    assert tool.name == "child_agent"
    assert tool.description == "A child agent"
    assert tool.arguments_type is AgentToolInput


@pytest.mark.asyncio
async def test_agent_as_tool_schema_has_input_field() -> None:
    child = Agent[None, str](
        model=ScriptedModel((_text_result("ok"),)), output=text_output_config()
    )
    tool = child.as_tool(name="child", description="test")

    schema = tool.definition.input_schema
    properties = schema.get("properties")
    assert isinstance(properties, dict)
    assert "input" in properties


@pytest.mark.asyncio
async def test_agent_as_tool_runs_child_and_returns_text_output() -> None:
    child_model = ScriptedModel((_text_result("child says hello"),))
    child = Agent[None, str](model=child_model, output=text_output_config())

    parent_model = ScriptedModel(
        (
            _tool_call_result("summarize", "call-1", {"input": "What is 2+2?"}),
            _text_result("The child said hello"),
        )
    )
    parent = Agent[None, str](model=parent_model, output=text_output_config()).with_tool(
        child.as_tool(name="summarize", description="Summarize")
    )

    result = await parent.run("test", deps=None)
    assert result.output == "The child said hello"

    # Verify child was called with the correct prompt
    assert len(child_model.requests) == 1
    child_request = child_model.requests[0]
    assert any(
        hasattr(part, "text") and part.text == "What is 2+2?"  # type: ignore[union-attr]
        for part in child_request.parts
    )


@pytest.mark.asyncio
async def test_agent_as_tool_returns_tool_result_as_json() -> None:
    child_model = ScriptedModel((_text_result("42"),))
    child = Agent[None, str](model=child_model, output=text_output_config())

    parent_model = ScriptedModel(
        (
            _tool_call_result("calc", "call-1", {"input": "compute"}),
            _text_result("done"),
        )
    )
    parent = Agent[None, str](model=parent_model, output=text_output_config()).with_tool(
        child.as_tool(name="calc")
    )

    result = await parent.run("go", deps=None)
    assert result.output == "done"

    # The tool result should have been fed back to the parent model
    second_request = parent_model.requests[1]
    tool_returns = [p for p in second_request.parts if isinstance(p, ToolReturnPart)]
    assert len(tool_returns) == 1
    assert tool_returns[0].content == "42"


@pytest.mark.asyncio
async def test_agent_as_tool_with_structured_output() -> None:
    class Answer(BaseModel):
        value: int

    child_model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(TextPart(text='{"value": 42}'),),
                    model_name="scripted",
                ),
                usage=RunUsage.zero(),
            ),
        )
    )
    child = Agent[None, Answer](model=child_model, output=schema_output_config(Answer))

    parent_model = ScriptedModel(
        (
            _tool_call_result("compute", "call-1", {"input": "what is 6*7?"}),
            _text_result("The answer is 42"),
        )
    )
    parent = Agent[None, str](model=parent_model, output=text_output_config()).with_tool(
        child.as_tool(name="compute", description="Compute an answer")
    )

    result = await parent.run("go", deps=None)
    assert result.output == "The answer is 42"

    # Verify the tool result contains the structured output as JSON
    second_request = parent_model.requests[1]
    tool_returns = [p for p in second_request.parts if isinstance(p, ToolReturnPart)]
    assert len(tool_returns) == 1
    assert tool_returns[0].content == {"value": 42}


@pytest.mark.asyncio
async def test_agent_as_tool_forwards_deps() -> None:
    captured_deps: list[str] = []

    def capture_deps(ctx: RunContext[str], args: AgentToolInput) -> JsonValue:
        captured_deps.append(ctx.deps)
        return "captured"

    child_model = ScriptedModel(
        (
            _tool_call_result("capture", "c1", {"input": "x"}),
            _text_result("done"),
        )
    )
    child = Agent[str, str](model=child_model, output=text_output_config()).with_tool(
        define_function_tool(
            name="capture",
            arguments_type=AgentToolInput,
            func=capture_deps,
            description="Capture deps",
        )
    )

    parent_model = ScriptedModel(
        (
            _tool_call_result("child", "p1", {"input": "go"}),
            _text_result("ok"),
        )
    )
    parent = Agent[str, str](model=parent_model, output=text_output_config()).with_tool(
        child.as_tool(name="child")
    )

    await parent.run("test", deps="my-deps")
    assert captured_deps == ["my-deps"]


@pytest.mark.asyncio
async def test_agent_as_tool_emits_tool_call_and_result_events() -> None:
    child_model = ScriptedModel((_text_result("child output"),))
    child = Agent[None, str](model=child_model, output=text_output_config())

    parent_model = ScriptedModel(
        (
            _tool_call_result("child", "call-1", {"input": "hello"}),
            _text_result("final"),
        )
    )
    parent = Agent[None, str](model=parent_model, output=text_output_config()).with_tool(
        child.as_tool(name="child")
    )

    events = [event async for event in parent.iter("go", deps=None)]
    tool_call_events = [e for e in events if isinstance(e, ToolCallEvent)]
    tool_result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    output_events = [e for e in events if isinstance(e, OutputEvent)]

    assert len(tool_call_events) == 1
    assert tool_call_events[0].tool_call.tool_name == "child"
    assert len(tool_result_events) == 1
    assert tool_result_events[0].tool_return.content == "child output"
    assert len(output_events) == 1
    assert output_events[0].output == "final"


@pytest.mark.asyncio
async def test_agent_as_tool_records_child_usage_on_parent_run() -> None:
    child_model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(TextPart(text="child output"),), model_name="scripted"
                ),
                usage=RunUsage(input_tokens=7, output_tokens=3),
            ),
        )
    )
    child = Agent[None, str](model=child_model, output=text_output_config())

    parent_model = ScriptedModel(
        (
            _tool_call_result("child", "call-1", {"input": "hello"}),
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="final"),), model_name="scripted"),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
        )
    )
    parent = Agent[None, str](model=parent_model, output=text_output_config()).with_tool(
        child.as_tool(name="child")
    )

    result = await parent.run("go", deps=None)
    assert result.output == "final"
    assert result.usage == RunUsage(input_tokens=9, output_tokens=4)

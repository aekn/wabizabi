"""Tests for tool failure handling in the runtime loop."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from wabizabi.agent import Agent
from wabizabi.context import RunContext
from wabizabi.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from wabizabi.models import ModelResult
from wabizabi.output import text_output_config
from wabizabi.stream import OutputEvent, ToolResultEvent
from wabizabi.testing import ScriptedModel
from wabizabi.tools import define_function_tool
from wabizabi.types import JsonValue
from wabizabi.usage import RunUsage


class BoomArguments(BaseModel):
    value: str


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
async def test_tool_failure_returns_error_tool_result_and_run_continues() -> None:
    def boom(_context: RunContext[None], arguments: BoomArguments) -> str:
        raise ValueError(f"boom: {arguments.value}")

    model = ScriptedModel(
        (
            _tool_call_result("boom", "call-1", {"value": "explode"}),
            _text_result("recovered"),
        )
    )
    agent = Agent[None, str](model=model, output=text_output_config()).with_tool(
        define_function_tool(name="boom", arguments_type=BoomArguments, func=boom)
    )

    events = [event async for event in agent.iter("go", deps=None)]
    tool_result_events = [event for event in events if isinstance(event, ToolResultEvent)]
    output_events = [event for event in events if isinstance(event, OutputEvent)]

    assert len(tool_result_events) == 1
    assert tool_result_events[0].tool_return.is_error is True
    assert tool_result_events[0].tool_return.tool_name == "boom"
    assert tool_result_events[0].tool_return.content == {
        "error": "Tool 'boom' failed: boom: explode"
    }

    assert len(output_events) == 1
    assert output_events[0].output == "recovered"

    second_request = model.requests[1]
    tool_returns = [part for part in second_request.parts if isinstance(part, ToolReturnPart)]
    assert len(tool_returns) == 1
    assert tool_returns[0].is_error is True
    assert tool_returns[0].content == {"error": "Tool 'boom' failed: boom: explode"}

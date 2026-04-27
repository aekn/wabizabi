from __future__ import annotations

import pytest
from pydantic import BaseModel
from wabizabi.agent import Agent
from wabizabi.context import RunContext
from wabizabi.messages import ModelResponse, ReasoningPart, TextPart, ToolCallPart
from wabizabi.models import ModelResult
from wabizabi.output import OutputValidationError, TextOutputDecoder
from wabizabi.telemetry import (
    InMemoryTelemetryRecorder,
    OutputDecodingFailedEvent,
    OutputRecordedEvent,
    OutputValidationFailedEvent,
    RequestRecordedEvent,
    ResponseRecordedEvent,
    RunFailedEvent,
    RunFinishedEvent,
    RunStartedEvent,
    ToolCallRecordedEvent,
    ToolResultRecordedEvent,
)
from wabizabi.testing import ScriptedModel
from wabizabi.tools import define_function_tool
from wabizabi.usage import RunUsage


@pytest.mark.asyncio
async def test_telemetry_records_single_turn_run() -> None:
    recorder = InMemoryTelemetryRecorder[str]()
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="Hi"),), model_name="scripted"),
                usage=RunUsage(input_tokens=3, output_tokens=2),
            ),
        )
    )
    agent = Agent[tuple[str, str], str](
        model=model,
        decoder=TextOutputDecoder(),
    ).with_telemetry(recorder)

    result = await agent.run(
        "Hello",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == "Hi"
    assert len(recorder.events) == 5
    assert isinstance(recorder.events[0], RunStartedEvent)
    assert isinstance(recorder.events[1], RequestRecordedEvent)
    assert isinstance(recorder.events[2], ResponseRecordedEvent)
    assert isinstance(recorder.events[3], OutputRecordedEvent)
    assert isinstance(recorder.events[4], RunFinishedEvent)

    response_event = recorder.events[2]
    output_event = recorder.events[3]
    finished_event = recorder.events[4]

    assert isinstance(response_event, ResponseRecordedEvent)
    assert response_event.state.usage == RunUsage(input_tokens=3, output_tokens=2)

    assert isinstance(output_event, OutputRecordedEvent)
    assert output_event.output == "Hi"

    assert isinstance(finished_event, RunFinishedEvent)
    assert finished_event.output == "Hi"


class AddArguments(BaseModel):
    left: int
    right: int


@pytest.mark.asyncio
async def test_telemetry_records_tool_execution_events() -> None:
    def add_tool(context: RunContext[tuple[str, str]], arguments: AddArguments) -> int:
        assert context.tool_name == "add"
        return arguments.left + arguments.right

    recorder = InMemoryTelemetryRecorder[str]()
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(
                        ToolCallPart(
                            tool_name="add",
                            call_id="call-1",
                            arguments={"left": 2, "right": 3},
                        ),
                    ),
                    model_name="scripted",
                ),
                usage=RunUsage(input_tokens=3, output_tokens=0),
            ),
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="5"),), model_name="scripted"),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
        )
    )
    agent = (
        Agent[tuple[str, str], str](
            model=model,
            decoder=TextOutputDecoder(),
        )
        .with_tool(
            define_function_tool(
                name="add",
                arguments_type=AddArguments,
                func=add_tool,
            )
        )
        .with_telemetry(recorder)
    )

    result = await agent.run(
        "What is 2 + 3?",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == "5"

    event_types = tuple(type(event) for event in recorder.events)
    assert event_types == (
        RunStartedEvent,
        RequestRecordedEvent,
        ResponseRecordedEvent,
        ToolCallRecordedEvent,
        ToolResultRecordedEvent,
        RequestRecordedEvent,
        ResponseRecordedEvent,
        OutputRecordedEvent,
        RunFinishedEvent,
    )

    tool_call_event = recorder.events[3]
    tool_result_event = recorder.events[4]
    second_response_event = recorder.events[6]

    assert isinstance(tool_call_event, ToolCallRecordedEvent)
    assert tool_call_event.tool_call == ToolCallPart(
        tool_name="add",
        call_id="call-1",
        arguments={"left": 2, "right": 3},
    )

    assert isinstance(tool_result_event, ToolResultRecordedEvent)
    assert tool_result_event.tool_return.tool_name == "add"
    assert tool_result_event.tool_return.content == 5

    assert isinstance(second_response_event, ResponseRecordedEvent)
    assert second_response_event.state.usage == RunUsage(input_tokens=5, output_tokens=1)


@pytest.mark.asyncio
async def test_telemetry_records_output_validation_failure_before_retry() -> None:
    def require_good(
        context: RunContext[tuple[str, str]],
        output: str,
    ) -> str:
        del context
        if output != "good":
            raise OutputValidationError(
                "output must be good",
                retry_feedback="Return exactly the word good.",
            )
        return output

    recorder = InMemoryTelemetryRecorder[str]()
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="bad"),), model_name="scripted"),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="good"),), model_name="scripted"),
                usage=RunUsage(input_tokens=1, output_tokens=1),
            ),
        )
    )
    agent = (
        Agent[tuple[str, str], str](
            model=model,
            decoder=TextOutputDecoder(),
        )
        .with_output_validator(require_good)
        .with_telemetry(recorder)
    )

    result = await agent.run(
        "Hello",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == "good"

    event_types = tuple(type(event) for event in recorder.events)
    assert event_types == (
        RunStartedEvent,
        RequestRecordedEvent,
        ResponseRecordedEvent,
        OutputValidationFailedEvent,
        RequestRecordedEvent,
        ResponseRecordedEvent,
        OutputRecordedEvent,
        RunFinishedEvent,
    )

    failure_event = recorder.events[3]
    assert isinstance(failure_event, OutputValidationFailedEvent)
    assert failure_event.state.retries == 1
    assert failure_event.retry_feedback == "Return exactly the word good."


@pytest.mark.asyncio
async def test_telemetry_records_output_decoding_failure_before_retry() -> None:
    recorder = InMemoryTelemetryRecorder[str]()
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(ReasoningPart(text="thinking only"),), model_name="scripted"
                ),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="good"),), model_name="scripted"),
                usage=RunUsage(input_tokens=1, output_tokens=1),
            ),
        )
    )
    agent = Agent[tuple[str, str], str](
        model=model,
        decoder=TextOutputDecoder(),
    ).with_telemetry(recorder)

    result = await agent.run(
        "Hello",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == "good"

    event_types = tuple(type(event) for event in recorder.events)
    assert event_types == (
        RunStartedEvent,
        RequestRecordedEvent,
        ResponseRecordedEvent,
        OutputDecodingFailedEvent,
        RequestRecordedEvent,
        ResponseRecordedEvent,
        OutputRecordedEvent,
        RunFinishedEvent,
    )

    failure_event = recorder.events[3]
    assert isinstance(failure_event, OutputDecodingFailedEvent)
    assert failure_event.state.retries == 1
    assert "plain text" in failure_event.retry_feedback


@pytest.mark.asyncio
async def test_telemetry_records_run_failure_event_for_unhandled_exception() -> None:
    def explode(context: RunContext[tuple[str, str]], output: str) -> str:
        del context
        raise RuntimeError("boom")

    recorder = InMemoryTelemetryRecorder[str]()
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="Hi"),), model_name="scripted"),
                usage=RunUsage(input_tokens=3, output_tokens=2),
            ),
        )
    )
    agent = (
        Agent[tuple[str, str], str](model=model, decoder=TextOutputDecoder())
        .with_output_validator(explode)
        .with_telemetry(recorder)
    )

    with pytest.raises(RuntimeError, match="boom"):
        await agent.run("Hello", deps=("svc", "cfg"), run_id="run-1")

    failure_event = recorder.events[-1]
    assert isinstance(failure_event, RunFailedEvent)
    assert failure_event.error_message == "boom"

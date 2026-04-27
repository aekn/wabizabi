from __future__ import annotations

import pytest
from pydantic import BaseModel
from wabizabi.agent import Agent, ModelCapabilityError
from wabizabi.context import RunContext
from wabizabi.history import MessageHistory
from wabizabi.messages import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    ReasoningPart,
    RetryFeedbackPart,
    SystemInstructionPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from wabizabi.models import (
    ModelReasoningDeltaEvent,
    ModelResponseCompletedEvent,
    ModelResponsePartEvent,
    ModelResult,
    ModelTextDeltaEvent,
)
from wabizabi.output import (
    OutputDecodingError,
    OutputMode,
    OutputValidationError,
    TextOutputDecoder,
    json_output_config,
    schema_output_config,
    text_output_config,
    tool_output_config,
)
from wabizabi.providers.ollama import OllamaSettings
from wabizabi.stream import (
    OutputEvent,
    ReasoningChunkEvent,
    RequestEvent,
    ResponseEvent,
    TextChunkEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from wabizabi.telemetry import NoopTelemetryRecorder
from wabizabi.testing import ScriptedModel, StreamingScriptedModel
from wabizabi.tools import define_function_tool
from wabizabi.types import JsonValue
from wabizabi.usage import RunUsage


@pytest.mark.asyncio
async def test_agent_run_executes_one_turn_and_returns_result() -> None:
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
    )

    result = await agent.run(
        "Hello",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == "Hi"
    assert result.run_id == "run-1"
    assert len(result.all_messages) == 2
    assert result.state.run_step == 1
    assert result.usage == RunUsage(input_tokens=3, output_tokens=2)

    request = model.requests[0]
    assert request.parts == (UserPromptPart(text="Hello"),)
    assert result.requests == (request,)
    assert model.histories[0] == MessageHistory.empty()


@pytest.mark.asyncio
async def test_agent_run_can_override_model_instructions_and_settings() -> None:
    base_model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="base"),), model_name="base"),
                usage=RunUsage(input_tokens=1, output_tokens=1),
            ),
        )
    )
    override_model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="override"),), model_name="override"),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
        )
    )
    agent = Agent[tuple[str, str], str](
        model=base_model,
        decoder=TextOutputDecoder(),
        system_instructions=("Base instruction.",),
    ).with_model_settings(
        OllamaSettings(
            ollama_model="gpt-base",
            ollama_temperature=0.1,
        )
    )

    result = await agent.run(
        "Hello",
        deps=("svc", "cfg"),
        run_id="run-1",
        model=override_model,
        instructions=("Override instruction.",),
        settings=OllamaSettings(ollama_temperature=0.7),
    )

    assert result.output == "override"
    assert base_model.requests == []

    override_request = override_model.requests[0]
    assert override_request.parts == (
        SystemInstructionPart(text="Base instruction."),
        SystemInstructionPart(text="Override instruction."),
        UserPromptPart(text="Hello"),
    )
    assert override_model.received_settings[0] == OllamaSettings(
        ollama_model="gpt-base",
        ollama_temperature=0.7,
    )


@pytest.mark.asyncio
async def test_agent_merges_metadata_and_applies_output_validation() -> None:
    async def add_suffix(
        context: RunContext[tuple[str, str]],
        output: str,
    ) -> str:
        assert context.run_step == 1
        assert context.metadata == {"app": "wazi", "request_id": "req-1"}
        assert context.usage == RunUsage(input_tokens=3, output_tokens=2)
        return f"{output}!"

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
        system_instructions=("Be terse.",),
        metadata={"app": "wazi"},
    ).with_output_validator(add_suffix)

    result = await agent.run(
        "Hello",
        deps=("svc", "cfg"),
        run_id="run-1",
        metadata={"request_id": "req-1"},
    )

    request = model.requests[0]

    assert request.metadata == {"app": "wazi", "request_id": "req-1"}
    assert request.parts == (
        SystemInstructionPart(text="Be terse."),
        UserPromptPart(text="Hello"),
    )
    assert result.output == "Hi!"
    assert result.state.metadata == {"app": "wazi", "request_id": "req-1"}
    assert result.usage == RunUsage(input_tokens=3, output_tokens=2)


class AddArguments(BaseModel):
    left: int
    right: int


class FinalAnswer(BaseModel):
    total: int


class CityAnswer(BaseModel):
    city: str
    country: str


@pytest.mark.asyncio
async def test_agent_passes_registered_tool_definitions_to_model() -> None:
    def add_tool(context: RunContext[tuple[str, str]], arguments: AddArguments) -> int:
        del context
        return arguments.left + arguments.right

    tool = define_function_tool(
        name="add",
        arguments_type=AddArguments,
        func=add_tool,
        description="Add two integers.",
    )

    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="5"),), model_name="scripted"),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
        )
    )
    agent = Agent[tuple[str, str], str](
        model=model,
        decoder=TextOutputDecoder(),
    ).with_tool(tool)

    result = await agent.run(
        "What is 2 + 3?",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == "5"
    assert model.received_tools == [(tool.definition,)]


@pytest.mark.asyncio
async def test_agent_executes_single_tool_call_and_continues() -> None:
    def add_tool(context: RunContext[tuple[str, str]], arguments: AddArguments) -> int:
        assert context.tool_name == "add"
        assert context.tool_call_id == "call-1"
        assert context.run_step == 1
        assert context.usage == RunUsage(input_tokens=3, output_tokens=0)
        return arguments.left + arguments.right

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
    agent = Agent[tuple[str, str], str](
        model=model,
        decoder=TextOutputDecoder(),
    ).with_tool(
        define_function_tool(
            name="add",
            arguments_type=AddArguments,
            func=add_tool,
        )
    )

    result = await agent.run(
        "What is 2 + 3?",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == "5"
    assert result.state.run_step == 2
    assert len(result.all_messages) == 4
    assert result.usage == RunUsage(input_tokens=5, output_tokens=1)

    second_request = model.requests[1]
    assert second_request.parts == (
        ToolReturnPart(
            tool_name="add",
            call_id="call-1",
            content=5,
        ),
    )

    first_history = model.histories[0]
    second_history = model.histories[1]

    assert first_history == MessageHistory.empty()
    assert first_history is not None
    assert second_history is not None
    assert second_history.messages == result.all_messages[:2]


@pytest.mark.asyncio
async def test_agent_supports_repeated_tool_rounds() -> None:
    def add_tool(context: RunContext[tuple[str, str]], arguments: AddArguments) -> int:
        assert context.tool_name == "add"
        return arguments.left + arguments.right

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
                response=ModelResponse(
                    parts=(
                        ToolCallPart(
                            tool_name="add",
                            call_id="call-2",
                            arguments={"left": 5, "right": 4},
                        ),
                    ),
                    model_name="scripted",
                ),
                usage=RunUsage(input_tokens=2, output_tokens=0),
            ),
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="9"),), model_name="scripted"),
                usage=RunUsage(input_tokens=1, output_tokens=1),
            ),
        )
    )
    agent = Agent[tuple[str, str], str](
        model=model,
        decoder=TextOutputDecoder(),
    ).with_tool(
        define_function_tool(
            name="add",
            arguments_type=AddArguments,
            func=add_tool,
        )
    )

    result = await agent.run(
        "Compute step by step.",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == "9"
    assert result.state.run_step == 3
    assert result.usage == RunUsage(input_tokens=6, output_tokens=1)
    assert len(result.all_messages) == 6

    assert model.requests[1].parts == (
        ToolReturnPart(
            tool_name="add",
            call_id="call-1",
            content=5,
        ),
    )
    assert model.requests[2].parts == (
        ToolReturnPart(
            tool_name="add",
            call_id="call-2",
            content=9,
        ),
    )


@pytest.mark.asyncio
async def test_agent_supports_multiple_tool_calls_in_one_response() -> None:
    def add_tool(context: RunContext[tuple[str, str]], arguments: AddArguments) -> int:
        assert context.tool_name == "add"
        return arguments.left + arguments.right

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
                        ToolCallPart(
                            tool_name="add",
                            call_id="call-2",
                            arguments={"left": 4, "right": 5},
                        ),
                    ),
                    model_name="scripted",
                ),
                usage=RunUsage(input_tokens=3, output_tokens=0),
            ),
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="5 and 9"),), model_name="scripted"),
                usage=RunUsage(input_tokens=2, output_tokens=2),
            ),
        )
    )
    agent = Agent[tuple[str, str], str](
        model=model,
        decoder=TextOutputDecoder(),
    ).with_tool(
        define_function_tool(
            name="add",
            arguments_type=AddArguments,
            func=add_tool,
        )
    )

    result = await agent.run(
        "Compute both sums.",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == "5 and 9"
    assert result.state.run_step == 2
    assert result.usage == RunUsage(input_tokens=5, output_tokens=2)
    assert model.requests[1].parts == (
        ToolReturnPart(
            tool_name="add",
            call_id="call-1",
            content=5,
        ),
        ToolReturnPart(
            tool_name="add",
            call_id="call-2",
            content=9,
        ),
    )


@pytest.mark.asyncio
async def test_agent_respects_max_tool_rounds() -> None:
    def add_tool(context: RunContext[tuple[str, str]], arguments: AddArguments) -> int:
        del context
        return arguments.left + arguments.right

    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(
                        ToolCallPart(
                            tool_name="add",
                            call_id="call-1",
                            arguments={"left": 1, "right": 1},
                        ),
                    ),
                    model_name="scripted",
                ),
                usage=RunUsage(input_tokens=1, output_tokens=0),
            ),
            ModelResult(
                response=ModelResponse(
                    parts=(
                        ToolCallPart(
                            tool_name="add",
                            call_id="call-2",
                            arguments={"left": 2, "right": 2},
                        ),
                    ),
                    model_name="scripted",
                ),
                usage=RunUsage(input_tokens=1, output_tokens=0),
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
        .with_max_tool_rounds(1)
    )

    with pytest.raises(RuntimeError, match="Exceeded maximum tool rounds"):
        await agent.run(
            "Loop",
            deps=("svc", "cfg"),
            run_id="run-1",
        )


@pytest.mark.asyncio
async def test_agent_iter_yields_request_response_and_output_events() -> None:
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
    )

    events = [
        event
        async for event in agent.iter(
            "Hello",
            deps=("svc", "cfg"),
            run_id="run-1",
        )
    ]

    assert len(events) == 4
    assert isinstance(events[0], RequestEvent)
    assert isinstance(events[1], ResponseEvent)
    assert isinstance(events[2], TextChunkEvent)
    assert isinstance(events[3], OutputEvent)

    request_event = events[0]
    response_event = events[1]
    output_event = events[3]

    assert request_event.request.parts == (UserPromptPart(text="Hello"),)
    assert response_event.response.parts == (TextPart(text="Hi"),)
    assert response_event.state.usage == RunUsage(input_tokens=3, output_tokens=2)
    assert output_event.output == "Hi"
    assert output_event.state.run_step == 1
    assert output_event.state.usage == RunUsage(input_tokens=3, output_tokens=2)


@pytest.mark.asyncio
async def test_agent_iter_yields_tool_events_for_tool_execution() -> None:
    def add_tool(context: RunContext[tuple[str, str]], arguments: AddArguments) -> int:
        assert context.tool_name == "add"
        assert context.usage == RunUsage(input_tokens=3, output_tokens=0)
        return arguments.left + arguments.right

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
    agent = Agent[tuple[str, str], str](
        model=model,
        decoder=TextOutputDecoder(),
    ).with_tool(
        define_function_tool(
            name="add",
            arguments_type=AddArguments,
            func=add_tool,
        )
    )

    events = [
        event
        async for event in agent.iter(
            "What is 2 + 3?",
            deps=("svc", "cfg"),
            run_id="run-1",
        )
    ]

    assert len(events) == 8
    assert isinstance(events[0], RequestEvent)
    assert isinstance(events[1], ResponseEvent)
    assert isinstance(events[2], ToolCallEvent)
    assert isinstance(events[3], ToolResultEvent)
    assert isinstance(events[4], RequestEvent)
    assert isinstance(events[5], ResponseEvent)
    assert isinstance(events[6], TextChunkEvent)
    assert isinstance(events[7], OutputEvent)

    tool_call_event = events[2]
    tool_result_event = events[3]
    second_response_event = events[5]
    final_event = events[7]

    assert tool_call_event.tool_call == ToolCallPart(
        tool_name="add",
        call_id="call-1",
        arguments={"left": 2, "right": 3},
    )
    assert tool_result_event.tool_return == ToolReturnPart(
        tool_name="add",
        call_id="call-1",
        content=5,
    )
    assert second_response_event.state.usage == RunUsage(input_tokens=5, output_tokens=1)
    assert final_event.output == "5"
    assert final_event.state.usage == RunUsage(input_tokens=5, output_tokens=1)


@pytest.mark.asyncio
async def test_agent_stream_output_yields_final_output() -> None:
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
    )

    outputs = [
        output
        async for output in agent.stream_output(
            "Hello",
            deps=("svc", "cfg"),
            run_id="run-1",
        )
    ]

    assert outputs == ["Hi"]


@pytest.mark.asyncio
async def test_agent_stream_responses_yields_all_responses() -> None:
    def add_tool(context: RunContext[tuple[str, str]], arguments: AddArguments) -> int:
        del context
        return arguments.left + arguments.right

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
    agent = Agent[tuple[str, str], str](
        model=model,
        decoder=TextOutputDecoder(),
    ).with_tool(
        define_function_tool(
            name="add",
            arguments_type=AddArguments,
            func=add_tool,
        )
    )

    responses = [
        response
        async for response in agent.stream_responses(
            "What is 2 + 3?",
            deps=("svc", "cfg"),
            run_id="run-1",
        )
    ]

    assert responses == [
        ModelResponse(
            parts=(
                ToolCallPart(
                    tool_name="add",
                    call_id="call-1",
                    arguments={"left": 2, "right": 3},
                ),
            ),
            model_name="scripted",
        ),
        ModelResponse(parts=(TextPart(text="5"),), model_name="scripted"),
    ]


@pytest.mark.asyncio
async def test_agent_stream_text_yields_text_output() -> None:
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
    )

    chunks = [
        chunk
        async for chunk in agent.stream_text(
            "Hello",
            deps=("svc", "cfg"),
            run_id="run-1",
        )
    ]

    assert chunks == ["Hi"]


@pytest.mark.asyncio
async def test_streamed_run_is_single_consumer() -> None:
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
    )
    streamed_run = agent.stream(
        "Hello",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    first_pass = [event async for event in streamed_run]
    assert len(first_pass) == 4

    with pytest.raises(RuntimeError, match="already been consumed"):
        _ = [event async for event in streamed_run]


def test_agent_runsync_executes_one_turn() -> None:
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
    )

    result = agent.runsync(
        "Hello",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == "Hi"
    assert result.run_id == "run-1"
    assert result.usage == RunUsage(input_tokens=3, output_tokens=2)


@pytest.mark.asyncio
async def test_agent_retries_output_validation_with_retry_feedback() -> None:
    def require_good(
        context: RunContext[tuple[str, str]],
        output: str,
    ) -> str:
        assert context.run_step >= 1
        if output != "good":
            raise OutputValidationError(
                "output must be good",
                retry_feedback="Return exactly the word good.",
            )
        return output

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
    agent = Agent[tuple[str, str], str](
        model=model,
        decoder=TextOutputDecoder(),
    ).with_output_validator(require_good)

    result = await agent.run(
        "Hello",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == "good"
    assert result.state.retries == 1
    assert result.usage == RunUsage(input_tokens=3, output_tokens=2)
    assert model.requests[1].parts == (RetryFeedbackPart(message="Return exactly the word good."),)


@pytest.mark.asyncio
async def test_agent_raises_after_output_retries_are_exhausted() -> None:
    def require_good(
        context: RunContext[tuple[str, str]],
        output: str,
    ) -> str:
        del context
        raise OutputValidationError(
            "output must be good",
            retry_feedback="Return exactly the word good.",
        )

    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="bad"),), model_name="scripted"),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="still bad"),), model_name="scripted"),
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
        .with_max_output_retries(1)
    )

    with pytest.raises(OutputValidationError, match="output must be good"):
        await agent.run(
            "Hello",
            deps=("svc", "cfg"),
            run_id="run-1",
        )

    assert model.requests[1].parts == (RetryFeedbackPart(message="Return exactly the word good."),)


@pytest.mark.asyncio
async def test_agent_can_use_explicit_schema_output_config() -> None:
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(TextPart(text='{"city":"Paris","country":"France"}'),),
                    model_name="scripted",
                ),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
        )
    )
    agent = Agent[tuple[str, str], CityAnswer](
        model=model,
        output=schema_output_config(CityAnswer),
    )

    result = await agent.run(
        "Where am I?",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == CityAnswer(city="Paris", country="France")
    assert agent.output_mode is OutputMode.SCHEMA


@pytest.mark.asyncio
async def test_agent_can_use_explicit_tool_output_config() -> None:
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(
                        ToolCallPart(
                            tool_name="final_answer",
                            call_id="call-1",
                            arguments={"total": 42},
                        ),
                    ),
                    model_name="scripted",
                ),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
        )
    )
    agent = Agent[tuple[str, str], FinalAnswer](
        model=model,
        output=tool_output_config(FinalAnswer, tool_name="final_answer"),
    )

    result = await agent.run(
        "What is six times seven?",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == FinalAnswer(total=42)
    assert agent.output_mode is OutputMode.TOOL


@pytest.mark.asyncio
async def test_agent_can_use_explicit_json_output_config() -> None:
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(TextPart(text='{"city":"Paris","country":"France"}'),),
                    model_name="scripted",
                ),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
        )
    )
    agent = Agent[tuple[str, str], JsonValue](
        model=model,
        output=json_output_config(),
    )

    result = await agent.run(
        "Return the city answer.",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == {"city": "Paris", "country": "France"}
    assert agent.output_mode is OutputMode.JSON


def test_agent_rejects_output_tool_name_conflicts() -> None:
    def add_tool(context: RunContext[tuple[str, str]], arguments: AddArguments) -> int:
        del context
        return arguments.left + arguments.right

    with pytest.raises(ValueError, match="Output tool names conflict with registered tools"):
        Agent[tuple[str, str], FinalAnswer](
            model=ScriptedModel(()),
            output=tool_output_config(FinalAnswer, tool_name="add"),
        ).with_tool(
            define_function_tool(
                name="add",
                arguments_type=AddArguments,
                func=add_tool,
            )
        )


@pytest.mark.asyncio
async def test_agent_iter_projects_text_chunk_events_from_response_parts() -> None:
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(
                        TextPart(text="Hello"),
                        TextPart(text=", "),
                        TextPart(text="world"),
                    ),
                    model_name="scripted",
                ),
                usage=RunUsage(input_tokens=2, output_tokens=3),
            ),
        )
    )
    agent = Agent[tuple[str, str], str](
        model=model,
        output=text_output_config(),
    )

    events = [
        event
        async for event in agent.iter(
            "Greet me",
            deps=("svc", "cfg"),
            run_id="run-1",
        )
    ]

    assert isinstance(events[0], RequestEvent)
    assert isinstance(events[1], ResponseEvent)
    assert isinstance(events[2], TextChunkEvent)
    assert isinstance(events[3], TextChunkEvent)
    assert isinstance(events[4], TextChunkEvent)
    assert isinstance(events[5], OutputEvent)

    assert events[2].text == "Hello"
    assert events[3].text == ", "
    assert events[4].text == "world"


@pytest.mark.asyncio
async def test_agent_stream_text_yields_projected_response_text_chunks() -> None:
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(
                        TextPart(text="Hello"),
                        TextPart(text=", "),
                        TextPart(text="world"),
                    ),
                    model_name="scripted",
                ),
                usage=RunUsage(input_tokens=2, output_tokens=3),
            ),
        )
    )
    agent = Agent[tuple[str, str], str](
        model=model,
        output=text_output_config(),
    )

    chunks = [
        chunk
        async for chunk in agent.stream_text(
            "Greet me",
            deps=("svc", "cfg"),
            run_id="run-1",
        )
    ]

    assert chunks == ["Hello", ", ", "world"]


@pytest.mark.asyncio
async def test_agent_run_can_continue_from_existing_message_history() -> None:
    prior_request = ModelRequest(parts=(UserPromptPart(text="Earlier question"),))
    prior_response = ModelResponse(parts=(TextPart(text="Earlier answer"),), model_name="scripted")

    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="New answer"),), model_name="scripted"),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
        )
    )
    agent = Agent[tuple[str, str], str](
        model=model,
        output=text_output_config(),
    )

    result = await agent.run(
        "New question",
        deps=("svc", "cfg"),
        run_id="run-1",
        message_history=(prior_request, prior_response),
    )

    assert result.output == "New answer"
    assert model.histories[0] == MessageHistory(messages=(prior_request, prior_response))
    assert result.new_messages == (
        ModelRequest(parts=(UserPromptPart(text="New question"),), metadata=None),
        ModelResponse(parts=(TextPart(text="New answer"),), model_name="scripted"),
    )
    assert result.all_messages == (
        prior_request,
        prior_response,
        *result.new_messages,
    )


@pytest.mark.asyncio
async def test_agent_stream_responses_can_continue_from_existing_message_history() -> None:
    prior_request = ModelRequest(parts=(UserPromptPart(text="Earlier question"),))
    prior_response = ModelResponse(parts=(TextPart(text="Earlier answer"),), model_name="scripted")

    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(TextPart(text="Follow-up answer"),), model_name="scripted"
                ),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
        )
    )
    agent = Agent[tuple[str, str], str](
        model=model,
        output=text_output_config(),
    )

    responses = [
        response
        async for response in agent.stream_responses(
            "Follow-up question",
            deps=("svc", "cfg"),
            run_id="run-1",
            message_history=MessageHistory(messages=(prior_request, prior_response)),
        )
    ]

    assert responses == [
        ModelResponse(parts=(TextPart(text="Follow-up answer"),), model_name="scripted"),
    ]
    assert model.histories[0] == MessageHistory(messages=(prior_request, prior_response))


@pytest.mark.asyncio
async def test_agent_run_uses_model_stream_response() -> None:
    model = StreamingScriptedModel(
        (
            (
                ModelTextDeltaEvent(text="Hel"),
                ModelResponsePartEvent(part=TextPart(text="Hello")),
                ModelResponseCompletedEvent(
                    model_name="streamed",
                    usage=RunUsage(input_tokens=2, output_tokens=1),
                ),
            ),
        )
    )
    agent = Agent[tuple[str, str], str](
        model=model,
        decoder=TextOutputDecoder(),
    )

    result = await agent.run(
        "Hello",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == "Hello"
    assert result.usage == RunUsage(input_tokens=2, output_tokens=1)
    assert model.requests == [
        ModelRequest(parts=(UserPromptPart(text="Hello"),), metadata=None),
    ]


@pytest.mark.asyncio
async def test_agent_stream_text_uses_live_model_text_deltas() -> None:
    model = StreamingScriptedModel(
        (
            (
                ModelTextDeltaEvent(text="Hel"),
                ModelTextDeltaEvent(text="lo"),
                ModelResponsePartEvent(part=TextPart(text="Hello")),
                ModelResponseCompletedEvent(
                    model_name="streamed",
                    usage=RunUsage(input_tokens=2, output_tokens=1),
                ),
            ),
        )
    )
    agent = Agent[tuple[str, str], str](
        model=model,
        decoder=TextOutputDecoder(),
    )

    chunks = [
        chunk
        async for chunk in agent.stream_text(
            "Hello",
            deps=("svc", "cfg"),
            run_id="run-1",
        )
    ]

    assert chunks == ["Hel", "lo"]


@pytest.mark.asyncio
async def test_agent_streamed_response_preserves_finish_reason_and_metadata() -> None:
    model = StreamingScriptedModel(
        (
            (
                ModelResponsePartEvent(part=TextPart(text="Hello")),
                ModelResponseCompletedEvent(
                    model_name="streamed",
                    usage=RunUsage(input_tokens=2, output_tokens=1),
                    finish_reason=FinishReason.STOP,
                    metadata={"provider_request_id": "req-1"},
                ),
            ),
        )
    )
    agent = Agent[tuple[str, str], str](
        model=model,
        decoder=TextOutputDecoder(),
    )

    responses = [
        response
        async for response in agent.stream_responses(
            "Hello",
            deps=("svc", "cfg"),
            run_id="run-1",
        )
    ]

    assert responses == [
        ModelResponse(
            parts=(TextPart(text="Hello"),),
            model_name="streamed",
            finish_reason=FinishReason.STOP,
            metadata={"provider_request_id": "req-1"},
        )
    ]


@pytest.mark.asyncio
async def test_agent_retries_output_decoding_failure_with_feedback() -> None:
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(ReasoningPart(text="thinking only"),), model_name="scripted"
                ),
                usage=RunUsage(input_tokens=3, output_tokens=1),
            ),
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="final"),), model_name="scripted"),
                usage=RunUsage(input_tokens=1, output_tokens=1),
            ),
        )
    )
    agent = Agent[tuple[str, str], str](
        model=model,
        decoder=TextOutputDecoder(),
    )

    result = await agent.run(
        "Hello",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == "final"
    assert len(model.requests) == 2
    assert model.requests[1].parts == (
        RetryFeedbackPart(
            message=(
                "Return a final answer as plain text. Do not emit only reasoning or tool calls. "
                "Previous decoding error: ModelResponse did not contain any text parts.."
            )
        ),
    )


@pytest.mark.asyncio
async def test_agent_raises_output_decoding_error_after_retry_limit() -> None:
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(ReasoningPart(text="thinking only"),), model_name="scripted"
                ),
                usage=RunUsage(input_tokens=3, output_tokens=1),
            ),
        )
    )
    agent = Agent[tuple[str, str], str](
        model=model,
        decoder=TextOutputDecoder(),
        max_output_retries=0,
    )

    with pytest.raises(OutputDecodingError):
        await agent.run(
            "Hello",
            deps=("svc", "cfg"),
            run_id="run-1",
        )


def test_agent_requires_exactly_one_of_decoder_or_output() -> None:
    with pytest.raises(ValueError, match="Exactly one of decoder or output must be provided"):
        Agent[tuple[str, str], str](
            model=ScriptedModel(()),
        )

    with pytest.raises(ValueError, match="Exactly one of decoder or output must be provided"):
        Agent[tuple[str, str], str](
            model=ScriptedModel(()),
            decoder=TextOutputDecoder(),
            output=text_output_config(),
        )


def test_agent_rejects_invalid_retry_and_tool_round_limits() -> None:
    with pytest.raises(ValueError, match="max_tool_rounds must be at least 1"):
        Agent[tuple[str, str], str](
            model=ScriptedModel(()),
            decoder=TextOutputDecoder(),
            max_tool_rounds=0,
        )

    with pytest.raises(ValueError, match="max_output_retries must be non-negative"):
        Agent[tuple[str, str], str](
            model=ScriptedModel(()),
            decoder=TextOutputDecoder(),
            max_output_retries=-1,
        )


def test_builder_rejects_invalid_tool_round_and_retry_limits() -> None:
    agent = Agent[tuple[str, str], str](
        model=ScriptedModel(()),
        decoder=TextOutputDecoder(),
    )

    with pytest.raises(ValueError, match="max_tool_rounds must be at least 1"):
        agent.with_max_tool_rounds(0)

    with pytest.raises(ValueError, match="max_output_retries must be non-negative"):
        agent.with_max_output_retries(-1)


@pytest.mark.asyncio
async def test_agent_with_output_resets_validators_and_updates_output_mode() -> None:
    """with_output resets validators and telemetry since they are typed to the old output."""

    async def require_non_empty(
        context: RunContext[tuple[str, str]],
        output: FinalAnswer,
    ) -> FinalAnswer:
        del context
        assert output.total > 0
        return output

    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(TextPart(text='{"total":5}'),),
                    model_name="scripted",
                ),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
        )
    )
    agent = (
        Agent[tuple[str, str], FinalAnswer](
            model=model,
            output=tool_output_config(FinalAnswer, tool_name="final_answer"),
        )
        .with_output_validator(require_non_empty)
        .with_output(schema_output_config(FinalAnswer))
    )

    result = await agent.run(
        "Return the answer.",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == FinalAnswer(total=5)
    assert agent.output_mode is OutputMode.SCHEMA
    assert isinstance(agent.telemetry, NoopTelemetryRecorder)
    assert agent.output_pipeline.mode is OutputMode.SCHEMA
    assert len(agent.output_pipeline.validators) == 0


@pytest.mark.asyncio
async def test_agent_with_output_then_validator_applies_correctly() -> None:
    """Validators added after with_output are preserved."""

    async def require_non_empty(
        context: RunContext[tuple[str, str]],
        output: FinalAnswer,
    ) -> FinalAnswer:
        del context
        assert output.total > 0
        return output

    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(TextPart(text='{"total":5}'),),
                    model_name="scripted",
                ),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
        )
    )
    agent = (
        Agent[tuple[str, str], FinalAnswer](
            model=model,
            output=tool_output_config(FinalAnswer, tool_name="final_answer"),
        )
        .with_output(schema_output_config(FinalAnswer))
        .with_output_validator(require_non_empty)
    )

    result = await agent.run(
        "Return the answer.",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == FinalAnswer(total=5)
    assert agent.output_mode is OutputMode.SCHEMA
    assert len(agent.output_pipeline.validators) == 1


@pytest.mark.asyncio
async def test_agent_iter_projects_reasoning_chunk_events_from_response_parts() -> None:
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(
                        ReasoningPart(text="thinking step 1"),
                        ReasoningPart(text="thinking step 2"),
                        TextPart(text="final answer"),
                    ),
                    model_name="scripted",
                ),
                usage=RunUsage(input_tokens=2, output_tokens=3),
            ),
        )
    )
    agent = Agent[tuple[str, str], str](
        model=model,
        output=text_output_config(),
    )

    events = [
        event
        async for event in agent.iter(
            "Think about this",
            deps=("svc", "cfg"),
            run_id="run-1",
        )
    ]

    reasoning_events = [e for e in events if isinstance(e, ReasoningChunkEvent)]
    text_events = [e for e in events if isinstance(e, TextChunkEvent)]
    assert len(reasoning_events) == 2
    assert reasoning_events[0].text == "thinking step 1"
    assert reasoning_events[1].text == "thinking step 2"
    assert len(text_events) == 1
    assert text_events[0].text == "final answer"


@pytest.mark.asyncio
async def test_agent_iter_reasoning_deltas_suppress_fallback_projection() -> None:
    """When the provider emits ModelReasoningDeltaEvent, the runtime should
    not re-project reasoning chunks from the final response parts."""
    model = StreamingScriptedModel(
        (
            (
                ModelReasoningDeltaEvent(text="streaming thought"),
                ModelTextDeltaEvent(text="streamed text"),
                ModelResponsePartEvent(part=ReasoningPart(text="streaming thought")),
                ModelResponsePartEvent(part=TextPart(text="streamed text")),
                ModelResponseCompletedEvent(
                    model_name="scripted",
                    usage=RunUsage(input_tokens=1, output_tokens=2),
                ),
            ),
        )
    )
    agent = Agent[tuple[str, str], str](
        model=model,
        output=text_output_config(),
    )

    events = [
        event
        async for event in agent.iter(
            "Think",
            deps=("svc", "cfg"),
            run_id="run-1",
        )
    ]

    reasoning_events = [e for e in events if isinstance(e, ReasoningChunkEvent)]
    text_events = [e for e in events if isinstance(e, TextChunkEvent)]
    assert len(reasoning_events) == 1
    assert reasoning_events[0].text == "streaming thought"
    assert len(text_events) == 1
    assert text_events[0].text == "streamed text"


def test_with_system_instruction_rejects_empty() -> None:
    agent = Agent[None, str](
        model=ScriptedModel(()),
        output=text_output_config(),
    )
    with pytest.raises(ValueError, match="empty"):
        agent.with_system_instruction("")
    with pytest.raises(ValueError, match="empty"):
        agent.with_system_instruction("   \n\t")


def test_with_output_preserves_handoff_terminal_tool_names() -> None:
    from wabizabi.handoff import Handoff

    handoff = Handoff(name="expert")
    agent = Agent[None, str](
        model=ScriptedModel(()),
        output=text_output_config(),
        handoffs=(handoff,),
    )
    assert handoff.tool_name in agent.output.terminal_tool_names

    swapped = agent.with_output(json_output_config())
    assert handoff.tool_name in swapped.output.terminal_tool_names
    assert swapped.output_mode is OutputMode.JSON


def test_with_output_rejects_conflicting_tool_names() -> None:
    class _NoArgs(BaseModel):
        pass

    def _conflict(context: RunContext[None], arguments: _NoArgs) -> str:
        del context, arguments
        return "x"

    conflicting_tool = define_function_tool(
        name="handoff_expert",
        description="conflict",
        arguments_type=_NoArgs,
        func=_conflict,
    )
    agent = Agent[None, str](
        model=ScriptedModel(()),
        output=text_output_config(),
    ).with_tool(conflicting_tool)

    colliding_output = tool_output_config(FinalAnswer, tool_name="handoff_expert")
    with pytest.raises(ValueError, match="conflict"):
        agent.with_output(colliding_output)


def _no_tools_model() -> ScriptedModel:
    from wabizabi.models import ModelProfile

    return ScriptedModel(
        (),
        profile=ModelProfile(
            provider_name="test",
            model_name="no-tools",
            supports_tools=False,
        ),
    )


def test_agent_rejects_tools_on_model_without_tool_support() -> None:
    class _EchoArgs(BaseModel):
        text: str

    def _echo(_context: RunContext[None], arguments: _EchoArgs) -> str:
        return arguments.text

    tool = define_function_tool(name="echo", arguments_type=_EchoArgs, func=_echo)

    with pytest.raises(ModelCapabilityError, match="does not support tool calls"):
        Agent[None, str](
            model=_no_tools_model(),
            output=text_output_config(),
        ).with_tool(tool)


def test_agent_rejects_handoff_on_model_without_tool_support() -> None:
    from wabizabi.handoff import Handoff

    with pytest.raises(ModelCapabilityError, match="does not support tool calls"):
        Agent[None, str](
            model=_no_tools_model(),
            output=text_output_config(),
        ).with_handoff(Handoff(name="billing"))


@pytest.mark.asyncio
async def test_agent_mixed_executable_and_terminal_tool_calls_retries_with_feedback() -> None:
    class _EchoArgs(BaseModel):
        text: str

    def _echo(_context: RunContext[None], arguments: _EchoArgs) -> str:
        return arguments.text

    class FinalAnswer(BaseModel):
        answer: str

    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(
                        ToolCallPart(
                            tool_name="echo",
                            call_id="call-1",
                            arguments={"text": "hi"},
                        ),
                        ToolCallPart(
                            tool_name="final_answer",
                            call_id="call-2",
                            arguments={"answer": "done"},
                        ),
                    ),
                    model_name="scripted",
                ),
                usage=RunUsage.zero(),
            ),
            ModelResult(
                response=ModelResponse(
                    parts=(
                        ToolCallPart(
                            tool_name="final_answer",
                            call_id="call-3",
                            arguments={"answer": "done"},
                        ),
                    ),
                    model_name="scripted",
                ),
                usage=RunUsage.zero(),
            ),
        )
    )
    agent = Agent[None, FinalAnswer](
        model=model,
        output=tool_output_config(FinalAnswer, tool_name="final_answer"),
    ).with_tool(define_function_tool(name="echo", arguments_type=_EchoArgs, func=_echo))

    result = await agent.run("do it", deps=None)
    assert result.output == FinalAnswer(answer="done")
    assert len(model.calls) == 2
    retry_request = model.calls[1].request
    assert any(
        isinstance(part, RetryFeedbackPart) and "tool call" in part.message.lower()
        for part in retry_request.parts
    )

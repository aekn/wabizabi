from __future__ import annotations

import pytest
from wabizabi import Hooks, InMemoryTelemetryRecorder, Toolset
from wabizabi.agent import Agent
from wabizabi.context import RunContext
from wabizabi.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from wabizabi.models import ModelResult
from wabizabi.output import text_output_config
from wabizabi.telemetry import (
    RequestRecordedEvent,
    ResponseRecordedEvent,
    ToolCallRecordedEvent,
    ToolResultRecordedEvent,
)
from wabizabi.testing import ScriptedModel
from wabizabi.tools import tool


@tool
async def add(ctx: RunContext[tuple[str, str]], left: int, right: int) -> int:
    assert ctx.tool_name == "add"
    return left + right


def _identity_request(
    context: RunContext[tuple[str, str]],
    request: ModelRequest,
) -> ModelRequest:
    del context
    return request


def _identity_request_none(
    context: RunContext[None],
    request: ModelRequest,
) -> ModelRequest:
    del context
    return request


@pytest.mark.asyncio
async def test_hooks_apply_in_deterministic_order_and_affect_runtime_outputs() -> None:
    order: list[str] = []

    def prepare_tools(
        context: RunContext[tuple[str, str]],
        toolset: Toolset[tuple[str, str]],
    ) -> Toolset[tuple[str, str]]:
        assert context.run_step in {0, 1}
        order.append(f"prepare:{context.run_step}")
        return Toolset(tools=(toolset.tools[0],))

    async def before_request(
        context: RunContext[tuple[str, str]],
        request: ModelRequest,
    ) -> ModelRequest:
        assert context.run_step in {0, 1}
        order.append(f"before_request:{context.run_step}")
        if context.run_step == 0:
            first_part = request.parts[0]
            assert isinstance(first_part, UserPromptPart)
            return request.model_copy(
                update={
                    "parts": (UserPromptPart(text=f"{first_part.text} Please use the add tool."),)
                }
            )
        return request

    def after_response(
        context: RunContext[tuple[str, str]],
        response: ModelResponse,
    ) -> ModelResponse:
        assert context.run_step in {1, 2}
        order.append(f"after_response:{context.run_step}")
        if context.run_step == 2:
            return response.model_copy(update={"parts": (TextPart(text="done!"),)})
        return response

    async def before_tool_call(
        context: RunContext[tuple[str, str]],
        tool_call: ToolCallPart,
    ) -> ToolCallPart:
        assert context.tool_name == "add"
        order.append(f"before_tool_call:{context.run_step}")
        return tool_call.model_copy(update={"arguments": {"left": 2, "right": 4}})

    def after_tool_call(
        context: RunContext[tuple[str, str]],
        tool_call: ToolCallPart,
        tool_return: ToolReturnPart,
    ) -> ToolReturnPart:
        assert context.tool_call_id == tool_call.call_id
        order.append(f"after_tool_call:{context.run_step}")
        return tool_return.model_copy(update={"content": 7})

    hooks = (
        Hooks[tuple[str, str]]
        .empty()
        .with_prepare_tools(prepare_tools)
        .with_before_request(before_request)
        .with_after_response(after_response)
        .with_before_tool_call(before_tool_call)
        .with_after_tool_call(after_tool_call)
    )

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
            ),
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="done"),), model_name="scripted"),
            ),
        )
    )

    agent = (
        Agent[tuple[str, str], str](
            model=model,
            output=text_output_config(),
            hooks=Hooks[tuple[str, str]].empty().with_before_request(_identity_request),
            telemetry=recorder,
        )
        .with_tool(add)
        .with_hooks(hooks)
    )

    result = await agent.run(
        "What is 2 + 3?",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == "done!"
    assert model.received_tools[0][0].name == "add"
    assert len(model.received_tools[0]) == 1

    first_request_part = model.requests[0].parts[0]
    assert isinstance(first_request_part, UserPromptPart)
    assert first_request_part.text == "What is 2 + 3? Please use the add tool."

    second_request_part = model.requests[1].parts[0]
    assert isinstance(second_request_part, ToolReturnPart)
    assert second_request_part.content == 7

    request_event = recorder.events[1]
    tool_call_event = recorder.events[3]
    tool_result_event = recorder.events[4]
    response_event = recorder.events[6]

    assert isinstance(request_event, RequestRecordedEvent)
    first_recorded_part = request_event.request.parts[0]
    assert isinstance(first_recorded_part, UserPromptPart)
    assert first_recorded_part.text.endswith("Please use the add tool.")

    assert isinstance(tool_call_event, ToolCallRecordedEvent)
    assert tool_call_event.tool_call.arguments == {"left": 2, "right": 4}

    assert isinstance(tool_result_event, ToolResultRecordedEvent)
    assert tool_result_event.tool_return.content == 7

    assert isinstance(response_event, ResponseRecordedEvent)
    assert response_event.response.parts == (TextPart(text="done!"),)

    assert order == [
        "prepare:0",
        "before_request:0",
        "after_response:1",
        "before_tool_call:1",
        "after_tool_call:1",
        "prepare:1",
        "before_request:1",
        "after_response:2",
    ]


@pytest.mark.asyncio
async def test_hooks_merge_appends_in_registration_order() -> None:
    order: list[str] = []

    def first(
        context: RunContext[None],
        request: ModelRequest,
    ) -> ModelRequest:
        del context
        order.append("first")
        return request

    async def second(
        context: RunContext[None],
        request: ModelRequest,
    ) -> ModelRequest:
        del context
        order.append("second")
        return request

    hooks = (
        Hooks[None]
        .empty()
        .with_before_request(first)
        .merge(Hooks[None].empty().with_before_request(second))
    )

    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="ok"),), model_name="scripted"),
            ),
        )
    )
    agent = Agent[None, str](
        model=model,
        output=text_output_config(),
        hooks=Hooks[None].empty().with_before_request(_identity_request_none).merge(hooks),
    )

    result = await agent.run("Hello", deps=None, run_id="run-2")

    assert result.output == "ok"
    assert order == ["first", "second"]

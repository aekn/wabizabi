from __future__ import annotations

import pytest
from wabizabi import Agent, RunContext, TrimHistoryProcessor, TrimOldestRequestsProcessor
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
from wabizabi.testing import ScriptedModel
from wabizabi.tools import tool


@tool
async def add(ctx: RunContext[None], left: int, right: int) -> int:
    del ctx
    return left + right


def _request(text: str) -> ModelRequest:
    return ModelRequest(parts=(UserPromptPart(text=text),))


def _response(text: str) -> ModelResponse:
    return ModelResponse(parts=(TextPart(text=text),), model_name="scripted")


@pytest.mark.asyncio
async def test_history_processors_affect_only_model_visible_history() -> None:
    seen_lengths: list[int] = []

    def record_tail(
        ctx: RunContext[None],
        messages: tuple[ModelRequest | ModelResponse, ...],
    ) -> tuple[ModelRequest | ModelResponse, ...]:
        del ctx
        seen_lengths.append(len(messages))
        return messages[-2:]

    initial_history = (
        _request("first"),
        _response("first-response"),
        _request("second"),
        _response("second-response"),
    )
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
                )
            ),
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="done"),), model_name="scripted")
            ),
        )
    )
    agent = Agent[None, str](model=model, output=text_output_config()).with_tool(add)
    agent = agent.with_history_processor(record_tail)

    result = await agent.run("Compute 2 + 3", deps=None, message_history=initial_history)

    assert result.output == "done"
    assert seen_lengths == [4, 6]
    assert model.histories[0] is not None
    assert model.histories[0].messages == initial_history[-2:]
    assert model.histories[1] is not None
    assert model.histories[1].messages == (
        _request("Compute 2 + 3"),
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
    )
    assert result.state.message_history.messages[:4] == initial_history
    assert result.state.message_history.messages[4:] == (
        _request("Compute 2 + 3"),
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
        ModelRequest(parts=(ToolReturnPart(tool_name="add", call_id="call-1", content=5),)),
        _response("done"),
    )


@pytest.mark.asyncio
async def test_history_processors_support_sync_and_async_processors_in_order() -> None:
    order: list[str] = []

    def drop_oldest(
        ctx: RunContext[None],
        messages: tuple[ModelRequest | ModelResponse, ...],
    ) -> tuple[ModelRequest | ModelResponse, ...]:
        del ctx
        order.append("sync")
        return messages[1:]

    async def keep_last_two(
        ctx: RunContext[None],
        messages: tuple[ModelRequest | ModelResponse, ...],
    ) -> tuple[ModelRequest | ModelResponse, ...]:
        del ctx
        order.append("async")
        return messages[-2:]

    model = ScriptedModel((ModelResult(response=_response("ok")),))
    agent = Agent[None, str](model=model, output=text_output_config())
    agent = agent.with_history_processor(drop_oldest).with_history_processor(keep_last_two)

    history = (
        _request("one"),
        _response("one-response"),
        _request("two"),
        _response("two-response"),
    )
    result = await agent.run("Hello", deps=None, message_history=history)

    assert result.output == "ok"
    assert order == ["sync", "async"]
    assert model.histories[0] is not None
    assert model.histories[0].messages == history[-2:]
    assert result.state.message_history.messages[:4] == history


@pytest.mark.asyncio
async def test_trim_history_processor_keeps_last_messages_only() -> None:
    model = ScriptedModel((ModelResult(response=_response("ok")),))
    agent = Agent[None, str](model=model, output=text_output_config()).with_history_processor(
        TrimHistoryProcessor(max_messages=3)
    )

    history = (
        _request("one"),
        _response("one-response"),
        _request("two"),
        _response("two-response"),
    )
    await agent.run("Hello", deps=None, message_history=history)

    assert model.histories[0] is not None
    assert model.histories[0].messages == history[-3:]


@pytest.mark.asyncio
async def test_trim_oldest_requests_processor_keeps_latest_request_pairs() -> None:
    model = ScriptedModel((ModelResult(response=_response("ok")),))
    agent = Agent[None, str](model=model, output=text_output_config()).with_history_processor(
        TrimOldestRequestsProcessor(max_pairs=2)
    )

    history = (
        _request("one"),
        _response("one-response"),
        _request("two"),
        _response("two-response"),
        _request("three"),
        _response("three-response"),
    )
    await agent.run("Hello", deps=None, message_history=history)

    assert model.histories[0] is not None
    assert model.histories[0].messages == history[-4:]


def test_trim_history_processor_rejects_negative_max_messages() -> None:
    with pytest.raises(ValueError, match="max_messages"):
        TrimHistoryProcessor(max_messages=-1)


def test_trim_oldest_requests_processor_rejects_negative_max_pairs() -> None:
    with pytest.raises(ValueError, match="max_pairs"):
        TrimOldestRequestsProcessor(max_pairs=-1)

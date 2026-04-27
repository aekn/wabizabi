"""Tests for wazi display module."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from wabizabi.messages import ToolCallPart, ToolReturnPart
from wabizabi.state import RunState
from wabizabi.stream import (
    HandoffEvent,
    OutputEvent,
    ReasoningChunkEvent,
    RunEvent,
    TextChunkEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from wabizabi.usage import RunUsage
from wazi.display import StreamResult, render_stream, summarize_args, summarize_result


def _state(*, usage: RunUsage | None = None) -> RunState:
    return RunState(
        run_id="test",
        usage=usage or RunUsage.zero(),
    )


async def _events_from(items: list[RunEvent[str]]) -> AsyncIterator[RunEvent[str]]:
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_render_stream_text_only(capsys: pytest.CaptureFixture[str]) -> None:
    events: list[RunEvent[str]] = [
        TextChunkEvent(state=_state(), text="Hello "),
        TextChunkEvent(state=_state(), text="world"),
        OutputEvent(
            state=_state(usage=RunUsage(input_tokens=10, output_tokens=5)),
            output="Hello world",
        ),
    ]
    result = await render_stream(_events_from(events), trace=False)
    assert isinstance(result, StreamResult)
    assert result.output == "Hello world"
    captured = capsys.readouterr()
    assert "Hello world" in captured.out
    assert "15 tokens" in captured.err


@pytest.mark.asyncio
async def test_render_stream_returns_usage() -> None:
    usage = RunUsage(input_tokens=100, output_tokens=50)
    events: list[RunEvent[str]] = [OutputEvent(state=_state(usage=usage), output="done")]
    result = await render_stream(_events_from(events), trace=False)
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 50


@pytest.mark.asyncio
async def test_render_stream_handoff() -> None:
    usage = RunUsage(input_tokens=20, output_tokens=10)
    events: list[RunEvent[str]] = [
        HandoffEvent(
            state=_state(usage=usage),
            handoff_name="billing",
            tool_call=ToolCallPart(
                tool_name="handoff_billing",
                call_id="c1",
                arguments={"input": "help"},
            ),
        ),
    ]
    result = await render_stream(_events_from(events), trace=False)
    assert result.output is None
    assert result.usage.input_tokens == 20
    assert result.handoff_name == "billing"


@pytest.mark.asyncio
async def test_render_stream_trace_tool_events_and_reasoning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[RunEvent[str]] = [
        ReasoningChunkEvent(state=_state(), text="thinking"),
        ToolCallEvent(
            state=_state(),
            tool_call=ToolCallPart(tool_name="add", call_id="c1", arguments={"a": 1, "b": 2}),
        ),
        ToolResultEvent(
            state=_state(),
            tool_return=ToolReturnPart(tool_name="add", call_id="c1", content=3),
        ),
        OutputEvent(state=_state(), output="3"),
    ]
    await render_stream(_events_from(events), trace=True)
    captured = capsys.readouterr()
    assert "[thinking] thinking" in captured.err
    assert "add" in captured.err


def test_summarize_args_simple() -> None:
    result = summarize_args({"a": 1, "b": "hello"})
    assert "a=1" in result
    assert "b='hello'" in result


def test_summarize_args_truncates_long_values() -> None:
    result = summarize_args({"key": "x" * 100})
    assert len(result) <= 83


def test_summarize_args_empty() -> None:
    assert summarize_args({}) == ""


def test_summarize_result_short() -> None:
    assert summarize_result("hello") == "hello"


def test_summarize_result_long() -> None:
    long = "x" * 100
    result = summarize_result(long)
    assert len(result) <= 80
    assert result.endswith("...")

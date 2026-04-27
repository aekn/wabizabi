"""Integration tests for wazi CLI using ScriptedModel."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from wabizabi import Agent, text_output_config, tool_plain
from wabizabi.messages import TextPart
from wabizabi.stream import RunEvent
from wabizabi.testing import text_result, tool_call_result
from wabizabi.tools import Toolset
from wabizabi.usage import RunUsage
from wazi.display import StreamResult, render_stream


def _text_agent(
    *results: tuple[TextPart, ...],
) -> Agent[None, str]:
    from wabizabi.testing import ScriptedModel

    model = ScriptedModel(results=[text_result(p[0].text) for p in results])
    return Agent[None, str](model=model, output=text_output_config())


async def _collect_stream(
    events: AsyncIterator[RunEvent[str]],
    *,
    trace: bool = False,
) -> StreamResult[str]:
    return await render_stream(events, trace=trace)


@pytest.mark.asyncio
async def test_single_turn_text_output(capsys: pytest.CaptureFixture[str]) -> None:
    agent = _text_agent((TextPart(text="Hello from the agent!"),))
    result = await _collect_stream(agent.iter("Hi", deps=None))

    assert result.output == "Hello from the agent!"
    captured = capsys.readouterr()
    assert "Hello from the agent!" in captured.out


@pytest.mark.asyncio
async def test_single_turn_usage_tracking() -> None:
    from wabizabi.testing import ScriptedModel, response_result

    model = ScriptedModel(
        results=[
            response_result(
                parts=(TextPart(text="ok"),),
                usage=RunUsage(input_tokens=10, output_tokens=5),
            )
        ]
    )
    agent = Agent[None, str](model=model, output=text_output_config())
    result = await render_stream(agent.iter("test", deps=None), trace=False)

    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_tool_call_round_trip(capsys: pytest.CaptureFixture[str]) -> None:
    @tool_plain
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    from wabizabi.testing import ScriptedModel

    model = ScriptedModel(
        results=[
            tool_call_result(tool_name="add", call_id="c1", arguments={"a": 2, "b": 3}),
            text_result("The sum is 5."),
        ]
    )
    agent = Agent[None, str](
        model=model,
        output=text_output_config(),
        toolset=Toolset(tools=(add,)),
    )

    result = await _collect_stream(agent.iter("Add 2 and 3", deps=None), trace=True)
    assert result.output == "The sum is 5."
    captured = capsys.readouterr()
    assert "add" in captured.err


@pytest.mark.asyncio
async def test_trace_mode_shows_tool_events(capsys: pytest.CaptureFixture[str]) -> None:
    @tool_plain
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    from wabizabi.testing import ScriptedModel

    model = ScriptedModel(
        results=[
            tool_call_result(tool_name="greet", call_id="c1", arguments={"name": "Alice"}),
            text_result("Done."),
        ]
    )
    agent = Agent[None, str](
        model=model,
        output=text_output_config(),
        toolset=Toolset(tools=(greet,)),
    )

    await _collect_stream(agent.iter("Greet Alice", deps=None), trace=True)
    captured = capsys.readouterr()
    assert "greet" in captured.err
    assert "Done." in captured.out

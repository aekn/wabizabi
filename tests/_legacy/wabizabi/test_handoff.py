"""Tests for handoff primitives — terminal run outcomes for multi-agent orchestration."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from wabizabi.agent import Agent
from wabizabi.context import RunContext
from wabizabi.handoff import Handoff
from wabizabi.messages import ModelResponse, TextPart, ToolCallPart
from wabizabi.models import ModelResult
from wabizabi.output import text_output_config
from wabizabi.stream import HandoffEvent, OutputEvent
from wabizabi.telemetry import HandoffRecordedEvent, InMemoryTelemetryRecorder
from wabizabi.testing import ScriptedModel
from wabizabi.tools import define_function_tool
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


def test_handoff_tool_name_is_derived_from_name() -> None:
    handoff = Handoff(name="billing", description="Transfer to billing")
    assert handoff.tool_name == "handoff_billing"


def test_handoff_has_tool_definition_with_input_schema() -> None:
    handoff = Handoff(name="support", description="Hand off to support")
    defn = handoff.tool_definition
    assert defn.name == "handoff_support"
    assert defn.description == "Hand off to support"
    properties = defn.input_schema.get("properties")
    assert isinstance(properties, dict)
    assert "input" in properties


def test_agent_with_handoff_adds_terminal_tool_name() -> None:
    handoff = Handoff(name="support")
    agent = Agent[None, str](
        model=ScriptedModel((_text_result("ok"),)),
        output=text_output_config(),
    ).with_handoff(handoff)

    assert "handoff_support" in agent.output.terminal_tool_names


@pytest.mark.asyncio
async def test_handoff_terminates_run_with_handoff_event() -> None:
    handoff = Handoff(name="billing", description="Transfer to billing")
    model = ScriptedModel(
        (_tool_call_result("handoff_billing", "call-1", {"input": "transfer me"}),)
    )
    agent = Agent[None, str](
        model=model,
        output=text_output_config(),
    ).with_handoff(handoff)

    events = [event async for event in agent.iter("help", deps=None)]
    handoff_events = [e for e in events if isinstance(e, HandoffEvent)]
    output_events = [e for e in events if isinstance(e, OutputEvent)]

    assert len(handoff_events) == 1
    assert handoff_events[0].handoff_name == "billing"
    assert handoff_events[0].tool_call.tool_name == "handoff_billing"
    assert len(output_events) == 0


@pytest.mark.asyncio
async def test_run_returns_handoff_terminal() -> None:
    handoff = Handoff(name="billing", description="Billing dept")
    model = ScriptedModel((_tool_call_result("handoff_billing", "call-1", {"input": "transfer"}),))
    agent = Agent[None, str](
        model=model,
        output=text_output_config(),
    ).with_handoff(handoff)

    result = await agent.run("test", deps=None)
    assert result.is_handoff
    assert result.output is None
    assert result.handoff is not None
    assert result.handoff.handoff.name == "billing"
    assert result.handoff.tool_call.tool_name == "handoff_billing"


@pytest.mark.asyncio
async def test_run_returns_output_terminal_on_normal_output() -> None:
    model = ScriptedModel((_text_result("hello"),))
    agent = Agent[None, str](
        model=model,
        output=text_output_config(),
    )

    result = await agent.run("test", deps=None)
    assert not result.is_handoff
    assert result.handoff is None
    assert result.output == "hello"


@pytest.mark.asyncio
async def test_handoff_records_telemetry_event() -> None:
    handoff = Handoff(name="support")
    recorder = InMemoryTelemetryRecorder[str]()
    model = ScriptedModel((_tool_call_result("handoff_support", "call-1", {"input": "help"}),))
    agent = Agent[None, str](
        model=model,
        output=text_output_config(),
        telemetry=recorder,
    ).with_handoff(handoff)

    events = [event async for event in agent.iter("test", deps=None)]
    assert any(isinstance(e, HandoffEvent) for e in events)

    handoff_recorded = [e for e in recorder.events if isinstance(e, HandoffRecordedEvent)]
    assert len(handoff_recorded) == 1
    assert handoff_recorded[0].handoff_name == "support"
    assert handoff_recorded[0].tool_call.tool_name == "handoff_support"


@pytest.mark.asyncio
async def test_multiple_handoffs_select_correct_one() -> None:
    billing = Handoff(name="billing")
    support = Handoff(name="support")
    model = ScriptedModel((_tool_call_result("handoff_support", "call-1", {"input": "need help"}),))
    agent = (
        Agent[None, str](model=model, output=text_output_config())
        .with_handoff(billing)
        .with_handoff(support)
    )

    result = await agent.run("test", deps=None)
    assert result.is_handoff
    assert result.handoff is not None
    assert result.handoff.handoff.name == "support"


@pytest.mark.asyncio
async def test_handoff_tool_definitions_sent_to_model() -> None:
    handoff = Handoff(name="support", description="Support team")
    model = ScriptedModel((_text_result("no handoff"),))
    agent = Agent[None, str](
        model=model,
        output=text_output_config(),
    ).with_handoff(handoff)

    await agent.run("test", deps=None)

    assert len(model.calls) == 1
    tool_names = [t.name for t in model.calls[0].tools]
    assert "handoff_support" in tool_names


def test_handoff_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        Handoff(name="")


def test_agent_rejects_duplicate_handoff_names() -> None:
    agent = Agent[None, str](
        model=ScriptedModel((_text_result("ok"),)), output=text_output_config()
    )
    with pytest.raises(ValueError, match="Duplicate handoff name"):
        agent.with_handoff(Handoff(name="billing")).with_handoff(Handoff(name="billing"))


def test_agent_rejects_tool_and_handoff_name_conflicts() -> None:
    class EchoArgs(BaseModel):
        input: str

    def echo(_context: RunContext[None], arguments: EchoArgs) -> str:
        return arguments.input

    agent = Agent[None, str](
        model=ScriptedModel((_text_result("ok"),)), output=text_output_config()
    ).with_tool(define_function_tool(name="handoff_billing", arguments_type=EchoArgs, func=echo))

    with pytest.raises(ValueError, match="conflict"):
        agent.with_handoff(Handoff(name="billing"))

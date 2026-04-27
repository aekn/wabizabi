from __future__ import annotations

from collections.abc import Sequence

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from pydantic import BaseModel
from wabizabi.agent import Agent
from wabizabi.context import RunContext
from wabizabi.handoff import Handoff
from wabizabi.messages import ModelResponse, TextPart, ToolCallPart
from wabizabi.models import ModelResult
from wabizabi.output import OutputValidationError, text_output_config
from wabizabi.telemetry.otel import OpenTelemetryRecorder
from wabizabi.testing import ScriptedModel
from wabizabi.tools import define_function_tool
from wabizabi.usage import RunUsage


class CollectingSpanExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def make_tracer() -> tuple[CollectingSpanExporter, trace.Tracer]:
    exporter = CollectingSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = trace.get_tracer("wabizabi.tests.otel", tracer_provider=provider)
    return exporter, tracer


@pytest.mark.asyncio
async def test_otel_recorder_creates_run_and_model_spans() -> None:
    exporter, tracer = make_tracer()
    recorder = OpenTelemetryRecorder[str](tracer=tracer)

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
        output=text_output_config(),
    ).with_telemetry(recorder)

    result = await agent.run(
        "Hello",
        deps=("svc", "cfg"),
        run_id="run-1",
        metadata={"request_id": "req-1"},
    )

    assert result.output == "Hi"

    names = [span.name for span in exporter.spans]
    assert names.count("wabizabi.model") == 1
    assert names.count("wabizabi.run") == 1

    model_span = next(span for span in exporter.spans if span.name == "wabizabi.model")
    run_span = next(span for span in exporter.spans if span.name == "wabizabi.run")

    model_parent = model_span.parent
    assert model_parent is not None
    run_context = run_span.context
    assert run_context is not None
    assert model_parent.span_id == run_context.span_id

    run_attributes = run_span.attributes
    assert run_attributes is not None
    assert run_attributes["wabizabi.run_id"] == "run-1"
    assert run_attributes["wabizabi.usage.total_tokens"] == 5
    assert run_attributes["wabizabi.metadata.request_id"] == "req-1"

    event_names = [event.name for event in run_span.events]
    assert "wabizabi.output.recorded" in event_names


class AddArguments(BaseModel):
    left: int
    right: int


@pytest.mark.asyncio
async def test_otel_recorder_creates_tool_spans_and_validation_events() -> None:
    def add_tool(context: RunContext[tuple[str, str]], arguments: AddArguments) -> int:
        assert context.tool_name == "add"
        return arguments.left + arguments.right

    def require_done(
        context: RunContext[tuple[str, str]],
        output: str,
    ) -> str:
        del context
        if output != "done":
            raise OutputValidationError(
                "output must be done",
                retry_feedback="Return exactly done.",
            )
        return output

    exporter, tracer = make_tracer()
    recorder = OpenTelemetryRecorder[str](tracer=tracer)

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
                usage=RunUsage(input_tokens=2, output_tokens=0),
            ),
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="not yet"),), model_name="scripted"),
                usage=RunUsage(input_tokens=1, output_tokens=1),
            ),
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="done"),), model_name="scripted"),
                usage=RunUsage(input_tokens=1, output_tokens=1),
            ),
        )
    )
    agent = (
        Agent[tuple[str, str], str](
            model=model,
            output=text_output_config(),
        )
        .with_tool(
            define_function_tool(
                name="add",
                arguments_type=AddArguments,
                func=add_tool,
            )
        )
        .with_output_validator(require_done)
        .with_telemetry(recorder)
    )

    result = await agent.run(
        "Do the thing",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == "done"

    names = [span.name for span in exporter.spans]
    assert names.count("wabizabi.run") == 1
    assert names.count("wabizabi.model") == 3
    assert names.count("wabizabi.tool") == 1

    tool_span = next(span for span in exporter.spans if span.name == "wabizabi.tool")
    run_span = next(span for span in exporter.spans if span.name == "wabizabi.run")

    tool_attributes = tool_span.attributes
    assert tool_attributes is not None
    assert tool_attributes["wabizabi.tool_name"] == "add"
    assert tool_attributes["wabizabi.tool_call_id"] == "call-1"
    tool_parent = tool_span.parent
    assert tool_parent is not None
    run_context = run_span.context
    assert run_context is not None
    assert tool_parent.span_id == run_context.span_id

    event_names = [event.name for event in run_span.events]
    assert "wabizabi.output.validation_failed" in event_names
    assert "wabizabi.output.recorded" in event_names


@pytest.mark.asyncio
async def test_otel_recorder_closes_run_span_on_handoff() -> None:
    exporter, tracer = make_tracer()
    recorder = OpenTelemetryRecorder[str](tracer=tracer)

    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(
                        ToolCallPart(
                            tool_name="handoff_billing",
                            call_id="call-h1",
                            arguments={"input": "transfer me"},
                        ),
                    ),
                    model_name="scripted",
                ),
                usage=RunUsage(input_tokens=5, output_tokens=3),
            ),
        )
    )
    agent = Agent[None, str](
        model=model,
        output=text_output_config(),
        handoffs=(Handoff(name="billing", description="Transfer to billing"),),
    ).with_telemetry(recorder)

    result = await agent.run("Hello", deps=None, run_id="run-handoff")

    assert result.is_handoff

    names = [span.name for span in exporter.spans]
    assert names.count("wabizabi.run") == 1
    assert names.count("wabizabi.model") == 1

    run_span = next(span for span in exporter.spans if span.name == "wabizabi.run")

    assert run_span.end_time is not None
    assert run_span.end_time > 0

    event_names = [event.name for event in run_span.events]
    assert "wabizabi.handoff" in event_names

    handoff_event = next(e for e in run_span.events if e.name == "wabizabi.handoff")
    attrs = handoff_event.attributes
    assert attrs is not None
    assert attrs["wabizabi.handoff_name"] == "billing"
    assert attrs["wabizabi.tool_call_id"] == "call-h1"

    for span in exporter.spans:
        assert span.end_time is not None and span.end_time > 0

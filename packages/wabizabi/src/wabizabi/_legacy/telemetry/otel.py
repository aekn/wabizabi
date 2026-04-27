"""Optional OpenTelemetry bridge for Wabizabi telemetry events."""

from __future__ import annotations

from dataclasses import dataclass, field

from opentelemetry.trace import Span, Status, StatusCode, Tracer, set_span_in_context

from wabizabi.state import RunState
from wabizabi.telemetry.events import (
    HandoffRecordedEvent,
    OutputDecodingFailedEvent,
    OutputRecordedEvent,
    OutputValidationFailedEvent,
    RequestRecordedEvent,
    ResponseRecordedEvent,
    RunFailedEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TelemetryEvent,
    ToolCallRecordedEvent,
    ToolResultRecordedEvent,
)
from wabizabi.types import JsonObject


def _coerce_attribute_value(value: object) -> str | bool | int | float | None:
    if isinstance(value, bool | int | float | str):
        return value
    return None


def _metadata_attributes(
    metadata: JsonObject | None,
) -> dict[str, str | bool | int | float]:
    if metadata is None:
        return {}

    attributes: dict[str, str | bool | int | float] = {}
    for key, value in metadata.items():
        coerced = _coerce_attribute_value(value)
        if coerced is not None:
            attributes[f"wabizabi.metadata.{key}"] = coerced
    return attributes


def _state_attributes(state: RunState) -> dict[str, str | bool | int | float]:
    attributes: dict[str, str | bool | int | float] = {
        "wabizabi.run_id": state.run_id,
        "wabizabi.run_step": state.run_step,
        "wabizabi.retries": state.retries,
        "wabizabi.message_history_length": len(state.message_history.messages),
        "wabizabi.usage.input_tokens": state.usage.input_tokens,
        "wabizabi.usage.output_tokens": state.usage.output_tokens,
        "wabizabi.usage.total_tokens": state.usage.total_tokens,
    }
    attributes.update(_metadata_attributes(state.metadata))
    return attributes


def _empty_tool_span_map() -> dict[str, Span]:
    return {}


@dataclass(slots=True)
class OpenTelemetryRecorder[OutputDataT]:
    """Bridge Wabizabi telemetry events into OpenTelemetry spans and events."""

    tracer: Tracer
    run_span_name: str = "wabizabi.run"
    model_span_name: str = "wabizabi.model"
    tool_span_name: str = "wabizabi.tool"
    _run_span: Span | None = field(default=None, init=False, repr=False)
    _request_span: Span | None = field(default=None, init=False, repr=False)
    _tool_spans: dict[str, Span] = field(
        default_factory=_empty_tool_span_map,
        init=False,
        repr=False,
    )

    async def record(
        self,
        event: TelemetryEvent[OutputDataT],
    ) -> None:
        if isinstance(event, RunStartedEvent):
            self._on_run_started(event)
            return
        if isinstance(event, RequestRecordedEvent):
            self._on_request(event)
            return
        if isinstance(event, ResponseRecordedEvent):
            self._on_response(event)
            return
        if isinstance(event, ToolCallRecordedEvent):
            self._on_tool_call(event)
            return
        if isinstance(event, ToolResultRecordedEvent):
            self._on_tool_result(event)
            return
        if isinstance(event, OutputValidationFailedEvent):
            self._on_output_validation_failed(event)
            return
        if isinstance(event, OutputDecodingFailedEvent):
            self._on_output_decoding_failed(event)
            return
        if isinstance(event, OutputRecordedEvent):
            self._on_output_recorded(event)
            return
        if isinstance(event, HandoffRecordedEvent):
            self._on_handoff(event)
            return
        if isinstance(event, RunFinishedEvent):
            self._on_run_finished(event)
            return
        if isinstance(event, RunFailedEvent):
            self._on_run_failed(event)
            return

    def _on_run_started(self, event: RunStartedEvent) -> None:
        self._close_open_spans()
        self._run_span = self.tracer.start_span(self.run_span_name)
        self._run_span.set_attributes(_state_attributes(event.state))

    def _on_request(self, event: RequestRecordedEvent) -> None:
        run_span = self._ensure_run_span()
        if self._request_span is not None:
            self._request_span.end()

        self._request_span = self.tracer.start_span(
            self.model_span_name,
            context=set_span_in_context(run_span),
        )
        self._request_span.set_attributes(_state_attributes(event.state))
        self._request_span.set_attribute(
            "wabizabi.request.parts",
            len(event.request.parts),
        )

    def _on_response(self, event: ResponseRecordedEvent) -> None:
        request_span = self._request_span
        if request_span is None:
            self._ensure_run_span().add_event(
                "wabizabi.response.orphan",
                attributes=_state_attributes(event.state),
            )
            return

        request_span.set_attributes(_state_attributes(event.state))
        request_span.set_attribute(
            "wabizabi.response.parts",
            len(event.response.parts),
        )

        model_name = event.response.model_name
        if model_name is not None:
            request_span.set_attribute("wabizabi.model_name", model_name)

        finish_reason = event.response.finish_reason
        if finish_reason is not None:
            request_span.set_attribute(
                "wabizabi.finish_reason",
                str(finish_reason),
            )

        request_span.end()
        self._request_span = None

    def _on_tool_call(self, event: ToolCallRecordedEvent) -> None:
        run_span = self._ensure_run_span()
        tool_span = self.tracer.start_span(
            self.tool_span_name,
            context=set_span_in_context(run_span),
        )
        tool_span.set_attributes(_state_attributes(event.state))
        tool_span.set_attribute("wabizabi.tool_name", event.tool_call.tool_name)
        tool_span.set_attribute("wabizabi.tool_call_id", event.tool_call.call_id)
        self._tool_spans[event.tool_call.call_id] = tool_span

    def _on_tool_result(self, event: ToolResultRecordedEvent) -> None:
        tool_span = self._tool_spans.pop(event.tool_return.call_id, None)
        if tool_span is None:
            self._ensure_run_span().add_event(
                "wabizabi.tool_result.orphan",
                attributes={
                    **_state_attributes(event.state),
                    "wabizabi.tool_name": event.tool_return.tool_name,
                    "wabizabi.tool_call_id": event.tool_return.call_id,
                },
            )
            return

        tool_span.set_attributes(_state_attributes(event.state))
        tool_span.set_attribute("wabizabi.tool_name", event.tool_return.tool_name)
        tool_span.set_attribute("wabizabi.tool_call_id", event.tool_return.call_id)
        tool_span.set_attribute(
            "wabizabi.tool_result_type",
            type(event.tool_return.content).__name__,
        )
        if event.tool_return.is_error:
            tool_span.set_status(
                Status(StatusCode.ERROR, f"Tool {event.tool_return.tool_name!r} failed")
            )
            tool_span.set_attribute("wabizabi.tool_error", True)
        tool_span.end()

    def _on_output_validation_failed(
        self,
        event: OutputValidationFailedEvent,
    ) -> None:
        self._ensure_run_span().add_event(
            "wabizabi.output.validation_failed",
            attributes={
                **_state_attributes(event.state),
                "wabizabi.error_message": event.error_message,
                "wabizabi.retry_feedback": event.retry_feedback,
            },
        )

    def _on_output_decoding_failed(
        self,
        event: OutputDecodingFailedEvent,
    ) -> None:
        self._ensure_run_span().add_event(
            "wabizabi.output.decoding_failed",
            attributes={
                **_state_attributes(event.state),
                "wabizabi.error_message": event.error_message,
                "wabizabi.retry_feedback": event.retry_feedback,
            },
        )

    def _on_output_recorded(self, event: OutputRecordedEvent[OutputDataT]) -> None:
        self._ensure_run_span().add_event(
            "wabizabi.output.recorded",
            attributes={
                **_state_attributes(event.state),
                "wabizabi.output_type": type(event.output).__name__,
            },
        )

    def _on_handoff(self, event: HandoffRecordedEvent) -> None:
        run_span = self._ensure_run_span()
        run_span.add_event(
            "wabizabi.handoff",
            attributes={
                **_state_attributes(event.state),
                "wabizabi.handoff_name": event.handoff_name,
                "wabizabi.tool_call_id": event.tool_call.call_id,
            },
        )
        run_span.set_attributes(_state_attributes(event.state))
        self._close_open_spans()
        run_span.end()
        self._run_span = None

    def _on_run_finished(self, event: RunFinishedEvent[OutputDataT]) -> None:
        run_span = self._ensure_run_span()
        run_span.set_attributes(_state_attributes(event.state))
        run_span.set_attribute("wabizabi.output_type", type(event.output).__name__)
        self._close_open_spans()
        run_span.end()
        self._run_span = None

    def _on_run_failed(self, event: RunFailedEvent) -> None:
        run_span = self._ensure_run_span()
        run_span.set_attributes(_state_attributes(event.state))
        run_span.set_status(Status(StatusCode.ERROR, event.error_message))
        run_span.add_event(
            "wabizabi.run.failed",
            attributes={
                **_state_attributes(event.state),
                "wabizabi.error_message": event.error_message,
            },
        )
        self._close_open_spans()
        run_span.end()
        self._run_span = None

    def _ensure_run_span(self) -> Span:
        if self._run_span is None:
            raise RuntimeError("OpenTelemetryRecorder received events before RunStartedEvent.")
        return self._run_span

    def _close_open_spans(self) -> None:
        if self._request_span is not None:
            self._request_span.end()
            self._request_span = None

        if self._tool_spans:
            open_tool_spans = tuple(self._tool_spans.values())
            self._tool_spans.clear()
            for span in open_tool_spans:
                span.end()


__all__ = ["OpenTelemetryRecorder"]

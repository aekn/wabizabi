from wabizabi.messages import FinishReason, ModelResponse, TextPart, ToolCallPart
from wabizabi.models import ModelResult
from wabizabi.models.stream import (
    ModelReasoningDeltaEvent,
    ModelResponseCompletedEvent,
    ModelResponsePartEvent,
    ModelTextDeltaEvent,
    model_result_events,
)
from wabizabi.runtime import ResponseAccumulator
from wabizabi.usage import RunUsage


def test_response_accumulator_tracks_text_delta() -> None:
    accumulator = ResponseAccumulator()

    update = accumulator.add(ModelTextDeltaEvent(text="hel"))

    assert update.text_delta == "hel"
    assert update.reasoning_delta is None
    assert update.finalized_parts == ()
    assert accumulator.parts == ()
    assert accumulator.completed is False


def test_response_accumulator_tracks_reasoning_delta() -> None:
    accumulator = ResponseAccumulator()

    update = accumulator.add(ModelReasoningDeltaEvent(text="thinking"))

    assert update.reasoning_delta == "thinking"
    assert update.text_delta is None
    assert update.finalized_parts == ()
    assert accumulator.parts == ()


def test_response_accumulator_emits_finalized_parts_and_builds_response() -> None:
    accumulator = ResponseAccumulator()
    tool_call = ToolCallPart(
        tool_name="weather",
        call_id="call-1",
        arguments={"city": "SF"},
    )

    update = accumulator.add(ModelResponsePartEvent(part=tool_call))
    assert update.finalized_parts == (tool_call,)

    completed = accumulator.add(
        ModelResponseCompletedEvent(
            model_name="ollama-test",
            usage=RunUsage(input_tokens=3, output_tokens=5),
            finish_reason=FinishReason.TOOL_CALLS,
            metadata={"provider": "ollama"},
        )
    )
    assert completed.completed is True
    assert accumulator.model_name == "ollama-test"
    assert accumulator.finish_reason is FinishReason.TOOL_CALLS
    assert accumulator.metadata == {"provider": "ollama"}

    response = accumulator.build_response()
    assert response == ModelResponse(
        parts=(tool_call,),
        model_name="ollama-test",
        finish_reason=FinishReason.TOOL_CALLS,
        metadata={"provider": "ollama"},
    )
    assert accumulator.usage == RunUsage(input_tokens=3, output_tokens=5)


def test_response_accumulator_rejects_build_before_completion() -> None:
    accumulator = ResponseAccumulator()
    accumulator.add(ModelResponsePartEvent(part=TextPart(text="hello")))

    try:
        accumulator.build_response()
    except RuntimeError as exc:
        assert str(exc) == "Cannot build response before completion."
    else:  # pragma: no cover
        raise AssertionError("Expected RuntimeError.")


def test_response_accumulator_rejects_completed_stream_without_parts() -> None:
    accumulator = ResponseAccumulator()
    accumulator.add(
        ModelResponseCompletedEvent(
            model_name="ollama-test",
            usage=RunUsage(input_tokens=1, output_tokens=1),
            finish_reason=FinishReason.STOP,
        )
    )

    try:
        accumulator.build_response()
    except RuntimeError as exc:
        assert str(exc) == "Model stream completed without response parts."
    else:  # pragma: no cover
        raise AssertionError("Expected RuntimeError.")


def test_response_accumulator_supports_model_result_event_projection() -> None:
    result = ModelResult(
        response=ModelResponse(
            parts=(TextPart(text="hello"),),
            model_name="fallback",
            finish_reason=FinishReason.STOP,
            metadata={"request_id": "req-1"},
        ),
        usage=RunUsage(input_tokens=1, output_tokens=1),
    )
    accumulator = ResponseAccumulator()

    for event in model_result_events(result):
        accumulator.add(event)

    assert accumulator.completed is True
    assert accumulator.build_response() == result.response
    assert accumulator.usage == result.usage

from wabizabi.messages import FinishReason, ModelResponse, RefusalPart, TextPart, ToolCallPart
from wabizabi.models.stream import ModelResponseCompletedEvent, ModelResponsePartEvent
from wabizabi.testing import (
    refusal_result,
    response_message,
    response_result,
    stream_script_from_result,
    text_response,
    text_result,
    tool_call_response,
    tool_call_result,
)
from wabizabi.usage import RunUsage


def test_response_message_builder_preserves_parts_and_metadata() -> None:
    response = response_message(
        (TextPart(text="First"), RefusalPart(text="Second")),
        model_name="multi-part",
        finish_reason=FinishReason.STOP,
        metadata={"request_id": "req-1"},
    )

    assert response == ModelResponse(
        parts=(TextPart(text="First"), RefusalPart(text="Second")),
        model_name="multi-part",
        finish_reason=FinishReason.STOP,
        metadata={"request_id": "req-1"},
    )


def test_text_response_builder_defaults_model_name() -> None:
    response = text_response("Hello")

    assert response == ModelResponse(parts=(TextPart(text="Hello"),), model_name="scripted")


def test_tool_call_response_builder_preserves_arguments() -> None:
    response = tool_call_response(
        "lookup",
        call_id="call-1",
        arguments={"city": "Paris"},
        model_name="tool-model",
        finish_reason=FinishReason.TOOL_CALLS,
    )

    assert response == ModelResponse(
        parts=(
            ToolCallPart(
                tool_name="lookup",
                call_id="call-1",
                arguments={"city": "Paris"},
            ),
        ),
        model_name="tool-model",
        finish_reason=FinishReason.TOOL_CALLS,
    )


def test_response_result_builder_supports_multiple_parts_and_usage() -> None:
    result = response_result(
        (
            TextPart(text="First"),
            RefusalPart(text="Second"),
        ),
        model_name="multi-part",
        usage=RunUsage(input_tokens=4, output_tokens=2),
        finish_reason=FinishReason.STOP,
    )

    assert result.response == ModelResponse(
        parts=(TextPart(text="First"), RefusalPart(text="Second")),
        model_name="multi-part",
        finish_reason=FinishReason.STOP,
    )
    assert result.usage == RunUsage(input_tokens=4, output_tokens=2)


def test_text_result_builder_defaults_usage_to_zero() -> None:
    result = text_result("Hello")

    assert result.usage == RunUsage.zero()


def test_refusal_result_builder_projects_to_stream_script() -> None:
    result = refusal_result(
        "I can't help with that.",
        model_name="refusal-model",
        finish_reason=FinishReason.STOP,
    )

    events = stream_script_from_result(result)

    assert events == (
        ModelResponsePartEvent(part=RefusalPart(text="I can't help with that.")),
        ModelResponseCompletedEvent(
            model_name="refusal-model",
            usage=RunUsage.zero(),
            finish_reason=FinishReason.STOP,
        ),
    )


def test_tool_call_result_builder_projects_to_stream_script() -> None:
    result = tool_call_result(
        "lookup",
        call_id="call-1",
        arguments={"city": "Paris"},
        usage=RunUsage(input_tokens=2, output_tokens=0),
        metadata={"trace_id": "trace-1"},
    )

    events = stream_script_from_result(result)

    assert events == (
        ModelResponsePartEvent(
            part=ToolCallPart(
                tool_name="lookup",
                call_id="call-1",
                arguments={"city": "Paris"},
            )
        ),
        ModelResponseCompletedEvent(
            model_name="scripted",
            usage=RunUsage(input_tokens=2, output_tokens=0),
            metadata={"trace_id": "trace-1"},
        ),
    )

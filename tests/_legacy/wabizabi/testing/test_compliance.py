from __future__ import annotations

from collections.abc import AsyncIterator, Iterable

import pytest
from wabizabi.history import MessageHistory
from wabizabi.messages import FinishReason, ModelRequest, ModelResponse, TextPart, UserPromptPart
from wabizabi.models.stream import ModelResponseCompletedEvent, ModelStreamEvent
from wabizabi.output import json_output_config
from wabizabi.testing import (
    ScriptedModel,
    assert_model_contract_is_self_consistent,
    assert_model_contract_matches_expected,
    assert_model_result_matches_expected,
    assert_stream_matches_result,
    collect_model_contract_capture,
    collect_stream_capture,
    stream_script_from_result,
    text_result,
)
from wabizabi.tools import ToolDefinition
from wabizabi.usage import RunUsage


async def _iterate_events(events: Iterable[ModelStreamEvent]) -> AsyncIterator[ModelStreamEvent]:
    for event in events:
        yield event


@pytest.mark.asyncio
async def test_collect_stream_capture_reconstructs_model_result() -> None:
    result = text_result(
        "Hello",
        model_name="test-model",
        usage=RunUsage(input_tokens=2, output_tokens=1),
    )

    capture = await collect_stream_capture(_iterate_events(stream_script_from_result(result)))

    assert_stream_matches_result(capture, result)
    assert capture.events == stream_script_from_result(result)


@pytest.mark.asyncio
async def test_collect_stream_capture_rejects_stream_without_response_parts() -> None:
    events = (
        ModelResponseCompletedEvent(
            model_name="empty",
            usage=RunUsage.zero(),
            finish_reason=FinishReason.STOP,
        ),
    )

    with pytest.raises(RuntimeError, match="without response parts"):
        await collect_stream_capture(_iterate_events(events))


@pytest.mark.asyncio
async def test_collect_model_contract_capture_validates_request_and_stream_paths() -> None:
    expected = text_result(
        "Hello",
        model_name="contract-model",
        usage=RunUsage(input_tokens=3, output_tokens=1),
        finish_reason=FinishReason.STOP,
        metadata={"trace_id": "trace-1"},
    )
    model = ScriptedModel((expected, expected), model_name="contract-model")

    capture = await collect_model_contract_capture(
        model,
        ModelRequest(parts=(UserPromptPart(text="Hi"),)),
    )

    assert_model_contract_matches_expected(capture, expected)
    assert capture.result.response == ModelResponse(
        parts=(TextPart(text="Hello"),),
        model_name="contract-model",
        finish_reason=FinishReason.STOP,
        metadata={"trace_id": "trace-1"},
    )


@pytest.mark.asyncio
async def test_collect_model_contract_capture_passes_through_optional_inputs() -> None:
    expected = text_result("Hello", usage=RunUsage(input_tokens=3, output_tokens=1))
    model = ScriptedModel((expected, expected), model_name="contract-model")
    history = MessageHistory(messages=(ModelRequest(parts=(UserPromptPart(text="Earlier"),)),))
    tool_definition = ToolDefinition(name="lookup", input_schema={"type": "object"})
    output = json_output_config()

    capture = await collect_model_contract_capture(
        model,
        ModelRequest(parts=(UserPromptPart(text="Hi"),)),
        message_history=history,
        settings=None,
        tools=(tool_definition,),
        output=output,
    )

    assert capture.result == expected
    assert model.histories[0] is not None
    assert model.histories[0] == history
    assert model.received_tools[0] == (tool_definition,)
    assert model.calls[0].output == output


@pytest.mark.asyncio
async def test_model_contract_self_consistency_helper_checks_request_and_stream_parity() -> None:
    expected = text_result(
        "Hello",
        model_name="contract-model",
        usage=RunUsage(input_tokens=3, output_tokens=1),
        finish_reason=FinishReason.STOP,
    )
    model = ScriptedModel((expected, expected), model_name="contract-model")

    capture = await collect_model_contract_capture(
        model,
        ModelRequest(parts=(UserPromptPart(text="Hi"),)),
    )

    assert_model_contract_is_self_consistent(capture)


def test_assert_model_result_matches_expected_checks_one_shot_result() -> None:
    expected = text_result(
        "Hello",
        model_name="one-shot",
        usage=RunUsage(input_tokens=1, output_tokens=1),
    )

    assert_model_result_matches_expected(expected, expected)

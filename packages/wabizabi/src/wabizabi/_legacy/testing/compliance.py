"""Reusable compliance helpers for provider-neutral model behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from wabizabi.history import MessageHistory
from wabizabi.messages import ModelRequest, ModelResponse
from wabizabi.models import Model, ModelResult, ModelSettings, ModelStreamEvent
from wabizabi.output import OutputConfig
from wabizabi.runtime.response_accumulator import ResponseAccumulator
from wabizabi.tools import ToolDefinition
from wabizabi.usage import RunUsage


@dataclass(frozen=True, slots=True)
class StreamCapture:
    """A captured canonical response stream."""

    events: tuple[ModelStreamEvent, ...]
    response: ModelResponse
    usage: RunUsage


@dataclass(frozen=True, slots=True)
class ModelContractCapture:
    """Both request and stream paths captured for the same model contract."""

    result: ModelResult
    stream: StreamCapture


async def collect_stream_capture(
    events: AsyncIterator[ModelStreamEvent],
) -> StreamCapture:
    """Collect streamed events into a canonical response and usage record."""

    captured_events: list[ModelStreamEvent] = []
    accumulator = ResponseAccumulator()
    async for event in events:
        captured_events.append(event)
        accumulator.add(event)

    return StreamCapture(
        events=tuple(captured_events),
        response=accumulator.build_response(),
        usage=accumulator.usage,
    )


def assert_model_result_matches_expected(
    result: ModelResult,
    expected: ModelResult,
) -> None:
    """Assert that a one-shot request matches an expected canonical result."""

    assert result == expected


def assert_model_contract_is_self_consistent(capture: ModelContractCapture) -> None:
    """Assert that request and stream paths agree with each other."""

    assert capture.result.response == capture.stream.response
    assert capture.result.usage == capture.stream.usage


def assert_stream_matches_result(
    capture: StreamCapture,
    expected: ModelResult,
) -> None:
    """Assert that a streamed response reconstructs the expected result."""

    assert capture.response == expected.response
    assert capture.usage == expected.usage


async def collect_model_contract_capture(
    model: Model,
    request: ModelRequest,
    *,
    message_history: MessageHistory | None = None,
    settings: ModelSettings | None = None,
    tools: tuple[ToolDefinition, ...] = (),
    output: OutputConfig[object] | None = None,
) -> ModelContractCapture:
    """Capture both request and stream paths for the same normalized request."""

    result = await model.request(
        request,
        message_history=message_history,
        settings=settings,
        tools=tools,
        output=output,
    )
    stream = await collect_stream_capture(
        model.stream_response(
            request,
            message_history=message_history,
            settings=settings,
            tools=tools,
            output=output,
        )
    )
    return ModelContractCapture(result=result, stream=stream)


def assert_model_contract_matches_expected(
    capture: ModelContractCapture,
    expected: ModelResult,
) -> None:
    """Assert that request and stream paths both match the expected canonical result."""

    assert_model_contract_is_self_consistent(capture)
    assert_model_result_matches_expected(capture.result, expected)
    assert_stream_matches_result(capture.stream, expected)


__all__ = [
    "ModelContractCapture",
    "StreamCapture",
    "assert_model_contract_is_self_consistent",
    "assert_model_contract_matches_expected",
    "assert_model_result_matches_expected",
    "assert_stream_matches_result",
    "collect_model_contract_capture",
    "collect_stream_capture",
]

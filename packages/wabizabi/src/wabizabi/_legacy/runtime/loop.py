"""Main agent run loop orchestration."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING
from uuid import uuid4

from wabizabi.history import MessageHistory
from wabizabi.messages import ModelRequest, ModelResponse, RetryFeedbackPart, ToolReturnPart
from wabizabi.models import Model, ModelSettings, merge_model_settings
from wabizabi.output import OutputDecodingError, OutputValidationError
from wabizabi.runtime.output import decode_and_validate_output, output_decoding_retry_feedback
from wabizabi.runtime.requests import build_request, coerce_message_history, merge_metadata
from wabizabi.runtime.response_accumulator import ResponseAccumulator
from wabizabi.runtime.tools import invoke_tool, partition_tool_calls
from wabizabi.state import RunState
from wabizabi.stream import (
    HandoffEvent,
    OutputEvent,
    ReasoningChunkEvent,
    RequestEvent,
    ResponseEvent,
    RunEvent,
    TextChunkEvent,
    ToolCallEvent,
    ToolResultEvent,
    reasoning_chunk_events,
    text_chunk_events,
)
from wabizabi.telemetry import (
    HandoffRecordedEvent,
    OutputDecodingFailedEvent,
    OutputRecordedEvent,
    OutputValidationFailedEvent,
    RequestRecordedEvent,
    ResponseRecordedEvent,
    RunFailedEvent,
    RunFinishedEvent,
    RunStartedEvent,
    ToolCallRecordedEvent,
    ToolResultRecordedEvent,
)
from wabizabi.types import JsonObject
from wabizabi.usage import RunUsage

if TYPE_CHECKING:
    from wabizabi.agent import Agent


class _NestedUsageRecorder:
    """Accumulate nested usage produced during one tool invocation."""

    __slots__ = ("usage",)

    def __init__(self) -> None:
        self.usage = RunUsage.zero()

    def record_usage(self, usage: RunUsage) -> None:
        self.usage = self.usage + usage


async def iter_run[AgentDepsT, OutputDataT](
    agent: Agent[AgentDepsT, OutputDataT],
    user_prompt: str,
    *,
    deps: AgentDepsT,
    run_id: str | None = None,
    metadata: JsonObject | None = None,
    model: Model | None = None,
    instructions: tuple[str, ...] = (),
    settings: ModelSettings | None = None,
    message_history: MessageHistory | tuple[ModelRequest | ModelResponse, ...] | None = None,
) -> AsyncIterator[RunEvent[OutputDataT]]:
    """Iterate through structured events for a single agent run."""
    resolved_run_id = uuid4().hex if run_id is None else run_id
    resolved_metadata = merge_metadata(agent.metadata, metadata)
    resolved_model = agent.model if model is None else model
    resolved_settings = merge_model_settings(agent.model_settings, settings)
    resolved_instructions = (*agent.system_instructions, *instructions)
    initial_history = coerce_message_history(message_history)

    state = RunState(
        run_id=resolved_run_id,
        message_history=initial_history,
        metadata=resolved_metadata,
    )

    await agent.telemetry.record(RunStartedEvent(state=state))

    request = build_request(
        user_prompt=user_prompt,
        metadata=resolved_metadata,
        system_instructions=resolved_instructions,
    )
    tool_rounds = 0
    handoff_map = {h.tool_name: h for h in agent.handoffs} if agent.handoffs else {}

    try:
        while True:
            request_context = state.context_for(deps)
            model_visible_history = await agent.model_visible_history(
                request_context,
                state.message_history,
            )
            prepared_toolset = await agent.hooks.apply_prepare_tools(request_context, agent.toolset)
            request = await agent.hooks.apply_before_request(request_context, request)

            await agent.telemetry.record(RequestRecordedEvent(state=state, request=request))
            yield RequestEvent(state=state, request=request)

            accumulator = ResponseAccumulator()
            saw_text_deltas = False
            saw_reasoning_deltas = False

            async for model_event in resolved_model.stream_response(
                request,
                message_history=model_visible_history,
                settings=resolved_settings,
                tools=(
                    *prepared_toolset.definitions(),
                    *(h.tool_definition for h in agent.handoffs),
                ),
                output=agent.output,
            ):
                update = accumulator.add(model_event)
                if update.reasoning_delta is not None:
                    saw_reasoning_deltas = True
                    yield ReasoningChunkEvent(state=state, text=update.reasoning_delta)
                if update.text_delta is not None:
                    saw_text_deltas = True
                    yield TextChunkEvent(state=state, text=update.text_delta)

            if not accumulator.completed:
                raise RuntimeError("Model stream completed without a completion event.")

            raw_response = accumulator.build_response()
            hook_state = (
                state.with_messages((request, raw_response))
                .add_usage(accumulator.usage)
                .increment_run_step()
            )
            response = await agent.hooks.apply_after_response(
                hook_state.context_for(deps), raw_response
            )
            if response is raw_response:
                state = hook_state
            else:
                state = (
                    state.with_messages((request, response))
                    .add_usage(accumulator.usage)
                    .increment_run_step()
                )

            response_event = ResponseEvent(state=state, response=response)
            await agent.telemetry.record(ResponseRecordedEvent(state=state, response=response))
            yield response_event
            if not saw_reasoning_deltas:
                for reasoning_event in reasoning_chunk_events(response_event):
                    yield reasoning_event
            if not saw_text_deltas:
                for text_event in text_chunk_events(response_event):
                    yield text_event

            tool_calls = partition_tool_calls(
                response,
                terminal_tool_names=agent.output.terminal_tool_names,
            )
            if tool_calls.function_calls and tool_calls.terminal_calls:
                state = state.increment_retry()
                retry_feedback = (
                    "Your response contained both an executable tool call and a "
                    "terminal tool call in a single turn. Pick one: either call "
                    "tools to gather information, or emit the terminal tool call "
                    "once you have everything you need. Do not mix them."
                )
                await agent.telemetry.record(
                    OutputValidationFailedEvent(
                        state=state,
                        error_message="mixed executable and terminal tool calls in one response",
                        retry_feedback=retry_feedback,
                    )
                )
                if state.retries > agent.max_output_retries:
                    raise RuntimeError(
                        "Model repeatedly mixed executable and terminal tool calls in one response."
                    )
                request = ModelRequest(
                    parts=(RetryFeedbackPart(message=retry_feedback),),
                    metadata=resolved_metadata,
                )
                continue

            if tool_calls.function_calls:
                tool_rounds += 1
                if tool_rounds > agent.max_tool_rounds:
                    raise RuntimeError("Exceeded maximum tool rounds for this agent run.")

                tool_returns: list[ToolReturnPart] = []
                for initial_tool_call in tool_calls.function_calls:
                    initial_tool_context = state.context_for(
                        deps,
                        tool_name=initial_tool_call.tool_name,
                        tool_call_id=initial_tool_call.call_id,
                    )
                    tool_call = await agent.hooks.apply_before_tool_call(
                        initial_tool_context,
                        initial_tool_call,
                    )
                    nested_usage = _NestedUsageRecorder()
                    tool_context = state.context_for(
                        deps,
                        tool_name=tool_call.tool_name,
                        tool_call_id=tool_call.call_id,
                        _usage_recorder=nested_usage,
                    )
                    await agent.telemetry.record(
                        ToolCallRecordedEvent(state=state, tool_call=tool_call)
                    )
                    yield ToolCallEvent(state=state, tool_call=tool_call)

                    tool_return = await invoke_tool(
                        toolset=prepared_toolset,
                        context=tool_context,
                        tool_name=tool_call.tool_name,
                        call_id=tool_call.call_id,
                        arguments=tool_call.arguments,
                    )
                    if nested_usage.usage.total_tokens > 0:
                        state = state.add_usage(nested_usage.usage)
                    tool_result_context = state.context_for(
                        deps,
                        tool_name=tool_call.tool_name,
                        tool_call_id=tool_call.call_id,
                    )
                    tool_return = await agent.hooks.apply_after_tool_call(
                        tool_result_context,
                        tool_call,
                        tool_return,
                    )
                    tool_returns.append(tool_return)
                    await agent.telemetry.record(
                        ToolResultRecordedEvent(state=state, tool_return=tool_return)
                    )
                    yield ToolResultEvent(state=state, tool_return=tool_return)

                request = ModelRequest(
                    parts=tuple(tool_returns),
                    metadata=resolved_metadata,
                )
                continue

            for terminal_call in tool_calls.terminal_calls:
                if terminal_call.tool_name in handoff_map:
                    handoff = handoff_map[terminal_call.tool_name]
                    await agent.telemetry.record(
                        HandoffRecordedEvent(
                            state=state,
                            handoff_name=handoff.name,
                            tool_call=terminal_call,
                        )
                    )
                    yield HandoffEvent(
                        state=state,
                        handoff_name=handoff.name,
                        tool_call=terminal_call,
                    )
                    return

            try:
                validated_output = await decode_and_validate_output(
                    output=agent.output,
                    output_pipeline=agent.output_pipeline,
                    response=response,
                    context=state.context_for(deps),
                )
            except (OutputDecodingError, OutputValidationError) as error:
                state = state.increment_retry()
                if isinstance(error, OutputDecodingError):
                    retry_feedback = output_decoding_retry_feedback(error, mode=agent.output.mode)
                    await agent.telemetry.record(
                        OutputDecodingFailedEvent(
                            state=state,
                            error_message=str(error),
                            retry_feedback=retry_feedback,
                        )
                    )
                else:
                    retry_feedback = error.retry_feedback
                    await agent.telemetry.record(
                        OutputValidationFailedEvent(
                            state=state,
                            error_message=str(error),
                            retry_feedback=retry_feedback,
                        )
                    )
                if state.retries > agent.max_output_retries:
                    raise
                request = ModelRequest(
                    parts=(RetryFeedbackPart(message=retry_feedback),),
                    metadata=resolved_metadata,
                )
                continue

            await agent.telemetry.record(OutputRecordedEvent(state=state, output=validated_output))
            yield OutputEvent(state=state, output=validated_output)
            await agent.telemetry.record(RunFinishedEvent(state=state, output=validated_output))
            break
    except Exception as error:
        await agent.telemetry.record(RunFailedEvent(state=state, error_message=str(error)))
        raise

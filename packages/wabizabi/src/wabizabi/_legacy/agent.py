"""Public agent API."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from wabizabi.capabilities import CapabilityLike, normalize_capability
from wabizabi.context import RunContext
from wabizabi.handoff import Handoff, HandoffResult
from wabizabi.history import MessageHistory
from wabizabi.hooks import Hooks
from wabizabi.messages import ModelMessage, ModelRequest, ModelResponse
from wabizabi.models import Model, ModelSettings
from wabizabi.output import (
    OutputConfig,
    OutputDecoder,
    OutputMode,
    OutputPipeline,
    OutputValidatorLike,
    infer_output_mode,
)
from wabizabi.processors import (
    HistoryProcessor,
    NormalizedHistoryProcessor,
    apply_history_processors,
    normalize_history_processor,
)
from wabizabi.run import RunResult
from wabizabi.runtime.loop import iter_run
from wabizabi.runtime.requests import coerce_message_history
from wabizabi.state import RunState
from wabizabi.stream import HandoffEvent, OutputEvent, RunEvent, StreamedRun
from wabizabi.telemetry import NoopTelemetryRecorder, TelemetryRecorder
from wabizabi.tools import AsyncFunctionTool, Tool
from wabizabi.tools.agent import AgentToolInput
from wabizabi.tools.toolset import Toolset
from wabizabi.types import JsonObject


def _new_messages_suffix(
    state: RunState,
    initial_history: MessageHistory,
) -> tuple[ModelMessage, ...]:
    return state.message_history.messages[len(initial_history.messages) :]


def _resolve_output[OutputDataT](
    *,
    decoder: OutputDecoder[OutputDataT] | None,
    output: OutputConfig[OutputDataT] | None,
) -> OutputConfig[OutputDataT]:
    if (decoder is None) == (output is None):
        raise ValueError("Exactly one of decoder or output must be provided.")

    if output is not None:
        return output

    assert decoder is not None
    return OutputConfig(mode=infer_output_mode(decoder), decoder=decoder)


def _validate_output_tool_names[AgentDepsT, OutputDataT](
    *,
    toolset: Toolset[AgentDepsT],
    output: OutputConfig[OutputDataT],
) -> None:
    conflicting_tool_names = {
        tool.name for tool in toolset.tools if tool.name in output.terminal_tool_names
    }
    if conflicting_tool_names:
        joined = ", ".join(sorted(conflicting_tool_names))
        raise ValueError(f"Output tool names conflict with registered tools: {joined}")


def _validate_handoffs(
    handoffs: tuple[Handoff, ...],
) -> None:
    names: set[str] = set()
    tool_names: set[str] = set()
    for handoff in handoffs:
        if handoff.name in names:
            raise ValueError(f"Duplicate handoff name: {handoff.name}")
        names.add(handoff.name)
        if handoff.tool_name in tool_names:
            raise ValueError(f"Duplicate handoff tool name: {handoff.tool_name}")
        tool_names.add(handoff.tool_name)


def _merge_handoff_output[OutputDataT](
    output: OutputConfig[OutputDataT],
    handoffs: tuple[Handoff, ...],
) -> OutputConfig[OutputDataT]:
    if not handoffs:
        return output
    handoff_tool_names = frozenset(h.tool_name for h in handoffs)
    return OutputConfig(
        mode=output.mode,
        decoder=output.decoder,
        response_format=output.response_format,
        terminal_tool_names=output.terminal_tool_names | handoff_tool_names,
    )


class ModelCapabilityError(ValueError):
    """Raised when an agent configuration exceeds its model's advertised capabilities."""


def _validate_model_capabilities[AgentDepsT, OutputDataT](
    *,
    model: Model,
    output: OutputConfig[OutputDataT],
    toolset: Toolset[AgentDepsT],
    handoffs: tuple[Handoff, ...],
) -> None:
    profile = model.profile
    label = f"{profile.provider_name}:{profile.model_name}"

    if not profile.supports_streaming:
        raise ModelCapabilityError(
            f"Model {label} does not support streaming, which the runtime requires."
        )

    needs_tools = bool(toolset.tools) or bool(handoffs) or output.mode is OutputMode.TOOL
    if needs_tools and not profile.supports_tools:
        raise ModelCapabilityError(
            f"Model {label} does not support tool calls, "
            f"but the agent has tools, handoffs, or tool-output mode configured."
        )


def _resolve_output_pipeline[AgentDepsT, OutputDataT](
    *,
    output: OutputConfig[OutputDataT],
    output_pipeline: OutputPipeline[AgentDepsT, OutputDataT] | None,
) -> OutputPipeline[AgentDepsT, OutputDataT]:
    if output_pipeline is None:
        return OutputPipeline[AgentDepsT, OutputDataT].empty(mode=output.mode)
    return OutputPipeline(mode=output.mode, validators=output_pipeline.validators)


class Agent[AgentDepsT, OutputDataT]:
    """A typed agent with iterative execution and explicit output configuration."""

    __slots__ = (
        "model",
        "output",
        "output_pipeline",
        "system_instructions",
        "metadata",
        "toolset",
        "max_tool_rounds",
        "max_output_retries",
        "telemetry",
        "model_settings",
        "hooks",
        "history_processors",
        "handoffs",
    )

    model: Model
    output: OutputConfig[OutputDataT]
    output_pipeline: OutputPipeline[AgentDepsT, OutputDataT]
    system_instructions: tuple[str, ...]
    metadata: JsonObject | None
    toolset: Toolset[AgentDepsT]
    max_tool_rounds: int
    max_output_retries: int
    telemetry: TelemetryRecorder[OutputDataT]
    model_settings: ModelSettings | None
    hooks: Hooks[AgentDepsT]
    history_processors: tuple[NormalizedHistoryProcessor[AgentDepsT], ...]
    handoffs: tuple[Handoff, ...]

    def __init__(
        self,
        *,
        model: Model,
        decoder: OutputDecoder[OutputDataT] | None = None,
        output: OutputConfig[OutputDataT] | None = None,
        output_pipeline: OutputPipeline[AgentDepsT, OutputDataT] | None = None,
        system_instructions: tuple[str, ...] = (),
        metadata: JsonObject | None = None,
        toolset: Toolset[AgentDepsT] | None = None,
        max_tool_rounds: int = 8,
        max_output_retries: int = 2,
        telemetry: TelemetryRecorder[OutputDataT] | None = None,
        model_settings: ModelSettings | None = None,
        hooks: Hooks[AgentDepsT] | None = None,
        history_processors: tuple[HistoryProcessor[AgentDepsT], ...] = (),
        handoffs: tuple[Handoff, ...] = (),
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1.")
        if max_output_retries < 0:
            raise ValueError("max_output_retries must be non-negative.")

        resolved_output = _resolve_output(decoder=decoder, output=output)
        resolved_toolset = Toolset[AgentDepsT].empty() if toolset is None else toolset
        _validate_handoffs(handoffs)
        resolved_output = _merge_handoff_output(resolved_output, handoffs)
        _validate_output_tool_names(toolset=resolved_toolset, output=resolved_output)
        _validate_model_capabilities(
            model=model,
            output=resolved_output,
            toolset=resolved_toolset,
            handoffs=handoffs,
        )

        self.model = model
        self.output = resolved_output
        self.output_pipeline = _resolve_output_pipeline(
            output=resolved_output,
            output_pipeline=output_pipeline,
        )
        self.system_instructions = system_instructions
        self.metadata = metadata
        self.toolset = resolved_toolset
        self.max_tool_rounds = max_tool_rounds
        self.max_output_retries = max_output_retries
        self.telemetry = NoopTelemetryRecorder() if telemetry is None else telemetry
        self.model_settings = model_settings
        self.hooks = Hooks[AgentDepsT].empty() if hooks is None else hooks
        self.history_processors = tuple(
            normalize_history_processor(processor) for processor in history_processors
        )
        self.handoffs = handoffs

    @property
    def decoder(self) -> OutputDecoder[OutputDataT]:
        """Return the active output decoder."""
        return self.output.decoder

    @property
    def output_mode(self) -> OutputMode:
        """Return the active output mode."""
        return self.output.mode

    def _replace(self, **overrides: object) -> Agent[AgentDepsT, OutputDataT]:
        """Return a shallow copy with selected attributes replaced.

        Bypasses ``__init__`` validation since all values are already validated.
        """
        new: Agent[AgentDepsT, OutputDataT] = object.__new__(Agent)  # pyright: ignore[reportUnknownVariableType]
        for slot in self.__slots__:
            object.__setattr__(new, slot, overrides.get(slot, getattr(self, slot)))
        return new

    def with_output_validator(
        self,
        validator: OutputValidatorLike[AgentDepsT, OutputDataT],
    ) -> Agent[AgentDepsT, OutputDataT]:
        """Return a new agent with one output validator appended."""
        return self._replace(
            output_pipeline=self.output_pipeline.with_validator(validator),
        )

    def with_output[NewOutputDataT](
        self,
        output: OutputConfig[NewOutputDataT],
    ) -> Agent[AgentDepsT, NewOutputDataT]:
        """Return a new agent with a new output configuration.

        Handoff terminal tool names are merged into the new config and the
        resulting tool names are validated against the registered toolset.
        Output validators and telemetry are reset because they are typed to
        the previous output type and cannot safely carry over.
        """
        output = _merge_handoff_output(output, self.handoffs)
        _validate_output_tool_names(toolset=self.toolset, output=output)
        _validate_model_capabilities(
            model=self.model,
            output=output,
            toolset=self.toolset,
            handoffs=self.handoffs,
        )

        new: Agent[AgentDepsT, NewOutputDataT] = object.__new__(Agent)  # pyright: ignore[reportUnknownVariableType]
        for slot in self.__slots__:
            object.__setattr__(new, slot, getattr(self, slot))
        object.__setattr__(new, "output", output)
        object.__setattr__(
            new,
            "output_pipeline",
            OutputPipeline[AgentDepsT, NewOutputDataT].empty(mode=output.mode),
        )
        object.__setattr__(new, "telemetry", NoopTelemetryRecorder())
        return new

    def with_system_instruction(
        self,
        instruction: str,
    ) -> Agent[AgentDepsT, OutputDataT]:
        """Return a new agent with one additional system instruction."""
        if not instruction.strip():
            raise ValueError("system instruction must not be empty or whitespace.")
        return self._replace(
            system_instructions=(*self.system_instructions, instruction),
        )

    def with_tool(
        self,
        tool: Tool[AgentDepsT],
    ) -> Agent[AgentDepsT, OutputDataT]:
        """Return a new agent with one additional tool."""
        new_toolset = self.toolset.with_tool(tool)
        _validate_output_tool_names(toolset=new_toolset, output=self.output)
        _validate_model_capabilities(
            model=self.model,
            output=self.output,
            toolset=new_toolset,
            handoffs=self.handoffs,
        )
        return self._replace(toolset=new_toolset)

    def with_capability(
        self,
        capability: CapabilityLike[AgentDepsT, OutputDataT],
    ) -> Agent[AgentDepsT, OutputDataT]:
        """Return a new agent with one capability applied."""
        return normalize_capability(capability).apply(self)

    def with_capabilities(
        self,
        *capabilities: CapabilityLike[AgentDepsT, OutputDataT],
    ) -> Agent[AgentDepsT, OutputDataT]:
        """Return a new agent with multiple capabilities applied in order."""
        updated = self
        for capability in capabilities:
            updated = updated.with_capability(capability)
        return updated

    def with_max_tool_rounds(
        self,
        max_tool_rounds: int,
    ) -> Agent[AgentDepsT, OutputDataT]:
        """Return a new agent with an updated tool-round limit."""
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1.")
        return self._replace(max_tool_rounds=max_tool_rounds)

    def with_max_output_retries(
        self,
        max_output_retries: int,
    ) -> Agent[AgentDepsT, OutputDataT]:
        """Return a new agent with an updated output-retry limit."""
        if max_output_retries < 0:
            raise ValueError("max_output_retries must be non-negative.")
        return self._replace(max_output_retries=max_output_retries)

    def with_telemetry(
        self,
        telemetry: TelemetryRecorder[OutputDataT],
    ) -> Agent[AgentDepsT, OutputDataT]:
        """Return a new agent with a telemetry recorder."""
        return self._replace(telemetry=telemetry)

    def with_model(
        self,
        model: Model,
    ) -> Agent[AgentDepsT, OutputDataT]:
        """Return a new agent bound to a different model."""
        _validate_model_capabilities(
            model=model,
            output=self.output,
            toolset=self.toolset,
            handoffs=self.handoffs,
        )
        return self._replace(model=model)

    def with_model_settings(
        self,
        model_settings: ModelSettings | None,
    ) -> Agent[AgentDepsT, OutputDataT]:
        """Return a new agent with default model settings."""
        return self._replace(model_settings=model_settings)

    def with_hooks(
        self,
        hooks: Hooks[AgentDepsT],
    ) -> Agent[AgentDepsT, OutputDataT]:
        """Return a new agent with additional lifecycle hooks merged in order."""
        return self._replace(hooks=self.hooks.merge(hooks))

    def with_history_processor(
        self,
        processor: HistoryProcessor[AgentDepsT],
    ) -> Agent[AgentDepsT, OutputDataT]:
        """Return a new agent with one additional model-visible history processor."""
        return self._replace(
            history_processors=(
                *self.history_processors,
                normalize_history_processor(processor),
            ),
        )

    def with_handoff(
        self,
        handoff: Handoff,
    ) -> Agent[AgentDepsT, OutputDataT]:
        """Return a new agent with one additional handoff target registered."""
        new_handoffs = (*self.handoffs, handoff)
        _validate_handoffs(new_handoffs)
        new_output = _merge_handoff_output(self.output, new_handoffs)
        _validate_output_tool_names(toolset=self.toolset, output=new_output)
        _validate_model_capabilities(
            model=self.model,
            output=new_output,
            toolset=self.toolset,
            handoffs=new_handoffs,
        )
        return self._replace(
            handoffs=new_handoffs,
            output=new_output,
        )

    async def model_visible_history(
        self,
        context: RunContext[AgentDepsT],
        message_history: MessageHistory,
    ) -> MessageHistory:
        """Return the history visible to the model for the current request."""
        return MessageHistory(
            messages=await apply_history_processors(
                self.history_processors,
                context,
                message_history.messages,
            )
        )

    def as_tool(
        self,
        *,
        name: str,
        description: str | None = None,
    ) -> AsyncFunctionTool[AgentDepsT, AgentToolInput]:
        """Wrap this agent as an async tool for use by a parent agent.

        The parent's dependencies are forwarded to the child on invocation.
        """
        from wabizabi.tools.agent import agent_as_tool

        return agent_as_tool(self, name=name, description=description)

    def stream(
        self,
        user_prompt: str,
        *,
        deps: AgentDepsT,
        run_id: str | None = None,
        metadata: JsonObject | None = None,
        model: Model | None = None,
        instructions: tuple[str, ...] = (),
        settings: ModelSettings | None = None,
        message_history: MessageHistory | tuple[ModelRequest | ModelResponse, ...] | None = None,
    ) -> StreamedRun[OutputDataT]:
        """Return a single-consumer streamed run wrapper."""
        return StreamedRun(
            events_factory=lambda: self.iter(
                user_prompt,
                deps=deps,
                run_id=run_id,
                metadata=metadata,
                model=model,
                instructions=instructions,
                settings=settings,
                message_history=message_history,
            )
        )

    def stream_output(
        self,
        user_prompt: str,
        *,
        deps: AgentDepsT,
        run_id: str | None = None,
        metadata: JsonObject | None = None,
        model: Model | None = None,
        instructions: tuple[str, ...] = (),
        settings: ModelSettings | None = None,
        message_history: MessageHistory | tuple[ModelRequest | ModelResponse, ...] | None = None,
    ) -> AsyncIterator[OutputDataT]:
        """Stream validated outputs from the run."""
        return self.stream(
            user_prompt,
            deps=deps,
            run_id=run_id,
            metadata=metadata,
            model=model,
            instructions=instructions,
            settings=settings,
            message_history=message_history,
        ).outputs()

    def stream_responses(
        self,
        user_prompt: str,
        *,
        deps: AgentDepsT,
        run_id: str | None = None,
        metadata: JsonObject | None = None,
        model: Model | None = None,
        instructions: tuple[str, ...] = (),
        settings: ModelSettings | None = None,
        message_history: MessageHistory | tuple[ModelRequest | ModelResponse, ...] | None = None,
    ) -> AsyncIterator[ModelResponse]:
        """Stream normalized model responses from the run."""
        return self.stream(
            user_prompt,
            deps=deps,
            run_id=run_id,
            metadata=metadata,
            model=model,
            instructions=instructions,
            settings=settings,
            message_history=message_history,
        ).responses()

    def stream_text(
        self,
        user_prompt: str,
        *,
        deps: AgentDepsT,
        run_id: str | None = None,
        metadata: JsonObject | None = None,
        model: Model | None = None,
        instructions: tuple[str, ...] = (),
        settings: ModelSettings | None = None,
        message_history: MessageHistory | tuple[ModelRequest | ModelResponse, ...] | None = None,
    ) -> AsyncIterator[str]:
        """Stream projected text chunks from model responses."""
        return self.stream(
            user_prompt,
            deps=deps,
            run_id=run_id,
            metadata=metadata,
            model=model,
            instructions=instructions,
            settings=settings,
            message_history=message_history,
        ).text()

    async def iter(
        self,
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
        async for event in iter_run(
            self,
            user_prompt,
            deps=deps,
            run_id=run_id,
            metadata=metadata,
            model=model,
            instructions=instructions,
            settings=settings,
            message_history=message_history,
        ):
            yield event

    async def run(
        self,
        user_prompt: str,
        *,
        deps: AgentDepsT,
        run_id: str | None = None,
        metadata: JsonObject | None = None,
        model: Model | None = None,
        instructions: tuple[str, ...] = (),
        settings: ModelSettings | None = None,
        message_history: MessageHistory | tuple[ModelRequest | ModelResponse, ...] | None = None,
    ) -> RunResult[OutputDataT]:
        """Execute a run and return its terminal result.

        The returned :class:`RunResult` has either ``output`` (a normal
        terminal) or ``handoff`` (a handoff terminal) populated. Branch on
        ``result.handoff is not None`` to dispatch to the next agent.
        """
        initial_history = coerce_message_history(message_history)
        handoff_map = {h.name: h for h in self.handoffs} if self.handoffs else {}
        terminal: RunResult[OutputDataT] | None = None

        async for event in self.iter(
            user_prompt,
            deps=deps,
            run_id=run_id,
            metadata=metadata,
            model=model,
            instructions=instructions,
            settings=settings,
            message_history=message_history,
        ):
            if isinstance(event, HandoffEvent):
                terminal = RunResult[OutputDataT](
                    state=event.state,
                    new_messages=_new_messages_suffix(event.state, initial_history),
                    handoff=HandoffResult(
                        handoff=handoff_map[event.handoff_name],
                        tool_call=event.tool_call,
                        state=event.state,
                    ),
                )
            elif isinstance(event, OutputEvent):
                terminal = RunResult[OutputDataT](
                    state=event.state,
                    new_messages=_new_messages_suffix(event.state, initial_history),
                    output=event.output,
                )

        if terminal is None:
            raise RuntimeError("Agent run completed without producing output or handoff.")
        return terminal

    def runsync(
        self,
        user_prompt: str,
        *,
        deps: AgentDepsT,
        run_id: str | None = None,
        metadata: JsonObject | None = None,
        model: Model | None = None,
        instructions: tuple[str, ...] = (),
        settings: ModelSettings | None = None,
        message_history: MessageHistory | tuple[ModelRequest | ModelResponse, ...] | None = None,
    ) -> RunResult[OutputDataT]:
        """Synchronous wrapper around :meth:`run`."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.run(
                    user_prompt,
                    deps=deps,
                    run_id=run_id,
                    metadata=metadata,
                    model=model,
                    instructions=instructions,
                    settings=settings,
                    message_history=message_history,
                )
            )

        raise RuntimeError("runsync() cannot be called from a running event loop.")

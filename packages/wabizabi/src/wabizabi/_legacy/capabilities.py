"""Composable capability bundles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeGuard, runtime_checkable

from wabizabi.hooks import Hooks
from wabizabi.models import ModelSettings, merge_model_settings
from wabizabi.output import OutputValidatorLike
from wabizabi.processors import HistoryProcessor
from wabizabi.tools import Tool
from wabizabi.tools.toolset import Toolset

if TYPE_CHECKING:
    from wabizabi.agent import Agent


@runtime_checkable
class Capability[AgentDepsT, OutputDataT](Protocol):
    """A reusable bundle that can transform an agent."""

    def apply(
        self,
        agent: Agent[AgentDepsT, OutputDataT],
    ) -> Agent[AgentDepsT, OutputDataT]:
        """Apply this capability to an agent."""
        ...


type CapabilityFn[AgentDepsT, OutputDataT] = Callable[
    [Agent[AgentDepsT, OutputDataT]],
    Agent[AgentDepsT, OutputDataT],
]

type CapabilityLike[AgentDepsT, OutputDataT] = (
    Capability[AgentDepsT, OutputDataT] | CapabilityFn[AgentDepsT, OutputDataT]
)


def _is_capability[AgentDepsT, OutputDataT](
    value: CapabilityLike[AgentDepsT, OutputDataT],
) -> TypeGuard[Capability[AgentDepsT, OutputDataT]]:
    return isinstance(value, Capability)


@dataclass(frozen=True, slots=True)
class FunctionCapability[AgentDepsT, OutputDataT]:
    """Wrap a capability function as a capability object."""

    func: CapabilityFn[AgentDepsT, OutputDataT]

    def apply(
        self,
        agent: Agent[AgentDepsT, OutputDataT],
    ) -> Agent[AgentDepsT, OutputDataT]:
        return self.func(agent)


@dataclass(frozen=True, slots=True)
class StaticCapability[AgentDepsT, OutputDataT]:
    """A narrow reusable bundle of static agent contributions."""

    system_instructions: tuple[str, ...] = ()
    toolset: Toolset[AgentDepsT] | None = None
    tools: tuple[Tool[AgentDepsT], ...] = ()
    output_validators: tuple[OutputValidatorLike[AgentDepsT, OutputDataT], ...] = ()
    hooks: Hooks[AgentDepsT] | None = None
    history_processors: tuple[HistoryProcessor[AgentDepsT], ...] = ()
    model_settings: ModelSettings | None = None

    def apply(
        self,
        agent: Agent[AgentDepsT, OutputDataT],
    ) -> Agent[AgentDepsT, OutputDataT]:
        updated = agent

        for instruction in self.system_instructions:
            updated = updated.with_system_instruction(instruction)

        if self.toolset is not None:
            for tool in self.toolset.tools:
                updated = updated.with_tool(tool)

        for tool in self.tools:
            updated = updated.with_tool(tool)

        if self.model_settings is not None:
            updated = updated.with_model_settings(
                merge_model_settings(updated.model_settings, self.model_settings)
            )

        if self.hooks is not None:
            updated = updated.with_hooks(self.hooks)

        for processor in self.history_processors:
            updated = updated.with_history_processor(processor)

        for validator in self.output_validators:
            updated = updated.with_output_validator(validator)

        return updated


def normalize_capability[AgentDepsT, OutputDataT](
    capability: CapabilityLike[AgentDepsT, OutputDataT],
) -> Capability[AgentDepsT, OutputDataT]:
    """Normalize a public capability input into a capability object."""

    if _is_capability(capability):
        return capability
    if callable(capability):
        return FunctionCapability(func=capability)
    raise TypeError("Unsupported capability.")


def define_static_capability[AgentDepsT, OutputDataT](
    *,
    system_instructions: tuple[str, ...] = (),
    toolset: Toolset[AgentDepsT] | None = None,
    tools: tuple[Tool[AgentDepsT], ...] = (),
    output_validators: tuple[OutputValidatorLike[AgentDepsT, OutputDataT], ...] = (),
    hooks: Hooks[AgentDepsT] | None = None,
    history_processors: tuple[HistoryProcessor[AgentDepsT], ...] = (),
    model_settings: ModelSettings | None = None,
) -> StaticCapability[AgentDepsT, OutputDataT]:
    """Define a reusable static capability bundle."""

    return StaticCapability(
        system_instructions=system_instructions,
        toolset=toolset,
        tools=tools,
        output_validators=output_validators,
        hooks=hooks,
        history_processors=history_processors,
        model_settings=model_settings,
    )


__all__ = [
    "Capability",
    "CapabilityFn",
    "CapabilityLike",
    "FunctionCapability",
    "StaticCapability",
    "define_static_capability",
    "normalize_capability",
]

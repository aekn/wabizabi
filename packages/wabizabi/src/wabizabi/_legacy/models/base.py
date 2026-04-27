"""Model abstraction and provider-neutral request/response contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from wabizabi.history import MessageHistory
from wabizabi.messages import ModelRequest, ModelResponse
from wabizabi.output import OutputConfig
from wabizabi.tools import ToolDefinition
from wabizabi.usage import RunUsage

if TYPE_CHECKING:
    from wabizabi.models.stream import ModelStreamEvent


class ModelSettings(BaseModel):
    """Base settings model for provider-specific model settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def merge_model_settings[SettingsT: ModelSettings](
    base: SettingsT | None,
    override: SettingsT | None,
) -> SettingsT | None:
    """Merge two settings objects of the same type, preferring override values."""

    if base is None:
        return override
    if override is None:
        return base
    if type(base) is not type(override):
        raise TypeError("Cannot merge model settings of different types.")

    return base.model_copy(update=override.model_dump(exclude_none=True))


class ModelProfile(BaseModel):
    """Capability profile advertised by a model adapter.

    Only flags that the runtime actually enforces are modeled here. Other
    capabilities (reasoning, multimodal input, parallel tool calls, native
    structured output) are provider-local concerns decided at request time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    supports_tools: bool = True
    supports_streaming: bool = True


@dataclass(frozen=True, slots=True)
class ModelResult:
    """The normalized result of one model request."""

    response: ModelResponse
    usage: RunUsage = RunUsage.zero()


@runtime_checkable
class Model(Protocol):
    """Provider-neutral model adapter protocol."""

    @property
    def profile(self) -> ModelProfile:
        """Return the advertised capability profile."""
        ...

    async def request(
        self,
        request: ModelRequest,
        *,
        message_history: MessageHistory | None = None,
        settings: ModelSettings | None = None,
        tools: tuple[ToolDefinition, ...] = (),
        output: OutputConfig[object] | None = None,
    ) -> ModelResult:
        """Execute a single normalized model request."""
        ...

    def stream_response(
        self,
        request: ModelRequest,
        *,
        message_history: MessageHistory | None = None,
        settings: ModelSettings | None = None,
        tools: tuple[ToolDefinition, ...] = (),
        output: OutputConfig[object] | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream one normalized model response."""
        ...


__all__ = [
    "Model",
    "ModelProfile",
    "ModelResult",
    "ModelSettings",
    "merge_model_settings",
]

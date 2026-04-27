"""Lifecycle hooks for request, response, and tool execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

from wabizabi._async import resolve
from wabizabi.context import RunContext
from wabizabi.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from wabizabi.tools.toolset import Toolset

type PrepareToolsHook[AgentDepsT] = Callable[
    [RunContext[AgentDepsT], Toolset[AgentDepsT]],
    Toolset[AgentDepsT] | Awaitable[Toolset[AgentDepsT]],
]

type BeforeRequestHook[AgentDepsT] = Callable[
    [RunContext[AgentDepsT], ModelRequest],
    ModelRequest | Awaitable[ModelRequest],
]

type AfterResponseHook[AgentDepsT] = Callable[
    [RunContext[AgentDepsT], ModelResponse],
    ModelResponse | Awaitable[ModelResponse],
]

type BeforeToolCallHook[AgentDepsT] = Callable[
    [RunContext[AgentDepsT], ToolCallPart],
    ToolCallPart | Awaitable[ToolCallPart],
]

type AfterToolCallHook[AgentDepsT] = Callable[
    [RunContext[AgentDepsT], ToolCallPart, ToolReturnPart],
    ToolReturnPart | Awaitable[ToolReturnPart],
]

type NormalizedPrepareToolsHook[AgentDepsT] = Callable[
    [RunContext[AgentDepsT], Toolset[AgentDepsT]],
    Awaitable[Toolset[AgentDepsT]],
]

type NormalizedBeforeRequestHook[AgentDepsT] = Callable[
    [RunContext[AgentDepsT], ModelRequest],
    Awaitable[ModelRequest],
]

type NormalizedAfterResponseHook[AgentDepsT] = Callable[
    [RunContext[AgentDepsT], ModelResponse],
    Awaitable[ModelResponse],
]

type NormalizedBeforeToolCallHook[AgentDepsT] = Callable[
    [RunContext[AgentDepsT], ToolCallPart],
    Awaitable[ToolCallPart],
]

type NormalizedAfterToolCallHook[AgentDepsT] = Callable[
    [RunContext[AgentDepsT], ToolCallPart, ToolReturnPart],
    Awaitable[ToolReturnPart],
]


def _normalize_prepare_tools_hook[AgentDepsT](
    hook: PrepareToolsHook[AgentDepsT],
) -> NormalizedPrepareToolsHook[AgentDepsT]:
    async def wrapped(
        context: RunContext[AgentDepsT],
        toolset: Toolset[AgentDepsT],
    ) -> Toolset[AgentDepsT]:
        return await resolve(hook(context, toolset))

    return wrapped


def _normalize_before_request_hook[AgentDepsT](
    hook: BeforeRequestHook[AgentDepsT],
) -> NormalizedBeforeRequestHook[AgentDepsT]:
    async def wrapped(
        context: RunContext[AgentDepsT],
        request: ModelRequest,
    ) -> ModelRequest:
        return await resolve(hook(context, request))

    return wrapped


def _normalize_after_response_hook[AgentDepsT](
    hook: AfterResponseHook[AgentDepsT],
) -> NormalizedAfterResponseHook[AgentDepsT]:
    async def wrapped(
        context: RunContext[AgentDepsT],
        response: ModelResponse,
    ) -> ModelResponse:
        return await resolve(hook(context, response))

    return wrapped


def _normalize_before_tool_call_hook[AgentDepsT](
    hook: BeforeToolCallHook[AgentDepsT],
) -> NormalizedBeforeToolCallHook[AgentDepsT]:
    async def wrapped(
        context: RunContext[AgentDepsT],
        tool_call: ToolCallPart,
    ) -> ToolCallPart:
        return await resolve(hook(context, tool_call))

    return wrapped


def _normalize_after_tool_call_hook[AgentDepsT](
    hook: AfterToolCallHook[AgentDepsT],
) -> NormalizedAfterToolCallHook[AgentDepsT]:
    async def wrapped(
        context: RunContext[AgentDepsT],
        tool_call: ToolCallPart,
        tool_return: ToolReturnPart,
    ) -> ToolReturnPart:
        return await resolve(hook(context, tool_call, tool_return))

    return wrapped


@dataclass(frozen=True, slots=True)
class Hooks[AgentDepsT]:
    """Lifecycle hooks applied by the shared runtime."""

    prepare_tools: tuple[NormalizedPrepareToolsHook[AgentDepsT], ...] = ()
    before_request: tuple[NormalizedBeforeRequestHook[AgentDepsT], ...] = ()
    after_response: tuple[NormalizedAfterResponseHook[AgentDepsT], ...] = ()
    before_tool_call: tuple[NormalizedBeforeToolCallHook[AgentDepsT], ...] = ()
    after_tool_call: tuple[NormalizedAfterToolCallHook[AgentDepsT], ...] = ()

    @classmethod
    def empty(cls) -> Hooks[AgentDepsT]:
        return cls()

    def merge(self, other: Hooks[AgentDepsT]) -> Hooks[AgentDepsT]:
        return Hooks(
            prepare_tools=(*self.prepare_tools, *other.prepare_tools),
            before_request=(*self.before_request, *other.before_request),
            after_response=(*self.after_response, *other.after_response),
            before_tool_call=(*self.before_tool_call, *other.before_tool_call),
            after_tool_call=(*self.after_tool_call, *other.after_tool_call),
        )

    def with_prepare_tools(self, hook: PrepareToolsHook[AgentDepsT]) -> Hooks[AgentDepsT]:
        return replace(
            self, prepare_tools=(*self.prepare_tools, _normalize_prepare_tools_hook(hook))
        )

    def with_before_request(self, hook: BeforeRequestHook[AgentDepsT]) -> Hooks[AgentDepsT]:
        return replace(
            self, before_request=(*self.before_request, _normalize_before_request_hook(hook))
        )

    def with_after_response(self, hook: AfterResponseHook[AgentDepsT]) -> Hooks[AgentDepsT]:
        return replace(
            self, after_response=(*self.after_response, _normalize_after_response_hook(hook))
        )

    def with_before_tool_call(
        self,
        hook: BeforeToolCallHook[AgentDepsT],
    ) -> Hooks[AgentDepsT]:
        return replace(
            self,
            before_tool_call=(*self.before_tool_call, _normalize_before_tool_call_hook(hook)),
        )

    def with_after_tool_call(
        self,
        hook: AfterToolCallHook[AgentDepsT],
    ) -> Hooks[AgentDepsT]:
        return replace(
            self,
            after_tool_call=(*self.after_tool_call, _normalize_after_tool_call_hook(hook)),
        )

    async def apply_prepare_tools(
        self,
        context: RunContext[AgentDepsT],
        toolset: Toolset[AgentDepsT],
    ) -> Toolset[AgentDepsT]:
        current = toolset
        for hook in self.prepare_tools:
            current = await hook(context, current)
        return current

    async def apply_before_request(
        self,
        context: RunContext[AgentDepsT],
        request: ModelRequest,
    ) -> ModelRequest:
        current = request
        for hook in self.before_request:
            current = await hook(context, current)
        return current

    async def apply_after_response(
        self,
        context: RunContext[AgentDepsT],
        response: ModelResponse,
    ) -> ModelResponse:
        current = response
        for hook in self.after_response:
            current = await hook(context, current)
        return current

    async def apply_before_tool_call(
        self,
        context: RunContext[AgentDepsT],
        tool_call: ToolCallPart,
    ) -> ToolCallPart:
        current = tool_call
        for hook in self.before_tool_call:
            current = await hook(context, current)
        return current

    async def apply_after_tool_call(
        self,
        context: RunContext[AgentDepsT],
        tool_call: ToolCallPart,
        tool_return: ToolReturnPart,
    ) -> ToolReturnPart:
        current = tool_return
        for hook in self.after_tool_call:
            current = await hook(context, tool_call, current)
        return current


__all__ = [
    "AfterResponseHook",
    "AfterToolCallHook",
    "BeforeRequestHook",
    "BeforeToolCallHook",
    "Hooks",
    "PrepareToolsHook",
]

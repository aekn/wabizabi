"""Function-backed tool implementations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from wabizabi.context import RunContext
from wabizabi.tools.base import ToolDefinition, ToolInvocationError
from wabizabi.tools.schema import tool_input_schema
from wabizabi.types import JsonObject, JsonValue, json_value_from_unknown

type SyncToolFn[AgentDepsT, ToolArgsT: BaseModel] = Callable[
    [RunContext[AgentDepsT], ToolArgsT],
    JsonValue,
]

type AsyncToolFn[AgentDepsT, ToolArgsT: BaseModel] = Callable[
    [RunContext[AgentDepsT], ToolArgsT],
    Awaitable[JsonValue],
]


def _tool_error_message(name: str, error: Exception) -> str:
    return f"Tool {name!r} failed: {error}"


@dataclass(frozen=True, slots=True)
class FunctionTool[AgentDepsT, ToolArgsT: BaseModel]:
    """A synchronous function-backed tool."""

    name: str
    arguments_type: type[ToolArgsT]
    func: SyncToolFn[AgentDepsT, ToolArgsT]
    description: str | None = None
    definition: ToolDefinition = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "definition",
            ToolDefinition(
                name=self.name,
                description=self.description,
                input_schema=tool_input_schema(self.arguments_type),
            ),
        )

    async def invoke(
        self,
        context: RunContext[AgentDepsT],
        arguments: JsonObject,
    ) -> JsonValue:
        try:
            parsed_arguments = self.arguments_type.model_validate(arguments)
        except ValidationError as exc:
            raise ToolInvocationError(_tool_error_message(self.name, exc)) from exc

        try:
            result = self.func(context, parsed_arguments)
        except ToolInvocationError:
            raise
        except Exception as exc:
            raise ToolInvocationError(_tool_error_message(self.name, exc)) from exc

        try:
            return json_value_from_unknown(result)
        except Exception as exc:
            raise ToolInvocationError(_tool_error_message(self.name, exc)) from exc


@dataclass(frozen=True, slots=True)
class AsyncFunctionTool[AgentDepsT, ToolArgsT: BaseModel]:
    """An asynchronous function-backed tool."""

    name: str
    arguments_type: type[ToolArgsT]
    func: AsyncToolFn[AgentDepsT, ToolArgsT]
    description: str | None = None
    definition: ToolDefinition = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "definition",
            ToolDefinition(
                name=self.name,
                description=self.description,
                input_schema=tool_input_schema(self.arguments_type),
            ),
        )

    async def invoke(
        self,
        context: RunContext[AgentDepsT],
        arguments: JsonObject,
    ) -> JsonValue:
        try:
            parsed_arguments = self.arguments_type.model_validate(arguments)
        except ValidationError as exc:
            raise ToolInvocationError(_tool_error_message(self.name, exc)) from exc

        try:
            result = await self.func(context, parsed_arguments)
        except ToolInvocationError:
            raise
        except Exception as exc:
            raise ToolInvocationError(_tool_error_message(self.name, exc)) from exc

        try:
            return json_value_from_unknown(result)
        except Exception as exc:
            raise ToolInvocationError(_tool_error_message(self.name, exc)) from exc


def define_function_tool[AgentDepsT, ToolArgsT: BaseModel](
    *,
    name: str,
    arguments_type: type[ToolArgsT],
    func: SyncToolFn[AgentDepsT, ToolArgsT],
    description: str | None = None,
) -> FunctionTool[AgentDepsT, ToolArgsT]:
    """Define a synchronous tool."""
    return FunctionTool(
        name=name,
        arguments_type=arguments_type,
        func=func,
        description=description,
    )


def define_async_function_tool[AgentDepsT, ToolArgsT: BaseModel](
    *,
    name: str,
    arguments_type: type[ToolArgsT],
    func: AsyncToolFn[AgentDepsT, ToolArgsT],
    description: str | None = None,
) -> AsyncFunctionTool[AgentDepsT, ToolArgsT]:
    """Define an asynchronous tool."""
    return AsyncFunctionTool(
        name=name,
        arguments_type=arguments_type,
        func=func,
        description=description,
    )


__all__ = [
    "AsyncFunctionTool",
    "AsyncToolFn",
    "FunctionTool",
    "SyncToolFn",
    "define_async_function_tool",
    "define_function_tool",
]

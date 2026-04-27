"""Decorator-based tool definitions."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from inspect import Parameter, Signature
from typing import (
    Annotated,
    Concatenate,
    ParamSpec,
    TypeGuard,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from wabizabi._async import is_async_callable
from wabizabi.context import RunContext
from wabizabi.tools.function import (
    AsyncFunctionTool,
    FunctionTool,
    define_async_function_tool,
    define_function_tool,
)
from wabizabi.types import JsonValue

P = ParamSpec("P")
Q = ParamSpec("Q")

type _ModelFieldDefinition = tuple[object, object]


@dataclass(frozen=True, slots=True)
class _DocstringInfo:
    summary: str | None
    parameter_descriptions: dict[str, str]


def _as_callable(func: object) -> Callable[..., object]:
    if not callable(func):
        raise TypeError("Tool decorators require a callable.")
    return func


def _annotation_is_run_context(annotation: object) -> bool:
    return annotation is RunContext or get_origin(annotation) is RunContext


def _field_info_from_annotation(annotation: object) -> FieldInfo | None:
    if get_origin(annotation) is not Annotated:
        return None
    for metadata in get_args(annotation)[1:]:
        if isinstance(metadata, FieldInfo):
            return metadata
    return None


def _field_has_description(annotation: object) -> bool:
    field_info = _field_info_from_annotation(annotation)
    return field_info is not None and field_info.description is not None


def _is_json_value(value: object) -> TypeGuard[JsonValue]:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if not isinstance(value, list | dict):
        return False
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError, OverflowError):
        return False


def _json_value(value: object) -> JsonValue:
    if not _is_json_value(value):
        raise TypeError("Tool decorators require JSON-serializable return values.")
    return value


def _google_docstring_info(func: Callable[..., object]) -> _DocstringInfo:
    docstring = inspect.getdoc(func)
    if docstring is None:
        return _DocstringInfo(summary=None, parameter_descriptions={})

    lines = docstring.splitlines()
    summary = next((line.strip() for line in lines if line.strip()), None)
    parameter_descriptions: dict[str, str] = {}
    in_args_section = False
    current_name: str | None = None
    current_lines: list[str] = []

    def flush_current() -> None:
        nonlocal current_name, current_lines
        if current_name is None:
            return
        description = " ".join(line for line in current_lines if line)
        parameter_descriptions[current_name] = " ".join(description.split())
        current_name = None
        current_lines = []

    for raw_line in lines:
        stripped = raw_line.strip()
        indent = len(raw_line) - len(raw_line.lstrip())

        if stripped in {"Args:", "Arguments:"}:
            flush_current()
            in_args_section = True
            continue

        if not in_args_section or not stripped:
            continue

        if indent == 0 and stripped.endswith(":"):
            flush_current()
            in_args_section = False
            continue

        if indent >= 4 and ":" in stripped:
            candidate_name, candidate_description = stripped.split(":", maxsplit=1)
            if candidate_name.isidentifier():
                flush_current()
                current_name = candidate_name
                description_text = candidate_description.strip()
                if description_text:
                    current_lines.append(description_text)
                continue

        if current_name is not None and indent >= 8:
            current_lines.append(stripped)
            continue

        flush_current()
        in_args_section = False

    flush_current()
    return _DocstringInfo(summary=summary, parameter_descriptions=parameter_descriptions)


def _tool_parameters(
    func: Callable[..., object],
    *,
    skip_context: bool,
) -> tuple[Parameter, ...]:
    parameters = tuple(inspect.signature(func).parameters.values())

    if skip_context:
        if not parameters:
            raise TypeError("@tool requires a first RunContext parameter.")
        context_parameter = parameters[0]
        annotations = get_type_hints(func, include_extras=True)
        context_annotation = annotations.get(context_parameter.name, context_parameter.annotation)
        if context_annotation is Signature.empty or not _annotation_is_run_context(
            context_annotation
        ):
            raise TypeError(
                "@tool requires the first parameter to be annotated as RunContext[...]."
            )
        parameters = parameters[1:]

    for parameter in parameters:
        if parameter.kind not in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}:
            raise TypeError(
                "Tool decorators only support positional-or-keyword and keyword-only parameters."
            )

    return parameters


def _field_default(
    *,
    parameter: Parameter,
    annotation: object,
    docstring_info: _DocstringInfo,
) -> object:
    if parameter.default is Parameter.empty:
        if _field_has_description(annotation):
            return ...
        description = docstring_info.parameter_descriptions.get(parameter.name)
        return Field(description=description) if description is not None else ...

    if _field_has_description(annotation):
        return parameter.default

    description = docstring_info.parameter_descriptions.get(parameter.name)
    if description is None:
        return parameter.default
    return Field(default=parameter.default, description=description)


def _create_arguments_model(
    *,
    model_name: str,
    module_name: str,
    field_definitions: Mapping[str, _ModelFieldDefinition],
) -> type[BaseModel]:
    fields = dict(field_definitions)
    return create_model(model_name, __module__=module_name, **fields)  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownVariableType]


def _callable_module_name(func: Callable[..., object]) -> str:
    module_name = getattr(func, "__module__", None)
    return module_name if isinstance(module_name, str) else "__main__"


def _tool_name(func: Callable[..., object], explicit_name: str | None) -> str:
    if explicit_name is not None:
        return explicit_name
    name = getattr(func, "__name__", None)
    if not isinstance(name, str) or not name:
        raise TypeError("Tool decorators require a named function.")
    return name


def _tool_arguments_model(
    func: Callable[..., object],
    *,
    skip_context: bool,
) -> type[BaseModel]:
    parameters = _tool_parameters(func, skip_context=skip_context)
    annotations = get_type_hints(func, include_extras=True)
    docstring_info = _google_docstring_info(func)

    field_definitions: dict[str, _ModelFieldDefinition] = {}
    for parameter in parameters:
        annotation = annotations.get(parameter.name, parameter.annotation)
        if annotation is Signature.empty:
            raise TypeError(f"Tool parameter {parameter.name!r} must have a type annotation.")
        field_definitions[parameter.name] = (
            annotation,
            _field_default(
                parameter=parameter,
                annotation=annotation,
                docstring_info=docstring_info,
            ),
        )

    tool_name = _tool_name(func, None)
    model_name = "".join(part.capitalize() for part in tool_name.split("_")) or "Tool"
    return _create_arguments_model(
        model_name=f"{model_name}Arguments",
        module_name=_callable_module_name(func),
        field_definitions=field_definitions,
    )


def _tool_description(func: Callable[..., object], explicit_description: str | None) -> str | None:
    if explicit_description is not None:
        return explicit_description
    return _google_docstring_info(func).summary


def _arguments_payload(arguments: BaseModel) -> dict[str, object]:
    return arguments.model_dump()


def _build_context_tool[AgentDepsT](
    func: Callable[..., object],
    *,
    explicit_name: str | None,
    explicit_description: str | None,
) -> FunctionTool[AgentDepsT, BaseModel] | AsyncFunctionTool[AgentDepsT, BaseModel]:
    callable_func = _as_callable(func)
    arguments_model = _tool_arguments_model(callable_func, skip_context=True)
    name = _tool_name(callable_func, explicit_name)
    description = _tool_description(callable_func, explicit_description)

    if is_async_callable(callable_func):

        async def async_invoke(context: RunContext[AgentDepsT], arguments: BaseModel) -> JsonValue:
            result = callable_func(context, **_arguments_payload(arguments))
            if not inspect.isawaitable(result):
                raise TypeError("Async tool wrappers must produce awaitable results.")
            return _json_value(await result)

        return define_async_function_tool(
            name=name,
            arguments_type=arguments_model,
            func=async_invoke,
            description=description,
        )

    def sync_invoke(context: RunContext[AgentDepsT], arguments: BaseModel) -> JsonValue:
        result = callable_func(context, **_arguments_payload(arguments))
        return _json_value(result)

    return define_function_tool(
        name=name,
        arguments_type=arguments_model,
        func=sync_invoke,
        description=description,
    )


def _build_plain_tool(
    func: Callable[..., object],
    *,
    explicit_name: str | None,
    explicit_description: str | None,
) -> FunctionTool[object, BaseModel] | AsyncFunctionTool[object, BaseModel]:
    callable_func = _as_callable(func)
    arguments_model = _tool_arguments_model(callable_func, skip_context=False)
    name = _tool_name(callable_func, explicit_name)
    description = _tool_description(callable_func, explicit_description)

    if is_async_callable(callable_func):

        async def async_invoke(_context: RunContext[object], arguments: BaseModel) -> JsonValue:
            result = callable_func(**_arguments_payload(arguments))
            if not inspect.isawaitable(result):
                raise TypeError("Async tool wrappers must produce awaitable results.")
            return _json_value(await result)

        return define_async_function_tool(
            name=name,
            arguments_type=arguments_model,
            func=async_invoke,
            description=description,
        )

    def sync_invoke(_context: RunContext[object], arguments: BaseModel) -> JsonValue:
        result = callable_func(**_arguments_payload(arguments))
        return _json_value(result)

    return define_function_tool(
        name=name,
        arguments_type=arguments_model,
        func=sync_invoke,
        description=description,
    )


@overload
def tool[AgentDepsT, **P](
    func: Callable[Concatenate[RunContext[AgentDepsT], P], JsonValue],
    /,
) -> FunctionTool[AgentDepsT, BaseModel]: ...


@overload
def tool[AgentDepsT, **P](
    func: Callable[Concatenate[RunContext[AgentDepsT], P], Awaitable[JsonValue]],
    /,
) -> AsyncFunctionTool[AgentDepsT, BaseModel]: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., object]], object]: ...


def tool(
    func: object | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> object:
    """Define a tool from a function that receives ``RunContext`` first."""

    def decorator(decorated_func: object) -> object:
        return _build_context_tool(  # pyright: ignore[reportUnknownVariableType]
            _as_callable(decorated_func),
            explicit_name=name,
            explicit_description=description,
        )

    if func is None:
        return decorator
    return decorator(func)


@overload
def tool_plain[**Q](
    func: Callable[Q, JsonValue],
    /,
) -> FunctionTool[object, BaseModel]: ...


@overload
def tool_plain[**Q](
    func: Callable[Q, Awaitable[JsonValue]],
    /,
) -> AsyncFunctionTool[object, BaseModel]: ...


@overload
def tool_plain(
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., object]], object]: ...


def tool_plain(
    func: object | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> object:
    """Define a tool from a function that does not require ``RunContext``."""

    def decorator(decorated_func: object) -> object:
        return _build_plain_tool(
            _as_callable(decorated_func),
            explicit_name=name,
            explicit_description=description,
        )

    if func is None:
        return decorator
    return decorator(func)


__all__ = ["tool", "tool_plain"]

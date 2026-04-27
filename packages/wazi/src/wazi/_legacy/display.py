"""Rich display and streaming rendering for wazi CLI."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass

from pydantic import BaseModel
from rich.console import Console
from rich.text import Text
from rich.theme import Theme
from wabizabi.messages import ModelMessage
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
)
from wabizabi.usage import RunUsage

_THEME = Theme(
    {
        "info": "dim",
        "accent": "steel_blue1",
        "muted": "grey50",
        "error": "red",
        "tool": "yellow",
        "handoff": "magenta",
        "model": "cyan",
        "timing": "dim italic",
        "usage_label": "dim",
        "cmd": "dim cyan",
        "trace_event": "dim yellow",
        "trace_detail": "dim",
    }
)


def _build_console(file: object | None = None) -> Console:
    kwargs: dict[str, object] = {"theme": _THEME}
    if file is not None:
        kwargs["file"] = file
    return Console(**kwargs)  # pyright: ignore[reportArgumentType]


def _stdout_console() -> Console:
    return _build_console(file=sys.stdout)


def _stderr_console() -> Console:
    return _build_console(file=sys.stderr)


def print_info(text: str) -> None:
    """Print dim informational text to stderr."""
    _stderr_console().print(Text(text, style="info"))


def print_error(text: str) -> None:
    """Print error text to stderr."""
    _stderr_console().print(Text(f"error: {text}", style="error"))


def print_command_result(text: str) -> None:
    """Print a command result to stderr."""
    _stderr_console().print(Text(text, style="cmd"))


@dataclass(frozen=True, slots=True)
class StreamResult[OutputDataT]:
    """Result from rendering a streamed run."""

    output: OutputDataT | None
    messages: tuple[ModelMessage, ...]
    usage: RunUsage
    handoff_name: str | None = None
    rendered_text: bool = False


async def render_stream[OutputDataT](
    events: AsyncIterator[RunEvent[OutputDataT]],
    *,
    trace: bool = False,
) -> StreamResult[OutputDataT]:
    """Consume an event stream, rendering text incrementally and traces to stderr."""
    err = _stderr_console()
    output: OutputDataT | None = None
    messages: tuple[ModelMessage, ...] = ()
    usage = RunUsage.zero()
    handoff_name: str | None = None
    rendered_text = False
    last_text_had_newline = True
    t0 = time.monotonic()
    tool_t0: float | None = None

    async for event in events:
        if isinstance(event, TextChunkEvent):
            sys.stdout.write(event.text)
            sys.stdout.flush()
            rendered_text = True
            last_text_had_newline = event.text.endswith("\n")
            continue

        if isinstance(event, ReasoningChunkEvent):
            if trace:
                err.print(Text(f"  [thinking] {event.text}", style="trace_detail"))
            continue

        if isinstance(event, RequestEvent):
            usage = event.state.usage
            if trace:
                err.print(Text(f"  [request] step={event.state.run_step}", style="trace_event"))
            continue

        if isinstance(event, ResponseEvent):
            usage = event.state.usage
            if trace:
                err.print(
                    Text(f"  [response] {len(event.response.parts)} parts", style="trace_event")
                )
            continue

        if isinstance(event, ToolCallEvent):
            tool_t0 = time.monotonic()
            if trace:
                err.print(Text(f"  >> {event.tool_call.tool_name}", style="tool"))
                args_summary = summarize_args(event.tool_call.arguments)
                if args_summary:
                    err.print(Text(f"     {args_summary}", style="trace_detail"))
            continue

        if isinstance(event, ToolResultEvent):
            usage = event.state.usage
            if trace:
                elapsed = _elapsed_since(tool_t0)
                err.print(Text(f"  << {event.tool_return.tool_name}{elapsed}", style="muted"))
                result_summary = summarize_result(event.tool_return.content)
                if result_summary:
                    err.print(Text(f"     {result_summary}", style="trace_detail"))
            tool_t0 = None
            continue

        if isinstance(event, HandoffEvent):
            messages = event.state.message_history.messages
            usage = event.state.usage
            handoff_name = event.handoff_name
            if trace:
                err.print(Text(f"  -> handoff: {event.handoff_name}", style="handoff"))
            continue

        if isinstance(event, OutputEvent):
            output = event.output
            messages = event.state.message_history.messages
            usage = event.state.usage
            continue

    if rendered_text and not last_text_had_newline:
        sys.stdout.write("\n")
        sys.stdout.flush()

    _print_footer(t0, usage)
    return StreamResult(
        output=output,
        messages=messages,
        usage=usage,
        handoff_name=handoff_name,
        rendered_text=rendered_text,
    )


def print_welcome(model: str, *, trace: bool = False) -> None:
    """Print the welcome banner for interactive chat."""
    err = _stderr_console()
    err.print()
    title = Text()
    title.append("wazi", style="bold accent")
    title.append("  model=", style="info")
    title.append(model, style="model")
    if trace:
        title.append("  trace=on", style="trace_event")
    err.print(title)
    err.print(Text("  type /help for commands, /quit to exit", style="info"))
    err.print()


def print_session_end(turn_count: int) -> None:
    """Print the session end message."""
    err = _stderr_console()
    err.print()
    err.print(Text(f"  {turn_count} turns", style="info"))


def _print_footer(t0: float, usage: RunUsage) -> None:
    elapsed = time.monotonic() - t0
    parts: list[str] = [f"{elapsed:.1f}s"]
    if usage.total_tokens > 0:
        parts.append(f"{usage.total_tokens:,} tokens")
    _stderr_console().print(Text(f"  {'  '.join(parts)}", style="timing"))


def _elapsed_since(t0: float | None) -> str:
    if t0 is None:
        return ""
    elapsed = time.monotonic() - t0
    return f" ({elapsed:.1f}s)"


def summarize_args(arguments: Mapping[str, object]) -> str:
    """Produce a short summary of tool call arguments for trace mode."""
    parts: list[str] = []
    for key, value in arguments.items():
        s = repr(value)
        if len(s) > 40:
            s = s[:37] + "..."
        parts.append(f"{key}={s}")
    summary = ", ".join(parts)
    if len(summary) > 80:
        return summary[:77] + "..."
    return summary


def summarize_result(content: object) -> str:
    """Produce a short summary of a tool result for trace mode."""
    if isinstance(content, BaseModel):
        s = content.model_dump_json()
    elif isinstance(content, dict | list):
        s = json.dumps(content, separators=(",", ":"), sort_keys=True)
    else:
        s = str(content)
    if len(s) > 80:
        return s[:77] + "..."
    return s

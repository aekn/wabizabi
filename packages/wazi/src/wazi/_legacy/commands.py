"""Slash command registry and dispatch for the wazi REPL."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wazi.session import ChatSession


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Result of executing a slash command."""

    message: str | None = None
    should_exit: bool = False


@dataclass(frozen=True, slots=True)
class CommandDef:
    """A registered slash command."""

    name: str
    description: str
    usage: str | None = None


@dataclass(frozen=True, slots=True)
class AgentView:
    """Shell-facing snapshot of one registered agent."""

    name: str
    model_name: str
    tool_names: tuple[str, ...] = ()
    base_system_instructions: tuple[str, ...] = ()
    temperature: float | None = None
    supports_model_override: bool = False


COMMANDS: dict[str, CommandDef] = {
    "help": CommandDef("help", "Show available commands"),
    "model": CommandDef("model", "Show or switch model", usage="/model [name|clear]"),
    "agent": CommandDef("agent", "Show or switch active agent", usage="/agent [name]"),
    "clear": CommandDef("clear", "Clear conversation history"),
    "history": CommandDef("history", "Show conversation summary"),
    "trace": CommandDef("trace", "Toggle trace mode"),
    "tools": CommandDef("tools", "Show registered tools"),
    "system": CommandDef(
        "system", "Show or add session system instructions", usage="/system [text|clear]"
    ),
    "usage": CommandDef("usage", "Show token usage for this session"),
    "config": CommandDef("config", "Show current shell configuration"),
    "quit": CommandDef("quit", "Exit the session"),
    "exit": CommandDef("exit", "Exit the session"),
}


def is_command(text: str) -> bool:
    """Return True if *text* looks like a slash command."""
    return text.startswith("/") and len(text) > 1 and not text.startswith("//")


def parse_command(text: str) -> tuple[str, str]:
    """Split a command string into (command_name, args_string)."""
    stripped = text[1:].strip()
    parts = stripped.split(None, 1)
    name = parts[0].lower() if parts else ""
    args = parts[1] if len(parts) > 1 else ""
    return name, args


def resolve_command(name: str) -> str | None:
    """Resolve a command name, supporting unambiguous prefix matching."""
    if name in COMMANDS:
        return name
    matches = [cmd for cmd in COMMANDS if cmd.startswith(name)]
    if len(matches) == 1:
        return matches[0]
    return None


@dataclass(slots=True)
class REPLState:
    """Mutable state shared across the REPL for command access."""

    session: ChatSession
    trace: bool
    agent_views: Mapping[str, AgentView] = field(default_factory=dict)
    active_agent: str | None = None
    model_name: str = ""
    tool_names: tuple[str, ...] = ()
    base_system_instructions: tuple[str, ...] = ()
    session_system_instructions: list[str] = field(default_factory=list)
    temperature: float | None = None
    supports_model_override: bool = False
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.agent_views and self.active_agent is None:
            self.active_agent = next(iter(self.agent_views))
        if self.active_agent is not None:
            self.switch_agent(self.active_agent)

    @property
    def available_agents(self) -> tuple[str, ...]:
        return tuple(self.agent_views.keys())

    @property
    def effective_system_instructions(self) -> tuple[str, ...]:
        return tuple(self.session_system_instructions)

    def current_agent_view(self) -> AgentView | None:
        if self.active_agent is None:
            return None
        return self.agent_views.get(self.active_agent)

    def switch_agent(self, name: str) -> None:
        view = self.agent_views[name]
        self.active_agent = name
        self.model_name = view.model_name
        self.tool_names = view.tool_names
        self.base_system_instructions = view.base_system_instructions
        self.temperature = view.temperature
        self.supports_model_override = view.supports_model_override


type _CommandHandler = Callable[[str, REPLState], CommandResult]


def dispatch(name: str, args: str, state: REPLState) -> CommandResult:
    """Execute a resolved command name and return its result."""
    resolved = resolve_command(name)
    if resolved is None:
        suggestions = [cmd for cmd in COMMANDS if cmd.startswith(name)]
        if suggestions:
            return CommandResult(
                message=f"Ambiguous command. Did you mean: {', '.join(suggestions)}"
            )
        return CommandResult(
            message=f"Unknown command: /{name}. Type /help for available commands."
        )

    handler = _HANDLERS.get(resolved)
    if handler is None:
        return CommandResult(message=f"Command /{resolved} is not implemented.")
    return handler(args, state)


def _cmd_help(_args: str, _state: REPLState) -> CommandResult:
    lines: list[str] = ["Available commands:"]
    for cmd in COMMANDS.values():
        label = cmd.usage if cmd.usage else f"/{cmd.name}"
        lines.append(f"  {label:<24s} {cmd.description}")
    return CommandResult(message="\n".join(lines))


def _cmd_model(args: str, state: REPLState) -> CommandResult:
    view = state.current_agent_view()
    base_model = view.model_name if view is not None else state.model_name
    if not args:
        if view is None:
            return CommandResult(message=f"Current model: {state.model_name}")
        override = "yes" if state.model_name != base_model else "no"
        return CommandResult(
            message=(
                f"Current model: {state.model_name}\n"
                f"Base model: {base_model}\n"
                f"Override active: {override}\n"
                f"Model override supported: {state.supports_model_override}"
            )
        )

    value = args.strip()
    if value == "clear":
        state.model_name = base_model
        return CommandResult(message=f"Model reset to base agent model: {base_model}")

    if not state.supports_model_override:
        return CommandResult(message="Current agent does not support truthful model overrides.")

    state.model_name = value
    return CommandResult(message=f"Switched to model: {state.model_name}")


def _cmd_agent(args: str, state: REPLState) -> CommandResult:
    if state.active_agent is None:
        return CommandResult(message="No agent registry loaded.")
    if not args:
        lines = [f"Active: {state.active_agent}", "Available:"]
        for name in state.available_agents:
            marker = "*" if name == state.active_agent else " "
            lines.append(f"  {marker} {name}")
        return CommandResult(message="\n".join(lines))
    target = args.strip()
    if target not in state.available_agents:
        return CommandResult(
            message=f"Unknown agent: {target!r}. Registered: {', '.join(state.available_agents)}"
        )
    state.switch_agent(target)
    return CommandResult(message=f"Switched to agent: {target}")


def _cmd_clear(_args: str, state: REPLState) -> CommandResult:
    state.session.clear()
    return CommandResult(message="Conversation cleared.")


def _cmd_history(_args: str, state: REPLState) -> CommandResult:
    session = state.session
    msg_count = len(session.history.messages)
    return CommandResult(message=f"{session.turn_count} turns, {msg_count} messages")


def _cmd_trace(_args: str, state: REPLState) -> CommandResult:
    state.trace = not state.trace
    label = "on" if state.trace else "off"
    return CommandResult(message=f"Trace mode: {label}")


def _cmd_tools(_args: str, state: REPLState) -> CommandResult:
    if not state.tool_names:
        return CommandResult(message="No tools registered.")
    lines = ["Registered tools:"]
    for name in state.tool_names:
        lines.append(f"  {name}")
    return CommandResult(message="\n".join(lines))


def _cmd_system(args: str, state: REPLState) -> CommandResult:
    if not args:
        lines: list[str] = []
        if state.base_system_instructions:
            lines.append("Agent system instructions:")
            for i, inst in enumerate(state.base_system_instructions, 1):
                lines.append(f"  {i}. {inst}")
        if state.session_system_instructions:
            lines.append("Session system instructions:")
            for i, inst in enumerate(state.session_system_instructions, 1):
                lines.append(f"  {i}. {inst}")
        if not lines:
            return CommandResult(message="No system instructions set.")
        return CommandResult(message="\n".join(lines))

    value = args.strip()
    if value == "clear":
        state.session_system_instructions.clear()
        return CommandResult(message="Cleared session system instructions.")

    state.session_system_instructions.append(value)
    return CommandResult(message=f"Added session system instruction: {value}")


def _cmd_usage(_args: str, state: REPLState) -> CommandResult:
    total = state.total_input_tokens + state.total_output_tokens
    return CommandResult(
        message=(
            f"Session usage: {total:,} tokens "
            f"({state.total_input_tokens:,} in, {state.total_output_tokens:,} out)"
        )
    )


def _cmd_config(_args: str, state: REPLState) -> CommandResult:
    view = state.current_agent_view()
    base_model = view.model_name if view is not None else state.model_name
    temperature_display = "default" if state.temperature is None else f"{state.temperature}"
    lines = [
        f"  active_agent: {state.active_agent or 'none'}",
        f"  model: {state.model_name}",
        f"  base_model: {base_model}",
        f"  model_override_supported: {state.supports_model_override}",
        f"  temperature: {temperature_display}",
        f"  session_system_instructions: {len(state.session_system_instructions)}",
        f"  trace: {state.trace}",
    ]
    return CommandResult(message="\n".join(lines))


def _cmd_quit(_args: str, _state: REPLState) -> CommandResult:
    return CommandResult(should_exit=True)


_HANDLERS: dict[str, _CommandHandler] = {
    "help": _cmd_help,
    "model": _cmd_model,
    "agent": _cmd_agent,
    "clear": _cmd_clear,
    "history": _cmd_history,
    "trace": _cmd_trace,
    "tools": _cmd_tools,
    "system": _cmd_system,
    "usage": _cmd_usage,
    "config": _cmd_config,
    "quit": _cmd_quit,
    "exit": _cmd_quit,
}


__all__ = [
    "AgentView",
    "COMMANDS",
    "CommandDef",
    "CommandResult",
    "REPLState",
    "dispatch",
    "is_command",
    "parse_command",
    "resolve_command",
]

"""Tests for wazi slash command system."""

from __future__ import annotations

from wazi.commands import AgentView, REPLState, dispatch, is_command, parse_command, resolve_command
from wazi.session import ChatSession


def _agent_view(
    *,
    name: str = "main",
    model_name: str = "qwen3:4b",
    tool_names: tuple[str, ...] = (),
    base_system_instructions: tuple[str, ...] = (),
    temperature: float | None = None,
    supports_model_override: bool = True,
) -> AgentView:
    return AgentView(
        name=name,
        model_name=model_name,
        tool_names=tool_names,
        base_system_instructions=base_system_instructions,
        temperature=temperature,
        supports_model_override=supports_model_override,
    )


def _make_state(**kwargs: object) -> REPLState:
    defaults: dict[str, object] = {
        "trace": False,
        "session": ChatSession(),
        "agent_views": {
            "main": _agent_view(
                tool_names=("add", "search"),
                base_system_instructions=("Be concise.",),
                temperature=0.3,
                supports_model_override=True,
            )
        },
        "active_agent": "main",
    }
    defaults.update(kwargs)
    return REPLState(**defaults)  # pyright: ignore[reportArgumentType]


# is_command


def test_is_command_slash() -> None:
    assert is_command("/help") is True


def test_is_command_double_slash() -> None:
    assert is_command("//not a command") is False


def test_is_command_plain_text() -> None:
    assert is_command("hello world") is False


def test_is_command_empty() -> None:
    assert is_command("") is False


def test_is_command_bare_slash() -> None:
    assert is_command("/") is False


# parse_command


def test_parse_command_no_args() -> None:
    name, args = parse_command("/help")
    assert name == "help"
    assert args == ""


def test_parse_command_with_args() -> None:
    name, args = parse_command("/model llama3")
    assert name == "model"
    assert args == "llama3"


def test_parse_command_with_multi_word_args() -> None:
    name, args = parse_command("/system Be concise and helpful")
    assert name == "system"
    assert args == "Be concise and helpful"


def test_parse_command_case_insensitive() -> None:
    name, _ = parse_command("/HELP")
    assert name == "help"


# resolve_command


def test_resolve_exact() -> None:
    assert resolve_command("help") == "help"


def test_resolve_prefix_unique() -> None:
    assert resolve_command("he") == "help"


def test_resolve_prefix_ambiguous() -> None:
    assert resolve_command("c") is None


def test_resolve_nonexistent() -> None:
    assert resolve_command("foobar") is None


# dispatch: /help


def test_help_lists_commands() -> None:
    result = dispatch("help", "", _make_state())
    assert result.message is not None
    assert "/help" in result.message
    assert "/quit" in result.message
    assert "/model" in result.message


# dispatch: /model


def test_model_show_current() -> None:
    state = _make_state()
    result = dispatch("model", "", state)
    assert result.message is not None
    assert "Current model: qwen3:4b" in result.message
    assert "Override active: no" in result.message
    assert "Model override supported: True" in result.message


def test_model_switch_when_supported() -> None:
    state = _make_state()
    result = dispatch("model", "llama3", state)
    assert state.model_name == "llama3"
    assert result.message == "Switched to model: llama3"


def test_model_clear_resets_to_base_model() -> None:
    state = _make_state()
    dispatch("model", "llama3", state)
    result = dispatch("model", "clear", state)
    assert state.model_name == "qwen3:4b"
    assert result.message == "Model reset to base agent model: qwen3:4b"


def test_model_switch_rejected_when_agent_does_not_support_override() -> None:
    state = _make_state(
        agent_views={
            "main": _agent_view(supports_model_override=False),
        }
    )
    result = dispatch("model", "llama3", state)
    assert state.model_name == "qwen3:4b"
    assert result.message == "Current agent does not support truthful model overrides."


# dispatch: /agent


def test_agent_show_lists_active_and_available() -> None:
    state = _make_state(
        agent_views={
            "main": _agent_view(name="main"),
            "billing": _agent_view(name="billing", model_name="llama3"),
        }
    )
    result = dispatch("agent", "", state)
    assert result.message is not None
    assert "Active: main" in result.message
    assert "* main" in result.message
    assert "billing" in result.message


def test_agent_switch_updates_snapshot() -> None:
    state = _make_state(
        agent_views={
            "main": _agent_view(name="main", model_name="qwen3:4b"),
            "billing": _agent_view(
                name="billing",
                model_name="llama3",
                tool_names=("lookup_invoice",),
                base_system_instructions=("Route billing issues.",),
                temperature=0.7,
                supports_model_override=False,
            ),
        }
    )
    result = dispatch("agent", "billing", state)
    assert result.message == "Switched to agent: billing"
    assert state.active_agent == "billing"
    assert state.model_name == "llama3"
    assert state.tool_names == ("lookup_invoice",)
    assert state.base_system_instructions == ("Route billing issues.",)
    assert state.temperature == 0.7
    assert state.supports_model_override is False


# dispatch: /clear


def test_clear_resets_session() -> None:
    state = _make_state()
    state.session.record_turn()
    result = dispatch("clear", "", state)
    assert state.session.turn_count == 0
    assert result.message == "Conversation cleared."


# dispatch: /history


def test_history_shows_counts() -> None:
    state = _make_state()
    state.session.record_turn()
    state.session.record_turn()
    result = dispatch("history", "", state)
    assert result.message == "2 turns, 0 messages"


# dispatch: /trace


def test_trace_toggles() -> None:
    state = _make_state(trace=False)
    result = dispatch("trace", "", state)
    assert state.trace is True
    assert result.message == "Trace mode: on"

    result = dispatch("trace", "", state)
    assert state.trace is False
    assert result.message == "Trace mode: off"


# dispatch: /tools


def test_tools_empty() -> None:
    state = _make_state(agent_views={"main": _agent_view(tool_names=())})
    result = dispatch("tools", "", state)
    assert result.message == "No tools registered."


def test_tools_lists_names() -> None:
    state = _make_state()
    result = dispatch("tools", "", state)
    assert result.message is not None
    assert "add" in result.message
    assert "search" in result.message


# dispatch: /system


def test_system_show_empty() -> None:
    state = _make_state(agent_views={"main": _agent_view(base_system_instructions=())})
    result = dispatch("system", "", state)
    assert result.message == "No system instructions set."


def test_system_add_instruction() -> None:
    state = _make_state()
    result = dispatch("system", "Be concise", state)
    assert state.session_system_instructions == ["Be concise"]
    assert result.message == "Added session system instruction: Be concise"


def test_system_show_after_add() -> None:
    state = _make_state()
    dispatch("system", "Be concise", state)
    result = dispatch("system", "", state)
    assert result.message is not None
    assert "Agent system instructions:" in result.message
    assert "Be concise." in result.message
    assert "Session system instructions:" in result.message
    assert "Be concise" in result.message


def test_system_clear_clears_only_session_instructions() -> None:
    state = _make_state()
    dispatch("system", "Be concise", state)
    result = dispatch("system", "clear", state)
    assert result.message == "Cleared session system instructions."
    assert state.session_system_instructions == []
    assert state.base_system_instructions == ("Be concise.",)


# dispatch: /usage


def test_usage_shows_tokens() -> None:
    state = _make_state(total_input_tokens=100, total_output_tokens=50)
    result = dispatch("usage", "", state)
    assert result.message == "Session usage: 150 tokens (100 in, 50 out)"


# dispatch: /config


def test_config_shows_state() -> None:
    state = _make_state(trace=True)
    dispatch("system", "Be concise", state)
    result = dispatch("config", "", state)
    assert result.message is not None
    assert "active_agent: main" in result.message
    assert "model: qwen3:4b" in result.message
    assert "base_model: qwen3:4b" in result.message
    assert "model_override_supported: True" in result.message
    assert "temperature: 0.3" in result.message
    assert "session_system_instructions: 1" in result.message
    assert "trace: True" in result.message


# dispatch: /quit, /exit


def test_quit_exits() -> None:
    result = dispatch("quit", "", _make_state())
    assert result.should_exit is True


def test_exit_exits() -> None:
    result = dispatch("exit", "", _make_state())
    assert result.should_exit is True


# dispatch: unknown / prefix matching


def test_unknown_command() -> None:
    result = dispatch("foobar", "", _make_state())
    assert result.message == "Unknown command: /foobar. Type /help for available commands."


def test_prefix_dispatch_unique() -> None:
    result = dispatch("he", "", _make_state())
    assert result.message is not None
    assert "/help" in result.message


def test_prefix_dispatch_ambiguous() -> None:
    result = dispatch("c", "", _make_state())
    assert result.message is not None
    assert "ambiguous" in result.message.lower()

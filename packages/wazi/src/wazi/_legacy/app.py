"""App logic for wazi CLI: interactive chat, single-shot runs, and agent runs."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Literal

from pydantic import BaseModel
from wabizabi import Agent, Model, ModelSettings, text_output_config
from wabizabi.providers.ollama import OllamaChatModel, OllamaSettings

from wazi.app_registry import AgentApp
from wazi.commands import AgentView, REPLState, dispatch, is_command, parse_command
from wazi.config import (
    ConfigOverrides,
    ConfigScope,
    WaziConfig,
    load_config,
    load_scope_config,
    resolve_config,
    save_config_data,
)
from wazi.display import (
    StreamResult,
    print_command_result,
    print_error,
    print_info,
    print_session_end,
    print_welcome,
    render_stream,
)
from wazi.loader import load_agent_or_app, reference_name
from wazi.session import ChatSession


def _clone_ollama_model(model: OllamaChatModel, model_name: str) -> OllamaChatModel:
    return OllamaChatModel(
        model_name=model_name,
        host=model.host,
        chat_fn=model.chat_fn,
        stream_chat_fn=model.stream_chat_fn,
    )


def _apply_temperature(
    current: ModelSettings | None,
    temperature: float,
) -> OllamaSettings:
    """Apply a temperature override, merging with existing Ollama settings."""
    if current is None:
        return OllamaSettings(ollama_temperature=temperature)
    if not isinstance(current, OllamaSettings):
        raise ValueError(
            "Cannot apply --temperature: agent uses non-Ollama model_settings. "
            "Configure temperature on the agent directly."
        )
    return current.model_copy(update={"ollama_temperature": temperature})


def _agent_supports_model_override(agent: Agent[object, object]) -> bool:
    return isinstance(agent.model, OllamaChatModel)


def _agent_temperature(agent: Agent[object, object]) -> float | None:
    if isinstance(agent.model_settings, OllamaSettings):
        return agent.model_settings.ollama_temperature
    return None


def _agent_view(name: str, agent: Agent[object, object]) -> AgentView:
    return AgentView(
        name=name,
        model_name=_model_display_name(agent.model),
        tool_names=tuple(tool.name for tool in agent.toolset.tools),
        base_system_instructions=agent.system_instructions,
        temperature=_agent_temperature(agent),
        supports_model_override=_agent_supports_model_override(agent),
    )


def _apply_overrides(
    agent: Agent[object, object],
    overrides: ConfigOverrides,
) -> Agent[object, object]:
    """Apply CLI overrides to a user-defined agent truthfully."""
    updated = agent
    if overrides.model is not None:
        if not isinstance(updated.model, OllamaChatModel):
            raise ValueError(
                "Cannot apply --model: agent does not use OllamaChatModel. "
                "Configure the provider/model on the agent itself."
            )
        updated = updated.with_model(_clone_ollama_model(updated.model, overrides.model))
    if overrides.system is not None:
        updated = updated.with_system_instruction(overrides.system)
    if overrides.temperature is not None:
        if not isinstance(updated.model, OllamaChatModel):
            raise ValueError("Cannot apply --temperature: agent does not use OllamaChatModel.")
        updated = updated.with_model_settings(
            _apply_temperature(updated.model_settings, overrides.temperature)
        )
    return updated


def _apply_overrides_to_app(app: AgentApp, overrides: ConfigOverrides) -> AgentApp:
    return AgentApp(
        agents={name: _apply_overrides(agent, overrides) for name, agent in app.agents.items()},
        default=app.default,
    )


def _single_agent_app(agent: Agent[object, object], *, name: str = "main") -> AgentApp:
    return AgentApp(agents={name: agent}, default=name)


def _build_agent(config: WaziConfig) -> Agent[object, object]:
    settings = OllamaSettings(ollama_temperature=config.temperature)
    instructions: tuple[str, ...] = ()
    if config.system is not None:
        instructions = (config.system,)
    return Agent[object, object](
        model=OllamaChatModel(config.model),
        output=text_output_config(),
        system_instructions=instructions,
        model_settings=settings,
    )


def _format_connection_error(exc: Exception) -> str:
    msg = str(exc)
    if "Connection refused" in msg or "ConnectError" in msg:
        return "Cannot connect to Ollama. Is it running? (ollama serve)"
    return msg


def _model_display_name(model: Model) -> str:
    return model.profile.model_name


def _agent_views(app: AgentApp) -> dict[str, AgentView]:
    return {name: _agent_view(name, agent) for name, agent in app.agents.items()}


def _build_repl_state(app: AgentApp, *, trace: bool) -> REPLState:
    return REPLState(
        session=ChatSession(),
        trace=trace,
        agent_views=_agent_views(app),
        active_agent=app.default,
    )


def _print_structured_output_if_needed(result: object, rendered_text: bool) -> None:
    if rendered_text or result is None:
        return
    if isinstance(result, BaseModel):
        print(result.model_dump_json(indent=2))
        return
    if isinstance(result, dict | list | bool | int | float) or result is None:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(result)


async def _run_stream(
    agent: Agent[object, object],
    user_input: str,
    *,
    trace: bool = False,
    message_history: object | None = None,
    model: Model | None = None,
    instructions: tuple[str, ...] = (),
) -> StreamResult[object]:
    return await render_stream(
        agent.iter(
            user_input,
            deps=None,
            model=model,
            message_history=message_history,
            settings=agent.model_settings,
            instructions=instructions,
        ),
        trace=trace,
    )


async def _interactive_chat(
    app: AgentApp,
    repl_state: REPLState,
) -> None:
    while True:
        active_name = repl_state.active_agent or app.default
        prompt = f"[{active_name}]> " if len(app.agents) > 1 else "> "
        try:
            user_input = await asyncio.to_thread(input, prompt)
        except EOFError:
            break

        stripped = user_input.strip()
        if not stripped:
            continue

        if is_command(stripped):
            name, args = parse_command(stripped)
            result = dispatch(name, args, repl_state)
            if result.message:
                print_command_result(result.message)
            if result.should_exit:
                break
            continue

        active_agent = app.get(active_name)
        active_view = repl_state.current_agent_view()

        model_override: Model | None = None
        if (
            active_view is not None
            and repl_state.supports_model_override
            and repl_state.model_name != active_view.model_name
            and isinstance(active_agent.model, OllamaChatModel)
        ):
            model_override = _clone_ollama_model(active_agent.model, repl_state.model_name)

        try:
            result = await _run_stream(
                active_agent,
                stripped,
                trace=repl_state.trace,
                message_history=repl_state.session.history,
                model=model_override,
                instructions=repl_state.effective_system_instructions,
            )
            prior = len(repl_state.session.history.messages)
            repl_state.session.append(*result.messages[prior:])
            repl_state.session.record_turn()
            repl_state.total_input_tokens += result.usage.input_tokens
            repl_state.total_output_tokens += result.usage.output_tokens
            _print_structured_output_if_needed(result.output, result.rendered_text)

            if result.handoff_name is not None:
                if result.handoff_name in app.agents:
                    repl_state.switch_agent(result.handoff_name)
                    print_info(f"[switched to {result.handoff_name}]")
                else:
                    print_info(
                        f"[handoff to {result.handoff_name!r} has no registered agent — session ended]"
                    )
                    break
        except KeyboardInterrupt:
            print_info("[interrupted]")
            continue
        except Exception as exc:
            print_error(_format_connection_error(exc))
            continue


def run_chat(overrides: ConfigOverrides, *, trace: bool = False) -> int:
    config = overrides.apply(load_config())
    agent = _build_agent(config)
    app = _single_agent_app(agent)
    repl_state = _build_repl_state(app, trace=trace)

    print_welcome(repl_state.model_name, trace=trace)

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_interactive_chat(app, repl_state))

    print_session_end(repl_state.session.turn_count)
    return 0


def run_single(prompt: str, overrides: ConfigOverrides, *, trace: bool = False) -> int:
    config = overrides.apply(load_config())
    agent = _build_agent(config)

    try:
        result = asyncio.run(_run_stream(agent, prompt, trace=trace))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print_error(_format_connection_error(exc))
        return 1

    _print_structured_output_if_needed(result.output, result.rendered_text)
    return 0


def _load_app_with_overrides(ref: str, overrides: ConfigOverrides) -> AgentApp:
    target = load_agent_or_app(ref)
    if isinstance(target, AgentApp):
        return _apply_overrides_to_app(target, overrides)
    return _single_agent_app(_apply_overrides(target, overrides), name=reference_name(ref))


def run_agent_chat(ref: str, overrides: ConfigOverrides, *, trace: bool = False) -> int:
    try:
        app = _load_app_with_overrides(ref, overrides)
    except (ValueError, RuntimeError) as exc:
        print_error(str(exc))
        return 1

    repl_state = _build_repl_state(app, trace=trace)
    print_welcome(repl_state.model_name, trace=trace)

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_interactive_chat(app, repl_state))

    print_session_end(repl_state.session.turn_count)
    return 0


def run_agent(ref: str, user_input: str, overrides: ConfigOverrides, *, trace: bool = False) -> int:
    try:
        app = _load_app_with_overrides(ref, overrides)
    except (ValueError, RuntimeError) as exc:
        print_error(str(exc))
        return 1

    try:
        result = asyncio.run(_run_stream(app.get(app.default), user_input, trace=trace))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print_error(_format_connection_error(exc))
        return 1

    _print_structured_output_if_needed(result.output, result.rendered_text)
    if result.handoff_name is not None:
        print_info(f"[handoff to {result.handoff_name}]")
    return 0


def show_config(*, scope: ConfigScope | Literal["effective"] = "effective") -> int:
    if scope == "effective":
        resolved = resolve_config()
        print(f"model = {resolved.config.model!r}  # {resolved.sources['model']}")
        print(f"temperature = {resolved.config.temperature!r}  # {resolved.sources['temperature']}")
        print(f"system = {resolved.config.system!r}  # {resolved.sources['system']}")
        print(f"user_path = {str(resolved.user_path)!r}")
        print(
            f"project_path = {None if resolved.project_path is None else str(resolved.project_path)!r}"
        )
        return 0

    path, data = load_scope_config(scope)
    print(f"path = {str(path)!r}")
    if not data:
        print("<empty>")
        return 0
    for key, value in data.items():
        print(f"{key} = {value!r}")
    return 0


def _parse_config_value(key: str, value: str) -> tuple[bool, object | None]:
    lowered = value.strip().lower()
    if key == "temperature":
        if lowered in {"clear", "default", "none", "null"}:
            return True, None
        try:
            return False, float(value)
        except ValueError:
            raise ValueError(f"Invalid temperature value: {value!r}") from None
    if key in {"model", "system"} and lowered in {"clear", "default", "none", "null"}:
        return True, None
    return False, value


def set_config(key: str, value: str, *, scope: ConfigScope = "user") -> int:
    if key not in {"model", "temperature", "system"}:
        print_error(f"Unknown config key: {key!r}. Valid keys: model, temperature, system")
        return 1

    try:
        clear_value, parsed_value = _parse_config_value(key, value)
    except ValueError as exc:
        print_error(str(exc))
        return 1

    path, data = load_scope_config(scope)
    if clear_value:
        data.pop(key, None)
    else:
        data[key] = parsed_value
    saved_path = save_config_data(data, scope=scope)
    action = "Cleared" if clear_value else "Set"
    print_info(f"{action} {key} in {scope} config ({saved_path})")
    return 0

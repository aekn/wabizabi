"""Tests for wazi app logic."""

from __future__ import annotations

from pathlib import Path

import pytest
from wabizabi import Agent, Handoff, text_output_config
from wabizabi.messages import ModelResponse, ToolCallPart
from wabizabi.models import ModelResult
from wabizabi.providers.ollama import OllamaChatModel, OllamaSettings
from wabizabi.testing import ScriptedModel
from wabizabi.types import JsonValue
from wabizabi.usage import RunUsage
from wazi.app import (
    _apply_overrides,  # pyright: ignore[reportPrivateUsage]
    _apply_temperature,  # pyright: ignore[reportPrivateUsage]
    run_agent,
    set_config,
    show_config,
)
from wazi.app_registry import AgentApp
from wazi.config import ConfigOverrides, load_scope_config, save_config_data, user_config_path


@pytest.fixture
def config_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    project = tmp_path / "workspace" / "proj"
    project.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.chdir(project)
    return project


def test_show_config_effective_default(
    capsys: pytest.CaptureFixture[str], config_env: Path
) -> None:
    exit_code = show_config()
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "model = 'qwen3:4b'  # default" in output
    assert "temperature = None  # default" in output
    assert "system = None  # default" in output
    assert f"user_path = {str(user_config_path())!r}" in output
    assert "project_path = None" in output


def test_show_config_effective_with_project_and_user_sources(
    capsys: pytest.CaptureFixture[str],
    config_env: Path,
) -> None:
    save_config_data({"model": "user-model", "temperature": 0.2}, scope="user")
    save_config_data({"model": "project-model", "system": "Be terse."}, scope="project")

    exit_code = show_config()
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "model = 'project-model'  # project:" in output
    assert "temperature = 0.2  # user:" in output
    assert "system = 'Be terse.'  # project:" in output


def test_show_config_project_scope(capsys: pytest.CaptureFixture[str], config_env: Path) -> None:
    save_config_data({"model": "project-model"}, scope="project")
    exit_code = show_config(scope="project")
    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"path = {str(config_env / '.wazi' / 'config.json')!r}" in output
    assert "model = 'project-model'" in output


def test_set_config_model_user_scope(config_env: Path) -> None:
    exit_code = set_config("model", "llama3")
    assert exit_code == 0
    _, data = load_scope_config("user")
    assert data == {"model": "llama3"}


def test_set_config_temperature_project_scope(config_env: Path) -> None:
    exit_code = set_config("temperature", "0.7", scope="project")
    assert exit_code == 0
    _, data = load_scope_config("project")
    assert data == {"temperature": 0.7}


def test_set_config_clear_value_removes_key(config_env: Path) -> None:
    save_config_data({"temperature": 0.7}, scope="project")
    exit_code = set_config("temperature", "clear", scope="project")
    assert exit_code == 0
    _, data = load_scope_config("project")
    assert data == {}


def test_set_config_invalid_key(config_env: Path) -> None:
    exit_code = set_config("invalid_key", "value")
    assert exit_code == 1


def test_set_config_invalid_temperature(config_env: Path) -> None:
    exit_code = set_config("temperature", "not_a_number")
    assert exit_code == 1


def _basic_agent() -> Agent[object, object]:
    return Agent[object, object](
        model=OllamaChatModel("test-model"),
        output=text_output_config(),
    )


def test_apply_overrides_model_preserves_ollama_transport_fields() -> None:
    async def fake_chat(**kwargs: object) -> object:
        return kwargs

    async def fake_stream(**kwargs: object):
        if False:
            yield kwargs

    agent = Agent[object, object](
        model=OllamaChatModel(
            "test-model",
            host="http://localhost:11434",
            chat_fn=fake_chat,
            stream_chat_fn=fake_stream,
        ),
        output=text_output_config(),
    )
    overrides = ConfigOverrides(model="llama3")
    updated = _apply_overrides(agent, overrides)
    assert isinstance(updated.model, OllamaChatModel)
    assert updated.model.profile.model_name == "llama3"
    assert updated.model.host == "http://localhost:11434"
    assert updated.model.chat_fn is fake_chat
    assert updated.model.stream_chat_fn is fake_stream


def test_apply_overrides_temperature_builds_ollama_settings() -> None:
    agent = _basic_agent()
    overrides = ConfigOverrides(temperature=0.3)
    updated = _apply_overrides(agent, overrides)
    assert isinstance(updated.model_settings, OllamaSettings)
    assert updated.model_settings.ollama_temperature == 0.3


def test_apply_overrides_temperature_merges_existing() -> None:
    agent = _basic_agent().with_model_settings(
        OllamaSettings(ollama_temperature=0.9, ollama_think=False)
    )
    overrides = ConfigOverrides(temperature=0.1)
    updated = _apply_overrides(agent, overrides)
    assert isinstance(updated.model_settings, OllamaSettings)
    assert updated.model_settings.ollama_temperature == 0.1
    assert updated.model_settings.ollama_think is False


def test_apply_overrides_system_instruction() -> None:
    agent = _basic_agent()
    overrides = ConfigOverrides(system="be concise")
    updated = _apply_overrides(agent, overrides)
    assert "be concise" in updated.system_instructions


class _FakeSettings:
    pass


def test_apply_temperature_rejects_foreign_settings() -> None:
    with pytest.raises(ValueError, match="non-Ollama"):
        _apply_temperature(_FakeSettings(), 0.5)  # pyright: ignore[reportArgumentType]


def test_run_agent_prints_terminal_handoff_notice(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _tool_call_result(
        tool_name: str, call_id: str, arguments: dict[str, JsonValue]
    ) -> ModelResult:
        return ModelResult(
            response=ModelResponse(
                parts=(ToolCallPart(tool_name=tool_name, call_id=call_id, arguments=arguments),),
                model_name="scripted",
            ),
            usage=RunUsage.zero(),
        )

    scripted = Agent[object, str](
        model=ScriptedModel(
            (_tool_call_result("handoff_billing", "call-1", {"input": "transfer"}),)
        ),
        output=text_output_config(),
    ).with_handoff(Handoff(name="billing"))
    app = AgentApp(agents={"main": scripted}, default="main")

    monkeypatch.setattr("wazi.app._load_app_with_overrides", lambda ref, overrides: app)

    exit_code = run_agent("fake.module:agent", "hello", ConfigOverrides())
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "[handoff to billing]" in captured.err


def test_agent_app_rejects_unknown_default() -> None:
    agent = _basic_agent()
    with pytest.raises(ValueError, match="Default agent"):
        AgentApp(agents={"main": agent}, default="other")


def test_agent_app_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        AgentApp(agents={}, default="main")


def test_agent_app_get_and_names() -> None:
    a1 = _basic_agent()
    a2 = _basic_agent()
    app = AgentApp(agents={"first": a1, "second": a2}, default="first")
    assert app.get("first") is a1
    assert app.get("second") is a2
    assert app.names() == ("first", "second")

"""Tests for wazi config layering and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from wazi.config import (
    ConfigOverrides,
    WaziConfig,
    find_project_config_path,
    load_config,
    load_scope_config,
    resolve_config,
    save_config,
    save_config_data,
    user_config_path,
)


def _set_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))


def test_default_config_values() -> None:
    config = WaziConfig()
    assert config.model == "qwen3:4b"
    assert config.temperature is None
    assert config.system is None


def test_to_dict_minimal() -> None:
    config = WaziConfig()
    assert config.to_dict() == {"model": "qwen3:4b"}


def test_to_dict_full() -> None:
    config = WaziConfig(model="llama3", temperature=0.5, system="You are helpful.")
    assert config.to_dict() == {
        "model": "llama3",
        "temperature": 0.5,
        "system": "You are helpful.",
    }


def test_from_dict_full() -> None:
    data: dict[str, object] = {
        "model": "llama3",
        "temperature": 0.7,
        "system": "Be concise.",
    }
    config = WaziConfig.from_dict(data)
    assert config.model == "llama3"
    assert config.temperature == 0.7
    assert config.system == "Be concise."


def test_from_dict_empty() -> None:
    config = WaziConfig.from_dict({})
    assert config.model == "qwen3:4b"
    assert config.temperature is None
    assert config.system is None


def test_from_dict_ignores_invalid_types() -> None:
    data: dict[str, object] = {"model": 123, "temperature": "not a number", "system": 7}
    config = WaziConfig.from_dict(data)
    assert config == WaziConfig()


def test_roundtrip_user_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)

    original = WaziConfig(model="llama3", temperature=0.3, system="Be terse.")
    save_config(original, scope="user")

    loaded = load_config(cwd=tmp_path)
    assert loaded == original
    assert user_config_path() == home / ".config" / "wazi" / "config.json"


def test_user_config_path_prefers_xdg_config_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    xdg = tmp_path / "xdg"
    _set_home(monkeypatch, home)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

    assert user_config_path() == xdg / "wazi" / "config.json"


def test_save_project_scope_creates_project_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace" / "proj"
    workspace.mkdir(parents=True)
    _set_home(monkeypatch, home)

    save_config_data({"model": "mistral", "temperature": 0.2}, scope="project", cwd=workspace)
    path, data = load_scope_config("project", cwd=workspace)

    assert path == workspace / ".wazi" / "config.json"
    assert data == {"model": "mistral", "temperature": 0.2}


def test_resolve_config_prefers_project_over_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "workspace" / "proj"
    nested = project / "src"
    nested.mkdir(parents=True)
    _set_home(monkeypatch, home)

    save_config_data(
        {"model": "user-model", "temperature": 0.1, "system": "user system"},
        scope="user",
    )
    save_config_data(
        {"model": "project-model", "system": "project system"},
        scope="project",
        cwd=project,
    )

    resolved = resolve_config(cwd=nested)
    assert resolved.config.model == "project-model"
    assert resolved.config.temperature == 0.1
    assert resolved.config.system == "project system"
    assert resolved.sources["model"].startswith("project:")
    assert resolved.sources["temperature"].startswith("user:")
    assert resolved.sources["system"].startswith("project:")
    assert resolved.project_path == project / ".wazi" / "config.json"


def test_find_project_config_path_searches_upwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "workspace" / "proj"
    nested = project / "a" / "b"
    nested.mkdir(parents=True)
    _set_home(monkeypatch, home)

    save_config_data({"model": "project-model"}, scope="project", cwd=project)
    assert find_project_config_path(cwd=nested) == project / ".wazi" / "config.json"


def test_load_config_missing_files_uses_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _set_home(monkeypatch, home)

    assert load_config(cwd=workspace) == WaziConfig()


def test_load_config_invalid_json_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _set_home(monkeypatch, home)

    path = user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json")

    assert load_config(cwd=workspace) == WaziConfig()


def test_load_config_non_dict_json_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _set_home(monkeypatch, home)

    path = user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([1, 2, 3]))

    assert load_config(cwd=workspace) == WaziConfig()


def test_overrides_apply() -> None:
    base = WaziConfig(model="base_model", temperature=0.5)
    overrides = ConfigOverrides(model="override_model")
    result = overrides.apply(base)
    assert result.model == "override_model"
    assert result.temperature == 0.5


def test_overrides_none_keeps_base() -> None:
    base = WaziConfig(model="base", temperature=0.5)
    overrides = ConfigOverrides()
    assert overrides.apply(base) == base

"""Config file management for wazi CLI."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from platformdirs import user_config_path as platformdirs_user_config_path

DEFAULT_MODEL = "qwen3:4b"
PROJECT_CONFIG_DIRNAME = ".wazi"
CONFIG_FILENAME = "config.json"
_XDG_PLATFORMS = ("linux", "freebsd", "openbsd", "netbsd", "dragonfly")

type ConfigScope = Literal["user", "project"]


@dataclass(frozen=True, slots=True)
class WaziConfig:
    """Effective configuration for wazi CLI defaults."""

    model: str = DEFAULT_MODEL
    temperature: float | None = None
    system: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"model": self.model}
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.system is not None:
            result["system"] = self.system
        return result

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> WaziConfig:
        fragment = _normalize_config_fragment(data)
        model = fragment.get("model")
        temperature = fragment.get("temperature")
        system = fragment.get("system")
        return cls(
            model=model if isinstance(model, str) else DEFAULT_MODEL,
            temperature=temperature if isinstance(temperature, float) else None,
            system=system if isinstance(system, str) else None,
        )


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """Effective configuration plus path and source metadata."""

    config: WaziConfig
    sources: dict[str, str]
    user_path: Path
    project_path: Path | None


def _normalize_config_fragment(raw: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    model = raw.get("model")
    if isinstance(model, str):
        result["model"] = model
    temperature = raw.get("temperature")
    if isinstance(temperature, int | float):
        result["temperature"] = float(temperature)
    system = raw.get("system")
    if isinstance(system, str):
        result["system"] = system
    return result


def user_config_path() -> Path:
    """Return the user-scoped config path."""
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home) / "wazi" / CONFIG_FILENAME
    if sys.platform.startswith(_XDG_PLATFORMS):
        return Path.home() / ".config" / "wazi" / CONFIG_FILENAME
    return Path(platformdirs_user_config_path("wazi")) / CONFIG_FILENAME


def default_project_config_path(*, cwd: Path | None = None) -> Path:
    """Return the default project-scoped config path for the current directory."""
    base = Path.cwd() if cwd is None else cwd
    return base / PROJECT_CONFIG_DIRNAME / CONFIG_FILENAME


def find_project_config_path(*, cwd: Path | None = None) -> Path | None:
    """Find the nearest project-scoped config file by searching upward."""
    base = Path.cwd() if cwd is None else cwd
    current = base.resolve()
    for directory in (current, *current.parents):
        candidate = directory / PROJECT_CONFIG_DIRNAME / CONFIG_FILENAME
        if candidate.exists():
            return candidate
    return None


def load_config_data(path: Path) -> dict[str, object] | None:
    """Load one config file into a validated raw fragment."""
    if not path.exists():
        return None
    try:
        raw: object = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    string_dict: dict[str, object] = raw  # pyright: ignore[reportUnknownVariableType]
    return _normalize_config_fragment(string_dict)


def _raw_config_for_scope(
    scope: ConfigScope, *, cwd: Path | None = None
) -> tuple[Path, dict[str, object]]:
    path = (
        user_config_path()
        if scope == "user"
        else find_project_config_path(cwd=cwd) or default_project_config_path(cwd=cwd)
    )
    data = load_config_data(path)
    return path, {} if data is None else data


def resolve_config(*, cwd: Path | None = None) -> ResolvedConfig:
    """Resolve effective config with project-over-user precedence."""
    user_path = user_config_path()
    project_path = find_project_config_path(cwd=cwd)
    sources = {
        "model": "default",
        "temperature": "default",
        "system": "default",
    }
    data: dict[str, object] = {}

    user_data = load_config_data(user_path)
    if user_data is not None:
        for key, value in user_data.items():
            data[key] = value
            sources[key] = f"user:{user_path}"

    project_data = load_config_data(project_path) if project_path is not None else None
    if project_data is not None:
        for key, value in project_data.items():
            data[key] = value
            sources[key] = f"project:{project_path}"

    return ResolvedConfig(
        config=WaziConfig.from_dict(data),
        sources=sources,
        user_path=user_path,
        project_path=project_path,
    )


def load_config(*, cwd: Path | None = None) -> WaziConfig:
    """Load the effective config."""
    return resolve_config(cwd=cwd).config


def load_scope_config(
    scope: ConfigScope, *, cwd: Path | None = None
) -> tuple[Path, dict[str, object]]:
    """Load one scope's raw config fragment."""
    return _raw_config_for_scope(scope, cwd=cwd)


def save_config_data(
    data: dict[str, object],
    *,
    scope: ConfigScope = "user",
    cwd: Path | None = None,
) -> Path:
    """Persist a validated raw config fragment to disk."""
    path = (
        user_config_path()
        if scope == "user"
        else find_project_config_path(cwd=cwd) or default_project_config_path(cwd=cwd)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_config_fragment(data)
    path.write_text(json.dumps(normalized, indent=2) + "\n")
    return path


def save_config(
    config: WaziConfig,
    *,
    scope: ConfigScope = "user",
    cwd: Path | None = None,
) -> Path:
    """Persist an effective config object to disk."""
    return save_config_data(config.to_dict(), scope=scope, cwd=cwd)


@dataclass(slots=True)
class ConfigOverrides:
    """CLI flag overrides that take precedence over saved config."""

    model: str | None = field(default=None)
    temperature: float | None = field(default=None)
    system: str | None = field(default=None)

    def apply(self, base: WaziConfig) -> WaziConfig:
        return WaziConfig(
            model=self.model if self.model is not None else base.model,
            temperature=self.temperature if self.temperature is not None else base.temperature,
            system=self.system if self.system is not None else base.system,
        )

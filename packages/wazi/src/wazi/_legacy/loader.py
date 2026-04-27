"""Load agent or AgentApp references from ``module.path:attribute`` strings."""

from __future__ import annotations

import importlib
import os
import sys

from wabizabi import Agent

from wazi.app_registry import AgentApp


def _ensure_cwd_importable() -> None:
    """Ensure the current working directory is on sys.path.

    This lets users reference local modules like ``examples.foo:agent``
    without installing them as packages.
    """
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def _resolve_attribute(ref: str) -> object:
    if ":" not in ref:
        raise ValueError(f"Invalid reference {ref!r}. Expected format: 'module.path:attribute'.")

    module_path, attr_name = ref.rsplit(":", 1)

    _ensure_cwd_importable()

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Could not import module {module_path!r}: {exc}") from exc

    attr = getattr(module, attr_name, None)
    if attr is None:
        raise RuntimeError(f"Module {module_path!r} has no attribute {attr_name!r}.")
    return attr


def reference_name(ref: str) -> str:
    """Return the attribute name component from a module reference."""
    if ":" not in ref:
        raise ValueError(f"Invalid reference {ref!r}. Expected format: 'module.path:attribute'.")
    return ref.rsplit(":", 1)[1]


def load_agent(ref: str) -> Agent[object, object]:
    """Resolve a reference to a single ``Agent`` instance.

    Raises ``RuntimeError`` if the target is not an Agent.
    """
    target = _resolve_attribute(ref)
    if not isinstance(target, Agent):
        raise RuntimeError(f"{ref!r} resolved to {type(target).__name__}, not an Agent.")
    loaded: Agent[object, object] = target  # pyright: ignore[reportUnknownVariableType]
    return loaded


def load_agent_or_app(ref: str) -> Agent[object, object] | AgentApp:
    """Resolve a reference that may be either an ``Agent`` or an ``AgentApp``.

    Raises ``RuntimeError`` if the target is neither.
    """
    target = _resolve_attribute(ref)
    if isinstance(target, AgentApp):
        return target
    if isinstance(target, Agent):
        loaded: Agent[object, object] = target  # pyright: ignore[reportUnknownVariableType]
        return loaded
    raise RuntimeError(f"{ref!r} resolved to {type(target).__name__}, not an Agent or AgentApp.")


__all__ = ["load_agent", "load_agent_or_app", "reference_name"]

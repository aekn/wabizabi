"""Tests for wazi agent loader."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest
from wabizabi import Agent, text_output_config
from wabizabi.providers.ollama import OllamaChatModel
from wazi.app_registry import AgentApp
from wazi.loader import load_agent, load_agent_or_app


def test_load_agent_invalid_format() -> None:
    with pytest.raises(ValueError, match="Expected format"):
        load_agent("no_colon_here")


def test_load_agent_missing_module() -> None:
    with pytest.raises(RuntimeError, match="Could not import"):
        load_agent("nonexistent.module.path:agent")


def test_load_agent_missing_attribute() -> None:
    with pytest.raises(RuntimeError, match="has no attribute"):
        load_agent("wazi:nonexistent_agent")


def test_load_agent_wrong_type() -> None:
    with pytest.raises(RuntimeError, match="not an Agent"):
        load_agent("wazi:__version__")


@pytest.fixture
def fake_module() -> Iterator[str]:
    name = "_wazi_loader_fake"
    module = types.ModuleType(name)
    agent = Agent[object, object](
        model=OllamaChatModel("test-model"),
        output=text_output_config(),
    )
    module.agent = agent  # pyright: ignore[reportAttributeAccessIssue]
    module.app = AgentApp(agents={"main": agent}, default="main")  # pyright: ignore[reportAttributeAccessIssue]
    module.not_an_agent = 42  # pyright: ignore[reportAttributeAccessIssue]
    sys.modules[name] = module
    yield name
    del sys.modules[name]


def test_load_agent_or_app_returns_agent(fake_module: str) -> None:
    result = load_agent_or_app(f"{fake_module}:agent")
    assert isinstance(result, Agent)


def test_load_agent_or_app_returns_app(fake_module: str) -> None:
    result = load_agent_or_app(f"{fake_module}:app")
    assert isinstance(result, AgentApp)
    assert result.names() == ("main",)


def test_load_agent_or_app_rejects_other(fake_module: str) -> None:
    with pytest.raises(RuntimeError, match="not an Agent or AgentApp"):
        load_agent_or_app(f"{fake_module}:not_an_agent")

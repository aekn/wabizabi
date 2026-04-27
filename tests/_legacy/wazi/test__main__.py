"""Tests for wazi CLI entry point."""

from __future__ import annotations

import pytest
from wazi.__main__ import build_parser, main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        main(["--version"])
    captured = capsys.readouterr()
    assert "wazi 0.1.0" in captured.out


def test_parser_default_command() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.command is None


def test_parser_chat_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["chat"])
    assert args.command == "chat"


def test_parser_run_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "hello world"])
    assert args.command == "run"
    assert args.prompt == "hello world"


def test_parser_trace_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["--trace", "run", "hello"])
    assert args.trace is True


def test_parser_trace_default_off() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "hello"])
    assert args.trace is False


def test_parser_global_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["--model", "llama3", "--temperature", "0.5", "--system", "Be brief"])
    assert args.model == "llama3"
    assert args.temperature == 0.5
    assert args.system == "Be brief"


def test_parser_agent_run_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["agent", "run", "mymod:agent", "--input", "hello"])
    assert args.command == "agent"
    assert args.agent_command == "run"
    assert args.ref == "mymod:agent"
    assert args.input == "hello"


def test_parser_agent_chat_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["agent", "chat", "mymod:agent"])
    assert args.command == "agent"
    assert args.agent_command == "chat"
    assert args.ref == "mymod:agent"


def test_parser_config_show_defaults_to_effective_scope() -> None:
    parser = build_parser()
    args = parser.parse_args(["config", "show"])
    assert args.command == "config"
    assert args.config_command == "show"
    assert args.user is False
    assert args.project is False


def test_parser_config_show_project_scope() -> None:
    parser = build_parser()
    args = parser.parse_args(["config", "show", "--project"])
    assert args.project is True
    assert args.user is False


def test_parser_config_set_user_scope() -> None:
    parser = build_parser()
    args = parser.parse_args(["config", "set", "model", "llama3", "--user"])
    assert args.command == "config"
    assert args.config_command == "set"
    assert args.key == "model"
    assert args.value == "llama3"
    assert args.user is True
    assert args.project is False

"""Entry point for the wazi CLI."""

from __future__ import annotations

import argparse

from wazi._version import __version__
from wazi.config import ConfigOverrides


def _add_scope_flags(parser: argparse.ArgumentParser) -> None:
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--user", action="store_true", default=False, help="use user config scope")
    scope.add_argument(
        "--project",
        action="store_true",
        default=False,
        help="use project config scope",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the wazi CLI."""
    parser = argparse.ArgumentParser(
        prog="wazi",
        description="CLI for Wabizabi agent orchestration.",
    )
    parser.add_argument("--version", action="version", version=f"wazi {__version__}")
    parser.add_argument("--model", type=str, default=None, help="model name override")
    parser.add_argument("--temperature", type=float, default=None, help="sampling temperature")
    parser.add_argument("--system", type=str, default=None, help="system instruction")
    parser.add_argument("--trace", action="store_true", default=False, help="enable trace mode")

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("chat", help="interactive chat session (default)")

    run_p = sub.add_parser("run", help="run a single prompt")
    run_p.add_argument("prompt", type=str, help="the prompt to send")

    agent_p = sub.add_parser("agent", help="run a user-defined agent")
    agent_sub = agent_p.add_subparsers(dest="agent_command")
    agent_run = agent_sub.add_parser("run", help="run an agent from a module reference")
    agent_run.add_argument("ref", type=str, help="module.path:agent_name")
    agent_run.add_argument("--input", type=str, required=True, help="input prompt")
    agent_chat = agent_sub.add_parser("chat", help="interactive chat with a user-defined agent")
    agent_chat.add_argument("ref", type=str, help="module.path:agent_name")

    config_p = sub.add_parser("config", help="manage configuration")
    config_sub = config_p.add_subparsers(dest="config_command")
    config_show = config_sub.add_parser("show", help="show current config")
    _add_scope_flags(config_show)
    config_set = config_sub.add_parser("set", help="set a config value")
    _add_scope_flags(config_set)
    config_set.add_argument("key", type=str, help="config key")
    config_set.add_argument("value", type=str, help="config value")

    return parser


def _show_scope(args: argparse.Namespace) -> str:
    if args.project:
        return "project"
    if args.user:
        return "user"
    return "effective"


def _set_scope(args: argparse.Namespace) -> str:
    if args.project:
        return "project"
    return "user"


def main(argv: list[str] | None = None) -> int:
    """Run the wazi CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    overrides = ConfigOverrides(
        model=args.model,
        temperature=args.temperature,
        system=args.system,
    )
    trace: bool = args.trace

    from wazi.app import (
        run_agent,
        run_agent_chat,
        run_chat,
        run_single,
        set_config,
        show_config,
    )

    if args.command is None or args.command == "chat":
        return run_chat(overrides, trace=trace)

    if args.command == "run":
        return run_single(args.prompt, overrides, trace=trace)

    if args.command == "agent":
        if args.agent_command == "run":
            return run_agent(args.ref, args.input, overrides, trace=trace)
        if args.agent_command == "chat":
            return run_agent_chat(args.ref, overrides, trace=trace)
        parser.parse_args(["agent", "--help"])
        return 1

    if args.command == "config":
        if args.config_command == "show":
            return show_config(scope=_show_scope(args))
        if args.config_command == "set":
            return set_config(args.key, args.value, scope=_set_scope(args))
        parser.parse_args(["config", "--help"])
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

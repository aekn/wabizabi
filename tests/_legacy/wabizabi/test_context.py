from __future__ import annotations

import pytest
from wabizabi.context import RunContext
from wabizabi.usage import RunUsage


def test_with_run_step_returns_updated_copy() -> None:
    context = RunContext(
        deps=("service", "demo"),
        run_id="run-1",
        run_step=0,
        usage=RunUsage(input_tokens=3, output_tokens=5),
        metadata={"request_id": "req-1"},
    )

    updated = context.with_run_step(1)

    assert updated is not context
    assert updated.deps == ("service", "demo")
    assert updated.run_id == "run-1"
    assert updated.run_step == 1
    assert updated.usage == RunUsage(input_tokens=3, output_tokens=5)
    assert updated.metadata == {"request_id": "req-1"}
    assert updated.tool_name is None
    assert updated.tool_call_id is None


def test_run_context_rejects_empty_run_id() -> None:
    with pytest.raises(ValueError, match="run_id must not be empty"):
        RunContext(deps=None, run_id="")


def test_run_context_supports_tool_metadata() -> None:
    context = RunContext(
        deps=("service", "demo"),
        run_id="run-1",
        tool_name="weather",
        tool_call_id="call-1",
    )

    assert context.tool_name == "weather"
    assert context.tool_call_id == "call-1"


def test_run_context_requires_tool_fields_together() -> None:
    with pytest.raises(ValueError, match="tool_name and tool_call_id must either both be set"):
        RunContext(
            deps=("service", "demo"),
            run_id="run-1",
            tool_name="weather",
        )

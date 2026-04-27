from __future__ import annotations

import pytest
from wabizabi.messages import ModelRequest, UserPromptPart
from wabizabi.state import RunState
from wabizabi.usage import RunUsage


def test_create_run_state_has_sensible_defaults() -> None:
    state = RunState.create("run-1", metadata={"request_id": "req-1"})

    assert state.run_id == "run-1"
    assert state.message_history.messages == ()
    assert state.usage == RunUsage.zero()
    assert state.retries == 0
    assert state.run_step == 0
    assert state.metadata == {"request_id": "req-1"}


def test_run_state_transitions_return_updated_copies() -> None:
    request = ModelRequest(parts=(UserPromptPart(text="Hello"),))
    original = RunState.create("run-1")

    updated = (
        original.with_message(request)
        .add_usage(RunUsage(input_tokens=3, output_tokens=5))
        .increment_retry()
        .increment_run_step()
    )

    assert original.message_history.messages == ()
    assert original.usage == RunUsage.zero()
    assert original.retries == 0
    assert original.run_step == 0

    assert updated.message_history.messages == (request,)
    assert updated.usage == RunUsage(input_tokens=3, output_tokens=5)
    assert updated.retries == 1
    assert updated.run_step == 1


def test_context_for_reflects_state_fields() -> None:
    state = (
        RunState.create("run-1", metadata={"workflow": "demo"})
        .add_usage(RunUsage(input_tokens=2, output_tokens=4))
        .increment_run_step()
    )

    context = state.context_for(("svc", "cfg"), tool_name="search", tool_call_id="call-1")

    assert context.deps == ("svc", "cfg")
    assert context.run_id == "run-1"
    assert context.run_step == 1
    assert context.usage == RunUsage(input_tokens=2, output_tokens=4)
    assert context.metadata == {"workflow": "demo"}
    assert context.tool_name == "search"
    assert context.tool_call_id == "call-1"


def test_run_state_rejects_negative_retry_count() -> None:
    with pytest.raises(ValueError, match="retries must be non-negative"):
        RunState(run_id="run-1", retries=-1)


def test_run_state_with_messages_appends_multiple_messages() -> None:
    request = ModelRequest(parts=(UserPromptPart(text="Hello"),))
    follow_up = ModelRequest(parts=(UserPromptPart(text="Follow up"),))

    state = RunState.create("run-1").with_messages((request, follow_up))

    assert state.message_history.messages == (request, follow_up)

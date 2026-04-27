from __future__ import annotations

import pytest
from wabizabi.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from wabizabi.run import RunResult
from wabizabi.state import RunState
from wabizabi.usage import RunUsage


def test_run_result_exposes_state_derived_views() -> None:
    request = ModelRequest(parts=(UserPromptPart(text="Hello"),))
    response = ModelResponse(parts=(TextPart(text="Hi"),), model_name="demo")
    state = (
        RunState.create("run-1")
        .with_messages((request, response))
        .add_usage(RunUsage(input_tokens=3, output_tokens=2))
    )

    result = RunResult[str](
        output="Hi",
        state=state,
        new_messages=(request, response),
    )

    assert result.output == "Hi"
    assert result.run_id == "run-1"
    assert result.all_messages == (request, response)
    assert result.requests == (request,)
    assert result.responses == (response,)
    assert result.usage == RunUsage(input_tokens=3, output_tokens=2)


def test_run_result_rejects_non_suffix_new_messages() -> None:
    request = ModelRequest(parts=(UserPromptPart(text="Hello"),))
    response = ModelResponse(parts=(TextPart(text="Hi"),), model_name="demo")
    other = ModelResponse(parts=(TextPart(text="Other"),), model_name="demo")
    state = RunState.create("run-1").with_messages((request, response))

    with pytest.raises(
        ValueError,
        match="new_messages must match the suffix of state.message_history",
    ):
        RunResult[str](
            output="Hi",
            state=state,
            new_messages=(request, other),
        )

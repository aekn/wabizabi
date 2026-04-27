from __future__ import annotations

from wabizabi.history import MessageHistory
from wabizabi.messages import (
    ModelRequest,
    ModelResponse,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)


def test_message_history_append_and_filters() -> None:
    request = ModelRequest(parts=(UserPromptPart(text="Hello"),))
    response = ModelResponse(parts=(TextPart(text="Hi"),))

    history = MessageHistory.empty().append(request).append(response)

    assert len(history) == 2
    assert history.messages == (request, response)
    assert history.requests == (request,)
    assert history.responses == (response,)


def test_message_history_extend_returns_new_history() -> None:
    request = ModelRequest(parts=(UserPromptPart(text="Question"),))
    response = ModelResponse(parts=(TextPart(text="Answer"),))

    history = MessageHistory.empty()
    updated = history.extend((request, response))

    assert history.messages == ()
    assert updated.messages == (request, response)


def test_message_history_json_round_trip() -> None:
    request = ModelRequest(parts=(UserPromptPart(text="Hello"),))
    response = ModelResponse(parts=(TextPart(text="Hi"), ReasoningPart(text="I thought")))

    history = MessageHistory(messages=(request, response))
    serialized = history.to_json()
    restored = MessageHistory.from_json(serialized)

    assert restored == history
    assert restored.messages == history.messages


def test_message_history_json_round_trip_with_tool_calls() -> None:
    request = ModelRequest(parts=(UserPromptPart(text="Use a tool"),))
    response = ModelResponse(
        parts=(ToolCallPart(tool_name="add", call_id="c1", arguments={"a": 1, "b": 2}),)
    )
    tool_return = ModelRequest(parts=(ToolReturnPart(tool_name="add", call_id="c1", content=3),))
    final = ModelResponse(parts=(TextPart(text="The answer is 3"),))

    history = MessageHistory(messages=(request, response, tool_return, final))
    serialized = history.to_json()
    restored = MessageHistory.from_json(serialized)

    assert restored == history


def test_message_history_list_round_trip() -> None:
    request = ModelRequest(parts=(UserPromptPart(text="Hello"),))
    response = ModelResponse(parts=(TextPart(text="Hi"),))

    history = MessageHistory(messages=(request, response))
    as_list = history.to_list()
    assert isinstance(as_list, list)
    restored = MessageHistory.from_list(as_list)

    assert restored == history


def test_message_history_from_json_bytes() -> None:
    request = ModelRequest(parts=(UserPromptPart(text="Hello"),))
    history = MessageHistory(messages=(request,))
    serialized = history.to_json().encode()
    restored = MessageHistory.from_json(serialized)

    assert restored == history


def test_message_history_empty_round_trip() -> None:
    history = MessageHistory.empty()
    serialized = history.to_json()
    restored = MessageHistory.from_json(serialized)

    assert restored == history
    assert len(restored) == 0


def test_message_history_to_json_indent() -> None:
    request = ModelRequest(parts=(UserPromptPart(text="Hello"),))
    history = MessageHistory(messages=(request,))
    serialized = history.to_json(indent=2)

    assert "\n" in serialized
    as_list = history.to_list()
    assert len(as_list) == 1

"""Tests for wazi session tracking."""

from __future__ import annotations

from wabizabi import MessageHistory
from wabizabi.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from wazi.session import ChatSession


def test_initial_state() -> None:
    session = ChatSession()
    assert session.turn_count == 0
    assert session.history == MessageHistory(messages=())


def test_record_turn_increments_count() -> None:
    session = ChatSession()
    session.record_turn()
    session.record_turn()
    assert session.turn_count == 2


def test_append_and_history() -> None:
    session = ChatSession()
    req = ModelRequest(parts=(UserPromptPart(text="hello"),))
    resp = ModelResponse(parts=(TextPart(text="hi"),), model_name="test")
    session.append(req, resp)
    assert session.history.messages == (req, resp)


def test_clear_resets_everything() -> None:
    session = ChatSession()
    req = ModelRequest(parts=(UserPromptPart(text="hello"),))
    session.append(req)
    session.record_turn()
    session.clear()
    assert session.turn_count == 0
    assert session.history.messages == ()

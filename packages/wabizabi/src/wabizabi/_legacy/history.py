"""Conversation history models and helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import RootModel

from wabizabi.messages import ModelMessage, ModelRequest, ModelResponse

type _JsonStr = str


class _MessageListModel(RootModel[tuple[ModelMessage, ...]]):
    """Typed validation wrapper for serializing message lists."""


@dataclass(frozen=True, slots=True)
class MessageHistory:
    """An immutable wrapper around canonical message history."""

    messages: tuple[ModelMessage, ...] = ()

    @classmethod
    def empty(cls) -> MessageHistory:
        return cls()

    def append(self, message: ModelMessage) -> MessageHistory:
        return MessageHistory(messages=(*self.messages, message))

    def extend(self, messages: Iterable[ModelMessage]) -> MessageHistory:
        return MessageHistory(messages=(*self.messages, *tuple(messages)))

    @property
    def requests(self) -> tuple[ModelRequest, ...]:
        result: list[ModelRequest] = []
        for message in self.messages:
            if isinstance(message, ModelRequest):
                result.append(message)
        return tuple(result)

    @property
    def responses(self) -> tuple[ModelResponse, ...]:
        result: list[ModelResponse] = []
        for message in self.messages:
            if isinstance(message, ModelResponse):
                result.append(message)
        return tuple(result)

    def to_json(self, *, indent: int | None = None) -> _JsonStr:
        """Serialize this history to a JSON string."""
        wrapper = _MessageListModel(self.messages)
        return wrapper.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str | bytes) -> MessageHistory:
        """Deserialize a history from a JSON string or bytes."""
        wrapper = _MessageListModel.model_validate_json(data)
        return cls(messages=wrapper.root)

    def to_list(self) -> list[object]:
        """Serialize this history to a JSON-compatible list of dicts."""
        result: list[object] = json.loads(self.to_json())
        return result

    @classmethod
    def from_list(cls, data: list[object]) -> MessageHistory:
        """Deserialize a history from a list of dicts."""
        wrapper = _MessageListModel.model_validate(data)
        return cls(messages=wrapper.root)

    def __len__(self) -> int:
        return len(self.messages)


__all__ = ["MessageHistory"]

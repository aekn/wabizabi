"""Conversation session tracking for interactive wazi chat."""

from __future__ import annotations

from wabizabi import MessageHistory, ModelMessage


class ChatSession:
    """Tracks conversation history and turn count across an interactive session."""

    __slots__ = ("_messages", "_turn_count")

    def __init__(self) -> None:
        self._messages: list[ModelMessage] = []
        self._turn_count = 0

    @property
    def history(self) -> MessageHistory:
        """Return the current message history for passing to agent runs."""
        return MessageHistory(messages=tuple(self._messages))

    @property
    def turn_count(self) -> int:
        """Return the number of completed user turns."""
        return self._turn_count

    def append(self, *messages: ModelMessage) -> None:
        """Append messages from a completed turn."""
        self._messages.extend(messages)

    def record_turn(self) -> None:
        """Record a completed turn."""
        self._turn_count += 1

    def clear(self) -> None:
        """Clear all conversation history."""
        self._messages.clear()
        self._turn_count = 0

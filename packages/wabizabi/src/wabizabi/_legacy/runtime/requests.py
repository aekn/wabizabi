"""Request construction and message history helpers."""

from __future__ import annotations

from wabizabi.history import MessageHistory
from wabizabi.messages import (
    ModelRequest,
    ModelResponse,
    RequestPart,
    SystemInstructionPart,
    UserPromptPart,
)
from wabizabi.types import JsonObject


def merge_metadata(
    base: JsonObject | None,
    override: JsonObject | None,
) -> JsonObject | None:
    """Merge base metadata with a per-run override."""
    if base is None:
        if override is None:
            return None
        return dict(override)

    if override is None:
        return dict(base)

    return {**base, **override}


def coerce_message_history(
    message_history: MessageHistory | tuple[ModelRequest | ModelResponse, ...] | None,
) -> MessageHistory:
    """Normalize optional history inputs into a :class:`MessageHistory`."""
    if message_history is None:
        return MessageHistory.empty()
    if isinstance(message_history, MessageHistory):
        return message_history
    return MessageHistory(messages=message_history)


def build_request(
    *,
    user_prompt: str,
    metadata: JsonObject | None,
    system_instructions: tuple[str, ...],
) -> ModelRequest:
    """Build the next canonical request for the model."""
    parts: list[RequestPart] = [SystemInstructionPart(text=text) for text in system_instructions]
    parts.append(UserPromptPart(text=user_prompt))
    return ModelRequest(parts=tuple(parts), metadata=metadata)


__all__ = ["build_request", "coerce_message_history", "merge_metadata"]

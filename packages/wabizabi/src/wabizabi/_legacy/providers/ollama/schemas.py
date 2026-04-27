"""Typed Ollama chat payload schemas used at the provider boundary."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from wabizabi.types import JsonObject


class _OllamaBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class OllamaToolFunctionSchema(_OllamaBaseModel):
    name: str
    arguments: JsonObject
    index: int | None = None


class OllamaToolCallSchema(_OllamaBaseModel):
    type: Literal["function"] | None = None
    function: OllamaToolFunctionSchema


class OllamaMessageSchema(_OllamaBaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    thinking: str | None = None
    tool_calls: tuple[OllamaToolCallSchema, ...] | None = None
    images: tuple[str, ...] | None = None
    tool_name: str | None = None


class OllamaChatResponseSchema(_OllamaBaseModel):
    model: str | None = None
    created_at: str | None = None
    message: OllamaMessageSchema
    done: bool | None = None
    done_reason: str | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


__all__ = [
    "OllamaChatResponseSchema",
    "OllamaMessageSchema",
    "OllamaToolCallSchema",
    "OllamaToolFunctionSchema",
]

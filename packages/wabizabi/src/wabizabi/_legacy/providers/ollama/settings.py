"""Ollama provider settings."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from wabizabi.models import ModelSettings

type OllamaThinkOption = bool | Literal["low", "medium", "high"]


class OllamaSettings(ModelSettings):
    """Typed settings for Ollama-based model adapters."""

    ollama_model: str | None = Field(default=None, min_length=1)
    ollama_temperature: float | None = Field(default=None, ge=0.0)
    ollama_top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    ollama_num_predict: int | None = Field(default=None, ge=1)
    ollama_think: OllamaThinkOption | None = None


__all__ = ["OllamaSettings", "OllamaThinkOption"]

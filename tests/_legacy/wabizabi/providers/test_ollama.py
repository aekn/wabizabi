from __future__ import annotations

import pytest
from pydantic import ValidationError
from wabizabi.providers.ollama import OllamaSettings


def test_ollama_settings_accepts_prefixed_fields() -> None:
    settings = OllamaSettings(
        ollama_model="llama3.2",
        ollama_temperature=0.2,
        ollama_top_p=0.8,
        ollama_num_predict=256,
    )

    assert settings.ollama_model == "llama3.2"
    assert settings.ollama_temperature == 0.2
    assert settings.ollama_top_p == 0.8
    assert settings.ollama_num_predict == 256


def test_ollama_settings_rejects_invalid_ranges() -> None:
    with pytest.raises(ValidationError):
        OllamaSettings(ollama_num_predict=0)

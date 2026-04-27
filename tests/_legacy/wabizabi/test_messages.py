from __future__ import annotations

import pytest
from pydantic import ValidationError
from wabizabi.messages import (
    FinishReason,
    ImagePart,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
    validate_message,
)


def test_validate_message_parses_request_union() -> None:
    message = validate_message(
        {
            "message_kind": "request",
            "parts": [
                {
                    "part_kind": "user_prompt",
                    "text": "Describe the image.",
                },
                {
                    "part_kind": "image",
                    "source_kind": "url",
                    "source": "https://example.com/cat.png",
                    "media_type": "image/png",
                },
            ],
            "metadata": {"request_id": "req-1"},
        }
    )

    assert isinstance(message, ModelRequest)
    assert isinstance(message.parts[0], UserPromptPart)
    assert isinstance(message.parts[1], ImagePart)
    assert message.metadata == {"request_id": "req-1"}


def test_model_response_supports_text_and_finish_reason() -> None:
    response = ModelResponse(
        parts=(TextPart(text="Hello from the model."),),
        model_name="demo-model",
        finish_reason=FinishReason.STOP,
    )

    assert response.model_name == "demo-model"
    assert response.finish_reason is FinishReason.STOP
    assert isinstance(response.parts[0], TextPart)


def test_tool_call_requires_json_object_arguments() -> None:
    with pytest.raises(ValidationError):
        ToolCallPart.model_validate(
            {
                "tool_name": "search",
                "call_id": "call-1",
                "arguments": "not-an-object",
            }
        )


def test_message_requires_non_empty_parts() -> None:
    with pytest.raises(ValidationError, match="parts must not be empty"):
        ModelRequest(parts=())

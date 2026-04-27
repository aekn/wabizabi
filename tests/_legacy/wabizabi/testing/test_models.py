from __future__ import annotations

import pytest
from wabizabi.history import MessageHistory
from wabizabi.messages import ModelRequest, TextPart, UserPromptPart
from wabizabi.models import ModelResponseCompletedEvent, ModelResponsePartEvent, ModelSettings
from wabizabi.output import json_output_config
from wabizabi.providers.ollama import OllamaSettings
from wabizabi.testing import ScriptedModel, StreamingScriptedModel, text_result
from wabizabi.tools import ToolDefinition
from wabizabi.usage import RunUsage


@pytest.mark.asyncio
async def test_scripted_model_captures_requests_and_settings() -> None:
    model = ScriptedModel(
        (
            text_result(
                "Hello",
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
        )
    )
    history = MessageHistory(messages=(ModelRequest(parts=(UserPromptPart(text="Earlier"),)),))
    settings: ModelSettings = OllamaSettings(ollama_temperature=0.2)
    tool_definition = ToolDefinition(name="lookup", input_schema={"type": "object"})
    output = json_output_config()

    result = await model.request(
        ModelRequest(parts=(UserPromptPart(text="Hi"),)),
        message_history=history,
        settings=settings,
        tools=(tool_definition,),
        output=output,
    )

    assert result.response.parts == (TextPart(text="Hello"),)
    assert model.requests == [ModelRequest(parts=(UserPromptPart(text="Hi"),), metadata=None)]
    assert model.histories == [history]
    assert model.received_settings == [settings]
    assert model.received_tools == [(tool_definition,)]
    assert model.calls[0].output == output


@pytest.mark.asyncio
async def test_scripted_model_rejects_more_requests_than_scripted_results() -> None:
    model = ScriptedModel((text_result("Hello"),))

    await model.request(ModelRequest(parts=(UserPromptPart(text="Hi"),)))

    with pytest.raises(AssertionError, match="more requests than scripted results"):
        await model.request(ModelRequest(parts=(UserPromptPart(text="Again"),)))


@pytest.mark.asyncio
async def test_streaming_scripted_model_yields_scripted_events_and_captures_call() -> None:
    tool_definition = ToolDefinition(name="lookup", input_schema={"type": "object"})
    output = json_output_config()
    model = StreamingScriptedModel(
        (
            (
                ModelResponsePartEvent(part=TextPart(text="Hello")),
                ModelResponseCompletedEvent(
                    model_name="streamed",
                    usage=RunUsage(input_tokens=2, output_tokens=1),
                ),
            ),
        )
    )

    events = [
        event
        async for event in model.stream_response(
            ModelRequest(parts=(UserPromptPart(text="Hi"),)),
            tools=(tool_definition,),
            output=output,
        )
    ]

    assert events == [
        ModelResponsePartEvent(part=TextPart(text="Hello")),
        ModelResponseCompletedEvent(
            model_name="streamed",
            usage=RunUsage(input_tokens=2, output_tokens=1),
        ),
    ]
    assert model.received_tools == [(tool_definition,)]
    assert model.calls[0].output == output


@pytest.mark.asyncio
async def test_streaming_scripted_model_request_path_is_not_supported() -> None:
    model = StreamingScriptedModel(())

    with pytest.raises(AssertionError, match="should not be called"):
        await model.request(ModelRequest(parts=(UserPromptPart(text="Hi"),)))

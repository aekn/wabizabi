from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel, TypeAdapter
from wabizabi import MessageHistory, json_output_config, schema_output_config
from wabizabi.messages import (
    DocumentPart,
    FinishReason,
    ImagePart,
    ModelRequest,
    ModelResponse,
    NativeOutputPart,
    ReasoningPart,
    SystemInstructionPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from wabizabi.models import ModelResult
from wabizabi.output import OutputConfig
from wabizabi.providers.ollama import OllamaChatModel, OllamaSettings
from wabizabi.testing import (
    assert_model_contract_is_self_consistent,
    assert_model_contract_matches_expected,
    collect_model_contract_capture,
)
from wabizabi.tools import ToolDefinition
from wabizabi.types import JsonObject
from wabizabi.usage import RunUsage

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


class WeatherAnswer(BaseModel):
    city: str
    condition: str
    temperature_c: int


def _json_object(value: object) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(value)


def _json_objects(*values: object) -> tuple[JsonObject, ...]:
    return tuple(_json_object(value) for value in values)


class FakeOllamaChatCall:
    def __init__(
        self,
        *,
        model: str,
        messages: tuple[JsonObject, ...],
        tools: tuple[JsonObject, ...],
        settings: OllamaSettings | None,
        output: OutputConfig[object] | None,
    ) -> None:
        self.model = model
        self.messages = messages
        self.tools = tools
        self.settings = settings
        self.output = output


class FakeOllamaChat:
    def __init__(
        self,
        *,
        response: JsonObject,
    ) -> None:
        self.response = response
        self.calls: list[FakeOllamaChatCall] = []

    async def __call__(
        self,
        *,
        model: str,
        messages: tuple[JsonObject, ...],
        tools: tuple[JsonObject, ...],
        settings: OllamaSettings | None,
        output: OutputConfig[object] | None,
    ) -> JsonObject:
        self.calls.append(
            FakeOllamaChatCall(
                model=model,
                messages=messages,
                tools=tools,
                settings=settings,
                output=output,
            )
        )
        return self.response


class FakeOllamaStreamCall:
    def __init__(
        self,
        *,
        model: str,
        messages: tuple[JsonObject, ...],
        tools: tuple[JsonObject, ...],
        settings: OllamaSettings | None,
        output: OutputConfig[object] | None,
    ) -> None:
        self.model = model
        self.messages = messages
        self.tools = tools
        self.settings = settings
        self.output = output


class FakeOllamaStream:
    def __init__(self, *, chunks: tuple[JsonObject, ...]) -> None:
        self.chunks = chunks
        self.calls: list[FakeOllamaStreamCall] = []

    async def __call__(
        self,
        *,
        model: str,
        messages: tuple[JsonObject, ...],
        tools: tuple[JsonObject, ...],
        settings: OllamaSettings | None,
        output: OutputConfig[object] | None,
    ) -> AsyncIterator[JsonObject]:
        self.calls.append(
            FakeOllamaStreamCall(
                model=model,
                messages=messages,
                tools=tools,
                settings=settings,
                output=output,
            )
        )
        for chunk in self.chunks:
            yield chunk


@pytest.mark.asyncio
async def test_ollama_chat_model_text_contract_is_self_consistent() -> None:
    request_payload = _json_object(
        {
            "model": "qwen3",
            "created_at": "2026-04-02T12:00:00Z",
            "message": {
                "role": "assistant",
                "content": "Hello from Ollama.",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 12,
            "eval_count": 4,
        }
    )
    stream_payloads = _json_objects(
        {
            "model": "qwen3",
            "created_at": "2026-04-02T12:00:00Z",
            "message": {
                "role": "assistant",
                "content": "Hello ",
            },
            "done": False,
        },
        {
            "model": "qwen3",
            "created_at": "2026-04-02T12:00:00Z",
            "message": {
                "role": "assistant",
                "content": "from Ollama.",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 12,
            "eval_count": 4,
        },
    )
    model = OllamaChatModel(
        model_name="qwen3",
        chat_fn=FakeOllamaChat(response=request_payload),
        stream_chat_fn=FakeOllamaStream(chunks=stream_payloads),
    )

    capture = await collect_model_contract_capture(
        model,
        ModelRequest(parts=(UserPromptPart(text="Say hello"),)),
        output=None,
    )

    assert_model_contract_is_self_consistent(capture)
    assert capture.result == model_result(
        parts=(TextPart(text="Hello from Ollama."),),
        usage=RunUsage(input_tokens=12, output_tokens=4),
        metadata={
            "ollama_created_at": "2026-04-02T12:00:00Z",
            "ollama_done_reason": "stop",
        },
    )


def model_result(
    *,
    parts: tuple[TextPart | ReasoningPart | ToolCallPart, ...],
    usage: RunUsage,
    metadata: JsonObject,
    finish_reason: FinishReason = FinishReason.STOP,
) -> ModelResult:
    return ModelResult(
        response=ModelResponse(
            parts=parts,
            model_name="qwen3",
            finish_reason=finish_reason,
            metadata=metadata,
        ),
        usage=usage,
    )


@pytest.mark.asyncio
async def test_ollama_chat_model_passes_tools_and_history() -> None:
    chat = FakeOllamaChat(
        response=_json_object(
            {
                "model": "qwen3",
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Paris"},
                                "index": 0,
                            },
                        }
                    ],
                },
                "done": True,
                "done_reason": "stop",
            }
        )
    )
    stream = FakeOllamaStream(
        chunks=_json_objects(
            {
                "model": "qwen3",
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Paris"},
                                "index": 0,
                            },
                        }
                    ],
                },
                "done": True,
                "done_reason": "stop",
            }
        )
    )
    model = OllamaChatModel(model_name="qwen3", chat_fn=chat, stream_chat_fn=stream)
    history = MessageHistory(
        messages=(
            ModelRequest(parts=(SystemInstructionPart(text="Be concise."),)),
            ModelResponse(parts=(TextPart(text="Earlier answer."),)),
            ModelRequest(
                parts=(
                    ToolReturnPart(
                        tool_name="get_weather",
                        call_id="call_123",
                        content={"temp_c": 21},
                    ),
                )
            ),
        )
    )
    tool = ToolDefinition(
        name="get_weather",
        description="Get the weather.",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )

    capture = await collect_model_contract_capture(
        model,
        ModelRequest(parts=(UserPromptPart(text="Weather in Paris?"),)),
        message_history=history,
        tools=(tool,),
        settings=OllamaSettings(
            ollama_model="qwen3:8b",
            ollama_temperature=0.1,
            ollama_top_p=0.9,
            ollama_num_predict=128,
            ollama_think=False,
        ),
    )

    assert_model_contract_matches_expected(
        capture,
        model_result(
            parts=(
                ToolCallPart(
                    tool_name="get_weather",
                    call_id="ollama-call-0",
                    arguments={"city": "Paris"},
                ),
            ),
            usage=RunUsage.zero(),
            metadata={"ollama_done_reason": "stop"},
            finish_reason=FinishReason.TOOL_CALLS,
        ),
    )

    request_call = chat.calls[0]
    assert request_call.model == "qwen3:8b"
    assert request_call.settings == OllamaSettings(
        ollama_model="qwen3:8b",
        ollama_temperature=0.1,
        ollama_top_p=0.9,
        ollama_num_predict=128,
        ollama_think=False,
    )
    assert request_call.tools == (
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        },
    )
    assert request_call.messages == (
        {"role": "system", "content": "Be concise."},
        {"role": "assistant", "content": "Earlier answer."},
        {"role": "tool", "tool_name": "get_weather", "content": '{"temp_c":21}'},
        {"role": "user", "content": "Weather in Paris?"},
    )


@pytest.mark.asyncio
async def test_ollama_chat_model_normalizes_reasoning_and_text() -> None:
    request_payload = _json_object(
        {
            "model": "qwen3",
            "message": {
                "role": "assistant",
                "thinking": "Need to add 2 + 2.",
                "content": "The answer is 4.",
            },
            "done": True,
            "done_reason": "stop",
        }
    )
    stream_payloads = _json_objects(
        {
            "model": "qwen3",
            "message": {"role": "assistant", "thinking": "Need to add 2 + 2."},
            "done": False,
        },
        {
            "model": "qwen3",
            "message": {"role": "assistant", "content": "The answer is 4."},
            "done": True,
            "done_reason": "stop",
        },
    )
    model = OllamaChatModel(
        model_name="qwen3",
        chat_fn=FakeOllamaChat(response=request_payload),
        stream_chat_fn=FakeOllamaStream(chunks=stream_payloads),
    )

    capture = await collect_model_contract_capture(
        model,
        ModelRequest(parts=(UserPromptPart(text="What is 2 + 2?"),)),
    )

    assert_model_contract_matches_expected(
        capture,
        model_result(
            parts=(
                ReasoningPart(text="Need to add 2 + 2."),
                TextPart(text="The answer is 4."),
            ),
            usage=RunUsage.zero(),
            metadata={"ollama_done_reason": "stop"},
        ),
    )


@pytest.mark.asyncio
async def test_ollama_chat_model_rejects_wrong_settings_type() -> None:
    model = OllamaChatModel(
        model_name="qwen3",
        chat_fn=FakeOllamaChat(
            response=_json_object(
                {
                    "model": "qwen3",
                    "message": {"role": "assistant", "content": "hello"},
                    "done": True,
                }
            )
        ),
        stream_chat_fn=FakeOllamaStream(
            chunks=_json_objects(
                {
                    "model": "qwen3",
                    "message": {"role": "assistant", "content": "hello"},
                    "done": True,
                }
            )
        ),
    )

    with pytest.raises(TypeError):
        await model.request(
            ModelRequest(parts=(UserPromptPart(text="Hi"),)),
            settings=object(),  # pyright: ignore[reportArgumentType]
        )


@pytest.mark.asyncio
async def test_ollama_chat_model_accumulates_streamed_tool_calls_without_type() -> None:
    stream = FakeOllamaStream(
        chunks=_json_objects(
            {
                "model": "qwen3",
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "add",
                                "arguments": {"a": 2, "b": 3},
                            }
                        }
                    ],
                },
                "done": False,
            },
            {
                "model": "qwen3",
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "multiply",
                                "arguments": {"a": 5, "b": 4},
                            }
                        }
                    ],
                },
                "done": True,
                "done_reason": "stop",
            },
        )
    )
    model = OllamaChatModel(
        model_name="qwen3",
        chat_fn=FakeOllamaChat(
            response=_json_object(
                {
                    "model": "qwen3",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "add",
                                    "arguments": {"a": 2, "b": 3},
                                }
                            },
                            {
                                "function": {
                                    "name": "multiply",
                                    "arguments": {"a": 5, "b": 4},
                                }
                            },
                        ],
                    },
                    "done": True,
                    "done_reason": "stop",
                }
            )
        ),
        stream_chat_fn=stream,
    )

    capture = await collect_model_contract_capture(
        model,
        ModelRequest(parts=(UserPromptPart(text="Calculate"),)),
    )

    assert_model_contract_matches_expected(
        capture,
        model_result(
            parts=(
                ToolCallPart(tool_name="add", call_id="ollama-call-0", arguments={"a": 2, "b": 3}),
                ToolCallPart(
                    tool_name="multiply", call_id="ollama-call-1", arguments={"a": 5, "b": 4}
                ),
            ),
            usage=RunUsage.zero(),
            metadata={"ollama_done_reason": "stop"},
            finish_reason=FinishReason.TOOL_CALLS,
        ),
    )


@pytest.mark.asyncio
async def test_ollama_chat_model_passes_json_output_format() -> None:
    chat = FakeOllamaChat(
        response=_json_object(
            {
                "model": "qwen3",
                "message": {"role": "assistant", "content": '{"city":"Paris","ok":true}'},
                "done": True,
                "done_reason": "stop",
            }
        )
    )
    stream = FakeOllamaStream(
        chunks=_json_objects(
            {
                "model": "qwen3",
                "message": {"role": "assistant", "content": '{"city":"Paris","ok":true}'},
                "done": True,
                "done_reason": "stop",
            }
        )
    )
    model = OllamaChatModel(model_name="qwen3", chat_fn=chat, stream_chat_fn=stream)

    await collect_model_contract_capture(
        model,
        ModelRequest(parts=(UserPromptPart(text="Return JSON."),)),
        output=json_output_config(),
    )

    assert chat.calls[0].output is not None
    assert chat.calls[0].output.response_format == "json"


@pytest.mark.asyncio
async def test_ollama_chat_model_passes_schema_output_format() -> None:
    chat = FakeOllamaChat(
        response=_json_object(
            {
                "model": "qwen3",
                "message": {
                    "role": "assistant",
                    "content": '{"city":"Paris","condition":"sunny","temperature_c":21}',
                },
                "done": True,
                "done_reason": "stop",
            }
        )
    )
    stream = FakeOllamaStream(
        chunks=_json_objects(
            {
                "model": "qwen3",
                "message": {
                    "role": "assistant",
                    "content": '{"city":"Paris","condition":"sunny","temperature_c":21}',
                },
                "done": True,
                "done_reason": "stop",
            }
        )
    )
    model = OllamaChatModel(model_name="qwen3", chat_fn=chat, stream_chat_fn=stream)

    output = schema_output_config(WeatherAnswer)
    await collect_model_contract_capture(
        model,
        ModelRequest(parts=(UserPromptPart(text="Return schema."),)),
        output=output,
    )

    captured_output = chat.calls[0].output
    assert captured_output is not None
    response_format = captured_output.response_format
    assert isinstance(response_format, dict)
    assert response_format["type"] == "object"
    properties = response_format["properties"]
    assert isinstance(properties, dict)
    assert sorted(properties.keys()) == [
        "city",
        "condition",
        "temperature_c",
    ]


@pytest.mark.asyncio
async def test_ollama_chat_model_translates_image_parts() -> None:
    chat = FakeOllamaChat(
        response=_json_object(
            {
                "model": "qwen3",
                "message": {"role": "assistant", "content": "done"},
                "done": True,
                "done_reason": "stop",
            }
        )
    )
    stream = FakeOllamaStream(
        chunks=_json_objects(
            {
                "model": "qwen3",
                "message": {"role": "assistant", "content": "done"},
                "done": True,
                "done_reason": "stop",
            }
        )
    )
    model = OllamaChatModel(model_name="qwen3", chat_fn=chat, stream_chat_fn=stream)

    await collect_model_contract_capture(
        model,
        ModelRequest(
            parts=(
                UserPromptPart(text="Describe this image."),
                ImagePart(
                    source_kind="url",
                    source="https://example.test/image.png",
                    media_type="image/png",
                ),
            )
        ),
    )

    assert chat.calls[0].messages == (
        {"role": "user", "content": "Describe this image."},
        {"role": "user", "content": "", "images": ["https://example.test/image.png"]},
    )


@pytest.mark.asyncio
async def test_ollama_chat_model_translates_text_document_parts() -> None:
    chat = FakeOllamaChat(
        response=_json_object(
            {
                "model": "qwen3",
                "message": {"role": "assistant", "content": "done"},
                "done": True,
                "done_reason": "stop",
            }
        )
    )
    stream = FakeOllamaStream(
        chunks=_json_objects(
            {
                "model": "qwen3",
                "message": {"role": "assistant", "content": "done"},
                "done": True,
                "done_reason": "stop",
            }
        )
    )
    model = OllamaChatModel(model_name="qwen3", chat_fn=chat, stream_chat_fn=stream)

    await collect_model_contract_capture(
        model,
        ModelRequest(
            parts=(
                DocumentPart(
                    source_kind="text",
                    source="Quarterly report summary.",
                    media_type="text/plain",
                    name="report.txt",
                ),
            )
        ),
    )

    assert chat.calls[0].messages == ({"role": "user", "content": "Quarterly report summary."},)


@pytest.mark.asyncio
async def test_ollama_chat_model_emits_native_output_part_for_structured_response() -> None:
    request_payload = _json_object(
        {
            "model": "qwen3",
            "message": {
                "role": "assistant",
                "content": '{"city":"Paris","condition":"sunny","temperature_c":21}',
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 8,
            "eval_count": 5,
        }
    )
    chat = FakeOllamaChat(response=request_payload)
    output = schema_output_config(WeatherAnswer)
    model = OllamaChatModel(
        model_name="qwen3",
        chat_fn=chat,
        stream_chat_fn=FakeOllamaStream(chunks=(request_payload,)),
    )

    result = await model.request(
        ModelRequest(parts=(UserPromptPart(text="Weather?"),)),
        output=output,
    )

    assert result.response.parts == (
        NativeOutputPart(data={"city": "Paris", "condition": "sunny", "temperature_c": 21}),
    )
    assert chat.calls[0].output is output

"""Ollama SDK client wrapping and callable protocols."""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator
from types import ModuleType
from typing import Protocol, runtime_checkable

from pydantic import TypeAdapter

from wabizabi.output import OutputConfig
from wabizabi.providers.ollama.settings import OllamaSettings
from wabizabi.types import JsonObject, JsonValue

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


@runtime_checkable
class _SupportsToDict(Protocol):
    def to_dict(self) -> JsonObject: ...


@runtime_checkable
class _SupportsModelDump(Protocol):
    def model_dump(self) -> JsonObject: ...


@runtime_checkable
class _OllamaAsyncStream(Protocol):
    def __aiter__(self) -> AsyncIterator[object]: ...


@runtime_checkable
class _OllamaClosableAsyncStream(_OllamaAsyncStream, Protocol):
    async def aclose(self) -> None: ...


@runtime_checkable
class _OllamaAsyncClient(Protocol):
    async def chat(self, **kwargs: object) -> object: ...


class OllamaChatFn(Protocol):
    """Execute one Ollama chat request."""

    async def __call__(
        self,
        *,
        model: str,
        messages: tuple[JsonObject, ...],
        tools: tuple[JsonObject, ...],
        settings: OllamaSettings | None,
        output: OutputConfig[object] | None,
    ) -> JsonObject: ...


class OllamaStreamChatFn(Protocol):
    """Stream raw Ollama chat chunks as JSON objects."""

    def __call__(
        self,
        *,
        model: str,
        messages: tuple[JsonObject, ...],
        tools: tuple[JsonObject, ...],
        settings: OllamaSettings | None,
        output: OutputConfig[object] | None,
    ) -> AsyncIterator[JsonObject]: ...


def json_object_from_unknown(value: object) -> JsonObject:
    """Convert an unknown SDK response object into a typed JSON dict."""
    if isinstance(value, _SupportsToDict):
        return value.to_dict()
    if isinstance(value, _SupportsModelDump):
        return _JSON_OBJECT_ADAPTER.validate_python(value.model_dump())
    return _JSON_OBJECT_ADAPTER.validate_python(value)


def _options_from_settings(settings: OllamaSettings | None) -> JsonObject | None:
    if settings is None:
        return None

    options: JsonObject = {}
    if settings.ollama_temperature is not None:
        options["temperature"] = settings.ollama_temperature
    if settings.ollama_top_p is not None:
        options["top_p"] = settings.ollama_top_p
    if settings.ollama_num_predict is not None:
        options["num_predict"] = settings.ollama_num_predict
    return options or None


def _think_from_settings(settings: OllamaSettings | None) -> JsonValue | None:
    if settings is None:
        return None
    return settings.ollama_think


def chat_kwargs(
    *,
    model: str,
    messages: tuple[JsonObject, ...],
    tools: tuple[JsonObject, ...],
    settings: OllamaSettings | None,
    output: OutputConfig[object] | None,
    stream: bool,
) -> dict[str, object]:
    """Build the kwargs dict for an Ollama chat API call."""
    kwargs: dict[str, object] = {
        "model": model,
        "messages": list(messages),
        "stream": stream,
    }
    if tools:
        kwargs["tools"] = list(tools)
    options = _options_from_settings(settings)
    if options is not None:
        kwargs["options"] = options
    think = _think_from_settings(settings)
    if think is not None:
        kwargs["think"] = think
    if output is not None and output.response_format is not None:
        kwargs["format"] = output.response_format
    return kwargs


def chat_from_sdk(client: object) -> OllamaChatFn:
    """Wrap an SDK ``AsyncClient`` behind a typed one-shot callable."""

    if not isinstance(client, _OllamaAsyncClient):
        raise TypeError("client must provide an async chat(**kwargs) method")

    async def chat(
        *,
        model: str,
        messages: tuple[JsonObject, ...],
        tools: tuple[JsonObject, ...],
        settings: OllamaSettings | None,
        output: OutputConfig[object] | None,
    ) -> JsonObject:
        kwargs = chat_kwargs(
            model=model,
            messages=messages,
            tools=tools,
            settings=settings,
            output=output,
            stream=False,
        )
        response = await client.chat(**kwargs)
        return json_object_from_unknown(response)

    return chat


def stream_chat_from_sdk(client: object) -> OllamaStreamChatFn:
    """Wrap an SDK ``AsyncClient`` behind a typed streaming callable."""

    if not isinstance(client, _OllamaAsyncClient):
        raise TypeError("client must provide an async chat(**kwargs) method")

    async def stream_chat(
        *,
        model: str,
        messages: tuple[JsonObject, ...],
        tools: tuple[JsonObject, ...],
        settings: OllamaSettings | None,
        output: OutputConfig[object] | None,
    ) -> AsyncIterator[JsonObject]:
        kwargs = chat_kwargs(
            model=model,
            messages=messages,
            tools=tools,
            settings=settings,
            output=output,
            stream=True,
        )
        stream = await client.chat(**kwargs)
        if not isinstance(stream, _OllamaAsyncStream):
            raise TypeError("client.chat(..., stream=True) must return an async iterator")

        try:
            async for chunk in stream:
                yield json_object_from_unknown(chunk)
        finally:
            if isinstance(stream, _OllamaClosableAsyncStream):
                try:
                    await stream.aclose()
                except RuntimeError as exc:
                    if "already running" not in str(exc):
                        raise

    return stream_chat


def build_default_sdk_client(host: str | None, *, purpose: str) -> object:
    """Import and instantiate the Ollama async client."""
    try:
        ollama_module: ModuleType = importlib.import_module("ollama")
    except ImportError as exc:
        raise RuntimeError(
            f"Install wabizabi[ollama] to use OllamaChatModel {purpose} without "
            + "supplying a custom callable. In a workspace, run "
            + "`uv sync --all-packages --extra ollama`."
        ) from exc

    async_client_factory = ollama_module.__dict__.get("AsyncClient")
    if async_client_factory is None or not callable(async_client_factory):
        raise RuntimeError("The installed ollama package does not expose AsyncClient.")

    return async_client_factory(host=host) if host is not None else async_client_factory()


def build_default_chat(host: str | None) -> OllamaChatFn:
    """Build a default one-shot chat callable from the SDK."""
    return chat_from_sdk(build_default_sdk_client(host, purpose="chat"))


def build_default_stream_chat(host: str | None) -> OllamaStreamChatFn:
    """Build a default streaming chat callable from the SDK."""
    return stream_chat_from_sdk(build_default_sdk_client(host, purpose="streaming"))


__all__ = [
    "OllamaChatFn",
    "OllamaStreamChatFn",
    "build_default_chat",
    "build_default_stream_chat",
    "chat_from_sdk",
    "chat_kwargs",
    "json_object_from_unknown",
    "stream_chat_from_sdk",
]

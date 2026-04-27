from wabizabi import providers


def test_providers_namespace_exports_only_ollama_surface() -> None:
    assert set(providers.__all__) == {
        "OllamaChatFn",
        "OllamaChatModel",
        "OllamaSettings",
        "OllamaStreamChatFn",
        "chat_from_sdk",
        "stream_chat_from_sdk",
    }

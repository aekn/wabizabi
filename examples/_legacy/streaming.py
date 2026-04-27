"""Streaming text output with Agent.iter().

Usage:
    python examples/streaming.py
"""

import asyncio

from wabizabi import Agent, text_output_config
from wabizabi.providers.ollama import OllamaChatModel, OllamaSettings
from wabizabi.stream import TextChunkEvent


async def main() -> None:
    agent = Agent[None, str](
        model=OllamaChatModel("qwen3:4b"),
        output=text_output_config(),
        model_settings=OllamaSettings(ollama_temperature=0.5, ollama_think=False),
    )

    async for event in agent.iter("Tell me a short joke.", deps=None):
        if isinstance(event, TextChunkEvent):
            print(event.text, end="", flush=True)

    print()


if __name__ == "__main__":
    asyncio.run(main())

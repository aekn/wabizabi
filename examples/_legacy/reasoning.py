"""Observing model reasoning with thinking enabled.

Usage:
    python examples/reasoning.py
"""

import asyncio

from wabizabi import Agent, text_output_config
from wabizabi.providers.ollama import OllamaChatModel, OllamaSettings
from wabizabi.stream import ReasoningChunkEvent, TextChunkEvent


async def main() -> None:
    agent = Agent[None, str](
        model=OllamaChatModel("qwen3:14b"),
        output=text_output_config(),
        model_settings=OllamaSettings(ollama_temperature=0.5, ollama_think=True),
    )

    saw_text = False
    print("--- Reasoning ---")
    async for event in agent.iter("What is 17 * 23?", deps=None):
        if isinstance(event, ReasoningChunkEvent):
            print(event.text, end="", flush=True)
        elif isinstance(event, TextChunkEvent):
            if not saw_text:
                print("\n--- Response ---")
                saw_text = True
            print(event.text, end="", flush=True)

    print()


if __name__ == "__main__":
    asyncio.run(main())

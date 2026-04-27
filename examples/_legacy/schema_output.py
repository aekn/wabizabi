"""Structured output with schema validation.

Usage:
    python examples/schema_output.py
"""

import asyncio

from pydantic import BaseModel
from wabizabi import Agent, schema_output_config
from wabizabi.providers.ollama import OllamaChatModel, OllamaSettings


class CityInfo(BaseModel):
    name: str
    country: str
    population: int
    known_for: str


async def main() -> None:
    agent = Agent[None, CityInfo](
        model=OllamaChatModel("qwen3:4b"),
        output=schema_output_config(CityInfo),
        model_settings=OllamaSettings(ollama_temperature=0.0, ollama_think=False),
    )

    result = await agent.run("Tell me about Tokyo.", deps=None)
    city = result.output
    print(f"{city.name}, {city.country}")
    print(f"Population: {city.population:,}")
    print(f"Known for: {city.known_for}")


if __name__ == "__main__":
    asyncio.run(main())

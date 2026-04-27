"""Reusable capabilities that bundle tools and instructions.

Usage:
    wazi agent chat examples.capabilities:agent
    wazi agent run examples.capabilities:agent --input "What time is it in Tokyo?"
"""

import datetime

from wabizabi import (
    Agent,
    RunContext,
    StaticCapability,
    Toolset,
    define_static_capability,
    text_output_config,
    tool,
)
from wabizabi.providers.ollama import OllamaChatModel, OllamaSettings


@tool
def current_time(ctx: RunContext[None], timezone: str) -> str:
    """Get the current time in a timezone (e.g. 'UTC', 'US/Eastern')."""
    import zoneinfo

    tz = zoneinfo.ZoneInfo(timezone)
    now = datetime.datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


time_capability: StaticCapability[None, str] = define_static_capability(
    system_instructions=("You can look up the current time in any timezone.",),
    toolset=Toolset[None]((current_time,)),
)

agent = Agent[None, str](
    model=OllamaChatModel("qwen3:14b"),
    output=text_output_config(),
    model_settings=OllamaSettings(ollama_temperature=0.0, ollama_think=False),
).with_capability(time_capability)

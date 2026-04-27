"""A tool-using agent optimized for running through wazi.

Usage:
    wazi agent chat examples.wazi_agent:agent
    wazi agent run examples.wazi_agent:agent --input "What is 42 + 58?"
    wazi --trace agent run examples.wazi_agent:agent --input "What is 42 + 58?"
"""

from wabizabi import Agent, RunContext, Toolset, text_output_config, tool
from wabizabi.providers.ollama import OllamaChatModel, OllamaSettings


@tool
def add(ctx: RunContext[None], a: float, b: float) -> float:
    """Add two numbers, performs (a+b)"""
    return a + b


@tool
def multiply(ctx: RunContext[None], a: float, b: float) -> float:
    """Multiply two numbers, performs (a*b)."""
    return a * b


@tool
def subtract(ctx: RunContext[None], a: float, b: float) -> float:
    """Subtract two numbers, performs (a-b)."""
    return a - b


@tool
def divide(ctx: RunContext[None], a: float, b: float) -> float:
    """Divide two integers, performs (a/b)."""
    return a / b


calculator = Toolset[None]((add, subtract, multiply, divide))

agent = Agent[None, str](
    model=OllamaChatModel("qwen3:14b"),
    output=text_output_config(),
    system_instructions=(
        "You are a helpful math assistant.",
        "Use the available tools to perform calculations.",
        "Always show your work by using tools, then explain the result.",
    ),
    toolset=calculator,
    model_settings=OllamaSettings(ollama_temperature=0.0, ollama_think=False),
    max_tool_rounds=100,
)

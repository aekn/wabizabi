"""Multi-agent handoff between a triage agent and a specialist.

Usage:
    python examples/handoffs.py
"""

import asyncio

from wabizabi import Agent, Handoff, text_output_config
from wabizabi.providers.ollama import OllamaChatModel, OllamaSettings
from wabizabi.stream import HandoffEvent, OutputEvent, TextChunkEvent
from wazi.app_registry import AgentApp

model = OllamaChatModel("qwen3:14b")
settings = OllamaSettings(ollama_temperature=0.0, ollama_think=False)

billing_agent = Agent[None, str](
    model=model,
    output=text_output_config(),
    system_instructions=("You are a billing specialist. Help with invoices and payments.",),
    model_settings=settings,
)

triage_agent = Agent[None, str](
    model=model,
    output=text_output_config(),
    system_instructions=(
        "You are a triage agent. Route billing questions to the billing specialist.",
        "For billing questions, use the handoff_billing tool.",
    ),
    model_settings=settings,
    handoffs=(Handoff(name="billing", description="Hand off to the billing specialist."),),
)

app = AgentApp(agents={"billing": billing_agent, "triage": triage_agent}, default="triage")


async def main() -> None:
    prompt = "I have a question about my invoice."

    async for event in triage_agent.iter(prompt, deps=None):
        if isinstance(event, TextChunkEvent):
            print(event.text, end="", flush=True)
        elif isinstance(event, HandoffEvent):
            print(f"\n[Handoff to {event.handoff_name}]")

            result = await billing_agent.run(prompt, deps=None)
            print(result.output)
            return
        elif isinstance(event, OutputEvent):
            break

    print()


if __name__ == "__main__":
    asyncio.run(main())

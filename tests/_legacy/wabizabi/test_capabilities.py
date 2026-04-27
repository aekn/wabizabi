from __future__ import annotations

import pytest
from pydantic import BaseModel
from wabizabi.agent import Agent
from wabizabi.capabilities import StaticCapability, define_static_capability
from wabizabi.context import RunContext
from wabizabi.hooks import Hooks
from wabizabi.messages import (
    ModelRequest,
    ModelResponse,
    SystemInstructionPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from wabizabi.models import ModelResult
from wabizabi.output import TextOutputDecoder
from wabizabi.processors import TrimHistoryProcessor
from wabizabi.providers.ollama import OllamaSettings
from wabizabi.testing import ScriptedModel
from wabizabi.tools import Toolset, define_function_tool
from wabizabi.usage import RunUsage


@pytest.mark.asyncio
async def test_static_capability_applies_instruction_and_validator() -> None:
    async def add_suffix(
        context: RunContext[tuple[str, str]],
        output: str,
    ) -> str:
        assert context.usage == RunUsage(input_tokens=3, output_tokens=2)
        return f"{output}!"

    capability: StaticCapability[tuple[str, str], str] = define_static_capability(
        system_instructions=("Be terse.",),
        output_validators=(add_suffix,),
    )

    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="Hi"),), model_name="scripted"),
                usage=RunUsage(input_tokens=3, output_tokens=2),
            ),
        )
    )

    agent = Agent[tuple[str, str], str](
        model=model,
        decoder=TextOutputDecoder(),
    ).with_capability(capability)

    result = await agent.run(
        "Hello",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    request = model.requests[0]

    assert request.parts == (
        SystemInstructionPart(text="Be terse."),
        UserPromptPart(text="Hello"),
    )
    assert result.output == "Hi!"


class AddArguments(BaseModel):
    left: int
    right: int


@pytest.mark.asyncio
async def test_static_capability_can_provide_tools() -> None:
    def add_tool(context: RunContext[tuple[str, str]], arguments: AddArguments) -> int:
        assert context.tool_name == "add"
        return arguments.left + arguments.right

    capability: StaticCapability[tuple[str, str], str] = define_static_capability(
        tools=(
            define_function_tool(
                name="add",
                arguments_type=AddArguments,
                func=add_tool,
            ),
        ),
    )

    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(
                        ToolCallPart(
                            tool_name="add",
                            call_id="call-1",
                            arguments={"left": 2, "right": 3},
                        ),
                    ),
                    model_name="scripted",
                ),
                usage=RunUsage(input_tokens=3, output_tokens=0),
            ),
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="5"),), model_name="scripted"),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
        )
    )

    agent = Agent[tuple[str, str], str](
        model=model,
        decoder=TextOutputDecoder(),
    ).with_capability(capability)

    result = await agent.run(
        "What is 2 + 3?",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    assert result.output == "5"
    assert result.usage == RunUsage(input_tokens=5, output_tokens=1)


@pytest.mark.asyncio
async def test_static_capability_can_contribute_toolset_hooks_processors_and_settings() -> None:
    def multiply_tool(context: RunContext[tuple[str, str]], arguments: AddArguments) -> int:
        del context
        return arguments.left * arguments.right

    def rewrite_request(
        context: RunContext[tuple[str, str]],
        request: ModelRequest,
    ) -> ModelRequest:
        del context
        first_part = request.parts[-1]
        assert isinstance(first_part, UserPromptPart)
        return request.model_copy(
            update={
                "parts": (
                    *request.parts[:-1],
                    UserPromptPart(text=f"{first_part.text} Use the multiply tool if needed."),
                )
            }
        )

    capability: StaticCapability[tuple[str, str], str] = define_static_capability(
        system_instructions=("Be concise.",),
        toolset=Toolset[tuple[str, str]]
        .empty()
        .with_tool(
            define_function_tool(
                name="multiply",
                arguments_type=AddArguments,
                func=multiply_tool,
            )
        ),
        hooks=Hooks[tuple[str, str]].empty().with_before_request(rewrite_request),
        history_processors=(TrimHistoryProcessor(max_messages=1),),
        model_settings=OllamaSettings(ollama_temperature=0.25),
    )

    history = (
        ModelRequest(parts=(UserPromptPart(text="old request"),)),
        ModelResponse(parts=(TextPart(text="old response"),), model_name="scripted"),
    )
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="ok"),), model_name="scripted"),
            ),
        )
    )

    agent = Agent[tuple[str, str], str](
        model=model,
        decoder=TextOutputDecoder(),
    ).with_capability(capability)

    result = await agent.run(
        "Hello",
        deps=("svc", "cfg"),
        message_history=history,
    )

    assert result.output == "ok"
    assert model.received_settings == [OllamaSettings(ollama_temperature=0.25)]
    assert len(model.received_tools[0]) == 1
    assert model.received_tools[0][0].name == "multiply"
    assert model.histories[0] is not None
    assert model.histories[0].messages == history[-1:]
    assert model.requests[0].parts == (
        SystemInstructionPart(text="Be concise."),
        UserPromptPart(text="Hello Use the multiply tool if needed."),
    )


@pytest.mark.asyncio
async def test_function_capability_can_transform_agent() -> None:
    def add_instruction(
        agent: Agent[tuple[str, str], str],
    ) -> Agent[tuple[str, str], str]:
        return agent.with_system_instruction("Be helpful.")

    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="Hi"),), model_name="scripted"),
                usage=RunUsage(input_tokens=3, output_tokens=2),
            ),
        )
    )

    agent = Agent[tuple[str, str], str](
        model=model,
        decoder=TextOutputDecoder(),
    ).with_capability(add_instruction)

    await agent.run(
        "Hello",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    request = model.requests[0]
    assert request.parts == (
        SystemInstructionPart(text="Be helpful."),
        UserPromptPart(text="Hello"),
    )


@pytest.mark.asyncio
async def test_with_capabilities_applies_multiple_capabilities_in_order() -> None:
    def add_instruction(
        agent: Agent[tuple[str, str], str],
    ) -> Agent[tuple[str, str], str]:
        return agent.with_system_instruction("Be concise.")

    async def add_suffix(
        context: RunContext[tuple[str, str]],
        output: str,
    ) -> str:
        del context
        return f"{output}!"

    capability: StaticCapability[tuple[str, str], str] = define_static_capability(
        output_validators=(add_suffix,),
    )

    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="Hi"),), model_name="scripted"),
                usage=RunUsage(input_tokens=3, output_tokens=2),
            ),
        )
    )

    agent = Agent[tuple[str, str], str](
        model=model,
        decoder=TextOutputDecoder(),
    ).with_capabilities(
        add_instruction,
        capability,
    )

    result = await agent.run(
        "Hello",
        deps=("svc", "cfg"),
        run_id="run-1",
    )

    request = model.requests[0]
    assert request.parts == (
        SystemInstructionPart(text="Be concise."),
        UserPromptPart(text="Hello"),
    )
    assert result.output == "Hi!"

from __future__ import annotations

from pydantic import BaseModel
from wabizabi.agent import Agent
from wabizabi.context import RunContext
from wabizabi.hooks import Hooks
from wabizabi.messages import ModelRequest
from wabizabi.output import schema_output_config, text_output_config
from wabizabi.testing import ScriptedModel
from wabizabi.tools import FunctionTool, define_function_tool
from wabizabi.tools.base import ToolDefinition


class AddArgs(BaseModel):
    value: int


class Total(BaseModel):
    total: int


def test_agent_builder_methods_return_new_agents_without_mutating_original() -> None:
    base_agent = Agent[tuple[str, str], str](
        model=ScriptedModel(()),
        output=text_output_config(),
        system_instructions=("Keep replies short.",),
        metadata={"team": "core"},
        max_tool_rounds=3,
        max_output_retries=1,
    )

    updated_agent = base_agent.with_system_instruction("Use tools when helpful.").with_output(
        schema_output_config(Total)
    )

    assert base_agent is not updated_agent
    assert base_agent.system_instructions == ("Keep replies short.",)
    assert updated_agent.system_instructions == (
        "Keep replies short.",
        "Use tools when helpful.",
    )
    assert base_agent.output_mode.name == "TEXT"
    assert updated_agent.output_mode.name == "SCHEMA"
    assert updated_agent.metadata == {"team": "core"}
    assert updated_agent.max_tool_rounds == 3
    assert updated_agent.max_output_retries == 1


def test_tools_package_submodules_expose_existing_runtime_types() -> None:
    def tool(arguments_context: RunContext[object], arguments: AddArgs) -> int:
        del arguments_context
        return arguments.value

    defined_tool = define_function_tool(
        name="answer",
        arguments_type=AddArgs,
        func=tool,
    )

    assert isinstance(defined_tool, FunctionTool)
    assert isinstance(defined_tool.definition, ToolDefinition)
    assert defined_tool.definition.name == "answer"


def test_agent_with_hooks_merges_without_mutating_original() -> None:
    base_agent = Agent[None, str](
        model=ScriptedModel(()),
        output=text_output_config(),
    )

    def first_before_request(
        context: RunContext[None],
        request: ModelRequest,
    ) -> ModelRequest:
        del context
        return request

    def second_before_request(
        context: RunContext[None],
        request: ModelRequest,
    ) -> ModelRequest:
        del context
        return request

    first_hooks = Hooks[None].empty().with_before_request(first_before_request)
    second_hooks = Hooks[None].empty().with_before_request(second_before_request)

    updated_agent = base_agent.with_hooks(first_hooks).with_hooks(second_hooks)

    assert updated_agent is not base_agent
    assert len(base_agent.hooks.before_request) == 0
    assert len(updated_agent.hooks.before_request) == 2

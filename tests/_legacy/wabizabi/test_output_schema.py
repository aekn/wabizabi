from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError
from wabizabi import Agent, json_output_config, schema_output_config
from wabizabi.messages import ModelResponse, NativeOutputPart, TextPart
from wabizabi.models import ModelResult
from wabizabi.output import OutputDecodingError
from wabizabi.testing import ScriptedModel
from wabizabi.types import JsonValue
from wabizabi.usage import RunUsage


class WeatherAnswer(BaseModel):
    city: str
    condition: str
    temperature_c: int


@pytest.mark.asyncio
async def test_agent_json_output_config_decodes_structured_text() -> None:
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(TextPart(text='{"city":"Paris","ok":true}'),),
                    model_name="scripted",
                ),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
        )
    )
    agent = Agent[None, JsonValue](
        model=model,
        output=json_output_config(),
    )

    result = await agent.run("Return JSON.", deps=None, run_id="run-1")

    assert result.output == {"city": "Paris", "ok": True}
    assert model.calls[0].output == agent.output


@pytest.mark.asyncio
async def test_agent_json_output_config_decodes_native_output() -> None:
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(NativeOutputPart(data={"city": "Paris", "ok": True}),),
                    model_name="scripted",
                ),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
        )
    )
    agent = Agent[None, JsonValue](model=model, output=json_output_config())

    result = await agent.run("Return JSON.", deps=None)

    assert result.output == {"city": "Paris", "ok": True}


@pytest.mark.asyncio
async def test_agent_schema_output_config_decodes_structured_text() -> None:
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(
                        TextPart(text='{"city":"Paris","condition":"sunny","temperature_c":21}'),
                    ),
                    model_name="scripted",
                ),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
        )
    )
    agent = Agent[None, WeatherAnswer](
        model=model,
        output=schema_output_config(WeatherAnswer),
    )

    result = await agent.run("Return weather.", deps=None, run_id="run-1")

    assert result.output == WeatherAnswer(city="Paris", condition="sunny", temperature_c=21)
    assert model.calls[0].output == agent.output


@pytest.mark.asyncio
async def test_agent_schema_output_config_decodes_native_output() -> None:
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(
                        NativeOutputPart(
                            data={"city": "Paris", "condition": "sunny", "temperature_c": 21}
                        ),
                    ),
                    model_name="scripted",
                ),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
        )
    )
    agent = Agent[None, WeatherAnswer](model=model, output=schema_output_config(WeatherAnswer))

    result = await agent.run("Return weather.", deps=None)

    assert result.output == WeatherAnswer(city="Paris", condition="sunny", temperature_c=21)


@pytest.mark.asyncio
async def test_agent_json_output_config_raises_decoding_error_when_retries_disabled() -> None:
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(parts=(TextPart(text="not-json"),), model_name="scripted"),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
        )
    )
    agent = Agent[None, JsonValue](
        model=model,
        output=json_output_config(),
        max_output_retries=0,
    )

    with pytest.raises(OutputDecodingError, match="valid structured JSON output"):
        await agent.run("Return JSON.", deps=None)


@pytest.mark.asyncio
async def test_agent_schema_output_config_propagates_validation_error() -> None:
    model = ScriptedModel(
        (
            ModelResult(
                response=ModelResponse(
                    parts=(TextPart(text='{"city":"Paris"}'),), model_name="scripted"
                ),
                usage=RunUsage(input_tokens=2, output_tokens=1),
            ),
        )
    )
    agent = Agent[None, WeatherAnswer](
        model=model,
        output=schema_output_config(WeatherAnswer),
        max_output_retries=0,
    )

    with pytest.raises(ValidationError):
        await agent.run("Return weather.", deps=None)

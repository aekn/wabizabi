from __future__ import annotations

from typing import cast

import pytest
from pydantic import BaseModel, ValidationError
from wabizabi.context import RunContext
from wabizabi.messages import ModelResponse, NativeOutputPart, TextPart, ToolCallPart
from wabizabi.output import (
    JsonOutputDecoder,
    OutputConfig,
    OutputDecoder,
    OutputDecodingError,
    OutputMode,
    OutputPipeline,
    OutputValidationError,
    SchemaOutputDecoder,
    TextOutputDecoder,
    infer_output_mode,
    json_output_config,
    json_output_decoder,
    schema_output_config,
    schema_output_decoder,
    text_output_config,
    tool_output_config,
    tool_output_decoder,
)
from wabizabi.usage import RunUsage


class CityAnswer(BaseModel):
    city: str
    country: str


class MathAnswer(BaseModel):
    total: int


class StrictMathAnswer(BaseModel):
    total: int
    label: str


def make_context() -> RunContext[tuple[str, str]]:
    return RunContext(
        deps=("svc", "cfg"),
        run_id="run-1",
        run_step=2,
        usage=RunUsage(input_tokens=3, output_tokens=4),
        metadata={"request_id": "req-1"},
    )


def test_output_validation_error_defaults_retry_feedback() -> None:
    error = OutputValidationError("bad output")

    assert str(error) == "bad output"
    assert error.retry_feedback == "bad output"


def test_output_validation_error_accepts_explicit_retry_feedback() -> None:
    error = OutputValidationError(
        "bad output",
        retry_feedback="Please follow the required output contract.",
    )

    assert str(error) == "bad output"
    assert error.retry_feedback == "Please follow the required output contract."


def _output_config(value: object) -> OutputConfig[object]:
    assert isinstance(value, OutputConfig)
    config: OutputConfig[object] = value  # type: ignore[assignment]
    return config


@pytest.mark.parametrize(
    ("config", "expected_mode", "expected_response_format"),
    [
        (text_output_config(), OutputMode.TEXT, None),
        (json_output_config(), OutputMode.JSON, "json"),
        (schema_output_config(CityAnswer), OutputMode.SCHEMA, "object"),
        (tool_output_config(MathAnswer, tool_name="final_answer"), OutputMode.TOOL, None),
    ],
)
def test_output_configs_expose_expected_mode_and_format(
    config: object,
    expected_mode: OutputMode,
    expected_response_format: str | None,
) -> None:
    typed_config = _output_config(config)
    config_mode = typed_config.mode
    assert config_mode is expected_mode

    response_format = typed_config.response_format
    if expected_response_format is None:
        assert response_format is None
    elif isinstance(response_format, str):
        assert response_format == expected_response_format
    else:
        assert isinstance(response_format, dict)
        assert response_format["type"] == expected_response_format


def test_tool_output_config_marks_terminal_tool_name() -> None:
    config = tool_output_config(MathAnswer, tool_name="final_answer")

    assert config.terminal_tool_names == frozenset(("final_answer",))


def _output_decoder(value: object) -> OutputDecoder[object]:
    return cast(OutputDecoder[object], value)


@pytest.mark.parametrize(
    ("decoder", "expected_mode"),
    [
        (TextOutputDecoder(), OutputMode.TEXT),
        (JsonOutputDecoder(), OutputMode.JSON),
        (SchemaOutputDecoder(output_type=CityAnswer), OutputMode.SCHEMA),
        (tool_output_decoder(MathAnswer, tool_name="final_answer"), OutputMode.TOOL),
    ],
)
def test_infer_output_mode_matches_builtin_decoders(
    decoder: object,
    expected_mode: OutputMode,
) -> None:
    assert infer_output_mode(_output_decoder(decoder)) is expected_mode


def test_text_output_decoder_joins_text_parts_with_separator() -> None:
    decoder = TextOutputDecoder(separator=" | ")
    response = ModelResponse(
        parts=(TextPart(text="hello"), TextPart(text="world")),
        model_name="demo",
    )

    assert decoder.decode(response) == "hello | world"


def test_text_output_decoder_rejects_missing_text_parts() -> None:
    decoder = TextOutputDecoder()
    response = ModelResponse(
        parts=(NativeOutputPart(data={"status": "ok"}),),
        model_name="demo",
    )

    with pytest.raises(OutputDecodingError, match="did not contain any text parts"):
        decoder.decode(response)


def test_json_output_decoder_builds_json_value_from_text() -> None:
    decoder = json_output_decoder()
    response = ModelResponse(
        parts=(TextPart(text='{"city":"Paris","country":"France"}'),),
        model_name="demo",
    )

    assert decoder.decode(response) == {"city": "Paris", "country": "France"}


def test_json_output_decoder_accepts_native_output_parts() -> None:
    decoder = JsonOutputDecoder()
    response = ModelResponse(
        parts=(NativeOutputPart(data={"city": "Paris", "country": "France"}),),
        model_name="demo",
    )

    assert decoder.decode(response) == {"city": "Paris", "country": "France"}


def test_json_output_decoder_rejects_multiple_native_output_parts() -> None:
    decoder = JsonOutputDecoder()
    response = ModelResponse(
        parts=(
            NativeOutputPart(data={"city": "Paris"}),
            NativeOutputPart(data={"city": "London"}),
        ),
        model_name="demo",
    )

    with pytest.raises(OutputDecodingError, match="multiple native output parts"):
        decoder.decode(response)


def test_json_output_decoder_rejects_invalid_json_text() -> None:
    decoder = json_output_decoder()
    response = ModelResponse(
        parts=(TextPart(text="not-json"),),
        model_name="demo",
    )

    with pytest.raises(OutputDecodingError, match="valid structured JSON output"):
        decoder.decode(response)


def test_schema_output_decoder_builds_pydantic_model_from_text() -> None:
    decoder = schema_output_decoder(CityAnswer)
    response = ModelResponse(
        parts=(TextPart(text='{"city":"Paris","country":"France"}'),),
        model_name="demo",
    )

    assert decoder.decode(response) == CityAnswer(city="Paris", country="France")


def test_schema_output_decoder_accepts_native_output_parts() -> None:
    decoder = SchemaOutputDecoder(output_type=CityAnswer)
    response = ModelResponse(
        parts=(NativeOutputPart(data={"city": "Paris", "country": "France"}),),
        model_name="demo",
    )

    assert decoder.decode(response) == CityAnswer(city="Paris", country="France")


def test_schema_output_decoder_propagates_model_validation_error() -> None:
    decoder = schema_output_decoder(StrictMathAnswer)
    response = ModelResponse(
        parts=(TextPart(text='{"total":42}'),),
        model_name="demo",
    )

    with pytest.raises(ValidationError):
        decoder.decode(response)


def test_tool_output_decoder_builds_pydantic_model_from_tool_arguments() -> None:
    decoder = tool_output_decoder(MathAnswer, tool_name="final_answer")
    response = ModelResponse(
        parts=(
            ToolCallPart(
                tool_name="final_answer",
                call_id="call-1",
                arguments={"total": 42},
            ),
        ),
        model_name="demo",
    )

    assert decoder.decode(response) == MathAnswer(total=42)


def test_tool_output_decoder_rejects_missing_tool_calls() -> None:
    decoder = tool_output_decoder(MathAnswer, tool_name="final_answer")
    response = ModelResponse(parts=(TextPart(text="42"),), model_name="demo")

    with pytest.raises(OutputDecodingError, match="did not contain a tool call"):
        decoder.decode(response)


def test_tool_output_decoder_requires_tool_name_when_multiple_tool_calls_exist() -> None:
    decoder = tool_output_decoder(MathAnswer)
    response = ModelResponse(
        parts=(
            ToolCallPart(tool_name="first", call_id="call-1", arguments={"total": 1}),
            ToolCallPart(tool_name="second", call_id="call-2", arguments={"total": 2}),
        ),
        model_name="demo",
    )

    with pytest.raises(OutputDecodingError, match="multiple tool calls"):
        decoder.decode(response)


def test_tool_output_decoder_rejects_multiple_matching_named_tool_calls() -> None:
    decoder = tool_output_decoder(MathAnswer, tool_name="final_answer")
    response = ModelResponse(
        parts=(
            ToolCallPart(
                tool_name="final_answer",
                call_id="call-1",
                arguments={"total": 1},
            ),
            ToolCallPart(
                tool_name="final_answer",
                call_id="call-2",
                arguments={"total": 2},
            ),
        ),
        model_name="demo",
    )

    with pytest.raises(OutputDecodingError, match="multiple tool calls for 'final_answer'"):
        decoder.decode(response)


def test_output_pipeline_empty_returns_same_output() -> None:
    pipeline = OutputPipeline[tuple[str, str], str].empty(mode=OutputMode.TEXT)

    assert pipeline.mode is OutputMode.TEXT
    assert pipeline.validators == ()


@pytest.mark.asyncio
async def test_sync_output_validator_can_be_passed_directly() -> None:
    def trim_output(context: RunContext[tuple[str, str]], output: str) -> str:
        assert context.run_id == "run-1"
        return output.strip()

    pipeline = (
        OutputPipeline[tuple[str, str], str].empty(mode=OutputMode.TEXT).with_validator(trim_output)
    )

    result = await pipeline.validate(make_context(), "  hello  ")

    assert result == "hello"


@pytest.mark.asyncio
async def test_async_output_validator_can_be_passed_directly() -> None:
    async def add_suffix(context: RunContext[tuple[str, str]], output: str) -> str:
        assert context.run_step == 2
        return f"{output}!"

    pipeline = (
        OutputPipeline[tuple[str, str], str].empty(mode=OutputMode.TEXT).with_validator(add_suffix)
    )

    result = await pipeline.validate(make_context(), "hello")

    assert result == "hello!"


@pytest.mark.asyncio
async def test_output_pipeline_runs_validators_in_order() -> None:
    order: list[str] = []

    def trim_output(context: RunContext[tuple[str, str]], output: str) -> str:
        assert context.metadata == {"request_id": "req-1"}
        order.append("trim")
        return output.strip()

    async def add_suffix(context: RunContext[tuple[str, str]], output: str) -> str:
        assert context.usage.total_tokens == 7
        order.append("suffix")
        return f"{output}!"

    pipeline = (
        OutputPipeline[tuple[str, str], str]
        .empty(mode=OutputMode.TEXT)
        .with_validator(trim_output)
        .with_validator(add_suffix)
    )

    result = await pipeline.validate(make_context(), "  hello  ")

    assert result == "hello!"
    assert order == ["trim", "suffix"]


@pytest.mark.asyncio
async def test_output_pipeline_accepts_sync_and_async_functions_directly() -> None:
    def trim_output(context: RunContext[tuple[str, str]], output: str) -> str:
        assert context.deps == ("svc", "cfg")
        return output.strip()

    async def add_suffix(context: RunContext[tuple[str, str]], output: str) -> str:
        assert context.run_id == "run-1"
        return f"{output}!"

    pipeline = (
        OutputPipeline[tuple[str, str], str]
        .empty(mode=OutputMode.TEXT)
        .with_validator(trim_output)
        .with_validator(add_suffix)
    )

    result = await pipeline.validate(make_context(), "  hello  ")

    assert result == "hello!"


@pytest.mark.asyncio
async def test_output_pipeline_propagates_validation_error() -> None:
    def reject_short_output(context: RunContext[tuple[str, str]], output: str) -> str:
        assert context.deps == ("svc", "cfg")
        if len(output) < 5:
            raise OutputValidationError(
                "output too short",
                retry_feedback="Please provide a more detailed answer.",
            )
        return output

    pipeline = (
        OutputPipeline[tuple[str, str], str]
        .empty(mode=OutputMode.TEXT)
        .with_validator(reject_short_output)
    )

    with pytest.raises(OutputValidationError, match="output too short"):
        await pipeline.validate(make_context(), "hey")

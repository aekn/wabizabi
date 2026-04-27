"""Output decoding and retry helpers."""

from __future__ import annotations

from wabizabi.context import RunContext
from wabizabi.messages import ModelResponse
from wabizabi.output import (
    OutputConfig,
    OutputDecodingError,
    OutputMode,
    OutputPipeline,
)


def output_decoding_retry_feedback(error: OutputDecodingError, *, mode: OutputMode) -> str:
    """Create retry guidance for an output decoding failure."""
    if mode is OutputMode.TEXT:
        requirement = (
            "Return a final answer as plain text. Do not emit only reasoning or tool calls."
        )
    elif mode is OutputMode.TOOL:
        requirement = "Return exactly one final output tool call in the required shape."
    elif mode is OutputMode.JSON:
        requirement = "Return exactly one valid JSON value and nothing else."
    else:
        requirement = "Return exactly one JSON object matching the required schema."
    return f"{requirement} Previous decoding error: {error}."


async def decode_and_validate_output[AgentDepsT, OutputDataT](
    *,
    output: OutputConfig[OutputDataT],
    output_pipeline: OutputPipeline[AgentDepsT, OutputDataT],
    response: ModelResponse,
    context: RunContext[AgentDepsT],
) -> OutputDataT:
    """Decode and validate the final model response."""
    decoded_output = output.decoder.decode(response)
    return await output_pipeline.validate(context, decoded_output)


__all__ = ["decode_and_validate_output", "output_decoding_retry_feedback"]

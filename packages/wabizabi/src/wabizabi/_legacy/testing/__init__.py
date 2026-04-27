"""Public testing helpers and test models."""

from wabizabi.testing.builders import (
    refusal_response,
    refusal_result,
    response_message,
    response_result,
    stream_script_from_result,
    text_response,
    text_result,
    tool_call_response,
    tool_call_result,
)
from wabizabi.testing.compliance import (
    ModelContractCapture,
    StreamCapture,
    assert_model_contract_is_self_consistent,
    assert_model_contract_matches_expected,
    assert_model_result_matches_expected,
    assert_stream_matches_result,
    collect_model_contract_capture,
    collect_stream_capture,
)
from wabizabi.testing.models import CapturedModelCall, ScriptedModel, StreamingScriptedModel

__all__ = [
    "CapturedModelCall",
    "ModelContractCapture",
    "ScriptedModel",
    "StreamCapture",
    "StreamingScriptedModel",
    "assert_model_contract_is_self_consistent",
    "assert_model_contract_matches_expected",
    "assert_model_result_matches_expected",
    "assert_stream_matches_result",
    "collect_model_contract_capture",
    "collect_stream_capture",
    "refusal_response",
    "refusal_result",
    "response_message",
    "response_result",
    "stream_script_from_result",
    "text_response",
    "text_result",
    "tool_call_response",
    "tool_call_result",
]

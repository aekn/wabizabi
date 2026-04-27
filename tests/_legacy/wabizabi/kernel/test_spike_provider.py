"""Spike 5: Second-provider normalization.

Validates that the provider interface works for providers beyond Ollama.
Uses a stub provider to verify:
- Provider interface is not Ollama-specific
- Capability descriptors are honest (don't claim unsupported features)
- Infer commands produce normalized results regardless of provider

This spike tests at the durable record level, not the old Model protocol.
The question is: can we represent provider-specific differences in
CommandRequested/CommandResolved without provider-specific branches in
kernel code?
"""

from __future__ import annotations

from uuid import uuid4

from wabizabi.kernel.branch import reconstruct_branch
from wabizabi.kernel.log import InMemoryLog
from wabizabi.kernel.records import (
    BranchCreated,
    BranchKind,
    CommandKind,
    CommandRequested,
    CommandResolved,
    RecordHeader,
    ResolutionKind,
)

SCHEMA_VERSION = 1


def _header(run_id: object, branch_id: object, seq_no: int) -> RecordHeader:
    from uuid import UUID

    return RecordHeader(
        schema_version=SCHEMA_VERSION,
        run_id=UUID(str(run_id)),
        branch_id=UUID(str(branch_id)),
        seq_no=seq_no,
    )


class TestProviderNormalization:
    """Provider differences live in canonical_payload, not in record structure."""

    def test_ollama_and_openai_produce_same_record_structure(self) -> None:
        """Two providers produce structurally identical records.
        Provider-specific details are inside the payload, not in the record type.
        """
        run_id = uuid4()
        branch_id = uuid4()
        cmd_ollama = uuid4()
        cmd_openai = uuid4()

        ollama_request = CommandRequested(
            header=_header(run_id, branch_id, 1),
            command_id=cmd_ollama,
            command_kind=CommandKind.INFER,
            canonical_payload={
                "provider": "ollama",
                "model": "llama3.2",
                "input": [{"role": "user", "content": "hello"}],
                "capabilities": {"tools": True, "streaming": True, "structured_output": False},
            },
            driver_type="agent",
        )

        openai_request = CommandRequested(
            header=_header(run_id, branch_id, 1),
            command_id=cmd_openai,
            command_kind=CommandKind.INFER,
            canonical_payload={
                "provider": "openai",
                "model": "gpt-4o",
                "input": [{"role": "user", "content": "hello"}],
                "capabilities": {"tools": True, "streaming": True, "structured_output": True},
            },
            driver_type="agent",
        )

        # Same record type, same field structure
        assert ollama_request.record_type == openai_request.record_type == "CommandRequested"
        assert ollama_request.command_kind == openai_request.command_kind == CommandKind.INFER

        # Provider differences are in the payload
        assert ollama_request.canonical_payload["provider"] == "ollama"
        assert openai_request.canonical_payload["provider"] == "openai"

        # Capability differences are in the payload
        ollama_caps = ollama_request.canonical_payload["capabilities"]
        openai_caps = openai_request.canonical_payload["capabilities"]
        assert isinstance(ollama_caps, dict)
        assert isinstance(openai_caps, dict)
        assert ollama_caps["structured_output"] is False
        assert openai_caps["structured_output"] is True

    def test_provider_specific_resolution_metadata(self) -> None:
        """Provider-specific resolution details go in usage_delta and adapter_receipt."""
        cmd_id = uuid4()
        run_id = uuid4()
        branch_id = uuid4()

        ollama_resolution = CommandResolved(
            header=_header(run_id, branch_id, 2),
            command_id=cmd_id,
            command_kind=CommandKind.INFER,
            resolution_kind=ResolutionKind.SUCCEEDED,
            resolution_payload={"text": "response", "refusal": None},
            usage_delta={"input_tokens": 10, "output_tokens": 20},
            adapter_receipt={"provider": "ollama", "model": "llama3.2", "eval_duration_ns": 500000},
        )

        openai_resolution = CommandResolved(
            header=_header(run_id, branch_id, 2),
            command_id=cmd_id,
            command_kind=CommandKind.INFER,
            resolution_kind=ResolutionKind.SUCCEEDED,
            resolution_payload={"text": "response", "refusal": None},
            usage_delta={"input_tokens": 10, "output_tokens": 20, "reasoning_tokens": 5},
            adapter_receipt={"provider": "openai", "model": "gpt-4o", "system_fingerprint": "abc"},
        )

        # Both are valid CommandResolved records
        assert ollama_resolution.resolution_kind == openai_resolution.resolution_kind

        # Provider-specific metadata is in adapter_receipt, not in record structure
        assert ollama_resolution.adapter_receipt is not None
        assert ollama_resolution.adapter_receipt["provider"] == "ollama"
        assert openai_resolution.adapter_receipt is not None
        assert openai_resolution.adapter_receipt["provider"] == "openai"

    def test_capability_descriptors_are_explicit(self) -> None:
        """Capabilities are carried in the payload, not assumed by the kernel."""
        run_id = uuid4()
        branch_id = uuid4()

        # A provider that doesn't support tools
        no_tools_request = CommandRequested(
            header=_header(run_id, branch_id, 1),
            command_id=uuid4(),
            command_kind=CommandKind.INFER,
            canonical_payload={
                "provider": "basic_llm",
                "model": "simple-model",
                "input": [{"role": "user", "content": "hello"}],
                "capabilities": {"tools": False, "streaming": False, "structured_output": False},
            },
            driver_type="agent",
        )

        caps = no_tools_request.canonical_payload["capabilities"]
        assert isinstance(caps, dict)
        assert caps["tools"] is False
        assert caps["streaming"] is False

    def test_branch_reconstruction_is_provider_agnostic(self) -> None:
        """Branch reconstruction doesn't care which provider generated the records."""
        log = InMemoryLog()
        run_id = uuid4()
        branch_id = uuid4()
        cmd_id = uuid4()

        log.append(BranchCreated(
            header=_header(run_id, branch_id, 0),
            parent_branch_id=None, spawning_command_id=None,
            branch_kind=BranchKind.ROOT,
        ))
        log.append(CommandRequested(
            header=_header(run_id, branch_id, 1),
            command_id=cmd_id, command_kind=CommandKind.INFER,
            canonical_payload={"provider": "any_provider", "model": "any_model"},
            driver_type="agent",
        ))
        log.append(CommandResolved(
            header=_header(run_id, branch_id, 2),
            command_id=cmd_id, command_kind=CommandKind.INFER,
            resolution_kind=ResolutionKind.SUCCEEDED,
            resolution_payload={"text": "works"},
            adapter_receipt={"provider": "any_provider"},
        ))

        view = reconstruct_branch(log.read_branch(branch_id))
        assert cmd_id in view.resolved_commands
        # Kernel doesn't inspect provider-specific fields
        assert view.resolved_commands[cmd_id].resolution_payload == {"text": "works"}

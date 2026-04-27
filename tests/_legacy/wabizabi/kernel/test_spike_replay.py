"""Spike 1: Durable log + replay boundary.

Validates the core assumption: we can append command requests and resolutions
to an ordered log, then reconstruct branch state from that log alone,
injecting committed results instead of re-executing.

Also validates:
- Branch lifecycle state transitions are derived from records
- Replay produces identical branch view from the same records
- Terminal state is only real when TerminalCommitted exists
"""

from __future__ import annotations

from uuid import uuid4

from wabizabi.kernel.branch import BranchState, BranchView, reconstruct_branch
from wabizabi.kernel.log import InMemoryLog
from wabizabi.kernel.records import (
    BranchCreated,
    BranchKind,
    CommandKind,
    CommandRequested,
    CommandResolved,
    RecordHeader,
    RecoveryMode,
    ReplayPolicy,
    ResolutionKind,
    TerminalCommitted,
    TerminalOutcome,
    WaitArmed,
    WaitKind,
    WaitResolved,
)

SCHEMA_VERSION = 1


def _header(run_id: object, branch_id: object, seq_no: int, **kw: object) -> RecordHeader:
    from uuid import UUID

    return RecordHeader(
        schema_version=SCHEMA_VERSION,
        run_id=UUID(str(run_id)),
        branch_id=UUID(str(branch_id)),
        seq_no=seq_no,
        **kw,  # type: ignore[arg-type]
    )


class TestDurableLogAppendAndRead:
    """Log stores records and retrieves them in order."""

    def test_append_and_read_branch(self) -> None:
        log = InMemoryLog()
        run_id = uuid4()
        branch_id = uuid4()

        rec = BranchCreated(
            header=_header(run_id, branch_id, 0),
            parent_branch_id=None,
            spawning_command_id=None,
            branch_kind=BranchKind.ROOT,
        )
        log.append(rec)

        records = log.read_branch(branch_id)
        assert len(records) == 1
        assert records[0] is rec

    def test_read_branch_filters_by_branch_id(self) -> None:
        log = InMemoryLog()
        run_id = uuid4()
        b1, b2 = uuid4(), uuid4()

        log.append(
            BranchCreated(
                header=_header(run_id, b1, 0),
                parent_branch_id=None,
                spawning_command_id=None,
                branch_kind=BranchKind.ROOT,
            )
        )
        log.append(
            BranchCreated(
                header=_header(run_id, b2, 0),
                parent_branch_id=b1,
                spawning_command_id=None,
                branch_kind=BranchKind.SPAWNED,
            )
        )

        assert len(log.read_branch(b1)) == 1
        assert len(log.read_branch(b2)) == 1

    def test_read_branch_from_filters_by_seq_no(self) -> None:
        log = InMemoryLog()
        run_id = uuid4()
        branch_id = uuid4()

        for i in range(5):
            if i == 0:
                log.append(
                    BranchCreated(
                        header=_header(run_id, branch_id, i),
                        parent_branch_id=None,
                        spawning_command_id=None,
                        branch_kind=BranchKind.ROOT,
                    )
                )
            else:
                log.append(
                    CommandRequested(
                        header=_header(run_id, branch_id, i),
                        command_id=uuid4(),
                        command_kind=CommandKind.INFER,
                        canonical_payload={"step": i},
                        driver_type="agent",
                    )
                )

        assert len(log.read_branch_from(branch_id, 3)) == 2


class TestBranchReconstruction:
    """Branch state is reconstructed from records alone."""

    def test_reconstruct_from_branch_created(self) -> None:
        run_id = uuid4()
        branch_id = uuid4()

        records = (
            BranchCreated(
                header=_header(run_id, branch_id, 0),
                parent_branch_id=None,
                spawning_command_id=None,
                branch_kind=BranchKind.ROOT,
            ),
        )

        view = reconstruct_branch(records)
        assert view.branch_id == branch_id
        assert view.run_id == run_id
        assert view.state == BranchState.RUNNING
        assert view.terminal_outcome is None
        assert view.head_seq_no == 0

    def test_command_lifecycle_tracked(self) -> None:
        """CommandRequested → pending, CommandResolved → resolved."""
        run_id = uuid4()
        branch_id = uuid4()
        cmd_id = uuid4()

        records = (
            BranchCreated(
                header=_header(run_id, branch_id, 0),
                parent_branch_id=None,
                spawning_command_id=None,
                branch_kind=BranchKind.ROOT,
            ),
            CommandRequested(
                header=_header(run_id, branch_id, 1),
                command_id=cmd_id,
                command_kind=CommandKind.INFER,
                canonical_payload={"model": "test", "input": "hello"},
                driver_type="agent",
            ),
        )

        view = reconstruct_branch(records)
        assert cmd_id in view.pending_commands
        assert cmd_id not in view.resolved_commands

        # Now add resolution
        records_with_resolution = records + (
            CommandResolved(
                header=_header(run_id, branch_id, 2),
                command_id=cmd_id,
                command_kind=CommandKind.INFER,
                resolution_kind=ResolutionKind.SUCCEEDED,
                resolution_payload={"text": "world"},
                usage_delta={"input_tokens": 5, "output_tokens": 3},
            ),
        )

        view2 = reconstruct_branch(records_with_resolution)
        assert cmd_id not in view2.pending_commands
        assert cmd_id in view2.resolved_commands
        assert view2.resolved_commands[cmd_id].resolution_payload == {"text": "world"}

    def test_terminal_state_requires_terminal_committed(self) -> None:
        """Branch is not terminal until TerminalCommitted record exists."""
        run_id = uuid4()
        branch_id = uuid4()
        cmd_id = uuid4()

        records_no_terminal = (
            BranchCreated(
                header=_header(run_id, branch_id, 0),
                parent_branch_id=None,
                spawning_command_id=None,
                branch_kind=BranchKind.ROOT,
            ),
            CommandRequested(
                header=_header(run_id, branch_id, 1),
                command_id=cmd_id,
                command_kind=CommandKind.COMPLETE,
                canonical_payload={"output": "done"},
                driver_type="agent",
            ),
            CommandResolved(
                header=_header(run_id, branch_id, 2),
                command_id=cmd_id,
                command_kind=CommandKind.COMPLETE,
                resolution_kind=ResolutionKind.TERMINAL_CANDIDATE,
            ),
        )

        view = reconstruct_branch(records_no_terminal)
        assert view.state == BranchState.RUNNING  # Not terminal yet!
        assert view.terminal_outcome is None

        records_with_terminal = records_no_terminal + (
            TerminalCommitted(
                header=_header(run_id, branch_id, 3),
                terminal_outcome=TerminalOutcome.COMPLETED,
                output_payload="done",
            ),
        )

        view2 = reconstruct_branch(records_with_terminal)
        assert view2.state == BranchState.TERMINAL
        assert view2.terminal_outcome == TerminalOutcome.COMPLETED
        assert view2.output_payload == "done"


class TestReplayInjection:
    """Replay reconstructs the same view and provides committed results."""

    def test_replay_produces_identical_view(self) -> None:
        """Same records → same branch view. This is the replay contract."""
        run_id = uuid4()
        branch_id = uuid4()
        cmd1 = uuid4()
        cmd2 = uuid4()

        records = (
            BranchCreated(
                header=_header(run_id, branch_id, 0),
                parent_branch_id=None,
                spawning_command_id=None,
                branch_kind=BranchKind.ROOT,
            ),
            CommandRequested(
                header=_header(run_id, branch_id, 1),
                command_id=cmd1,
                command_kind=CommandKind.INFER,
                canonical_payload={"input": "question"},
                driver_type="agent",
            ),
            CommandResolved(
                header=_header(run_id, branch_id, 2),
                command_id=cmd1,
                command_kind=CommandKind.INFER,
                resolution_kind=ResolutionKind.SUCCEEDED,
                resolution_payload={"text": "answer"},
            ),
            CommandRequested(
                header=_header(run_id, branch_id, 3),
                command_id=cmd2,
                command_kind=CommandKind.ACT,
                canonical_payload={"action": "save"},
                driver_type="agent",
                operation_id="save-001",
                recovery_mode=RecoveryMode.SAFE_RETRY,
            ),
            CommandResolved(
                header=_header(run_id, branch_id, 4),
                command_id=cmd2,
                command_kind=CommandKind.ACT,
                resolution_kind=ResolutionKind.SUCCEEDED,
                resolution_payload={"saved": True},
                adapter_receipt={"executor": "file_system", "op_id": "save-001"},
            ),
            TerminalCommitted(
                header=_header(run_id, branch_id, 5),
                terminal_outcome=TerminalOutcome.COMPLETED,
                output_payload="task done",
            ),
        )

        # Replay twice from same records
        view_a = reconstruct_branch(records)
        view_b = reconstruct_branch(records)

        assert view_a.state == view_b.state == BranchState.TERMINAL
        assert view_a.terminal_outcome == view_b.terminal_outcome == TerminalOutcome.COMPLETED
        assert view_a.head_seq_no == view_b.head_seq_no == 5
        assert view_a.output_payload == view_b.output_payload == "task done"

        # Resolved commands carry committed results for injection
        assert cmd1 in view_a.resolved_commands
        assert view_a.resolved_commands[cmd1].resolution_payload == {"text": "answer"}
        assert cmd2 in view_a.resolved_commands
        assert view_a.resolved_commands[cmd2].resolution_payload == {"saved": True}
        assert view_a.resolved_commands[cmd2].adapter_receipt == {
            "executor": "file_system",
            "op_id": "save-001",
        }

    def test_replay_from_log_matches_direct_construction(self) -> None:
        """Records written to log, then read back, produce same view."""
        log = InMemoryLog()
        run_id = uuid4()
        branch_id = uuid4()
        cmd_id = uuid4()

        records = [
            BranchCreated(
                header=_header(run_id, branch_id, 0),
                parent_branch_id=None,
                spawning_command_id=None,
                branch_kind=BranchKind.ROOT,
            ),
            CommandRequested(
                header=_header(run_id, branch_id, 1),
                command_id=cmd_id,
                command_kind=CommandKind.INFER,
                canonical_payload={"input": "test"},
                driver_type="agent",
            ),
            CommandResolved(
                header=_header(run_id, branch_id, 2),
                command_id=cmd_id,
                command_kind=CommandKind.INFER,
                resolution_kind=ResolutionKind.SUCCEEDED,
                resolution_payload={"text": "response"},
            ),
        ]

        for r in records:
            log.append(r)

        # Reconstruct from log
        from_log = reconstruct_branch(log.read_branch(branch_id))
        # Reconstruct from direct tuple
        from_tuple = reconstruct_branch(tuple(records))

        assert from_log.state == from_tuple.state == BranchState.RUNNING
        assert from_log.head_seq_no == from_tuple.head_seq_no == 2
        assert cmd_id in from_log.resolved_commands
        assert from_log.resolved_commands[cmd_id].resolution_payload == {"text": "response"}


class TestWaitLifecycle:
    """Wait records transition branch state correctly."""

    def test_wait_arms_and_resolves(self) -> None:
        run_id = uuid4()
        branch_id = uuid4()
        wait_id = uuid4()

        records_armed = (
            BranchCreated(
                header=_header(run_id, branch_id, 0),
                parent_branch_id=None,
                spawning_command_id=None,
                branch_kind=BranchKind.ROOT,
            ),
            WaitArmed(
                header=_header(run_id, branch_id, 1),
                wait_id=wait_id,
                wait_kind=WaitKind.SIGNAL,
                arm_payload={"signal_name": "user_input"},
            ),
        )

        view = reconstruct_branch(records_armed)
        assert view.state == BranchState.WAITING_EXTERNAL
        assert wait_id in view.pending_waits

        records_resolved = records_armed + (
            WaitResolved(
                header=_header(run_id, branch_id, 2),
                wait_id=wait_id,
                wait_kind=WaitKind.SIGNAL,
                resolution_payload={"value": "user said hello"},
            ),
        )

        view2 = reconstruct_branch(records_resolved)
        assert view2.state == BranchState.RUNNING
        assert wait_id not in view2.pending_waits


class TestActRecoveryMetadata:
    """Act commands carry recovery metadata in durable records."""

    def test_act_carries_operation_id_and_recovery_mode(self) -> None:
        run_id = uuid4()
        branch_id = uuid4()
        cmd_id = uuid4()

        records = (
            BranchCreated(
                header=_header(run_id, branch_id, 0),
                parent_branch_id=None,
                spawning_command_id=None,
                branch_kind=BranchKind.ROOT,
            ),
            CommandRequested(
                header=_header(run_id, branch_id, 1),
                command_id=cmd_id,
                command_kind=CommandKind.ACT,
                canonical_payload={"action": "deploy", "target": "prod"},
                driver_type="agent",
                operation_id="deploy-v2-001",
                recovery_mode=RecoveryMode.RECONCILE,
            ),
        )

        view = reconstruct_branch(records)
        pending = view.pending_commands[cmd_id]
        assert pending.operation_id == "deploy-v2-001"
        assert pending.recovery_mode == RecoveryMode.RECONCILE

    def test_act_resolution_carries_adapter_receipt(self) -> None:
        run_id = uuid4()
        branch_id = uuid4()
        cmd_id = uuid4()

        records = (
            BranchCreated(
                header=_header(run_id, branch_id, 0),
                parent_branch_id=None,
                spawning_command_id=None,
                branch_kind=BranchKind.ROOT,
            ),
            CommandRequested(
                header=_header(run_id, branch_id, 1),
                command_id=cmd_id,
                command_kind=CommandKind.ACT,
                canonical_payload={"action": "write_file"},
                driver_type="agent",
                operation_id="write-001",
                recovery_mode=RecoveryMode.SAFE_RETRY,
            ),
            CommandResolved(
                header=_header(run_id, branch_id, 2),
                command_id=cmd_id,
                command_kind=CommandKind.ACT,
                resolution_kind=ResolutionKind.SUCCEEDED,
                resolution_payload={"bytes_written": 1024},
                adapter_receipt={"executor": "fs", "path": "/tmp/out.txt", "op_id": "write-001"},
            ),
        )

        view = reconstruct_branch(records)
        resolved = view.resolved_commands[cmd_id]
        assert resolved.adapter_receipt is not None
        assert resolved.adapter_receipt["op_id"] == "write-001"

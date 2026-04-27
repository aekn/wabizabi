"""Spike 3: Uncertain Act recovery.

Validates the three recovery modes when a crash occurs after an Act command
request is persisted but before its resolution is committed:

- safe_retry: the executor declares it's safe to retry. Runtime retries.
- reconcile: the executor provides a reconciler that checks actual state.
  Runtime calls reconciler to determine outcome.
- manual: runtime cannot safely proceed. Branch blocks for human resolution.

The critical gap: between CommandRequested and CommandResolved for an Act,
the external world may have been mutated. The runtime must never guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from uuid import UUID, uuid4

from wabizabi.kernel.branch import BranchState, reconstruct_branch
from wabizabi.kernel.log import InMemoryLog
from wabizabi.kernel.records import (
    BranchCreated,
    BranchKind,
    CommandKind,
    CommandRequested,
    CommandResolved,
    RecordHeader,
    RecoveryMode,
    ResolutionKind,
    WaitArmed,
    WaitKind,
)

SCHEMA_VERSION = 1


def _header(run_id: UUID, branch_id: UUID, seq_no: int) -> RecordHeader:
    return RecordHeader(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        branch_id=branch_id,
        seq_no=seq_no,
    )


# ---------------------------------------------------------------------------
# Recovery executor spike
# ---------------------------------------------------------------------------


class RecoveryAction(Enum):
    RETRY = "retry"
    RECONCILE = "reconcile"
    BLOCK = "block"


@dataclass
class RecoveryDecision:
    action: RecoveryAction
    reconciled_result: object | None = None


def decide_recovery(
    pending_act: CommandRequested,
    *,
    reconciler: object | None = None,
) -> RecoveryDecision:
    """Decide what to do with a pending Act after crash.

    This is the kernel's recovery decision function. It uses the
    executor-declared recovery_mode on the command to decide.
    """
    if pending_act.command_kind != CommandKind.ACT:
        raise ValueError("Recovery only applies to Act commands.")

    match pending_act.recovery_mode:
        case RecoveryMode.SAFE_RETRY:
            return RecoveryDecision(action=RecoveryAction.RETRY)

        case RecoveryMode.RECONCILE:
            if reconciler is None:
                # No reconciler provided — fall back to manual
                return RecoveryDecision(action=RecoveryAction.BLOCK)
            # Call reconciler to check actual state
            result = _call_reconciler(reconciler, pending_act)
            return RecoveryDecision(action=RecoveryAction.RECONCILE, reconciled_result=result)

        case RecoveryMode.MANUAL:
            return RecoveryDecision(action=RecoveryAction.BLOCK)

        case None:
            # No recovery mode declared — must block
            return RecoveryDecision(action=RecoveryAction.BLOCK)


def _call_reconciler(reconciler: object, pending: CommandRequested) -> object:
    """Simulate calling a reconciler function."""
    if callable(reconciler):
        return reconciler(pending.canonical_payload, pending.operation_id)
    raise TypeError("Reconciler must be callable.")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_crash_scenario(
    recovery_mode: RecoveryMode,
) -> tuple[InMemoryLog, UUID, UUID, UUID]:
    """Create a scenario where Act was requested but not resolved (crash)."""
    log = InMemoryLog()
    run_id = uuid4()
    branch_id = uuid4()
    cmd_id = uuid4()

    log.append(
        BranchCreated(
            header=_header(run_id, branch_id, 0),
            parent_branch_id=None,
            spawning_command_id=None,
            branch_kind=BranchKind.ROOT,
        )
    )
    log.append(
        CommandRequested(
            header=_header(run_id, branch_id, 1),
            command_id=cmd_id,
            command_kind=CommandKind.ACT,
            canonical_payload={"action": "deploy", "version": "2.0"},
            driver_type="agent",
            operation_id="deploy-v2",
            recovery_mode=recovery_mode,
        )
    )

    return log, run_id, branch_id, cmd_id


class TestSafeRetryRecovery:
    """safe_retry: executor says retry is safe. Runtime retries the Act."""

    def test_safe_retry_decides_retry(self) -> None:
        log, run_id, branch_id, cmd_id = _make_crash_scenario(RecoveryMode.SAFE_RETRY)

        view = reconstruct_branch(log.read_branch(branch_id))
        assert cmd_id in view.pending_commands

        pending = view.pending_commands[cmd_id]
        decision = decide_recovery(pending)
        assert decision.action == RecoveryAction.RETRY

    def test_safe_retry_then_commit(self) -> None:
        """After retry succeeds, resolution is committed normally."""
        log, run_id, branch_id, cmd_id = _make_crash_scenario(RecoveryMode.SAFE_RETRY)

        view = reconstruct_branch(log.read_branch(branch_id))
        decision = decide_recovery(view.pending_commands[cmd_id])
        assert decision.action == RecoveryAction.RETRY

        # Simulate successful retry — commit resolution
        log.append(
            CommandResolved(
                header=_header(run_id, branch_id, 2),
                command_id=cmd_id,
                command_kind=CommandKind.ACT,
                resolution_kind=ResolutionKind.SUCCEEDED,
                resolution_payload={"deployed": True},
                adapter_receipt={"op_id": "deploy-v2", "retried": True},
            )
        )

        view2 = reconstruct_branch(log.read_branch(branch_id))
        assert cmd_id in view2.resolved_commands
        assert cmd_id not in view2.pending_commands


class TestReconcileRecovery:
    """reconcile: executor provides a reconciler to check actual state."""

    def test_reconcile_calls_reconciler(self) -> None:
        log, run_id, branch_id, cmd_id = _make_crash_scenario(RecoveryMode.RECONCILE)

        view = reconstruct_branch(log.read_branch(branch_id))
        pending = view.pending_commands[cmd_id]

        # Reconciler checks actual state and reports it succeeded
        def my_reconciler(payload: object, op_id: object) -> dict[str, object]:
            return {"already_deployed": True, "version": "2.0"}

        decision = decide_recovery(pending, reconciler=my_reconciler)
        assert decision.action == RecoveryAction.RECONCILE
        assert decision.reconciled_result == {"already_deployed": True, "version": "2.0"}

    def test_reconcile_without_reconciler_blocks(self) -> None:
        """If no reconciler is available, fall back to manual block."""
        log, run_id, branch_id, cmd_id = _make_crash_scenario(RecoveryMode.RECONCILE)

        view = reconstruct_branch(log.read_branch(branch_id))
        pending = view.pending_commands[cmd_id]

        decision = decide_recovery(pending, reconciler=None)
        assert decision.action == RecoveryAction.BLOCK

    def test_reconcile_result_can_be_committed(self) -> None:
        """Reconciled result is committed as the command resolution."""
        log, run_id, branch_id, cmd_id = _make_crash_scenario(RecoveryMode.RECONCILE)

        view = reconstruct_branch(log.read_branch(branch_id))
        pending = view.pending_commands[cmd_id]

        def reconciler(payload: object, op_id: object) -> dict[str, object]:
            return {"already_deployed": True}

        decision = decide_recovery(pending, reconciler=reconciler)

        # Commit reconciled result
        log.append(
            CommandResolved(
                header=_header(run_id, branch_id, 2),
                command_id=cmd_id,
                command_kind=CommandKind.ACT,
                resolution_kind=ResolutionKind.SUCCEEDED,
                resolution_payload=decision.reconciled_result,
                adapter_receipt={"op_id": "deploy-v2", "reconciled": True},
            )
        )

        view2 = reconstruct_branch(log.read_branch(branch_id))
        assert cmd_id in view2.resolved_commands
        assert view2.resolved_commands[cmd_id].resolution_payload == {"already_deployed": True}


class TestManualRecovery:
    """manual: runtime cannot proceed. Branch blocks for human resolution."""

    def test_manual_blocks(self) -> None:
        log, run_id, branch_id, cmd_id = _make_crash_scenario(RecoveryMode.MANUAL)

        view = reconstruct_branch(log.read_branch(branch_id))
        pending = view.pending_commands[cmd_id]

        decision = decide_recovery(pending)
        assert decision.action == RecoveryAction.BLOCK

    def test_manual_block_can_be_modeled_as_wait(self) -> None:
        """Blocking for human resolution is modeled as a durable wait."""
        log, run_id, branch_id, cmd_id = _make_crash_scenario(RecoveryMode.MANUAL)

        # Arm a wait for human resolution
        wait_id = uuid4()
        log.append(
            WaitArmed(
                header=_header(run_id, branch_id, 2),
                wait_id=wait_id,
                wait_kind=WaitKind.SIGNAL,
                originating_command_id=cmd_id,
                arm_payload={
                    "reason": "manual_recovery",
                    "pending_action": "deploy",
                    "operation_id": "deploy-v2",
                },
            )
        )

        view = reconstruct_branch(log.read_branch(branch_id))
        assert view.state == BranchState.WAITING_EXTERNAL
        assert wait_id in view.pending_waits
        assert view.pending_waits[wait_id].originating_command_id == cmd_id


class TestNoRecoveryModeBlocks:
    """Commands without recovery mode must block, not guess."""

    def test_missing_recovery_mode_blocks(self) -> None:
        log = InMemoryLog()
        run_id = uuid4()
        branch_id = uuid4()
        cmd_id = uuid4()

        log.append(
            BranchCreated(
                header=_header(run_id, branch_id, 0),
                parent_branch_id=None,
                spawning_command_id=None,
                branch_kind=BranchKind.ROOT,
            )
        )
        log.append(
            CommandRequested(
                header=_header(run_id, branch_id, 1),
                command_id=cmd_id,
                command_kind=CommandKind.ACT,
                canonical_payload={"action": "something_dangerous"},
                driver_type="agent",
                operation_id="danger-001",
                # No recovery_mode!
            )
        )

        view = reconstruct_branch(log.read_branch(branch_id))
        decision = decide_recovery(view.pending_commands[cmd_id])
        assert decision.action == RecoveryAction.BLOCK

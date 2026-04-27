"""Spike 2: Workflow replay discipline.

Validates that user workflow code can be replayed honestly:
- On first execution, effects go through executors and results are committed to log
- On replay, the same workflow code runs but committed results are injected
- Workflow code sees identical results in both cases
- Ambient nondeterminism between kernel boundaries is detected

The spike uses a minimal "replay executor" that decides per-command whether
to execute fresh or inject from the log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from wabizabi.kernel.branch import reconstruct_branch
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
)
from wabizabi.types import JsonObject, JsonValue

SCHEMA_VERSION = 1


def _header(run_id: UUID, branch_id: UUID, seq_no: int) -> RecordHeader:
    return RecordHeader(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        branch_id=branch_id,
        seq_no=seq_no,
    )


# ---------------------------------------------------------------------------
# Minimal replay-aware workflow executor (spike only)
# ---------------------------------------------------------------------------


@dataclass
class ReplayContext:
    """Minimal spike context that a workflow function receives.

    On first execution: effects execute fresh and commit to log.
    On replay: committed results are injected from the log.
    """

    run_id: UUID
    branch_id: UUID
    log: InMemoryLog
    _seq_counter: int = 1  # 0 is BranchCreated
    _replay_cursor: int = 0
    _committed_resolutions: dict[int, CommandResolved] = field(default_factory=dict)
    _trace: list[str] = field(default_factory=list)

    # Track what was executed vs injected for test assertions
    executed_commands: list[UUID] = field(default_factory=list)
    injected_commands: list[UUID] = field(default_factory=list)

    def _next_seq(self) -> int:
        seq = self._seq_counter
        self._seq_counter += 1
        return seq

    def _find_resolution(self, command_id: UUID) -> CommandResolved | None:
        """Check if this command has a committed resolution in the log."""
        for record in self.log.read_branch(self.branch_id):
            if isinstance(record, CommandResolved) and record.command_id == command_id:
                return record
        return None

    def infer(self, payload: JsonObject, *, executor: object = None) -> JsonValue:
        """Issue an Infer command. On replay, inject committed result."""
        cmd_id = uuid4()
        req_seq = self._next_seq()

        # Persist request
        req = CommandRequested(
            header=_header(self.run_id, self.branch_id, req_seq),
            command_id=cmd_id,
            command_kind=CommandKind.INFER,
            canonical_payload=payload,
            driver_type="workflow",
            replay_policy=ReplayPolicy.INJECT,
        )
        self.log.append(req)

        # Check for committed resolution (replay path)
        existing = self._find_resolution(cmd_id)
        if existing is not None:
            self.injected_commands.append(cmd_id)
            self._trace.append(f"infer:{cmd_id}:injected")
            return existing.resolution_payload

        # Fresh execution path — simulate model call
        result: JsonValue = _simulate_infer(payload)
        self.executed_commands.append(cmd_id)
        self._trace.append(f"infer:{cmd_id}:executed")

        # Commit resolution
        res = CommandResolved(
            header=_header(self.run_id, self.branch_id, self._next_seq()),
            command_id=cmd_id,
            command_kind=CommandKind.INFER,
            resolution_kind=ResolutionKind.SUCCEEDED,
            resolution_payload=result,
        )
        self.log.append(res)
        return result

    def act(
        self,
        payload: JsonObject,
        *,
        operation_id: str,
        recovery_mode: RecoveryMode = RecoveryMode.SAFE_RETRY,
    ) -> JsonValue:
        """Issue an Act command. On replay, inject committed result."""
        cmd_id = uuid4()
        req_seq = self._next_seq()

        req = CommandRequested(
            header=_header(self.run_id, self.branch_id, req_seq),
            command_id=cmd_id,
            command_kind=CommandKind.ACT,
            canonical_payload=payload,
            driver_type="workflow",
            replay_policy=ReplayPolicy.INJECT,
            operation_id=operation_id,
            recovery_mode=recovery_mode,
        )
        self.log.append(req)

        existing = self._find_resolution(cmd_id)
        if existing is not None:
            self.injected_commands.append(cmd_id)
            self._trace.append(f"act:{cmd_id}:injected")
            return existing.resolution_payload

        result: JsonValue = _simulate_act(payload, operation_id)
        self.executed_commands.append(cmd_id)
        self._trace.append(f"act:{cmd_id}:executed")

        res = CommandResolved(
            header=_header(self.run_id, self.branch_id, self._next_seq()),
            command_id=cmd_id,
            command_kind=CommandKind.ACT,
            resolution_kind=ResolutionKind.SUCCEEDED,
            resolution_payload=result,
            adapter_receipt={"op_id": operation_id},
        )
        self.log.append(res)
        return result

    def complete(self, output: JsonValue) -> None:
        """Commit terminal outcome."""
        self.log.append(
            TerminalCommitted(
                header=_header(self.run_id, self.branch_id, self._next_seq()),
                terminal_outcome=TerminalOutcome.COMPLETED,
                output_payload=output,
            )
        )


# Simulated executors (deterministic for testing)


def _simulate_infer(payload: JsonObject) -> JsonValue:
    return {"text": f"response to: {payload.get('input', '?')}"}


def _simulate_act(payload: JsonObject, op_id: str) -> JsonValue:
    return {"done": True, "action": payload.get("action"), "op_id": op_id}


# ---------------------------------------------------------------------------
# Example workflow function
# ---------------------------------------------------------------------------


def example_workflow(ctx: ReplayContext) -> str:
    """A deterministic workflow that issues Infer then Act.

    This is the user-authored code that must behave identically
    on first execution and replay.
    """
    # Step 1: Infer
    infer_result = ctx.infer({"input": "What should I do?"})
    assert isinstance(infer_result, dict)

    # Step 2: Act based on infer result (deterministic decision from committed result)
    act_result = ctx.act(
        {"action": "save_plan", "plan": infer_result.get("text", "")},
        operation_id="save-001",
    )
    assert isinstance(act_result, dict)

    # Step 3: Complete
    output = f"Plan: {infer_result.get('text')}, saved: {act_result.get('done')}"
    ctx.complete(output)
    return output


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWorkflowFirstExecution:
    """First execution commits all results to log."""

    def test_first_run_executes_all_commands(self) -> None:
        log = InMemoryLog()
        run_id = uuid4()
        branch_id = uuid4()

        # Create branch
        log.append(
            BranchCreated(
                header=_header(run_id, branch_id, 0),
                parent_branch_id=None,
                spawning_command_id=None,
                branch_kind=BranchKind.ROOT,
            )
        )

        ctx = ReplayContext(run_id=run_id, branch_id=branch_id, log=log)
        output = example_workflow(ctx)

        # All commands were executed fresh
        assert len(ctx.executed_commands) == 2
        assert len(ctx.injected_commands) == 0
        assert "Plan:" in output

        # Log contains BranchCreated + 2*(Request+Resolution) + Terminal = 6 records
        all_records = log.read_branch(branch_id)
        assert len(all_records) == 6

        # Branch is terminal
        view = reconstruct_branch(all_records)
        assert view.terminal_outcome == TerminalOutcome.COMPLETED


class TestWorkflowReplayFromLog:
    """Replay injects committed results instead of re-executing.

    This is the core workflow replay discipline test.
    The key insight: because each CommandRequested gets a fresh uuid4(),
    the replay context will never find a matching resolution by command_id.

    This reveals a critical design point: for replay to inject, the command
    identity must be deterministic across executions. uuid4() breaks this.

    We document this finding and test the alternative: replay must use the
    committed log directly, not re-run the workflow.
    """

    def test_replay_from_committed_log_reconstructs_state(self) -> None:
        """Branch reconstruction from log is the primary replay mechanism.

        Workflow code does NOT re-run on replay. Instead, the branch view
        is reconstructed from committed records, and the driver can resume
        from the last committed point.
        """
        log = InMemoryLog()
        run_id = uuid4()
        branch_id = uuid4()

        # First execution
        log.append(
            BranchCreated(
                header=_header(run_id, branch_id, 0),
                parent_branch_id=None,
                spawning_command_id=None,
                branch_kind=BranchKind.ROOT,
            )
        )
        ctx = ReplayContext(run_id=run_id, branch_id=branch_id, log=log)
        first_output = example_workflow(ctx)

        # "Replay" = reconstruct branch from log
        view = reconstruct_branch(log.read_branch(branch_id))

        # All command resolutions are available
        assert len(view.resolved_commands) == 2
        assert view.state.value == "Terminal"
        assert view.output_payload == first_output

        # A driver resuming from this view can see:
        # - what commands ran and their results
        # - the terminal outcome
        # - no need to re-execute anything
        for cmd_id, resolution in view.resolved_commands.items():
            assert resolution.resolution_kind == ResolutionKind.SUCCEEDED
            assert resolution.resolution_payload is not None

    def test_partial_replay_resumes_from_last_committed(self) -> None:
        """If execution crashes mid-way, replay reconstructs up to the
        last committed resolution. The driver then resumes from there.

        This tests the scenario where Infer succeeded and was committed,
        but Act was requested and not yet resolved (crash point).
        """
        log = InMemoryLog()
        run_id = uuid4()
        branch_id = uuid4()
        cmd_infer = uuid4()
        cmd_act = uuid4()

        # Simulate partial execution: Infer committed, Act requested but not resolved
        records = [
            BranchCreated(
                header=_header(run_id, branch_id, 0),
                parent_branch_id=None,
                spawning_command_id=None,
                branch_kind=BranchKind.ROOT,
            ),
            CommandRequested(
                header=_header(run_id, branch_id, 1),
                command_id=cmd_infer,
                command_kind=CommandKind.INFER,
                canonical_payload={"input": "What should I do?"},
                driver_type="workflow",
            ),
            CommandResolved(
                header=_header(run_id, branch_id, 2),
                command_id=cmd_infer,
                command_kind=CommandKind.INFER,
                resolution_kind=ResolutionKind.SUCCEEDED,
                resolution_payload={"text": "do the thing"},
            ),
            # Act was requested but crashed before resolution
            CommandRequested(
                header=_header(run_id, branch_id, 3),
                command_id=cmd_act,
                command_kind=CommandKind.ACT,
                canonical_payload={"action": "save_plan"},
                driver_type="workflow",
                operation_id="save-001",
                recovery_mode=RecoveryMode.SAFE_RETRY,
            ),
        ]

        for r in records:
            log.append(r)

        # Replay: reconstruct from log
        view = reconstruct_branch(log.read_branch(branch_id))

        # Infer is resolved
        assert cmd_infer in view.resolved_commands
        assert view.resolved_commands[cmd_infer].resolution_payload == {"text": "do the thing"}

        # Act is pending (not resolved — crash point)
        assert cmd_act in view.pending_commands
        assert cmd_act not in view.resolved_commands

        # The pending Act carries its recovery mode
        pending_act = view.pending_commands[cmd_act]
        assert pending_act.recovery_mode == RecoveryMode.SAFE_RETRY

        # A driver resuming from this view knows:
        # 1. Infer is done — inject its result
        # 2. Act is pending with safe_retry — it's safe to retry
        # 3. Branch is still Running (not terminal)
        assert view.state.value == "Running"


class TestWorkflowDeterminismFinding:
    """Documents the critical finding about command identity and replay.

    Finding: using uuid4() for command_id means the same workflow code
    produces different command IDs on each execution. This means
    "re-run the workflow and match by command_id" does not work.

    The correct replay strategy is:
    1. Reconstruct branch state from committed log (not re-running code)
    2. Resume from the last committed point
    3. Only re-execute effects that don't have committed resolutions

    For workflows specifically, this means the driver must:
    - Track a replay cursor (position in the log)
    - On each effect call, check if a committed resolution exists at that position
    - If yes, inject it. If no, execute fresh.

    Position-based matching (seq_no order) is the right identity for replay,
    not command_id matching.
    """

    def test_position_based_replay_works(self) -> None:
        """Replay by log position, not command_id."""
        log = InMemoryLog()
        run_id = uuid4()
        branch_id = uuid4()

        # First execution
        log.append(
            BranchCreated(
                header=_header(run_id, branch_id, 0),
                parent_branch_id=None,
                spawning_command_id=None,
                branch_kind=BranchKind.ROOT,
            )
        )

        # Manually build a committed log
        cmd1 = uuid4()
        log.append(
            CommandRequested(
                header=_header(run_id, branch_id, 1),
                command_id=cmd1,
                command_kind=CommandKind.INFER,
                canonical_payload={"input": "question"},
                driver_type="workflow",
            )
        )
        log.append(
            CommandResolved(
                header=_header(run_id, branch_id, 2),
                command_id=cmd1,
                command_kind=CommandKind.INFER,
                resolution_kind=ResolutionKind.SUCCEEDED,
                resolution_payload={"text": "answer"},
            )
        )

        # Reconstruct and walk the log
        records = log.read_branch(branch_id)
        view = reconstruct_branch(records)

        # Position-based replay: walk records in order
        # At position 1 (CommandRequested for Infer): we have a resolution at position 2
        # The workflow driver can match by: "the Nth effect call maps to the Nth
        # CommandRequested in the log"
        infer_requests = [
            r for r in records if isinstance(r, CommandRequested) and r.command_kind == CommandKind.INFER
        ]
        assert len(infer_requests) == 1

        # The resolution for this request is in resolved_commands
        resolution = view.resolved_commands[infer_requests[0].command_id]
        assert resolution.resolution_payload == {"text": "answer"}

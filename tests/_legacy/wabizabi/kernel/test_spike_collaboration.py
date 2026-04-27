"""Spike 4: Delegate vs spawn vs handoff.

Validates that the three collaboration primitives produce correctly
different log structures:

- delegate: same-branch nested execution. No child branch created.
  Commands appear inline in the parent branch log.

- spawn: separately durable child branch. BranchCreated record
  on the child branch, with parent linkage. Parent and child logs
  are independent.

- handoff: ownership transfer. Current branch terminates with
  HandedOff outcome. Successor branch is created.
"""

from __future__ import annotations

from uuid import uuid4

from wabizabi.kernel.branch import BranchState, reconstruct_branch
from wabizabi.kernel.log import InMemoryLog
from wabizabi.kernel.records import (
    BranchCreated,
    BranchKind,
    CommandKind,
    CommandRequested,
    CommandResolved,
    RecordHeader,
    ResolutionKind,
    TerminalCommitted,
    TerminalOutcome,
    WaitArmed,
    WaitKind,
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


class TestDelegate:
    """delegate = same-branch nested execution.

    No child branch. The delegate's commands appear inline in the parent
    branch log. From the log's perspective, it's just more commands on
    the same branch.
    """

    def test_delegate_produces_no_child_branch(self) -> None:
        log = InMemoryLog()
        run_id = uuid4()
        branch_id = uuid4()

        # Parent branch with two Infer commands — second is "delegated"
        # but from the log's perspective, it's the same branch
        cmd_parent = uuid4()
        cmd_delegate = uuid4()

        records = [
            BranchCreated(
                header=_header(run_id, branch_id, 0),
                parent_branch_id=None,
                spawning_command_id=None,
                branch_kind=BranchKind.ROOT,
            ),
            # Parent's own infer
            CommandRequested(
                header=_header(run_id, branch_id, 1),
                command_id=cmd_parent,
                command_kind=CommandKind.INFER,
                canonical_payload={"input": "parent question"},
                driver_type="agent",
            ),
            CommandResolved(
                header=_header(run_id, branch_id, 2),
                command_id=cmd_parent,
                command_kind=CommandKind.INFER,
                resolution_kind=ResolutionKind.SUCCEEDED,
                resolution_payload={"text": "parent answer"},
            ),
            # Delegated specialist's infer — same branch
            CommandRequested(
                header=_header(run_id, branch_id, 3),
                command_id=cmd_delegate,
                command_kind=CommandKind.INFER,
                canonical_payload={"input": "specialist question"},
                driver_type="agent",
                context_snapshot={"delegated_from": "parent_agent", "delegate_to": "specialist"},
            ),
            CommandResolved(
                header=_header(run_id, branch_id, 4),
                command_id=cmd_delegate,
                command_kind=CommandKind.INFER,
                resolution_kind=ResolutionKind.SUCCEEDED,
                resolution_payload={"text": "specialist answer"},
            ),
        ]

        for r in records:
            log.append(r)

        # Only one branch exists
        parent_records = log.read_branch(branch_id)
        assert len(parent_records) == 5

        view = reconstruct_branch(parent_records)
        assert view.state == BranchState.RUNNING
        assert len(view.resolved_commands) == 2
        assert cmd_parent in view.resolved_commands
        assert cmd_delegate in view.resolved_commands

        # No other branches in the log
        all_branch_ids = {r.header.branch_id for r in log.all_records}
        assert len(all_branch_ids) == 1


class TestSpawn:
    """spawn = separately durable child branch.

    Parent issues a Spawn command. A new BranchCreated record appears
    on the child branch. Parent and child logs are independent.
    Parent may wait on child completion.
    """

    def test_spawn_creates_child_branch(self) -> None:
        log = InMemoryLog()
        run_id = uuid4()
        parent_branch = uuid4()
        child_branch = uuid4()
        spawn_cmd = uuid4()

        # Parent branch
        log.append(
            BranchCreated(
                header=_header(run_id, parent_branch, 0),
                parent_branch_id=None,
                spawning_command_id=None,
                branch_kind=BranchKind.ROOT,
            )
        )
        log.append(
            CommandRequested(
                header=_header(run_id, parent_branch, 1),
                command_id=spawn_cmd,
                command_kind=CommandKind.SPAWN,
                canonical_payload={"target": "research_agent", "input": "find information"},
                driver_type="agent",
            )
        )
        log.append(
            CommandResolved(
                header=_header(run_id, parent_branch, 2),
                command_id=spawn_cmd,
                command_kind=CommandKind.SPAWN,
                resolution_kind=ResolutionKind.SPAWN_COMMITTED,
                resolution_payload={"child_branch_id": str(child_branch)},
            )
        )

        # Child branch — separately durable
        child_cmd = uuid4()
        log.append(
            BranchCreated(
                header=_header(run_id, child_branch, 0),
                parent_branch_id=parent_branch,
                spawning_command_id=spawn_cmd,
                branch_kind=BranchKind.SPAWNED,
            )
        )
        log.append(
            CommandRequested(
                header=_header(run_id, child_branch, 1),
                command_id=child_cmd,
                command_kind=CommandKind.INFER,
                canonical_payload={"input": "researching..."},
                driver_type="agent",
            )
        )
        log.append(
            CommandResolved(
                header=_header(run_id, child_branch, 2),
                command_id=child_cmd,
                command_kind=CommandKind.INFER,
                resolution_kind=ResolutionKind.SUCCEEDED,
                resolution_payload={"text": "research results"},
            )
        )
        log.append(
            TerminalCommitted(
                header=_header(run_id, child_branch, 3),
                terminal_outcome=TerminalOutcome.COMPLETED,
                output_payload="research complete",
            )
        )

        # Two independent branches
        parent_view = reconstruct_branch(log.read_branch(parent_branch))
        child_view = reconstruct_branch(log.read_branch(child_branch))

        assert parent_view.state == BranchState.RUNNING
        assert child_view.state == BranchState.TERMINAL
        assert child_view.parent_branch_id == parent_branch
        assert child_view.branch_kind == BranchKind.SPAWNED
        assert child_view.terminal_outcome == TerminalOutcome.COMPLETED

        # Parent's spawn is resolved but parent itself continues
        assert spawn_cmd in parent_view.resolved_commands

    def test_parent_can_wait_on_child(self) -> None:
        """Parent arms a child_join wait after spawning."""
        log = InMemoryLog()
        run_id = uuid4()
        parent_branch = uuid4()
        child_branch = uuid4()
        spawn_cmd = uuid4()
        wait_id = uuid4()

        log.append(
            BranchCreated(
                header=_header(run_id, parent_branch, 0),
                parent_branch_id=None,
                spawning_command_id=None,
                branch_kind=BranchKind.ROOT,
            )
        )
        log.append(
            CommandRequested(
                header=_header(run_id, parent_branch, 1),
                command_id=spawn_cmd,
                command_kind=CommandKind.SPAWN,
                canonical_payload={"target": "child_agent"},
                driver_type="agent",
            )
        )
        log.append(
            CommandResolved(
                header=_header(run_id, parent_branch, 2),
                command_id=spawn_cmd,
                command_kind=CommandKind.SPAWN,
                resolution_kind=ResolutionKind.SPAWN_COMMITTED,
                resolution_payload={"child_branch_id": str(child_branch)},
            )
        )
        log.append(
            WaitArmed(
                header=_header(run_id, parent_branch, 3),
                wait_id=wait_id,
                wait_kind=WaitKind.CHILD_JOIN,
                originating_command_id=spawn_cmd,
                arm_payload={"child_branch_id": str(child_branch)},
            )
        )

        view = reconstruct_branch(log.read_branch(parent_branch))
        assert view.state == BranchState.WAITING_EXTERNAL
        assert wait_id in view.pending_waits


class TestHandoff:
    """handoff = ownership transfer.

    Current branch terminates with HandedOff outcome.
    A successor branch is created with handoff_successor kind.
    """

    def test_handoff_terminates_current_creates_successor(self) -> None:
        log = InMemoryLog()
        run_id = uuid4()
        current_branch = uuid4()
        successor_branch = uuid4()
        handoff_cmd = uuid4()

        # Current branch
        log.append(
            BranchCreated(
                header=_header(run_id, current_branch, 0),
                parent_branch_id=None,
                spawning_command_id=None,
                branch_kind=BranchKind.ROOT,
            )
        )
        log.append(
            CommandRequested(
                header=_header(run_id, current_branch, 1),
                command_id=handoff_cmd,
                command_kind=CommandKind.HANDOFF,
                canonical_payload={
                    "target": "specialist_agent",
                    "input": "take over this task",
                },
                driver_type="agent",
            )
        )
        log.append(
            CommandResolved(
                header=_header(run_id, current_branch, 2),
                command_id=handoff_cmd,
                command_kind=CommandKind.HANDOFF,
                resolution_kind=ResolutionKind.HANDOFF_COMMITTED,
                resolution_payload={"successor_branch_id": str(successor_branch)},
            )
        )
        log.append(
            TerminalCommitted(
                header=_header(run_id, current_branch, 3),
                terminal_outcome=TerminalOutcome.HANDED_OFF,
                terminal_metadata={"successor_branch_id": str(successor_branch)},
            )
        )

        # Successor branch
        log.append(
            BranchCreated(
                header=_header(run_id, successor_branch, 0),
                parent_branch_id=current_branch,
                spawning_command_id=handoff_cmd,
                branch_kind=BranchKind.HANDOFF_SUCCESSOR,
                projected_input="take over this task",
            )
        )

        # Current branch is terminal with HandedOff
        current_view = reconstruct_branch(log.read_branch(current_branch))
        assert current_view.state == BranchState.TERMINAL
        assert current_view.terminal_outcome == TerminalOutcome.HANDED_OFF

        # Successor branch is running
        successor_view = reconstruct_branch(log.read_branch(successor_branch))
        assert successor_view.state == BranchState.RUNNING
        assert successor_view.branch_kind == BranchKind.HANDOFF_SUCCESSOR
        assert successor_view.parent_branch_id == current_branch


class TestLogStructureDifferences:
    """The three primitives produce structurally different logs."""

    def test_delegate_single_branch(self) -> None:
        """Delegate: all commands on one branch."""
        log = InMemoryLog()
        run_id = uuid4()
        b = uuid4()

        log.append(BranchCreated(
            header=_header(run_id, b, 0),
            parent_branch_id=None, spawning_command_id=None,
            branch_kind=BranchKind.ROOT,
        ))
        # Two infer commands, inline
        for i, seq in enumerate([1, 2], start=1):
            cid = uuid4()
            log.append(CommandRequested(
                header=_header(run_id, b, seq * 2 - 1),
                command_id=cid, command_kind=CommandKind.INFER,
                canonical_payload={"step": i}, driver_type="agent",
            ))
            log.append(CommandResolved(
                header=_header(run_id, b, seq * 2),
                command_id=cid, command_kind=CommandKind.INFER,
                resolution_kind=ResolutionKind.SUCCEEDED,
                resolution_payload={"result": i},
            ))

        all_branches = {r.header.branch_id for r in log.all_records}
        assert len(all_branches) == 1

    def test_spawn_two_branches(self) -> None:
        """Spawn: parent + child = 2 branches."""
        log = InMemoryLog()
        run_id = uuid4()
        parent, child = uuid4(), uuid4()

        log.append(BranchCreated(
            header=_header(run_id, parent, 0),
            parent_branch_id=None, spawning_command_id=None,
            branch_kind=BranchKind.ROOT,
        ))
        log.append(BranchCreated(
            header=_header(run_id, child, 0),
            parent_branch_id=parent, spawning_command_id=None,
            branch_kind=BranchKind.SPAWNED,
        ))

        all_branches = {r.header.branch_id for r in log.all_records}
        assert len(all_branches) == 2

    def test_handoff_two_branches_current_terminal(self) -> None:
        """Handoff: current (terminal) + successor (running) = 2 branches."""
        log = InMemoryLog()
        run_id = uuid4()
        current, successor = uuid4(), uuid4()

        log.append(BranchCreated(
            header=_header(run_id, current, 0),
            parent_branch_id=None, spawning_command_id=None,
            branch_kind=BranchKind.ROOT,
        ))
        log.append(TerminalCommitted(
            header=_header(run_id, current, 1),
            terminal_outcome=TerminalOutcome.HANDED_OFF,
        ))
        log.append(BranchCreated(
            header=_header(run_id, successor, 0),
            parent_branch_id=current, spawning_command_id=None,
            branch_kind=BranchKind.HANDOFF_SUCCESSOR,
        ))

        current_view = reconstruct_branch(log.read_branch(current))
        successor_view = reconstruct_branch(log.read_branch(successor))

        assert current_view.state == BranchState.TERMINAL
        assert current_view.terminal_outcome == TerminalOutcome.HANDED_OFF
        assert successor_view.state == BranchState.RUNNING
        assert successor_view.branch_kind == BranchKind.HANDOFF_SUCCESSOR

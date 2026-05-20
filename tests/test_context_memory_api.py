from pathlib import Path

from cogneetree.distributed import (
    DistributedContextMemory,
    LeaderCommandHandler,
    LocalCommandSubmitter,
    LocalDecisionReader,
    MemoryCommand,
)
from cogneetree.memory import ProposalInput, StandaloneContextMemory
from cogneetree.protocol import ProposalStatus
from cogneetree.store import DecisionFileStore


def test_standalone_api_accepts_first_decision(tmp_path: Path) -> None:
    memory = StandaloneContextMemory("local", DecisionFileStore(tmp_path / "memory"))

    resolution = memory.propose_decision(
        ProposalInput(
            area="agent/context-policy",
            content="Read accepted context first.",
            rationale="Prevents stale proposals.",
            agent_id="agent-a",
        )
    )

    assert resolution.status == ProposalStatus.ACCEPTED
    assert "Read accepted context" in (memory.get_decision("agent/context-policy") or "")


def test_distributed_api_rejects_stale_decision(tmp_path: Path) -> None:
    store = DecisionFileStore(tmp_path / "memory")
    memory = build_distributed_memory(store)

    memory.propose_decision(
        ProposalInput(
            area="api/error-format",
            content="Use RFC 7807.",
            rationale="Standard shape.",
            agent_id="agent-a",
        )
    )
    resolution = memory.propose_decision(
        ProposalInput(
            area="api/error-format",
            content="Use custom errors.",
            rationale="Shorter payload.",
            agent_id="agent-b",
        )
    )

    assert resolution.status == ProposalStatus.REJECTED_STALE
    assert "Use RFC 7807" in (resolution.latest_state_markdown or "")


def test_distributed_api_reads_accepted_decisions(tmp_path: Path) -> None:
    store = DecisionFileStore(tmp_path / "memory")
    memory = build_distributed_memory(store)

    memory.propose_decision(
        ProposalInput(
            area="runtime/python-version",
            content="Use Python 3.12.",
            rationale="Matches project configuration.",
            agent_id="ci-agent",
            evidence=["pyproject.toml"],
        )
    )

    assert "Use Python 3.12" in (memory.get_decision("runtime/python-version") or "")
    assert memory.list_decisions() == ["runtime/python-version"]


def test_leader_command_handler_rejects_unknown_command(tmp_path: Path) -> None:
    handler = LeaderCommandHandler("node-a", DecisionFileStore(tmp_path / "memory"))

    command = MemoryCommand(
        command_id="cmd_unknown",
        command_type="unknown",
        payload={},
        submitted_by="agent-a",
    )

    try:
        handler.handle(command)
    except ValueError as error:
        assert "Unsupported command type" in str(error)
        return
    raise AssertionError("Expected unsupported command to raise ValueError")


def build_distributed_memory(store: DecisionFileStore) -> DistributedContextMemory:
    handler = LeaderCommandHandler("node-a", store)
    submitter = LocalCommandSubmitter(handler)
    reader = LocalDecisionReader(store)
    return DistributedContextMemory(submitter, reader)

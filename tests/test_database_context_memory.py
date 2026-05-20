from pathlib import Path

from cogneetree.db_memory import DatabaseContextMemory, DatabaseLeaderWorker
from cogneetree.memory import ProposalInput
from cogneetree.protocol import ProposalStatus
from cogneetree.sqlite_log import SQLiteDecisionLog


def test_database_memory_accepts_pending_context(tmp_path: Path) -> None:
    memory, worker = build_database_memory(tmp_path)

    proposal_id = memory.submit_context(
        ProposalInput(
            area="auth/session-storage",
            content="Use Redis for session storage.",
            rationale="TTL support.",
            agent_id="agent-a",
        )
    )

    assert memory.get_resolution(proposal_id) is None
    assert worker.claim_leadership()
    resolution = worker.process_next()

    assert resolution is not None
    assert resolution.status == ProposalStatus.ACCEPTED
    assert memory.get_resolution(proposal_id) == resolution
    assert "Use Redis" in (memory.get_context("auth/session-storage") or "")


def test_database_memory_rejects_stale_context(tmp_path: Path) -> None:
    memory, worker = build_database_memory(tmp_path)
    worker.claim_leadership()

    memory.submit_context(
        ProposalInput(
            area="api/error-format",
            content="Use RFC 7807.",
            rationale="Standard shape.",
            agent_id="agent-a",
        )
    )
    worker.process_next()
    stale_id = memory.submit_context(
        ProposalInput(
            area="api/error-format",
            content="Use custom errors.",
            rationale="Shorter payload.",
            agent_id="agent-b",
        )
    )

    resolution = worker.process_next()

    assert resolution is not None
    assert resolution.status == ProposalStatus.REJECTED_STALE
    assert memory.get_resolution(stale_id) == resolution
    assert "Use RFC 7807" in (resolution.latest_state_markdown or "")


def test_leader_lease_blocks_other_nodes_until_expired(tmp_path: Path) -> None:
    log = SQLiteDecisionLog(tmp_path / "memory.db")

    first = log.claim_leadership("node-a", lease_seconds=60)
    second = log.claim_leadership("node-b", lease_seconds=60)

    assert first is not None
    assert second is None


def test_leader_lease_can_be_claimed_after_expiry(tmp_path: Path) -> None:
    log = SQLiteDecisionLog(tmp_path / "memory.db")

    first = log.claim_leadership("node-a", lease_seconds=-1)
    second = log.claim_leadership("node-b", lease_seconds=60)

    assert first is not None
    assert second is not None
    assert second.node_id == "node-b"
    assert second.lease_epoch == first.lease_epoch + 1


def test_materialized_context_tree_and_search(tmp_path: Path) -> None:
    memory, worker = build_database_memory(tmp_path)
    worker.claim_leadership()

    submit_and_process(memory, worker, "auth/session-storage", "Use Redis.")
    submit_and_process(memory, worker, "auth/token-policy", "Use short JWT TTLs.")
    submit_and_process(memory, worker, "api/error-format", "Use RFC 7807.")

    root_children = memory.list_children()
    auth_children = memory.list_children("auth")
    subtree = memory.list_subtree("auth")
    search_results = memory.search_contexts("Redis")

    assert [node.context_key for node in root_children] == ["api", "auth"]
    assert [node.has_context for node in root_children] == [False, False]
    assert [node.context_key for node in auth_children] == [
        "auth/session-storage",
        "auth/token-policy",
    ]
    assert [node.context_key for node in subtree] == [
        "auth/session-storage",
        "auth/token-policy",
    ]
    assert [node.context_key for node in search_results] == ["auth/session-storage"]


def build_database_memory(tmp_path: Path) -> tuple[DatabaseContextMemory, DatabaseLeaderWorker]:
    log = SQLiteDecisionLog(tmp_path / "memory.db")
    return DatabaseContextMemory(log), DatabaseLeaderWorker("node-a", log)


def submit_and_process(
    memory: DatabaseContextMemory,
    worker: DatabaseLeaderWorker,
    area: str,
    content: str,
) -> None:
    memory.submit_context(
        ProposalInput(
            area=area,
            content=content,
            rationale="Test rationale.",
            agent_id="agent-a",
        )
    )
    worker.process_next()

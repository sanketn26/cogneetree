from pathlib import Path

from cogneetree.leader import MemoryLeader
from cogneetree.protocol import DecisionProposal, ProposalStatus
from cogneetree.store import DecisionFileStore


def test_accepts_first_decision_for_area(tmp_path: Path) -> None:
    store = DecisionFileStore(tmp_path / "memory")
    leader = MemoryLeader("node-a", store)

    resolution = leader.review(
        DecisionProposal(
            area="auth/session-storage",
            content="Use Redis for session storage.",
            rationale="TTL support.",
            agent_id="agent-a",
        )
    )

    assert resolution.status == ProposalStatus.ACCEPTED
    assert resolution.latest_version == 1
    assert "Use Redis" in (store.read_decision("auth/session-storage") or "")


def test_rejects_competing_decision_as_stale(tmp_path: Path) -> None:
    store = DecisionFileStore(tmp_path / "memory")
    leader = MemoryLeader("node-a", store)

    leader.review(
        DecisionProposal(
            area="auth/session-storage",
            content="Use Redis for session storage.",
            rationale="TTL support.",
            agent_id="agent-a",
        )
    )
    resolution = leader.review(
        DecisionProposal(
            area="auth/session-storage",
            content="Use Postgres for session storage.",
            rationale="Reduce infrastructure.",
            agent_id="agent-b",
        )
    )

    stored = store.read_decision("auth/session-storage") or ""
    assert resolution.status == ProposalStatus.REJECTED_STALE
    assert "Use Redis" in (resolution.latest_state_markdown or "")
    assert "Use Postgres" not in stored


def test_accepts_different_areas(tmp_path: Path) -> None:
    store = DecisionFileStore(tmp_path / "memory")
    leader = MemoryLeader("node-a", store)

    first = leader.review(
        DecisionProposal(
            area="auth/session-storage",
            content="Use Redis.",
            rationale="TTL support.",
            agent_id="agent-a",
        )
    )
    second = leader.review(
        DecisionProposal(
            area="database/audit-log",
            content="Use Postgres.",
            rationale="Durable relational audit data.",
            agent_id="agent-b",
        )
    )

    assert first.status == ProposalStatus.ACCEPTED
    assert second.status == ProposalStatus.ACCEPTED
    assert store.list_areas() == ["auth/session-storage", "database/audit-log"]


def test_records_events_and_rejection_audit(tmp_path: Path) -> None:
    store = DecisionFileStore(tmp_path / "memory")
    leader = MemoryLeader("node-a", store)

    leader.review(
        DecisionProposal(
            area="api/error-format",
            content="Use RFC 7807 problem details.",
            rationale="Standard JSON error shape.",
            agent_id="agent-a",
        )
    )
    rejected = leader.review(
        DecisionProposal(
            area="api/error-format",
            content="Use custom errors.",
            rationale="Shorter payload.",
            agent_id="agent-b",
        )
    )

    assert store.events_path.exists()
    assert "proposal_accepted" in store.events_path.read_text(encoding="utf-8")
    assert (store.rejected_dir / f"{rejected.proposal_id}.json").exists()


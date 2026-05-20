"""Database-backed context memory APIs."""

from __future__ import annotations

from cogneetree.db_protocol import ContextNode, LeaderLease
from cogneetree.memory import ProposalInput
from cogneetree.protocol import DecisionProposal, DecisionResolution
from cogneetree.sqlite_log import SQLiteDecisionLog


class DatabaseContextMemory:
    """Submit proposals and read materialized context through a decision log."""

    def __init__(self, log: SQLiteDecisionLog) -> None:
        self.log = log

    def initialize(self) -> None:
        self.log.initialize()

    def submit_context(self, proposal_input: ProposalInput) -> str:
        proposal = DecisionProposal(
            area=proposal_input.area,
            content=proposal_input.content,
            rationale=proposal_input.rationale,
            agent_id=proposal_input.agent_id,
            evidence=proposal_input.evidence,
        )
        return self.log.submit_proposal(proposal)

    def get_resolution(self, proposal_id: str) -> DecisionResolution | None:
        return self.log.get_resolution(proposal_id)

    def get_context(self, context_key: str) -> str | None:
        return self.log.get_context(context_key)

    def list_children(self, parent_key: str | None = None) -> list[ContextNode]:
        return self.log.list_children(parent_key)

    def list_subtree(self, parent_key: str) -> list[ContextNode]:
        return self.log.list_subtree(parent_key)

    def search_contexts(self, query: str) -> list[ContextNode]:
        return self.log.search_contexts(query)


class DatabaseLeaderWorker:
    """Leader worker that resolves pending proposals from the decision log."""

    def __init__(self, node_id: str, log: SQLiteDecisionLog) -> None:
        self.node_id = node_id
        self.log = log
        self.lease: LeaderLease | None = None

    def claim_leadership(self, lease_seconds: int = 300) -> bool:
        self.lease = self.log.claim_leadership(self.node_id, lease_seconds)
        return self.lease is not None

    def renew_leadership(self, lease_seconds: int = 300) -> bool:
        if self.lease is None:
            return self.claim_leadership(lease_seconds)
        self.lease = self.log.renew_leadership(self.lease, lease_seconds)
        return self.lease is not None

    def process_next(self) -> DecisionResolution | None:
        if self.lease is None:
            return None
        proposal = self.log.claim_pending(self.lease)
        if proposal is None:
            return None
        return self.log.resolve_claimed(proposal, self.lease)


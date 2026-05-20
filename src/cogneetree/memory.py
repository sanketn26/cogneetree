"""Standalone context memory API."""

from __future__ import annotations

from dataclasses import dataclass, field

from cogneetree.leader import MemoryLeader
from cogneetree.ports import DecisionStore
from cogneetree.protocol import DecisionProposal, DecisionResolution


@dataclass(frozen=True)
class ProposalInput:
    """Input accepted by public memory APIs."""

    area: str
    content: str
    rationale: str
    agent_id: str
    evidence: list[str] = field(default_factory=list)


class StandaloneContextMemory:
    """Local context memory backed by any decision store."""

    def __init__(self, leader_id: str, store: DecisionStore) -> None:
        self.store = store
        self.leader = MemoryLeader(leader_id, store)

    def initialize(self) -> None:
        self.store.initialize()

    def propose_decision(self, proposal_input: ProposalInput) -> DecisionResolution:
        proposal = DecisionProposal(
            area=proposal_input.area,
            content=proposal_input.content,
            rationale=proposal_input.rationale,
            agent_id=proposal_input.agent_id,
            evidence=proposal_input.evidence,
        )
        return self.leader.review(proposal)

    def get_decision(self, area: str) -> str | None:
        return self.store.read_decision(area)

    def list_decisions(self) -> list[str]:
        return self.store.list_areas()

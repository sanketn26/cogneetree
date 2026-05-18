"""Leader admission logic."""

from __future__ import annotations

from dataclasses import asdict

from cogneetree.protocol import (
    AcceptedDecision,
    DecisionProposal,
    DecisionResolution,
    ProposalStatus,
    new_id,
    utc_now,
)
from cogneetree.store import DecisionFileStore


class MemoryLeader:
    """Admit one active decision per area."""

    def __init__(self, leader_id: str, store: DecisionFileStore) -> None:
        self.leader_id = leader_id
        self.store = store

    def review(self, proposal: DecisionProposal) -> DecisionResolution:
        self.store.initialize()
        self.store.append_event("proposal_received", asdict(proposal))

        latest = self.store.read_decision(proposal.area)
        if latest is not None:
            resolution = DecisionResolution(
                status=ProposalStatus.REJECTED_STALE,
                area=proposal.area,
                proposal_id=proposal.proposal_id,
                leader_id=self.leader_id,
                latest_version=parse_version(latest),
                latest_state_markdown=latest,
                reason="An active decision already exists for this area.",
            )
            self.store.append_event("proposal_rejected_stale", asdict(resolution))
            self.store.write_rejection(resolution)
            return resolution

        decision = AcceptedDecision(
            area=proposal.area,
            content=proposal.content,
            rationale=proposal.rationale,
            version=1,
            decision_id=new_id("dec"),
            proposal_id=proposal.proposal_id,
            leader_id=self.leader_id,
            accepted_at=utc_now(),
            evidence=proposal.evidence,
        )
        markdown = render_decision(decision)
        self.store.write_decision(proposal.area, markdown)

        resolution = DecisionResolution(
            status=ProposalStatus.ACCEPTED,
            area=proposal.area,
            proposal_id=proposal.proposal_id,
            leader_id=self.leader_id,
            latest_version=decision.version,
            latest_state_markdown=markdown,
            decision=decision,
        )
        self.store.append_event("proposal_accepted", asdict(resolution))
        return resolution


def render_decision(decision: AcceptedDecision) -> str:
    """Render an accepted decision as canonical Markdown."""
    title = decision.area.replace("/", " ").title()
    evidence = "\n".join(f"- {item}" for item in decision.evidence) or "- none"
    return f"""# Decision: {title}

Area: {decision.area}
Status: accepted
Version: {decision.version}
Leader: {decision.leader_id}
Accepted-At: {decision.accepted_at}
Decision-ID: {decision.decision_id}

## Current Decision

{decision.content}

## Rationale

{decision.rationale}

## Sources

- proposal: {decision.proposal_id}
{evidence}
"""


def parse_version(markdown: str) -> int | None:
    """Extract a decision version from Markdown metadata."""
    for line in markdown.splitlines():
        if line.startswith("Version:"):
            return int(line.removeprefix("Version:").strip())
    return None


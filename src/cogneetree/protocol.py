"""Protocol records for decision memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    """Create a compact prefixed identifier."""
    return f"{prefix}_{uuid4().hex}"


class ProposalStatus(StrEnum):
    """Decision proposal outcomes."""

    ACCEPTED = "accepted"
    REJECTED_STALE = "rejected_stale"


@dataclass(frozen=True)
class DecisionProposal:
    """An agent's request to create an active decision for an area."""

    area: str
    content: str
    rationale: str
    agent_id: str
    evidence: list[str] = field(default_factory=list)
    base_version: int | None = None
    proposal_id: str = field(default_factory=lambda: new_id("prop"))
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class AcceptedDecision:
    """The one active decision admitted for an area."""

    area: str
    content: str
    rationale: str
    version: int
    decision_id: str
    proposal_id: str
    leader_id: str
    accepted_at: str
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DecisionResolution:
    """Leader response to a proposal."""

    status: ProposalStatus
    area: str
    proposal_id: str
    leader_id: str
    latest_version: int | None
    latest_state_markdown: str | None = None
    decision: AcceptedDecision | None = None
    reason: str | None = None


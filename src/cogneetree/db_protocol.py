"""Records for database-backed context memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class PendingDecisionStatus(StrEnum):
    """Database-backed proposal states."""

    PENDING = "pending"
    CLAIMED = "claimed"
    ACCEPTED = "accepted"
    REJECTED_STALE = "rejected_stale"


@dataclass(frozen=True)
class LeaderLease:
    """A database-backed leadership lease."""

    lease_name: str
    node_id: str
    lease_epoch: int
    expires_at: int


@dataclass(frozen=True)
class ContextPath:
    """Derived tree fields for a context key."""

    context_key: str
    parent_key: str | None
    leaf_name: str
    depth: int
    path_parts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContextNode:
    """A materialized context tree node."""

    context_key: str
    parent_key: str | None
    leaf_name: str
    depth: int
    has_context: bool


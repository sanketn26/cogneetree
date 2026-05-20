"""SQLite decision log and materialized context store."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cogneetree.db_protocol import ContextNode, ContextPath, LeaderLease, PendingDecisionStatus
from cogneetree.leader import parse_version, render_decision
from cogneetree.protocol import (
    AcceptedDecision,
    DecisionProposal,
    DecisionResolution,
    ProposalStatus,
    new_id,
    utc_now,
)


class SQLiteDecisionLog:
    """Database-backed proposal log with materialized accepted context."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            self.create_tables(connection)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def create_tables(self, connection: sqlite3.Connection) -> None:
        create_leader_lease(connection)
        create_pending_decisions(connection)
        create_accepted_contexts(connection)
        create_rejected_decisions(connection)
        create_decision_events(connection)

    def claim_leadership(self, node_id: str, lease_seconds: int = 300) -> LeaderLease | None:
        self.initialize()
        now = int(time.time())
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = get_leader_row(connection)
            lease = claim_or_renew_leader(connection, row, node_id, now, lease_seconds)
            connection.commit()
            return lease

    def renew_leadership(self, lease: LeaderLease, lease_seconds: int = 300) -> LeaderLease | None:
        self.initialize()
        now = int(time.time())
        expires_at = now + lease_seconds
        with self.connect() as connection:
            cursor = connection.execute(RENEW_LEASE_SQL, (expires_at, lease.node_id, lease.lease_epoch, now))
            row = cursor.fetchone()
            connection.commit()
        return lease_from_row(row) if row is not None else None

    def submit_proposal(self, proposal: DecisionProposal) -> str:
        self.initialize()
        path = parse_context_path(proposal.area)
        payload = proposal_to_row(proposal, path)
        with self.connect() as connection:
            connection.execute(INSERT_PENDING_SQL, payload)
            append_event(connection, "proposal_submitted", proposal.proposal_id, proposal.area, asdict(proposal))
            connection.commit()
        return proposal.proposal_id

    def claim_pending(self, lease: LeaderLease) -> DecisionProposal | None:
        self.initialize()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if not lease_is_current(connection, lease):
                connection.commit()
                return None
            row = connection.execute(SELECT_PENDING_SQL).fetchone()
            if row is None:
                connection.commit()
                return None
            mark_claimed(connection, row["proposal_id"], lease)
            connection.commit()
        return proposal_from_row(row)

    def resolve_claimed(self, proposal: DecisionProposal, lease: LeaderLease) -> DecisionResolution:
        self.initialize()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if not lease_is_current(connection, lease):
                raise RuntimeError("Leadership lease is no longer current.")
            resolution = resolve_proposal(connection, proposal, lease)
            connection.commit()
        return resolution

    def get_resolution(self, proposal_id: str) -> DecisionResolution | None:
        self.initialize()
        with self.connect() as connection:
            row = connection.execute(SELECT_RESOLUTION_SQL, (proposal_id,)).fetchone()
        if row is None or row["resolution_json"] is None:
            return None
        return resolution_from_json(row["resolution_json"])

    def get_context(self, context_key: str) -> str | None:
        self.initialize()
        with self.connect() as connection:
            row = connection.execute(SELECT_CONTEXT_SQL, (context_key,)).fetchone()
        return row["markdown"] if row is not None else None

    def list_children(self, parent_key: str | None = None) -> list[ContextNode]:
        self.initialize()
        with self.connect() as connection:
            rows = connection.execute(SELECT_CONTEXT_NODES_SQL).fetchall()
        return child_nodes_from_rows(rows, parent_key)

    def list_subtree(self, parent_key: str) -> list[ContextNode]:
        self.initialize()
        pattern = f"{parent_key}/%"
        with self.connect() as connection:
            rows = connection.execute(SELECT_SUBTREE_SQL, (parent_key, pattern)).fetchall()
        return [context_node_from_row(row) for row in rows]

    def search_contexts(self, query: str) -> list[ContextNode]:
        self.initialize()
        pattern = f"%{query}%"
        with self.connect() as connection:
            rows = connection.execute(SEARCH_CONTEXTS_SQL, (pattern, pattern, pattern)).fetchall()
        return [context_node_from_row(row) for row in rows]


def parse_context_path(context_key: str) -> ContextPath:
    parts = [part.strip() for part in context_key.split("/") if part.strip()]
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError(f"Invalid context key: {context_key!r}")
    normalized = "/".join(parts)
    parent_key = "/".join(parts[:-1]) or None
    return ContextPath(normalized, parent_key, parts[-1], len(parts), parts)


def proposal_to_row(proposal: DecisionProposal, path: ContextPath) -> dict[str, Any]:
    return {
        "proposal_id": proposal.proposal_id,
        "context_key": path.context_key,
        "parent_key": path.parent_key,
        "leaf_name": path.leaf_name,
        "depth": path.depth,
        "path_parts_json": json.dumps(path.path_parts),
        "content": proposal.content,
        "rationale": proposal.rationale,
        "agent_id": proposal.agent_id,
        "evidence_json": json.dumps(proposal.evidence),
        "status": PendingDecisionStatus.PENDING.value,
        "created_at": int(time.time()),
    }


def proposal_from_row(row: sqlite3.Row) -> DecisionProposal:
    return DecisionProposal(
        area=row["context_key"],
        content=row["content"],
        rationale=row["rationale"],
        agent_id=row["agent_id"],
        evidence=json.loads(row["evidence_json"]),
        proposal_id=row["proposal_id"],
    )


def resolve_proposal(
    connection: sqlite3.Connection,
    proposal: DecisionProposal,
    lease: LeaderLease,
) -> DecisionResolution:
    latest = read_context(connection, proposal.area)
    if latest is not None:
        return reject_stale(connection, proposal, latest, lease)
    return accept_proposal(connection, proposal, lease)


def accept_proposal(
    connection: sqlite3.Connection,
    proposal: DecisionProposal,
    lease: LeaderLease,
) -> DecisionResolution:
    decision = build_decision(proposal, lease)
    markdown = render_decision(decision)
    insert_accepted_context(connection, proposal, decision, markdown, lease)
    resolution = accepted_resolution(proposal, decision, markdown, lease)
    finish_pending(connection, proposal.proposal_id, resolution)
    append_event(connection, "proposal_accepted", proposal.proposal_id, proposal.area, asdict(resolution))
    return resolution


def reject_stale(
    connection: sqlite3.Connection,
    proposal: DecisionProposal,
    latest_markdown: str,
    lease: LeaderLease,
) -> DecisionResolution:
    resolution = DecisionResolution(
        status=ProposalStatus.REJECTED_STALE,
        area=proposal.area,
        proposal_id=proposal.proposal_id,
        leader_id=lease.node_id,
        latest_version=parse_version(latest_markdown),
        latest_state_markdown=latest_markdown,
        reason="An active decision already exists for this area.",
    )
    finish_pending(connection, proposal.proposal_id, resolution)
    insert_rejected(connection, resolution)
    append_event(connection, "proposal_rejected_stale", proposal.proposal_id, proposal.area, asdict(resolution))
    return resolution


def build_decision(proposal: DecisionProposal, lease: LeaderLease) -> AcceptedDecision:
    return AcceptedDecision(
        area=proposal.area,
        content=proposal.content,
        rationale=proposal.rationale,
        version=1,
        decision_id=new_id("dec"),
        proposal_id=proposal.proposal_id,
        leader_id=lease.node_id,
        accepted_at=utc_now(),
        evidence=proposal.evidence,
    )


def accepted_resolution(
    proposal: DecisionProposal,
    decision: AcceptedDecision,
    markdown: str,
    lease: LeaderLease,
) -> DecisionResolution:
    return DecisionResolution(
        status=ProposalStatus.ACCEPTED,
        area=proposal.area,
        proposal_id=proposal.proposal_id,
        leader_id=lease.node_id,
        latest_version=decision.version,
        latest_state_markdown=markdown,
        decision=decision,
    )


def insert_accepted_context(
    connection: sqlite3.Connection,
    proposal: DecisionProposal,
    decision: AcceptedDecision,
    markdown: str,
    lease: LeaderLease,
) -> None:
    path = parse_context_path(proposal.area)
    connection.execute(
        INSERT_ACCEPTED_SQL,
        accepted_context_row(proposal, decision, markdown, lease, path),
    )


def accepted_context_row(
    proposal: DecisionProposal,
    decision: AcceptedDecision,
    markdown: str,
    lease: LeaderLease,
    path: ContextPath,
) -> dict[str, Any]:
    return {
        "context_key": path.context_key,
        "parent_key": path.parent_key,
        "leaf_name": path.leaf_name,
        "depth": path.depth,
        "path_parts_json": json.dumps(path.path_parts),
        "decision_id": decision.decision_id,
        "proposal_id": proposal.proposal_id,
        "version": decision.version,
        "content": proposal.content,
        "rationale": proposal.rationale,
        "evidence_json": json.dumps(proposal.evidence),
        "markdown": markdown,
        "accepted_by": lease.node_id,
        "accepted_epoch": lease.lease_epoch,
        "accepted_at": int(time.time()),
    }


def finish_pending(
    connection: sqlite3.Connection,
    proposal_id: str,
    resolution: DecisionResolution,
) -> None:
    connection.execute(
        FINISH_PENDING_SQL,
        (resolution.status.value, int(time.time()), json.dumps(asdict(resolution), default=str), proposal_id),
    )


def read_context(connection: sqlite3.Connection, context_key: str) -> str | None:
    row = connection.execute(SELECT_CONTEXT_SQL, (context_key,)).fetchone()
    return row["markdown"] if row is not None else None


def insert_rejected(connection: sqlite3.Connection, resolution: DecisionResolution) -> None:
    connection.execute(
        INSERT_REJECTED_SQL,
        (
            resolution.proposal_id,
            resolution.area,
            resolution.latest_version,
            json.dumps(asdict(resolution), default=str),
            int(time.time()),
        ),
    )


def append_event(
    connection: sqlite3.Connection,
    event_type: str,
    proposal_id: str | None,
    area: str | None,
    payload: dict[str, Any],
) -> None:
    connection.execute(
        INSERT_EVENT_SQL,
        (event_type, proposal_id, area, json.dumps(payload, default=str), int(time.time())),
    )


def get_leader_row(connection: sqlite3.Connection) -> sqlite3.Row | None:
    return connection.execute(SELECT_LEADER_SQL, ("decision-leader",)).fetchone()


def claim_or_renew_leader(
    connection: sqlite3.Connection,
    row: sqlite3.Row | None,
    node_id: str,
    now: int,
    lease_seconds: int,
) -> LeaderLease | None:
    if row is not None and row["node_id"] != node_id and row["expires_at"] >= now:
        return None
    epoch = 1 if row is None else int(row["lease_epoch"]) + 1
    expires_at = now + lease_seconds
    connection.execute(UPSERT_LEASE_SQL, ("decision-leader", node_id, epoch, now, expires_at))
    return LeaderLease("decision-leader", node_id, epoch, expires_at)


def lease_is_current(connection: sqlite3.Connection, lease: LeaderLease) -> bool:
    row = get_leader_row(connection)
    if row is None:
        return False
    return (
        row["node_id"] == lease.node_id
        and row["lease_epoch"] == lease.lease_epoch
        and row["expires_at"] >= int(time.time())
    )


def mark_claimed(connection: sqlite3.Connection, proposal_id: str, lease: LeaderLease) -> None:
    connection.execute(
        MARK_CLAIMED_SQL,
        (PendingDecisionStatus.CLAIMED.value, lease.node_id, lease.lease_epoch, int(time.time()), proposal_id),
    )


def lease_from_row(row: sqlite3.Row) -> LeaderLease:
    return LeaderLease(row["lease_name"], row["node_id"], row["lease_epoch"], row["expires_at"])


def resolution_from_json(payload: str) -> DecisionResolution:
    data = json.loads(payload)
    decision = data.get("decision")
    if decision is not None:
        data["decision"] = AcceptedDecision(**decision)
    data["status"] = ProposalStatus(data["status"])
    return DecisionResolution(**data)


def context_node_from_row(row: sqlite3.Row) -> ContextNode:
    return ContextNode(
        context_key=row["context_key"],
        parent_key=row["parent_key"],
        leaf_name=row["leaf_name"],
        depth=row["depth"],
        has_context=True,
    )


def child_nodes_from_rows(rows: list[sqlite3.Row], parent_key: str | None) -> list[ContextNode]:
    parent_parts = [] if parent_key is None else parent_key.split("/")
    accepted_keys = {row["context_key"] for row in rows}
    nodes = {}
    for row in rows:
        path_parts = json.loads(row["path_parts_json"])
        child = child_path(path_parts, parent_parts)
        if child is None:
            continue
        has_context = child.context_key in accepted_keys
        nodes[child.context_key] = ContextNode(
            child.context_key,
            child.parent_key,
            child.leaf_name,
            child.depth,
            has_context,
        )
    return [nodes[key] for key in sorted(nodes)]


def child_path(path_parts: list[str], parent_parts: list[str]) -> ContextPath | None:
    if path_parts[: len(parent_parts)] != parent_parts:
        return None
    if len(path_parts) <= len(parent_parts):
        return None
    child_parts = path_parts[: len(parent_parts) + 1]
    context_key = "/".join(child_parts)
    parent_key = "/".join(child_parts[:-1]) or None
    return ContextPath(context_key, parent_key, child_parts[-1], len(child_parts), child_parts)


def create_leader_lease(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS leader_lease (
            lease_name TEXT PRIMARY KEY,
            node_id TEXT NOT NULL,
            lease_epoch INTEGER NOT NULL,
            renewed_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL
        )
        """
    )


def create_pending_decisions(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_decisions (
            proposal_id TEXT PRIMARY KEY,
            context_key TEXT NOT NULL,
            parent_key TEXT,
            leaf_name TEXT NOT NULL,
            depth INTEGER NOT NULL,
            path_parts_json TEXT NOT NULL,
            content TEXT NOT NULL,
            rationale TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            claimed_by TEXT,
            claimed_epoch INTEGER,
            claimed_at INTEGER,
            resolved_at INTEGER,
            resolution_json TEXT
        )
        """
    )


def create_accepted_contexts(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS accepted_contexts (
            context_key TEXT PRIMARY KEY,
            parent_key TEXT,
            leaf_name TEXT NOT NULL,
            depth INTEGER NOT NULL,
            path_parts_json TEXT NOT NULL,
            decision_id TEXT NOT NULL,
            proposal_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            content TEXT NOT NULL,
            rationale TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            markdown TEXT NOT NULL,
            accepted_by TEXT NOT NULL,
            accepted_epoch INTEGER NOT NULL,
            accepted_at INTEGER NOT NULL
        )
        """
    )


def create_rejected_decisions(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS rejected_decisions (
            proposal_id TEXT PRIMARY KEY,
            context_key TEXT NOT NULL,
            latest_version INTEGER,
            resolution_json TEXT NOT NULL,
            rejected_at INTEGER NOT NULL
        )
        """
    )


def create_decision_events(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            proposal_id TEXT,
            context_key TEXT,
            payload_json TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
        """
    )


INSERT_PENDING_SQL = """
INSERT INTO pending_decisions (
    proposal_id, context_key, parent_key, leaf_name, depth, path_parts_json,
    content, rationale, agent_id, evidence_json, status, created_at
) VALUES (
    :proposal_id, :context_key, :parent_key, :leaf_name, :depth, :path_parts_json,
    :content, :rationale, :agent_id, :evidence_json, :status, :created_at
)
"""

INSERT_ACCEPTED_SQL = """
INSERT INTO accepted_contexts (
    context_key, parent_key, leaf_name, depth, path_parts_json, decision_id,
    proposal_id, version, content, rationale, evidence_json, markdown,
    accepted_by, accepted_epoch, accepted_at
) VALUES (
    :context_key, :parent_key, :leaf_name, :depth, :path_parts_json, :decision_id,
    :proposal_id, :version, :content, :rationale, :evidence_json, :markdown,
    :accepted_by, :accepted_epoch, :accepted_at
)
"""

INSERT_REJECTED_SQL = """
INSERT INTO rejected_decisions (
    proposal_id, context_key, latest_version, resolution_json, rejected_at
) VALUES (?, ?, ?, ?, ?)
"""

INSERT_EVENT_SQL = """
INSERT INTO decision_events (
    event_type, proposal_id, context_key, payload_json, created_at
) VALUES (?, ?, ?, ?, ?)
"""

SELECT_PENDING_SQL = """
SELECT * FROM pending_decisions
WHERE status = 'pending'
ORDER BY created_at, proposal_id
LIMIT 1
"""

MARK_CLAIMED_SQL = """
UPDATE pending_decisions
SET status = ?, claimed_by = ?, claimed_epoch = ?, claimed_at = ?
WHERE proposal_id = ?
"""

FINISH_PENDING_SQL = """
UPDATE pending_decisions
SET status = ?, resolved_at = ?, resolution_json = ?
WHERE proposal_id = ?
"""

SELECT_CONTEXT_SQL = "SELECT markdown FROM accepted_contexts WHERE context_key = ?"
SELECT_RESOLUTION_SQL = "SELECT resolution_json FROM pending_decisions WHERE proposal_id = ?"
SELECT_LEADER_SQL = "SELECT * FROM leader_lease WHERE lease_name = ?"

SELECT_CONTEXT_NODES_SQL = "SELECT * FROM accepted_contexts ORDER BY context_key"

SELECT_SUBTREE_SQL = """
SELECT * FROM accepted_contexts
WHERE context_key = ? OR context_key LIKE ?
ORDER BY context_key
"""

SEARCH_CONTEXTS_SQL = """
SELECT * FROM accepted_contexts
WHERE context_key LIKE ? OR content LIKE ? OR rationale LIKE ?
ORDER BY context_key
"""

UPSERT_LEASE_SQL = """
INSERT INTO leader_lease (lease_name, node_id, lease_epoch, renewed_at, expires_at)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(lease_name) DO UPDATE SET
    node_id = excluded.node_id,
    lease_epoch = excluded.lease_epoch,
    renewed_at = excluded.renewed_at,
    expires_at = excluded.expires_at
"""

RENEW_LEASE_SQL = """
UPDATE leader_lease
SET renewed_at = strftime('%s', 'now'), expires_at = ?
WHERE node_id = ? AND lease_epoch = ? AND expires_at >= ?
RETURNING *
"""

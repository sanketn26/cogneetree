# Cogneetree

Governed residual memory for autonomous agents.

Cogneetree is restarting around one simple protocol:

> Agents propose memory changes. A scoped leader admits one active accepted
> Markdown state per organization, area, and node. Competing proposals are
> rejected as stale and the agent must re-evaluate from the latest state.

This repo intentionally keeps the implementation small. The file-backed path
stores accepted memory as Markdown and audit events as JSONL. The DB-backed path
stores proposals in a pending log, resolves them through a leased leader, and
exposes accepted context as a materialized tree.

## Why

Autonomous agents need durable memory, but they should not own the shared truth
directly. Cogneetree gives them a governed, human-readable knowledge tree:

- organization, area, and node identity
- one active accepted Markdown state per node
- one scoped leader admitting writes
- stale proposals rejected with latest state
- accepted updates recorded with attribution and reason
- derived indexes for lookup, not as source of truth

## Example

```bash
cogneetree --memory memory init

cogneetree --memory memory propose-decision auth/session-storage \
  --content "Use Redis for session storage." \
  --rationale "Redis supports TTLs and fast lookup." \
  --agent backend-agent

cogneetree --memory memory decisions show auth/session-storage
```

If another agent proposes a different decision for the same area, it gets:

```text
REJECTED_STALE auth/session-storage

An active decision already exists for this area.
Re-evaluate using the latest accepted state.
```

## Repository Shape

```text
src/cogneetree/
  protocol.py   # dataclasses and statuses
  db_protocol.py # database-backed records
  ports.py      # storage ports
  store.py      # Markdown and JSONL persistence
  sqlite_log.py # SQLite decision log and materialized context
  leader.py     # one-active-decision admission logic
  memory.py     # standalone context memory API
  distributed.py # distributed command API boundaries
  db_memory.py  # database-backed context memory API
  cli.py        # minimal CLI

docs/
  GUIDED_IMPLEMENTATION_GUIDE.md
  ROADMAP.md
  VALIDATION_AND_TESTING.md

tests/
  test_decision_protocol.py
  test_context_memory_api.py
  test_database_context_memory.py
```

## Design Docs

- [Guided Implementation Guide](docs/GUIDED_IMPLEMENTATION_GUIDE.md)
- [Roadmap](docs/ROADMAP.md)
- [Validation And Testing](docs/VALIDATION_AND_TESTING.md)

## Development

```bash
poetry install
poetry run pytest
```

## Current Status

Implemented today:

- file-backed protocol reference
- standalone context memory API
- distributed command boundary
- SQLite proposal log
- SQLite leader lease
- pending proposal resolution
- accepted materialized context tree
- stale rejection audit

Still planned: organization boundaries, versioned updates, change sets,
snapshots, node metadata, tombstones, schema migrations, graph relationships,
worker daemon, transport adapters, and derived indexes.

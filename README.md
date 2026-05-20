# Cogneetree

Distributed decision memory for agentic programming.

Cogneetree is restarting around one simple protocol:

> Agents propose decisions. A scoped leader admits one active decision per area.
> Accepted decisions become Markdown. Competing proposals are rejected as stale
> and the agent must re-evaluate from the latest state.

This repo intentionally keeps the implementation small. The file-backed path
stores accepted decisions as Markdown and audit events as JSONL. The DB-backed
path stores proposals in a pending log, resolves them through a leased leader,
and exposes accepted context as a materialized tree.

## Why

Distributed agents do not need a magical conflict resolver first. They need a
simple way to avoid polluting shared memory:

- one decision area
- one active accepted decision
- one leader admitting writes
- stale proposals rejected with latest state
- accepted decisions stored as human-readable Markdown

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
  CONTEXT_MEMORY_SYSTEM.md
  DISTRIBUTED_IMPLEMENTATION.md
  GUIDED_IMPLEMENTATION_GUIDE.md
  IMPLEMENTATION_ROADMAP.md

tests/
  test_decision_protocol.py
  test_context_memory_api.py
  test_database_context_memory.py
```

## Design Docs

- [Context Memory System](docs/CONTEXT_MEMORY_SYSTEM.md)
- [Guided Implementation Guide](docs/GUIDED_IMPLEMENTATION_GUIDE.md)
- [Implementation Roadmap](docs/IMPLEMENTATION_ROADMAP.md)
- [Distributed Implementation](docs/DISTRIBUTED_IMPLEMENTATION.md)

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

Still planned: schema migrations, supersede/update, worker daemon, HTTP/gRPC,
Postgres, MCP tools, and semantic recall.

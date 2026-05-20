# Implementation Roadmap

This roadmap tracks the work needed to move Cogneetree from the current
SQLite-backed protocol core to a production-ready context memory service.

Keep the rule simple:

```text
agents submit context proposals
one leased leader resolves proposals
accepted context is materialized by context_key
read models can be rendered by anyone
protocol writes stay centralized
```

## Current Baseline

Implemented:

- file-backed decision protocol
- standalone context memory API
- distributed command API boundaries
- SQLite proposal log
- SQLite leader lease
- pending decisions
- accepted materialized context tree
- stale rejection audit
- decision event log
- exact context read
- tree reads with `list_children` and `list_subtree`
- simple lexical search

Planned but not implemented:

- graph relationships between accepted contexts
- context neighborhood bundles

Verified by:

```bash
poetry run pytest
```

## Phase 1: Schema Versioning and Migrations

Goal: make the database schema safe to evolve.

Add:

- `schema_migrations` table
- current schema version constant
- idempotent migration runner
- migration tests from empty DB to latest schema
- migration tests that running migrations twice is safe

Suggested table:

```sql
create table schema_migrations (
  version integer primary key,
  name text not null,
  applied_at integer not null
);
```

Acceptance criteria:

- new SQLite DB initializes through migrations
- existing initialized DB does not fail on repeated initialization
- tests assert expected tables and migration rows exist
- no protocol code manually creates tables outside migrations

## Phase 2: Supersede and Update Protocol

Goal: allow accepted context to change safely.

Add:

- `SupersedeProposal` or explicit supersede fields
- expected current version
- old decision ID reference
- version increment
- supersede audit event
- stale rejection when expected version does not match

Rules:

```text
missing context -> reject_missing_current
expected_version mismatch -> reject_stale
expected_version match -> write version + 1
```

Acceptance criteria:

- accepted context can be superseded only with matching version
- stale supersede does not alter accepted context
- old and new decision IDs are linked in audit events
- materialized Markdown reflects the newest version
- tests cover missing, stale, and accepted supersede paths

## Phase 3: Long-Running Worker Loop

Goal: run a node as a process that claims leadership, renews it, and processes
pending proposals continuously.

Add:

- `DatabaseWorkerRunner`
- configurable lease duration
- configurable renew interval
- configurable idle sleep
- graceful stop signal
- error handling that records failed attempts without corrupting context

Loop shape:

```text
while running:
  claim or renew leadership
  if leader:
    process pending proposal
  else:
    sleep
```

Acceptance criteria:

- worker processes pending proposals until queue is empty
- worker renews lease while active
- worker stops cleanly
- worker does not resolve proposals after losing lease
- tests use short intervals and deterministic stop conditions

## Phase 4: Graph Relationships

Goal: support a graph over the materialized context tree.

Add:

- `context_edges` table
- edge records for accepted-context relationships
- edge types: `depends_on`, `related_to`, `supersedes`, `conflicts_with`
- APIs for edges, dependencies, dependents, related contexts, and neighborhoods

Suggested table:

```sql
create table context_edges (
  edge_id text primary key,
  from_context_key text not null,
  to_context_key text not null,
  edge_type text not null,
  rationale text,
  created_by text not null,
  created_at integer not null
);
```

Rules:

```text
tree identity stays context_key
graph edges connect accepted contexts
readers can traverse edges
only leader/materialization code creates accepted edges
```

Acceptance criteria:

- edges can be listed from and to a context
- dependencies and dependents are queryable
- neighborhood API returns context, ancestors, children, and graph neighbors
- edges cannot point to missing accepted contexts
- stale rejected proposals do not create edges

## Phase 5: Concurrent Multi-Process Stress Tests

Goal: prove the DB lease and pending queue behave under contention.

Add tests for:

- multiple workers competing for one lease
- one accepted context per key under concurrent submissions
- stale rejections under concurrent duplicate keys
- independent keys accepted under concurrent load
- expired leader handoff

Test strategy:

- use temporary SQLite files
- use `multiprocessing` or subprocesses
- keep test counts small enough for CI
- mark heavier stress tests separately if needed

Acceptance criteria:

- no duplicate accepted rows for one `context_key`
- all submitted proposals eventually resolve
- accepted count plus rejected count equals submitted count
- leadership handoff does not corrupt materialized context

## Phase 6: HTTP Transport

Goal: expose the DB model over a small network API.

Prefer HTTP before gRPC.

Core endpoints:

```text
POST /contexts/proposals
GET  /contexts/proposals/{proposal_id}
GET  /contexts/{context_key}
GET  /contexts
GET  /contexts/tree
GET  /contexts/{context_key}/edges
GET  /contexts/{context_key}/neighborhood
GET  /contexts/search?q=...
POST /worker/process-next
```

The HTTP layer must call:

```text
DatabaseContextMemory
DatabaseLeaderWorker
```

It must not duplicate protocol rules.

Acceptance criteria:

- proposal submission returns `proposal_id`
- resolution endpoint returns pending or resolved state
- context reads return materialized Markdown and metadata
- tree endpoint returns folder-like context nodes
- graph endpoints return accepted context relationships
- search endpoint returns lexical matches
- tests cover API serialization and stale proposal behavior

## Phase 7: Postgres Implementation

Goal: add a production database adapter.

Add:

- `PostgresDecisionLog`
- Postgres migrations
- transaction boundaries equivalent to SQLite implementation
- lease claiming with row locks or atomic update
- pending proposal claiming with `FOR UPDATE SKIP LOCKED`

Important Postgres primitives:

```sql
select ... for update skip locked
insert ... on conflict do nothing
```

Acceptance criteria:

- Postgres implementation passes the same behavior tests as SQLite
- adapter-specific tests cover lease claiming and pending claim behavior
- no protocol logic is duplicated outside shared helper functions
- local tests can be skipped when Postgres is unavailable

## Phase 8: gRPC Transport

Goal: add gRPC only after HTTP and DB behavior are stable.

Add:

- protobuf definitions for proposals, resolutions, and context nodes
- gRPC submitter
- gRPC read client
- gRPC worker service if needed

Acceptance criteria:

- gRPC supports the same operations as HTTP
- wire records map cleanly to protocol dataclasses
- tests verify serialization round trips

## Phase 9: MCP Tools

Goal: let agents access Cogneetree through MCP without direct DB access.

Tools:

```text
cogneetree_submit_context
cogneetree_get_resolution
cogneetree_get_context
cogneetree_list_children
cogneetree_list_subtree
cogneetree_list_edges
cogneetree_get_neighborhood
cogneetree_search_contexts
cogneetree_supersede_context
```

Acceptance criteria:

- MCP tools call public APIs only
- tool output includes context key, status, version, and Markdown when relevant
- stale rejection returns latest accepted context
- tests cover tool handlers without requiring a live MCP host

## Phase 10: Semantic and Vector Recall

Goal: add richer read-side recall without changing write semantics.

Rules:

- semantic recall is a read model
- accepted context remains canonical
- embeddings must never decide acceptance
- lexical and exact lookup remain available

Possible design:

```text
accepted_contexts -> recall_index
context_key
content
rationale
markdown
embedding
indexed_at
```

Acceptance criteria:

- exact context lookup is still authoritative
- semantic results include context keys and snippets
- index rebuild is possible from accepted contexts
- stale index never mutates accepted context

## Suggested Order

1. Schema versioning and migrations
2. Supersede/update protocol
3. Long-running worker loop
4. Graph relationships
5. Concurrent multi-process stress tests
6. HTTP transport
7. Postgres implementation
8. gRPC transport
9. MCP tools
10. Semantic/vector recall

The reason to do migrations first is simple: every later item changes the schema
or depends on a stable schema. The reason to do supersede before transports is
that API contracts are easier to expose once the write protocol is complete.

## Tracking Checklist

- [ ] Schema versioning and migrations
- [ ] Supersede/update protocol
- [ ] Long-running worker loop
- [ ] Graph relationships
- [ ] Concurrent multi-process stress tests
- [ ] HTTP transport
- [ ] Postgres implementation
- [ ] gRPC transport
- [ ] MCP tools
- [ ] Semantic/vector recall

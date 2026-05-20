# Guided Implementation Guide

This guide explains the implemented APIs and where new work should attach.

For pending work and acceptance criteria, use
[Implementation Roadmap](IMPLEMENTATION_ROADMAP.md).

## Module Map

```text
src/cogneetree/
  protocol.py       # base proposal, accepted decision, resolution records
  db_protocol.py    # DB lease, pending status, context tree records
  ports.py          # storage protocol for file-backed leader
  store.py          # file-backed Markdown and JSONL store
  leader.py         # one-active-decision admission logic
  memory.py         # standalone file-backed API
  distributed.py    # command/reader boundaries for remote adapters
  sqlite_log.py     # SQLite proposal log and materialized context tree
  db_memory.py      # DB-backed public API and leader worker
  cli.py            # minimal CLI for file-backed mode
```

## File-Backed API

Use this for local mode and protocol-level tests.

```python
from cogneetree import DecisionFileStore, ProposalInput, StandaloneContextMemory

memory = StandaloneContextMemory("local", DecisionFileStore("memory"))
memory.initialize()

resolution = memory.propose_decision(
    ProposalInput(
        area="api/error-format",
        content="Use RFC 7807 problem details.",
        rationale="Standard JSON error shape.",
        agent_id="api-agent",
    )
)

markdown = memory.get_decision("api/error-format")
areas = memory.list_decisions()
```

The file-backed path writes:

```text
memory/decisions/**/*.md
memory/events/decisions.jsonl
memory/audit/rejected/*.json
```

## DB-Backed API

Use this for the simplified distributed core.

```python
from cogneetree import DatabaseContextMemory, DatabaseLeaderWorker, ProposalInput, SQLiteDecisionLog

log = SQLiteDecisionLog("memory.db")
memory = DatabaseContextMemory(log)
worker = DatabaseLeaderWorker("node-a", log)
```

Submit a context proposal:

```python
proposal_id = memory.submit_context(
    ProposalInput(
        area="auth/session-storage",
        content="Use Redis for session storage.",
        rationale="TTL support.",
        agent_id="backend-agent",
    )
)
```

Resolve pending proposals:

```python
if worker.claim_leadership():
    resolution = worker.process_next()
```

Read materialized context:

```python
markdown = memory.get_context("auth/session-storage")
resolution = memory.get_resolution(proposal_id)
children = memory.list_children("auth")
subtree = memory.list_subtree("auth")
matches = memory.search_contexts("redis")
```

## DB Tables

SQLite currently creates:

```text
leader_lease          # active node lease and epoch
pending_decisions     # submitted proposals
accepted_contexts     # materialized context tree
rejected_decisions    # stale proposal audit
decision_events       # append-only audit stream
```

## Context Tree

The tree is derived from `context_key`.

```text
auth/session-storage
auth/token-policy
api/error-format
```

becomes:

```text
auth
  auth/session-storage
  auth/token-policy
api
  api/error-format
```

`list_children(None)` returns top-level folder nodes. `list_children("auth")`
returns immediate children. `list_subtree("auth")` returns accepted context rows
under that prefix.

## Future Graph Relationships

The next read-model expansion should add graph edges between accepted contexts.

The tree remains the canonical location model:

```text
context_key = auth/session-storage
```

Graph edges describe relationships:

```text
auth/session-storage depends_on project/runtime
auth/session-storage related_to auth/token-policy
auth/token-policy conflicts_with auth/session-storage
```

Expected table shape:

```text
context_edges
  edge_id
  from_context_key
  to_context_key
  edge_type
  rationale
  created_by
  created_at
```

Expected APIs:

```python
edges = memory.list_edges("auth/session-storage")
dependencies = memory.list_dependencies("auth/session-storage")
dependents = memory.list_dependents("project/runtime")
bundle = memory.get_context_neighborhood("auth/session-storage", depth=1)
```

Graph edges should be materialized by the leader as part of accepted proposals or
supersede operations. Readers may traverse edges, but readers must not create
accepted relationships directly.

## Distributed Adapter Boundary

Network transports should attach at the API boundary:

```text
HTTP/gRPC/MCP
  -> DatabaseContextMemory
  -> SQLiteDecisionLog or future PostgresDecisionLog
```

Worker processes should attach here:

```text
worker daemon
  -> DatabaseLeaderWorker
  -> decision log
```

Adapters must not duplicate stale rejection, acceptance, or materialization
rules.

## Tests

Run:

```bash
poetry run pytest
```

Important coverage:

- file-backed first proposal acceptance
- stale rejection
- independent context keys
- DB leader lease ownership
- DB lease expiry takeover
- pending proposal resolution
- materialized tree reads
- lexical search

## Rules to Preserve

- agents submit proposals instead of writing accepted context
- only the leader resolves pending proposals
- accepted context is addressed by `context_key`
- stale proposals do not mutate accepted context
- materialized views are read models
- graph edges are relationships, not context identity
- semantic search, MCP, HTTP, and gRPC must call public APIs

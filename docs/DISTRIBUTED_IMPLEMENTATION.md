# Distributed Implementation

Cogneetree's distributed core is intentionally small:

```text
submit proposal
  -> pending_decisions
  -> leased leader
  -> accepted_contexts or rejected_decisions
  -> decision_events
```

The protocol is not a custom consensus system. SQLite currently provides the
local DB-backed implementation. A future Postgres adapter should preserve the
same behavior with stronger production concurrency primitives.

## Leader Lease

`leader_lease` stores one active lease:

```text
lease_name
node_id
lease_epoch
renewed_at
expires_at
```

Only the current lease holder can claim and resolve pending proposals. The epoch
acts as a fencing token so an old leader cannot continue writing after another
node takes over.

## Pending Proposals

Agents submit proposals into `pending_decisions`.

```text
pending -> claimed -> accepted
pending -> claimed -> rejected_stale
```

The leader processes proposals in created order.

## Materialized Context

Accepted context is written to `accepted_contexts`.

The primary lookup key is:

```text
context_key
```

Path-like keys also create a walkable tree:

```text
auth/session-storage
auth/token-policy
project/runtime
```

Readers can use exact lookup, children, subtree, or lexical search. Readers do
not decide acceptance.

## Future Context Graph

The distributed read model should support graph relationships over accepted
contexts:

```text
auth/session-storage depends_on project/runtime
api/error-format related_to frontend/error-handling
auth/token-policy conflicts_with auth/session-storage
```

The tree remains the identity and location model. Graph edges are relationships
between accepted contexts and should be created only by leader-controlled
materialization.

## Current API

```python
from cogneetree import DatabaseContextMemory, DatabaseLeaderWorker, ProposalInput, SQLiteDecisionLog

log = SQLiteDecisionLog("memory.db")
memory = DatabaseContextMemory(log)
worker = DatabaseLeaderWorker("node-a", log)

proposal_id = memory.submit_context(
    ProposalInput(
        area="auth/session-storage",
        content="Use Redis for session storage.",
        rationale="TTL support.",
        agent_id="backend-agent",
    )
)

worker.claim_leadership()
resolution = worker.process_next()
markdown = memory.get_context("auth/session-storage")
```

## Future Transports

HTTP, gRPC, and MCP should be thin adapters:

```text
transport request
  -> public API
  -> decision log
```

They must not duplicate protocol rules.

## Future Postgres Adapter

The Postgres implementation should use:

```text
row locks or atomic updates for leader lease
FOR UPDATE SKIP LOCKED for pending proposal claims
INSERT ... ON CONFLICT for accepted context
```

It should pass the same behavior tests as the SQLite implementation.

## Do Not Add Yet

- custom Raft
- direct agent writes to accepted context
- vector search in the write path
- framework-specific adapters inside core protocol modules

For planned production work, see
[Implementation Roadmap](IMPLEMENTATION_ROADMAP.md).

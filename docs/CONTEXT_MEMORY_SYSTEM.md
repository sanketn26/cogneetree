# Context Memory System

Cogneetree is a small context memory protocol for agents.

It gives agents a stable answer to:

```text
What accepted context is currently true for this key?
```

Agents do not write accepted memory directly. They submit proposals. A leader
admits or rejects those proposals. Accepted context is materialized for readers.

## Core Rule

```text
agents submit context proposals
one leader admits writes
one context key has one active accepted context
accepted context is rendered as Markdown
proposal outcomes are audited
competing proposals are rejected as stale
```

## Context Identity

Cogneetree uses three identifiers:

```text
proposal_id  # immutable submission identity
context_key  # stable path-like memory address
decision_id  # accepted decision identity
```

Examples of context keys:

```text
project/runtime
auth/session-storage
frontend/state-management
agent/context-policy
```

The context key is also the tree path used by read models.

## Write Model

The write path is centralized:

```text
agent
  -> submit proposal
  -> pending log
  -> leased leader
  -> accepted context or stale rejection
  -> audit events
```

If a context key has no accepted context, the leader accepts the proposal.

If a context key already has accepted context, the leader rejects the proposal as
stale and returns the latest accepted Markdown.

## Read Model

Accepted context can be rendered by anyone.

Current read models:

```text
exact lookup       get_context("auth/session-storage")
tree children      list_children("auth")
tree subtree       list_subtree("auth")
lexical search     search_contexts("redis")
Markdown output    accepted context body
```

Folder nodes such as `auth` or `project` are derived from accepted context keys.
They are read-model nodes, not separate accepted decisions.

## Tree Plus Graph

Cogneetree should keep both a tree and a graph.

The tree is for location:

```text
auth/session-storage
auth/token-policy
api/error-format
project/runtime
```

The graph is for meaning:

```text
auth/session-storage depends_on project/runtime
api/error-format related_to frontend/error-handling
auth/token-policy supersedes auth/old-token-policy
```

The rule is:

```text
context_key gives the walkable tree
context_edges give relationships
```

Do not replace the tree with a graph. The tree gives humans and agents a stable
folder-like map. The graph adds cross-links for dependencies, conflicts,
supersession, and related context.

Planned edge types:

```text
depends_on
related_to
supersedes
conflicts_with
```

Future APIs should include:

```text
list_edges(context_key)
list_dependencies(context_key)
list_dependents(context_key)
list_related(context_key)
get_context_neighborhood(context_key, depth=1)
```

`get_context_neighborhood` should return a compact working-memory bundle:

```text
the context itself
tree ancestors
tree children
direct graph edges
linked accepted contexts
```

Graph edges should initially connect accepted contexts only. Later, proposals can
request edges, and the leader can materialize them only when the proposal is
accepted.

## Storage Modes

### File-Backed Reference

The file-backed reference implementation stores:

```text
memory/
  decisions/**/*.md
  events/decisions.jsonl
  audit/rejected/*.json
```

This path is useful for local use, CLI behavior, and protocol tests.

### SQLite Decision Log

The SQLite implementation stores:

```text
leader_lease
pending_decisions
accepted_contexts
rejected_decisions
decision_events
```

`accepted_contexts` is the materialized context table. It includes tree fields:

```text
context_key
parent_key
leaf_name
depth
path_parts_json
markdown
```

## Accepted Markdown

Accepted context is rendered as Markdown:

```markdown
# Decision: Agent Context Policy

Area: agent/context-policy
Status: accepted
Version: 1
Leader: node-a
Accepted-At: 2026-05-20T00:00:00+00:00
Decision-ID: dec_...

## Current Decision

Agents must cite accepted decisions before proposing changes.

## Rationale

Prevents stale context from overwriting accepted state.

## Sources

- proposal: prop_...
- docs/CONTEXT_MEMORY_SYSTEM.md
```

Markdown is a readable view of accepted state. The DB-backed model can also
serve structured rows and tree nodes.

## Local Mode

Local file-backed mode:

```python
from cogneetree import DecisionFileStore, ProposalInput, StandaloneContextMemory

memory = StandaloneContextMemory("local", DecisionFileStore("memory"))
memory.initialize()

resolution = memory.propose_decision(
    ProposalInput(
        area="agent/context-policy",
        content="Agents must read accepted context before proposing changes.",
        rationale="Prevents stale writes.",
        agent_id="planner-agent",
    )
)
```

## DB-Backed Mode

SQLite-backed submission and processing:

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

## Distributed Shape

The current distributed core is DB-coordinated:

```text
agents submit proposals into pending_decisions
nodes compete for leader_lease
the leased leader resolves pending proposals
readers consume accepted_contexts
```

Future HTTP, gRPC, Postgres, MCP, and semantic recall work should call the public
APIs and must not duplicate protocol rules.

## Related Docs

- [Guided Implementation Guide](GUIDED_IMPLEMENTATION_GUIDE.md)
- [Distributed Implementation](DISTRIBUTED_IMPLEMENTATION.md)
- [Implementation Roadmap](IMPLEMENTATION_ROADMAP.md)

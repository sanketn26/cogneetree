# Roadmap

This roadmap moves Cogneetree from a small decision protocol into a governed
organizational memory tree for autonomous agents.

Use [VALIDATION_AND_TESTING.md](VALIDATION_AND_TESTING.md) as the test contract
for every phase.

Keep the rule simple:

```text
agents propose memory changes
leaders admit memory changes
accepted memory is human-readable Markdown
audit is append-only JSONL
indexes are derived
```

## Current Baseline

Current phase: Phase 1 is the next active phase. The existing implementation
already proves the older context/decision protocol; new work should align the
language and identity model before adding more storage behavior.

Implemented:

- file-backed proposal and acceptance protocol
- standalone context memory API
- SQLite proposal log
- SQLite leader lease
- pending proposal resolution
- materialized accepted context tree
- stale rejection audit
- decision event log
- exact context read
- tree reads with `list_children` and `list_subtree`
- simple lexical search

Still needed:

- organization boundary
- node identity beyond path-only context keys
- accepted version updates
- change sets
- snapshots
- richer provenance
- node type and authority metadata
- tombstones
- validation and migration support
- explicit conflict and supersession edges
- derived manifest and lexical index contract

## Phase 1: Language Alignment

Goal: align code and docs around governed memory instead of one-off decisions.

Add:

- protocol names for organization, area, node, proposal, accepted version
- frontmatter fields for accepted Markdown
- compatibility layer for current `area` and `context_key` names
- tests that old context behavior still passes

Acceptance criteria:

- `org_id + area_id + node_id` is the canonical identity in new records
- accepted Markdown includes stable frontmatter
- existing tests pass without weakening protocol rules
- docs use memory node language consistently

## Phase 2: Schema Versioning And Migrations

Goal: make the database safe to evolve.

Add:

- `schema_migrations` table
- current schema version constant
- idempotent migration runner
- migration tests from empty DB to latest schema
- migration tests that repeated initialization is safe

Acceptance criteria:

- new SQLite DB initializes through migrations
- existing initialized DB does not fail on repeated initialization
- tests assert expected tables and migration rows exist
- no protocol code manually creates tables outside migrations

## Phase 3: Supersede And Update Protocol

Goal: allow accepted memory to change safely.

Add:

- explicit update proposal fields
- expected current version
- previous accepted version reference
- version increment
- supersede audit event
- stale rejection when expected version does not match

Rules:

```text
missing node on create -> accept
missing node on update -> reject_missing_current
expected version mismatch -> reject_stale
expected version match -> write version + 1
```

Acceptance criteria:

- accepted memory can be superseded only with matching version
- stale supersede does not alter accepted Markdown
- old and new accepted version IDs are linked in audit events
- materialized Markdown reflects the newest version
- tests cover missing, stale, and accepted update paths

## Phase 4: Provenance, Snapshots, And Change Sets

Goal: make every accepted update attributable and reconstructable.

Add:

- snapshot path for every accepted version
- `proposed_by`, `accepted_by`, `reason`, and `source_refs`
- `change_set_id` for multi-node updates
- all-or-nothing change set admission
- JSONL events for proposal, acceptance, rejection, and supersession

Acceptance criteria:

- accepted versions can be reconstructed from snapshots
- each accepted update records who, what, when, and why
- multi-node change sets reject if any expected version is stale
- rejected change sets do not alter any accepted Markdown

## Phase 5: Node Metadata And Validation

Goal: make memory interpretable by agents and safe to accept.

Add:

- `node_type` enum
- `authority` enum
- aliases and tags
- review and expiration fields
- access scope fields
- frontmatter validation
- link validation
- content size limits

Acceptance criteria:

- invalid frontmatter is rejected before acceptance
- node ID and path rules are enforced
- required metadata is present
- expired or review-due nodes are visible to readers
- tests cover each validation rule

## Phase 6: Tombstones And Lifecycle

Goal: make deletion and retirement explicit.

Add:

- `retired` status
- retirement reason
- superseded-by links
- tombstone Markdown rendering
- lifecycle audit events

Acceptance criteria:

- retired nodes remain discoverable
- retired nodes do not appear as active accepted guidance by default
- readers can request retired nodes explicitly
- tests prove deletion is not silent absence

## Phase 7: Graph Relationships

Goal: support meaning over the materialized tree.

Add:

- `context_edges` or `node_edges` table
- edge records for accepted-node relationships
- edge types: `depends_on`, `related_to`, `supersedes`, `conflicts_with`
- APIs for edges, dependencies, dependents, conflicts, and neighborhoods

Rules:

```text
tree identity stays org + area + node
graph edges connect accepted nodes
readers can traverse edges
only leader materialization creates accepted edges
```

Acceptance criteria:

- edges can be listed from and to a node
- dependencies and dependents are queryable
- conflicts are visible to proposal validation
- neighborhood API returns node, ancestors, children, and graph neighbors
- edges cannot point to missing accepted nodes

## Phase 8: Derived Indexes

Goal: make lookup fast without making indexes authoritative.

Add:

- generated manifest
- alias index
- lexical inverted index
- rebuild command
- freshness metadata

Non-goal:

```text
semantic search in the write path
```

Acceptance criteria:

- indexes can be deleted and rebuilt from Markdown plus audit
- exact lookup does not require semantic search
- lexical lookup returns node IDs and paths
- stale indexes are detectable

## Phase 9: Long-Running Worker Loop

Goal: run a node as a process that claims leadership, renews it, and processes
pending proposals continuously.

Add:

- long-running worker runner around the public memory API
- configurable lease duration
- configurable renew interval
- configurable idle sleep
- graceful stop signal
- error handling that records failed attempts without corrupting memory

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

- worker processes pending proposals until the queue is empty
- worker renews lease while active
- worker stops cleanly
- worker does not resolve proposals after losing lease
- tests use short intervals and deterministic stop conditions

## Phase 10: Transport Adapters

Goal: expose the memory protocol through small adapters.

Prefer HTTP before gRPC. MCP is deliberately out of scope for this roadmap until
the protocol, validation contract, and HTTP adapter are stable.

Core endpoints:

```text
POST /orgs/{org_id}/areas/{area_id}/nodes/{node_id}/proposals
GET  /orgs/{org_id}/areas/{area_id}/nodes/{node_id}
GET  /orgs/{org_id}/areas/{area_id}/tree
GET  /orgs/{org_id}/areas/{area_id}/nodes/{node_id}/history
GET  /orgs/{org_id}/areas/{area_id}/nodes/{node_id}/edges
GET  /orgs/{org_id}/areas/{area_id}/nodes/{node_id}/neighborhood
GET  /orgs/{org_id}/search?q=...
POST /worker/process-next
```

Acceptance criteria:

- proposal submission returns `proposal_id`
- resolution endpoint returns pending or resolved state
- node reads return accepted Markdown and metadata
- tree endpoint returns folder-like nodes
- graph endpoints return accepted relationships
- search endpoint returns lexical matches
- transport tests cover stale proposal behavior through public APIs

## Deferred: MCP Adapter

MCP should be added only after the transport-independent protocol is stable.
The MCP adapter must be a thin wrapper over the same public APIs as HTTP and
must not introduce a separate memory mutation path.

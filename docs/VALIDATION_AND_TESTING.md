# Validation And Testing

This document defines how to validate Cogneetree as governed residual memory for
autonomous agents.

The goal is not only that APIs work. The goal is to prove that accepted memory
is durable, attributable, human-readable, and protected from unsafe concurrent
agent writes.

## Validation Principles

Every test should protect one protocol invariant.

```text
accepted Markdown is truth
agents propose, leaders admit
one active state exists per org + area + node
updates require the expected base version
rejections never mutate accepted state
audit events explain who, what, when, and why
indexes are rebuildable caches
```

If a feature cannot be validated against an invariant, do not add it to the core
protocol yet.

## Test Layers

Use four layers of tests.

```text
unit tests          # dataclasses, validators, path rules, status transitions
protocol tests      # proposal, acceptance, stale rejection, supersession
storage tests       # Markdown, JSONL, snapshots, SQLite rows, migrations
integration tests   # worker lease, concurrent proposals, index rebuilds
```

Keep unit and protocol tests fast. Use integration tests only when behavior
requires real storage or multiple workers.

## Canonical Fixtures

Use a small fixed organization tree in tests:

```text
org_id: acme

areas:
  rbac
    rbac.auth
    rbac.acls
    rbac.authorizations
  billing
    billing.contractor_access
```

Use stable actors:

```text
agent-auth-reviewer
agent-billing-reviewer
rbac-leader
billing-leader
```

Stable fixtures make audit assertions readable and prevent accidental drift.

## Identity Tests

Validate that identity is stable and scoped.

Acceptance tests:

- same `area_id` and `node_id` can exist in different orgs independently
- same `node_id` in two areas is treated as different memory
- file moves or title changes do not change `node_id`
- path traversal values such as `../secrets` are rejected
- malformed org, area, and node IDs are rejected

Invariant:

```text
org_id + area_id + node_id is the accepted memory identity
```

## Proposal Admission Tests

Validate the core write protocol.

Acceptance tests:

- first proposal for a missing node is accepted
- second create proposal for the same node is rejected as stale
- create proposals for different nodes are accepted independently
- rejected proposal does not alter accepted Markdown
- rejection event includes latest accepted version reference
- pending proposals are resolved in deterministic order

Invariant:

```text
agents cannot directly write accepted memory
```

## Update And Version Tests

Validate safe mutation of accepted memory.

Acceptance tests:

- update with matching expected version is accepted
- accepted update increments version by one
- update with stale expected version is rejected
- update against missing current node is rejected as missing current
- accepted Markdown frontmatter reflects the new version
- previous accepted version remains reconstructable from snapshot

Invariant:

```text
accepted updates are compare-and-set writes
```

## Change Set Tests

Validate multi-node updates.

Acceptance tests:

- change set with all matching base versions is accepted
- change set writes one audit correlation ID across all node updates
- change set with one stale node rejects the entire set
- rejected change set mutates no accepted nodes
- change set reason is recorded on every affected node audit event

Invariant:

```text
multi-node memory changes are all-or-nothing
```

## Markdown Validation Tests

Validate human-readable accepted state.

Acceptance tests:

- accepted Markdown includes required frontmatter
- missing `id`, `org_id`, `area`, `status`, `version`, or `authority` is rejected
- invalid `node_type` is rejected
- invalid `authority` is rejected
- `status: retired` requires a retirement reason
- Markdown body cannot be empty for active nodes
- frontmatter values match proposal identity

Invariant:

```text
accepted memory is readable by humans and interpretable by agents
```

## Provenance And Audit Tests

Validate permanent accountability.

Acceptance tests:

- proposal event is written for every submission
- accepted event records proposer, leader, reason, source refs, and timestamp
- rejection event records rejection reason and current accepted version
- supersede event links old and new versions
- JSONL events are append-only
- audit can reconstruct the sequence of state transitions

Invariant:

```text
every accepted memory change is attributable and explainable
```

## Snapshot Tests

Validate historical reconstruction.

Acceptance tests:

- every accepted version writes a snapshot
- snapshot content matches accepted Markdown for that version
- current Markdown matches the latest accepted version
- missing snapshot is reported as validation failure
- generated diff between two snapshots is reproducible

Invariant:

```text
history survives after the current state changes
```

## Tombstone Tests

Validate deletion and retirement.

Acceptance tests:

- retiring a node writes accepted Markdown with `status: retired`
- retired node remains addressable by exact lookup
- active reads exclude retired nodes by default
- explicit history reads include retired nodes
- recreating a retired node requires an explicit new proposal rule

Invariant:

```text
deleted knowledge is explicit, not silent absence
```

## Access Boundary Tests

Validate organization and area governance.

Acceptance tests:

- agent without read scope cannot read restricted node
- agent without propose scope cannot submit proposal
- non-leader cannot accept proposal
- leader for one area cannot accept another area's proposal
- cross-org reads and writes are denied unless explicitly scoped

Invariant:

```text
organization and area are governance boundaries
```

## Conflict And Edge Tests

Validate relationships between nodes.

Acceptance tests:

- edge cannot point to a missing accepted node
- `conflicts_with` edge is visible during proposal validation
- `supersedes` edge is written when a node replaces another
- graph edge creation is leader-admitted
- stale rejected proposal does not create edges

Invariant:

```text
the tree locates memory; edges describe meaning
```

## Index Validation Tests

Validate lookup without making indexes authoritative.

Acceptance tests:

- manifest rebuilds from accepted Markdown
- alias index rebuilds from frontmatter
- lexical index rebuilds from Markdown body
- deleting indexes and rebuilding yields the same lookup results
- stale index metadata is detectable

Invariant:

```text
indexes are disposable read models
```

## Deferred Tests

Semantic-search validation:

- semantic search results resolve back to concrete node IDs
- semantic search cannot create, update, accept, or retire memory
- semantic search can be deleted without losing accepted knowledge

## Worker And Concurrency Tests

Validate autonomous worker behavior.

Acceptance tests:

- only one worker holds the leader lease at a time
- expired lease can be claimed by another worker
- old leader cannot write after losing lease
- concurrent duplicate proposals produce one accepted state and stale rejections
- concurrent independent proposals are accepted independently
- accepted count plus rejected count equals submitted count

Invariant:

```text
distributed workers cannot corrupt accepted memory
```

## Recovery Tests

Validate restart and repair behavior.

Acceptance tests:

- process restart preserves pending proposals
- process restart preserves accepted Markdown and audit
- rebuilding derived indexes after restart succeeds
- partially written proposal is ignored or marked failed
- failed leader processing attempt does not mutate accepted Markdown
- migrations are idempotent after restart

Invariant:

```text
memory survives process failure without becoming ambiguous
```

## Manual Validation Checklist

Before a release, run through one human-readable scenario:

1. Create `acme/rbac/rbac.acls`.
2. Submit a competing create proposal and confirm stale rejection.
3. Update `rbac.acls` from version 1 to version 2.
4. Submit a stale update based on version 1 and confirm rejection.
5. Retire a related legacy node with a tombstone.
6. Rebuild manifest and lexical index from disk.
7. Read the audit log and confirm a human can explain the full history.

The release is not valid if this scenario requires inspecting private runtime
state instead of accepted Markdown, snapshots, and JSONL audit.

## Release Gate

A protocol release is valid only when:

- `poetry run pytest` passes
- every new protocol rule has at least one failing-first test
- every accepted write path has stale rejection coverage
- every persisted format has a round-trip test
- every derived index has a rebuild test
- docs describe the invariant that each test protects

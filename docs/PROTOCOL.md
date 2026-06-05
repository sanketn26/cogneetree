# Cogneetree Protocol

Cogneetree is a governed, human-readable wiki for autonomous agents. It gives
agents durable residual memory without letting them mutate shared memory
directly.

This document defines the protocol — concepts, invariants, identity, write
model, lifecycle, and the test contract that defends every invariant. It is
language-agnostic. For the Go implementation plan, see
[IMPLEMENTATION.md](IMPLEMENTATION.md).

---

# Part 1: Concepts

## Core Invariant

```text
agents read accepted Markdown
agents submit proposals
one leader admits writes for a scope
accepted state is one Markdown node per org + area + node
proposals, acceptances, rejections, and updates are JSONL audit events
indexes are derived read models
```

Agents may be autonomous in their work, but they are not sovereign over
shared memory.

## Concept Model

```text
organization
  area
    content node
      accepted Markdown
      versions
      audit events
      edges
```

Identity vocabulary:

```text
org_id       # tenant, team, company, or governance boundary
area_id      # domain boundary within an org, such as rbac or billing
node_id      # stable node identity inside an area
proposal_id  # immutable proposed change identity
version      # accepted state version (monotonic per node)
```

The identity invariant:

```text
one active accepted state per org_id + area_id + node_id
```

Do not bake `org_id` into `node_id`. The conceptual node `rbac.acls` means
the same thing across organizations even though each org keeps its own
accepted version.

## Human-Readable State

Accepted memory is Markdown with structured frontmatter.

```markdown
---
id: rbac.acls
org_id: acme
area: rbac
title: ACLs
node_type: policy
authority: normative
status: accepted
version: 5
updated_at: 2026-05-21T10:16:01Z
updated_by: agent-auth-reviewer
accepted_by: rbac-leader
change_reason: Contractor access reduced after policy update.
review_after: 2026-08-21
aliases:
  - access control lists
  - authorization grants
tags:
  - rbac
  - access
---

# ACLs

Accepted access control policy lives here.
```

Humans and agents read the Markdown first. JSON is for audit and transport,
not the reading path.

## Node Types

```text
policy
decision
procedure
fact
runbook
lesson
interface_contract
open_question
```

Agents should treat node types differently. A `policy` constrains behavior.
A `lesson` is guidance. An `open_question` is not accepted truth.

## Authority Levels

```text
normative      # binding unless current instructions override it
advisory       # useful guidance
observational  # historical or empirical note
```

Authority is independent of node type. A procedure can be normative; a
lesson is usually advisory.

## Write Model

The write path is always centralized:

```text
agent
  -> proposal
  -> pending log
  -> scoped leader
  -> accepted Markdown or stale rejection
  -> audit event
  -> derived index refresh
```

A future long-running worker may wrap this flow as a daemon that claims a
leader lease, processes pending proposals, and renews. The worker is
orchestration around the public memory API, not a second protocol path.

Every accepted update must answer:

```text
what changed
who proposed it
who accepted it
why it changed
which version it replaced
which sources justified it
```

## Versioning and Staleness

Updates carry an expected current version.

```text
missing node on create -> accept if no accepted state exists
missing node on update -> reject_missing_current
expected version mismatch -> reject_stale
expected version match -> accept version + 1
```

Rejected proposals must not alter accepted Markdown.

## Version History

History is append-only audit plus snapshots of every accepted version. No
heavy VCS inside the product.

```text
wiki/
  acme/
    rbac/
      acls.md
audit/
  acme/
    rbac.acls.jsonl
snapshots/
  acme/
    rbac/
      acls/
        v0004.md
        v0005.md
```

An accepted update emits an event like:

```json
{
  "type":"accepted",
  "org_id":"acme",
  "area":"rbac",
  "node_id":"rbac.acls",
  "proposal_id":"p_123",
  "from_version":4,
  "to_version":5,
  "proposed_by":"agent-auth-reviewer",
  "accepted_by":"rbac-leader",
  "reason":"Contractor access reduced after policy update.",
  "timestamp":"2026-05-21T10:16:01Z"
}
```

In the Postgres implementation, snapshots are simply rows in
`accepted_versions`; the on-disk snapshot layout is optional human-browsable
export.

## Change Sets

A change set groups one or more node updates under one reason.

Use it when one real-world event touches multiple nodes:

```text
rbac.auth
rbac.acls
rbac.authorizations
```

The leader accepts a change set only when **every** included node version
still matches its expected base version. One stale node rejects the set.

## Access Scope

Organization is the governance boundary. Area is the first practical policy
boundary inside an organization.

Track at least:

```text
read_scope
propose_scope
accept_scope
```

The protocol carries these fields from day one even if enforcement starts
simple.

## Links and Conflicts

The tree locates memory. Edges describe meaning.

```text
depends_on
related_to
supersedes
conflicts_with
```

Edges connect accepted nodes and are admitted only by the leader. The tree
is not replaced by the graph.

Contradictions are explicit:

```yaml
conflicts_with:
  - billing.contractor_access
supersedes:
  - rbac.legacy_acl_policy
```

Version pinning rules:

| Edge | From | To | Why |
|---|---|---|---|
| `depends_on` | latest | latest | follows current state |
| `related_to` | latest | latest | structural, not version-specific |
| `supersedes` | pinned | pinned | "X replaced Y" pins to the moment |
| `conflicts_with` | pinned | pinned | conflict is between specific states |

## Tombstones

Deletion is an accepted state, not absence.

```markdown
---
id: rbac.old_acl_policy
status: retired
retired_reason: Replaced by rbac.acls.
superseded_by:
  - rbac.acls
---

# Retired: Old ACL Policy
```

Recreating a retired node requires an explicit `recreate` proposal, never a
plain `create`. Agents must decide deliberately.

## Lookup and Indexing

The source of truth is the tree.

```text
Markdown tree     = accepted knowledge
JSONL audit       = provenance and history
manifest          = derived navigation cache
inverted index    = derived exact lookup cache
semantic index    = optional discovery cache, deferred
```

Start with structural lookup by `org_id + area_id + node_id`. Add a manifest
and lexical index before considering semantic search.

Semantic search may *suggest* candidate nodes; it must not decide truth.

## Adapter Boundary

Network and tool adapters stay thin:

```text
HTTP/gRPC/MCP
  -> public memory API
  -> log/store
```

Adapters must not duplicate proposal admission, stale rejection,
versioning, or materialization rules.

---

# Part 2: Validation Contract

This part defines how to validate the protocol as governed residual memory
for autonomous agents. The goal is not only that APIs work; the goal is
that accepted memory is durable, attributable, human-readable, and
protected from unsafe concurrent agent writes.

## Validation Principles

Every test should protect one invariant.

```text
accepted Markdown is truth
agents propose, leaders admit
one active state exists per org + area + node
updates require the expected base version
rejections never mutate accepted state
audit events explain who, what, when, and why
indexes are rebuildable caches
```

If a feature cannot be validated against an invariant, do not add it to the
core protocol yet.

## Test Layers

```text
unit tests          # dataclasses, validators, path rules, status transitions
protocol tests      # proposal, acceptance, stale rejection, supersession
storage tests       # Markdown, JSONL, snapshots, SQL rows, migrations
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

Stable fixtures make audit assertions readable and prevent drift.

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

## Update and Version Tests

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

## Provenance and Audit Tests

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

- retiring a node writes accepted Markdown with `status: retired`
- retired node remains addressable by exact lookup
- active reads exclude retired nodes by default
- explicit history reads include retired nodes
- recreating a retired node requires an explicit recreate proposal

Invariant:

```text
deleted knowledge is explicit, not silent absence
```

## Access Boundary Tests

- agent without read scope cannot read restricted node
- agent without propose scope cannot submit proposal
- non-leader cannot accept proposal
- leader for one area cannot accept another area's proposal
- cross-org reads and writes are denied unless explicitly scoped

Invariant:

```text
organization and area are governance boundaries
```

## Conflict and Edge Tests

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

Semantic search is deferred until lexical search demonstrably falls short.
See `IMPLEMENTATION.md` Phase 9 for the rationale.

## Worker and Concurrency Tests

- only one accepted version exists per node at any time
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

Before a release, walk through one human-readable scenario:

1. Create `acme/rbac/rbac.acls`.
2. Submit a competing create proposal and confirm stale rejection.
3. Update `rbac.acls` from version 1 to version 2.
4. Submit a stale update based on version 1 and confirm rejection.
5. Retire a related legacy node with a tombstone.
6. Rebuild manifest and lexical index from disk.
7. Read the audit log and confirm a human can explain the full history.

The release is not valid if this scenario requires inspecting private
runtime state instead of accepted Markdown, snapshots, and JSONL audit.

## Release Gate

A protocol release is valid only when:

- the test suite passes
- every new protocol rule has at least one failing-first test
- every accepted write path has stale rejection coverage
- every persisted format has a round-trip test
- every derived index has a rebuild test
- docs describe the invariant that each test protects

---

# Part 3: Non-Goals

To prevent scope drift, the following are explicitly out of scope for the
protocol:

- **Semantic / vector search** in the write path. Lexical search with
  weighted columns and trigram fuzzy matching is sufficient at the target
  scale. See `IMPLEMENTATION.md` Phase 9 for the supported search path.
- **Dedicated search engines** (Sonic, Meilisearch, Typesense, Elasticsearch,
  OpenSearch). Postgres FTS plus `pg_trgm` is the supported path until a
  tenant exceeds ~1M current nodes.
- **MCP adapter** until the HTTP adapter is stable. MCP, when added, must be
  a thin wrapper over the same public API as HTTP.
- **Embedded UI**. Cogneetree exposes APIs; UIs are downstream concerns.
- **Free-form schema-on-read**. Frontmatter is validated at admission; agents
  cannot smuggle new node types or authorities through unsanitized fields.

These non-goals are durable. Revisit them only with evidence of a real,
specific need that the supported path cannot meet.

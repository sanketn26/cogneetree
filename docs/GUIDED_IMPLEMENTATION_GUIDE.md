# Guided Implementation Guide

Cogneetree is a governed, human-readable LLM wiki for autonomous agents.
It gives agents durable residual memory without letting them mutate shared
memory directly.

For sequence and acceptance criteria, use [ROADMAP.md](ROADMAP.md). For the
validation contract, use [VALIDATION_AND_TESTING.md](VALIDATION_AND_TESTING.md).

## Core Invariant

```text
agents read accepted Markdown
agents submit proposals
one leader admits writes for a scope
accepted state is one Markdown node per org + area + node
proposals, acceptances, rejections, and updates are JSONL audit events
indexes are derived read models
```

Agents may be autonomous in work, but they are not sovereign over shared memory.

## Concept Model

```text
organization
  area
    content node
      accepted Markdown
      versions
      audit events
```

Use these identities:

```text
org_id       # tenant, team, company, or governance boundary
area_id      # domain boundary, such as rbac or billing
node_id      # stable node identity inside the area
proposal_id  # immutable proposed change identity
version      # accepted state version
```

The invariant is:

```text
one active accepted state per org_id + area_id + node_id
```

Do not bake `org_id` into `node_id`. Let `rbac.acls` mean the same conceptual
node in different organizations.

## Human-Readable State

Accepted memory is Markdown with small frontmatter.

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

Keep JSON out of the normal reading path. Humans and agents should read accepted
Markdown first.

## Node Types

Start with a small enum:

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

Agents should treat node types differently. A `policy` can constrain behavior.
A `lesson` is guidance. An `open_question` is not accepted truth.

## Authority Levels

Use authority to tell agents how strongly to rely on a node.

```text
normative      # binding unless current instructions override it
advisory       # useful guidance
observational  # historical or empirical note
```

Authority is separate from node type. A procedure can be normative; a lesson is
usually advisory.

## Write Model

The write path remains centralized:

```text
agent
  -> proposal
  -> pending log
  -> scoped leader
  -> accepted Markdown or stale rejection
  -> audit event
  -> derived index refresh
```

Every accepted update must answer:

```text
what changed
who proposed it
who accepted it
why it changed
which version it replaced
which sources justified it
```

A future long-running worker may wrap this flow as a daemon that claims a
leader lease, renews it, and processes pending proposals. Keep that worker as
orchestration around the public memory API, not as a second protocol path.

## Version History

Keep version history as append-only audit and snapshots. Do not build a heavy
VCS inside the product.

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

An accepted update should create an audit event:

```json
{"type":"accepted","org_id":"acme","area":"rbac","node_id":"rbac.acls","proposal_id":"p_123","from_version":4,"to_version":5,"proposed_by":"agent-auth-reviewer","accepted_by":"rbac-leader","reason":"Contractor access reduced after policy update.","timestamp":"2026-05-21T10:16:01Z"}
```

Snapshots are enough for the first implementation. Generated diffs can be added
later as a read model.

## Change Sets

A change set groups one or more node updates under one reason.

Use it when one real-world event touches multiple nodes:

```text
rbac.auth
rbac.acls
rbac.authorizations
```

The leader may accept a change set only when all included node versions still
match their expected base versions. If one node is stale, reject the change set.

## Staleness

Updates must include the expected current version.

```text
missing node on create -> accept if no accepted state exists
missing node on update -> reject_missing_current
expected version mismatch -> reject_stale
expected version match -> accept version + 1
```

Rejected proposals must not alter accepted Markdown.

## Access Scope

Organization is a governance boundary. Area is the first practical policy
boundary inside an organization.

Track at least:

```text
read_scope
propose_scope
accept_scope
```

Enforcement can stay simple initially, but the protocol should carry the fields.

## Links And Conflicts

The tree is for location. Edges are for meaning.

```text
depends_on
related_to
supersedes
conflicts_with
```

Do not replace the tree with a graph. Graph edges connect accepted nodes and are
created only by leader-admitted updates.

Contradictions should be explicit:

```yaml
conflicts_with:
  - billing.contractor_access
supersedes:
  - rbac.legacy_acl_policy
```

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

Agents need to know when knowledge was intentionally retired.

## Lookup And Indexing

The source of truth is the tree.

```text
Markdown tree = accepted knowledge
JSONL audit = provenance and history
manifest = derived navigation cache
inverted index = derived exact lookup cache
semantic index = optional discovery cache, deferred
```

Start with structural lookup by `org_id + area_id + node_id`. Add a boring
manifest and lexical index before semantic search.

Semantic search may suggest candidate nodes. It must not decide truth.

## Current Code Attachment Points

Until Phase 1 lands, modules still use the earlier decision/context naming.
Evolve them without reintroducing old retrieval architecture.

Avoid relying on a hand-maintained file inventory in this guide. Use
`rg --files src/cogneetree` for the exact current module list, then attach new
work by responsibility:

- protocol records and statuses live with the protocol model
- leader admission logic owns accept/reject decisions
- stores and logs own persistence, snapshots, and audit events
- public memory APIs own reads, proposal submission, and worker boundaries
- CLI and future transports stay thin over public APIs

Prefer renaming concepts gradually:

```text
area/context_key -> org + area + node
decision -> accepted node version
decision_events -> audit events
accepted_contexts -> accepted node states
```

## Adapter Boundary

Network and tool adapters should be thin:

```text
HTTP/gRPC/MCP
  -> public memory API
  -> log/store
```

Adapters must not duplicate proposal admission, stale rejection, versioning, or
materialization rules.

## Tests

Run:

```bash
poetry run pytest
```

Important protocol coverage:

- first proposal for a node is accepted
- second proposal for the same base version is rejected as stale
- different nodes are accepted independently
- accepted update increments version
- stale update does not alter accepted Markdown
- proposal, acceptance, rejection, and supersede events are written
- snapshots are written for accepted versions
- tombstones are accepted states
- derived indexes can be rebuilt from Markdown and audit

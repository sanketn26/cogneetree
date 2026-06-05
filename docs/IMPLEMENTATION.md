# Cogneetree Implementation Plan

This document is the implementation roadmap for Cogneetree. It covers the Go
stack, the repository shape, the phase sequence, and the acceptance criteria
that gate each phase.

For protocol concepts and the test contract, see [PROTOCOL.md](PROTOCOL.md).
For per-phase deep dives (structs, signatures, SQL, tests), see the files
under [implementation/](implementation/).

---

# Part 1: Stack and Layout

## Implementation Stack

```text
language       Go (toolchain ≥ 1.23)
database       Postgres 16+
driver         github.com/jackc/pgx/v5
migrations     pressly/goose (single binary, embeds SQL)
http router    net/http with stdlib ServeMux (Go 1.22+ pattern syntax)
config         caarlos0/env/v11 + stdlib flag
logging        log/slog
metrics        prometheus/client_golang
tracing        go.opentelemetry.io/otel
testing        stdlib testing + testcontainers-go for Postgres
linter         golangci-lint
module path    github.com/sanketn26/cogneetree
```

No ORMs. Hand-written SQL with `pgx`. The schema is the contract.

## Repository Shape

```text
cogneetree/
  cmd/
    cogneetree/          # CLI + daemon entrypoint
  pkg/
    protocol/            # records, identity, errors, statuses (wire vocabulary)
    cogneetreeclient/    # external Go client SDK
  internal/
    store/               # Postgres-backed read/write
    leader/              # admission logic, CAS, supersede
    audit/               # DB + JSONL audit writers
    snapshot/            # snapshot storage and retrieval
    edges/               # graph relationships
    search/              # FTS, pg_trgm, manifest
    worker/              # long-running leader loop
    httpadapter/         # REST adapter
    telemetry/           # metrics, traces, structured logs
    config/              # env + flag parsing
    frontmatter/         # YAML+Markdown render/parse
  docs/
    implementation/      # one Markdown file per phase
  migrations/            # SQL migration files
  observability/         # Grafana dashboards, prometheus rules
  docker/                # local-dev compose
  go.mod
  go.sum
```

`internal/` is private to this module. `pkg/` is public and stable. The
protocol package lives in `pkg/` because its types appear in the public
SDK's method signatures — Go forbids importing `internal/...` from outside
the module, so any DTO that crosses the SDK boundary must live in `pkg/`.
`cmd/` is thin — it wires config, opens a DB pool, and starts a server or
daemon.

---

# Part 2: Phase Index

Each phase is implemented and shipped in order. Do not skip phases. Each
phase ends with a green test suite and a runnable subset of the system.

| Phase | Title | Doc |
|---|---|---|
| 0 | Foundations | [phase-00-foundations.md](implementation/phase-00-foundations.md) |
| 1 | Protocol Core | [phase-01-protocol-core.md](implementation/phase-01-protocol-core.md) |
| 2 | Postgres Schema and Migrations | [phase-02-postgres-schema.md](implementation/phase-02-postgres-schema.md) |
| 3 | Leader Admission | [phase-03-leader-admission.md](implementation/phase-03-leader-admission.md) |
| 4 | Supersede and Update | [phase-04-supersede-update.md](implementation/phase-04-supersede-update.md) |
| 5 | Provenance, Snapshots, Change Sets | [phase-05-provenance-snapshots.md](implementation/phase-05-provenance-snapshots.md) |
| 6 | Node Metadata and Validation | [phase-06-metadata-validation.md](implementation/phase-06-metadata-validation.md) |
| 7 | Tombstones and Lifecycle | [phase-07-tombstones.md](implementation/phase-07-tombstones.md) |
| 8 | Graph Edges | [phase-08-graph-edges.md](implementation/phase-08-graph-edges.md) |
| 9 | Search and Derived Indexes | [phase-09-search-indexes.md](implementation/phase-09-search-indexes.md) |
| 10 | Long-Running Worker Loop | [phase-10-worker-loop.md](implementation/phase-10-worker-loop.md) |
| 11 | HTTP Adapter | [phase-11-http-adapter.md](implementation/phase-11-http-adapter.md) |
| 12 | Observability | [phase-12-observability.md](implementation/phase-12-observability.md) |

## Guided Progress Path

```text
Phase 0  -> repo bootstrapped, CI passes empty test suite, Postgres connects
Phase 1  -> protocol records compile, table-driven unit tests pass
Phase 2  -> migrations create schema on a fresh DB, RLS active, partitions exist
Phase 3  -> first proposal accepted, second rejected as stale, audit row written
Phase 4  -> update with matching version accepted; stale update rejected
Phase 5  -> snapshots reconstruct prior versions; change sets are atomic
Phase 6  -> invalid frontmatter rejected before acceptance
Phase 7  -> retire writes accepted status; active reads exclude retired
Phase 8  -> edges queryable; neighborhood API works; conflicts surfaced
Phase 9  -> manifest, alias, and lexical indexes rebuild from Markdown
Phase 10 -> daemon claims work, processes pending, exits cleanly on SIGTERM
Phase 11 -> HTTP endpoints satisfy the validation contract end-to-end
Phase 12 -> metrics, structured logs, and traces emitted for hot paths
```

Each phase's doc closes with an "Exit Criteria" section. Do not move to the
next phase until those criteria are green on `main`.

---

# Part 3: Phase Summaries

This part is a quick scan of what each phase delivers, sourced from the
per-phase docs. Open the linked doc for the full plan.

## Phase 0: Foundations

Goal: bootstrap the toolchain, repository layout, and local development loop.

Delivers a working Go module, a Postgres container, the migration tool wired
to `make migrate`, an empty `go test ./...` that exits 0, a CI workflow, and
a `cogneetree version` binary.

## Phase 1: Protocol Core

Goal: model identity, proposals, accepted versions, and resolutions as pure
Go types with no Postgres dependency.

Delivers `pkg/protocol` with `Identity`, `Proposal`, `AcceptedVersion`,
`Resolution`, status enums, sentinel and typed errors, ID generation, and
table-driven validators. Phase 1 is fully unit-testable without a database.

Acceptance:

- `org_id + area_id + node_id` is the canonical identity in new records
- accepted Markdown will include stable frontmatter (full schema in Phase 6)
- existing protocol invariants compile and validate
- docs use memory node language consistently

## Phase 2: Postgres Schema and Migrations

Goal: stand up the durable storage layer.

Delivers `migrations/` with eight initial migrations: extensions, orgs,
partitioned `accepted_versions` with a partial unique index on `is_current`,
partitioned `pending_proposals`, range-partitioned `audit_events`, RLS
policies, and database roles for app vs. leader.

Acceptance:

- new Postgres database initializes through migrations
- repeated initialization is safe
- tests assert expected tables and migration rows exist
- no application code creates tables outside migrations
- partial unique index prevents two current versions per node
- RLS blocks cross-org reads on the app role

## Phase 3: Leader Admission

Goal: first proposal for a node is accepted, second rejected as stale.

Delivers `internal/store` write methods, `internal/leader` admission logic
using `SELECT ... FOR UPDATE SKIP LOCKED`, the `internal/audit` write path,
and `pkg/cogneetreeclient.DirectAPI` wiring them together.

Acceptance:

- create proposals admitted end-to-end against Postgres
- concurrent creates yield exactly one accepted state and N-1 stale rejections
- rejected proposals do not alter accepted Markdown
- audit events written for proposal, accepted, and rejected outcomes

## Phase 4: Supersede and Update

Goal: accepted memory can change safely.

Delivers update and retire proposal kinds with compare-and-set semantics
enforced by the `(org, area, node, version)` PK plus the partial unique
index. `from_version` recorded; `proposal_superseded` audit events emitted.

Acceptance:

- update with matching expected version is accepted; version increments
- update with stale expected version is rejected
- update against missing node is rejected as missing-current
- prior versions remain reconstructable
- materialized Markdown reflects the newest version
- tests cover missing, stale, and accepted update paths

## Phase 5: Provenance, Snapshots, Change Sets

Goal: every accepted update is attributable and reconstructable.

Delivers `change_sets` table, `internal/snapshot` package for historical
reconstruction, JSONL audit becomes co-authoritative with the DB, and a
`Rebuilder` that reconstructs SQL state from Markdown + JSONL.

Acceptance:

- accepted versions reconstructable from snapshots byte-for-byte
- each accepted update records who, what, when, and why
- multi-node change sets reject if any expected version is stale
- rejected change sets do not alter any accepted Markdown
- Rebuilder reproduces accepted state from authoritative sources

## Phase 6: Node Metadata and Validation

Goal: memory is interpretable by agents and safe to accept.

Delivers full `Frontmatter` schema with `NodeType`, `Authority`, aliases,
tags, review/expire fields; body size and UTF-8 validation; `internal/
frontmatter` render/parse round-trip with unknown-field preservation.

Acceptance:

- invalid frontmatter rejected before acceptance
- node ID and path rules enforced
- required metadata present
- expired or review-due nodes visible to readers
- tests cover each validation rule

## Phase 7: Tombstones and Lifecycle

Goal: deletion is explicit.

Delivers retire and recreate proposal kinds, default read filtering of
retired nodes, audit `node_retired` and `node_recreated` events, and
tombstone Markdown rendering.

Acceptance:

- retired nodes remain discoverable via explicit reads
- retired nodes do not appear as active accepted guidance by default
- recreate is required to revive a retired node
- audit distinguishes retire from supersede
- tests prove deletion is not silent absence

## Phase 8: Graph Edges

Goal: relationships are queryable.

Delivers `edges` table with version-pinning rules per edge type, edge
derivation from frontmatter on admission, neighborhood and dependency
closure queries with depth caps, and conflict surfacing during admission.

Acceptance:

- edges queryable from and to a node
- dependencies and dependents queryable
- conflicts visible to proposal validation
- neighborhood returns node, ancestors, children, and graph neighbors
- edges cannot point to missing accepted nodes
- only leader-admitted updates create or modify edges

## Phase 9: Search and Derived Indexes

Goal: lookup is fast without making indexes authoritative.

Delivers stored generated `body_tsv` column with weighted sections,
partial GIN index over current active nodes, `pg_trgm` fuzzy match on
titles, alias and tag lookup tables, a manifest, and `RebuildIndexes`.

Search is hot-tier only by default. Historical search requires explicit
on-demand index creation.

Acceptance:

- indexes deletable and rebuildable from Markdown plus audit
- exact lookup does not require search
- lexical lookup returns ranked node IDs and paths
- stale indexes detectable
- benchmark targets met (p50 < 5ms at 10k nodes)

Non-goal: semantic search in the write path.

## Phase 10: Long-Running Worker Loop

Goal: a daemon that listens for proposals and processes them.

Delivers `internal/worker` with a NOTIFY-driven listener (no polling), an
area-keyed worker pool, graceful shutdown on SIGTERM/SIGINT, and crash
recovery that requeues abandoned claims.

Acceptance:

- worker processes pending proposals until the queue is empty
- worker does not poll when idle
- worker stops cleanly within a configurable drain timeout
- worker does not resolve proposals after losing access
- tests use short intervals and deterministic stop conditions

## Phase 11: HTTP Adapter

Goal: expose the memory protocol over HTTP.

Delivers REST endpoints matching the validation contract, per-request RLS
via Postgres GUC, bearer-token authentication with scoped permissions,
structured request logging, and graceful shutdown.

Core endpoints:

```text
POST /v1/orgs/{org}/areas/{area}/nodes/{node}/proposals
GET  /v1/orgs/{org}/areas/{area}/nodes/{node}
GET  /v1/orgs/{org}/areas/{area}/nodes/{node}/versions
GET  /v1/orgs/{org}/areas/{area}/nodes/{node}/edges
GET  /v1/orgs/{org}/areas/{area}/nodes/{node}/neighborhood
GET  /v1/orgs/{org}/areas/{area}/tree
GET  /v1/orgs/{org}/search
GET  /v1/proposals/{proposal_id}
POST /v1/orgs/{org}/change-sets
GET  /healthz
GET  /readyz
```

Acceptance:

- proposal submission returns `proposal_id` with 202
- resolution endpoint returns pending or resolved state
- node reads return accepted Markdown and metadata with ETag
- tree endpoint returns folder-like nodes
- graph endpoints return accepted relationships
- search endpoint returns lexical matches
- RLS enforced per request via Postgres GUC
- transport tests cover stale proposal behavior through public APIs

## Phase 12: Observability

Goal: production-grade visibility.

Delivers Prometheus metrics for every hot path, OpenTelemetry traces
across HTTP → leader → store → pgx, structured slog logs with a stable
field vocabulary, `/healthz` vs `/readyz` distinction, and sample Grafana
dashboards.

Acceptance:

- every metric in the catalog exported on `/metrics`
- traces propagate end-to-end with consistent attributes
- `node_id`, `proposal_id` never used as metric labels (cardinality discipline)
- structured logs use standard field names
- benchmark overhead under 1% (metrics) and 2% (tracing) of request CPU

---

# Part 4: Deferred Phases

These are explicitly out of scope for the current plan and will be revisited
only when the supported path proves insufficient.

## Deferred: gRPC Adapter

A gRPC server may be added after HTTP is stable. It must reuse
`pkg/cogneetreeclient` as the API surface and add no new admission paths.

## Deferred: MCP Adapter

MCP should be added only after the transport-independent protocol is
stable. The MCP adapter must be a thin wrapper over the same public APIs
as HTTP and must not introduce a separate memory mutation path.

## Deferred: Semantic Search

Semantic / vector search is not part of the protocol. Lexical search with
weighted columns satisfies the agentic-memory use case at the target scale
(mid-org: ~10k current nodes; large-org: ~100k). Revisit only if a tenant
exceeds ~1M current nodes and demonstrates concrete retrieval failures that
lexical ranking cannot fix.

## Deferred: Multi-Region Replication

Single-region Postgres with logical replication for read replicas is the
supported path. Active-active multi-region would require a CRDT-shaped
protocol change and is out of scope.

---

# Part 5: Scale Assumptions

Target scale, used to validate that the architecture is appropriately sized.

| Tenant Size | Current Nodes | Versions Per Node | 3-Year Storage |
|---|---|---|---|
| Mid-org (200–2000 employees) | 5k–20k | 10–15 | 3–5 GB |
| Large org (5k+) | 20k–100k | 10–20 | 15–30 GB |
| Worst plausible single tenant | 100k–1M | 10–20 | 50–200 GB |

Cogneetree is not Wikipedia-scale. It is governed org memory. The
architecture assumes:

- one Postgres per fleet, partitioned by `org_id`
- hot tier = current accepted versions (indexed, in RAM)
- cold tier = prior versions (queryable, not indexed by default)
- search workload dominated by exact lookups; full-text search is the minority case
- write contention from many concurrent agents per area is the real
  scaling axis, not byte volume

If a tenant ever exceeds 1M current nodes, revisit. The org partition
boundary lets that tenant graduate to a dedicated Postgres without
rewriting the others.

# CLAUDE.md

Guidance for Claude Code working in this repository.

## What This Project Is

Cogneetree is a governed wiki for autonomous agents. The protocol is the
product. Read [docs/PROTOCOL.md](docs/PROTOCOL.md) before suggesting any
change that touches identity, admission, versioning, edges, or indexes.

Implementation language: **Go**. The Python implementation was removed; do
not suggest reintroducing it.

## Source Of Truth For Decisions

| Question | Where To Look |
|---|---|
| What is the protocol? | [docs/PROTOCOL.md](docs/PROTOCOL.md) |
| What gets built in what order? | [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) |
| What's in phase N? | [docs/implementation/phase-NN-*.md](docs/implementation/) |
| What style do I follow? | [AGENTS.md](AGENTS.md) |

If a phase doc and PROTOCOL.md disagree, PROTOCOL.md wins. If a phase doc
and IMPLEMENTATION.md disagree, the phase doc wins (it is the deeper plan).

## Hard Rules

- Do not reintroduce Python. The migration to Go is final.
- Do not propose SQLite. Postgres is the chosen engine; see
  [phase-02-postgres-schema.md](docs/implementation/phase-02-postgres-schema.md).
- Do not propose vector / semantic search. It is an explicit non-goal in
  PROTOCOL.md.
- Do not propose dedicated search engines (Sonic, Meilisearch, Typesense,
  Elasticsearch, OpenSearch). Postgres FTS plus `pg_trgm` is the path.
- Do not add ORMs. Hand-written SQL with `pgx`.
- Do not add MCP, gRPC, or UI work before HTTP (Phase 11) is stable.
- No function exceeds 50 lines.
- Adapters (HTTP, gRPC, MCP) must be thin wrappers over the public memory
  API. They never duplicate admission, validation, versioning, or
  materialization logic.

## Phase Discipline

Implementation is sequential. Do not jump phases. Each phase has explicit
**Exit Criteria** at the bottom of its doc. When asked to implement,
confirm which phase the work belongs to and check the prior phase's exit
criteria are green before proposing code.

If a request straddles multiple phases, split it and surface the split
before starting.

## Code Style (Go)

- Standard library first. Add a dependency only if it carries real weight.
- Errors wrap with `fmt.Errorf("...: %w", err)`. Never swallow.
- `context.Context` is the first parameter on any I/O-bound function.
- Use `time.Now().UTC()` everywhere; never local time.
- Small interfaces, defined at the consumer.
- Struct tags only when a serializer reads them. No speculative tags.
- Sentinel errors checked via `errors.Is`; typed errors via `errors.As`.
- No comments that explain what the code does. Only the non-obvious *why*.

## Testing

- Table-driven tests for every protocol invariant.
- Integration tests use `testcontainers-go/modules/postgres`.
- Every accepted write path needs at least one stale-rejection test.
- Every persisted format needs a round-trip test.
- Every derived index needs a rebuild test.

Run:

```bash
go test ./...
```

## When Asked To "Add A Feature"

1. Identify the invariant the feature must protect or extend.
2. Identify the phase it belongs to.
3. Check whether PROTOCOL.md already covers it. If not, propose a PROTOCOL.md
   amendment **before** writing code.
4. If it requires a non-goal (semantic search, MCP, SQLite, dedicated search
   engine), surface the conflict and stop. Do not silently work around it.

## When Asked To "Refactor"

Refactors are easy to justify and easy to overdo. Default answer: only if
it removes duplication, simplifies an interface used in multiple places, or
unblocks the next phase. Do not refactor for taste.

## When Asked To "Add A Test"

Identify the invariant. If there isn't one, the test is checking
implementation detail, not behavior — push back.

## File Layout Reminders

```text
cmd/cogneetree/              # entrypoint
pkg/protocol/                # records, errors, validation (public wire vocabulary)
pkg/cogneetreeclient/        # public Go SDK
internal/store/              # pgx-backed read/write
internal/leader/             # admission and CAS
internal/audit/              # DB + JSONL audit
internal/snapshot/           # history reads
internal/edges/              # graph
internal/search/             # FTS, manifest
internal/worker/             # NOTIFY-driven daemon
internal/httpadapter/        # REST
internal/telemetry/          # metrics, traces, logs
migrations/                  # NNNN_name.sql via goose
docs/                        # PROTOCOL.md, IMPLEMENTATION.md, implementation/*
```

`internal/` is private to this module. `pkg/` is public and stable.
Protocol types live in `pkg/protocol` because they appear in the public
SDK's method signatures; Go forbids importing `internal/...` from outside
the module.

## Scope Boundaries

- Storage: Postgres only. No SQLite, no Mongo, no DynamoDB.
- Search: Postgres FTS + `pg_trgm` only. No Sonic, Meili, Typesense,
  Elastic, OpenSearch, Vespa.
- Transport: HTTP first (Phase 11). gRPC and MCP are deferred until HTTP
  is stable.
- Agents: agents are clients, not part of this repo. Cogneetree exposes a
  protocol; agents consume it.

## How To Talk About Progress

When summarizing work in this repo, name the phase. "Phase 3 leader
admission is green" is useful. "Did some Postgres work" is not.

## When In Doubt

Open the relevant phase doc. The exit criteria at the bottom tell you what
"done" looks like. If a request is incompatible with the exit criteria,
surface the conflict before writing code.

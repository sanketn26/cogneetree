# AGENTS.md

Keep this repo simple.

## Product Rule

Agents do not write shared memory directly. They propose memory changes.

For each org + area + node:

- one active accepted version
- one leader admits writes for a scope
- accepted state is human-readable Markdown
- proposals, acceptances, rejections, and supersessions are JSONL audit events
- competing proposals are rejected as stale and the agent re-evaluates

## Implementation Language

Cogneetree is implemented in Go. Module path: `github.com/sanketn26/cogneetree`.

## Code Rules

- Prefer boring Go. Standard library first.
- Keep packages small and focused on one responsibility.
- No function exceeds 50 lines.
- Follow SOLID principles.
- Apply well-known patterns (GoF, POSA, PoEAA) where they fit; do not force them.
- Add a test for every protocol invariant.
- Do not add UI, vector search, or framework adapters until the protocol is stable.
- Do not reintroduce hierarchy/retrieval architecture from the prior implementation.
- Adapters (HTTP, gRPC, MCP) must be thin wrappers over the public memory API.
- Indexes are derived caches. Markdown plus audit is the source of truth.

## Current Shape

```text
docs/PROTOCOL.md
docs/IMPLEMENTATION.md
docs/implementation/phase-00-foundations.md
docs/implementation/phase-01-protocol-core.md
docs/implementation/...
```

Source layout grows phase by phase under `cmd/`, `internal/`, and `pkg/` per
[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

## Go Style

- Exported types only when consumers outside the package need them.
- Errors wrap with `fmt.Errorf("...: %w", err)`; do not swallow.
- Use `context.Context` as the first parameter on any I/O-bound function.
- Use `time.Time` from `time.Now().UTC()` consistently; never local time.
- Prefer small interfaces defined at the consumer, not the producer.
- Use struct tags only where a serializer reads them. No speculative tags.
- Use `errors.Is` / `errors.As` for sentinel and typed error checks.

## Testing

Once Phase 0 has created `go.mod` and the module layout, run:

```bash
go test ./...
```

Until then, the repository contains only specifications under `docs/`; there
is no Go code to test.

The important behaviors to cover once code exists:

- first proposal for a node is accepted
- second create proposal for the same node is rejected as stale
- update with matching expected version is accepted and increments
- update with stale expected version is rejected without mutation
- proposal, accepted, rejected, and supersede events are written
- snapshots reconstruct any prior accepted version
- tombstones are accepted states, not deletes

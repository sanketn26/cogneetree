# Cogneetree

Governed residual memory for autonomous agents.

Cogneetree is a human-readable wiki for AI agents. It gives agents durable
memory without letting them mutate shared truth directly.

> Agents propose memory changes. A scoped leader admits one active accepted
> Markdown version per organization, area, and node. Competing proposals are
> rejected as stale and the agent must re-evaluate from the latest state.

## Why

Autonomous agents need durable memory, but they should not own the shared
truth directly. Cogneetree gives them a governed, human-readable knowledge
tree:

- organization, area, and node identity
- one active accepted Markdown version per node
- one scoped leader admitting writes for that scope
- stale proposals rejected with the latest state
- accepted updates recorded with attribution, reason, and source refs
- derived indexes for lookup, not as source of truth

## Documentation

Two top-level docs, plus per-phase implementation notes:

- [PROTOCOL.md](docs/PROTOCOL.md) — concepts, invariants, validation contract
- [IMPLEMENTATION.md](docs/IMPLEMENTATION.md) — Go stack, repo layout, phase plan
- [docs/implementation/](docs/implementation/) — one Markdown file per phase
  with structs, function signatures, SQL, and required tests

Start with PROTOCOL.md if you want to understand what Cogneetree is. Start
with IMPLEMENTATION.md if you want to build it.

## Status

The Python reference implementation has been removed. No Go code exists yet
— the repository currently contains only specifications. The Go
implementation starts at
[Phase 0](docs/implementation/phase-00-foundations.md), which creates
`go.mod`, the module layout, and the build/test toolchain.

## Development

There is nothing to build yet. Once Phase 0 is implemented:

```bash
go test ./...
```

Detailed setup, tooling, and module layout live in
[docs/implementation/phase-00-foundations.md](docs/implementation/phase-00-foundations.md).

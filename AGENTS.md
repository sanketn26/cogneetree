# AGENTS.md

Keep this repo simple.

## Product Rule

Agents do not write shared memory directly. They propose decisions.

For each decision area:

- one active accepted decision
- one leader admits writes
- accepted state is Markdown
- proposals and resolutions are JSONL audit events
- competing proposals are rejected as stale

## Code Rules

- Prefer boring Python.
- Keep modules small.
- Keep the core dependency-free.
- Add tests for every protocol rule.
- Do not add UI, vector search, storage backends, or framework adapters until the
  protocol is stable.
- Do not reintroduce the old hierarchy/retrieval architecture.
- Do not have any methods or functions exceed more that 50 lines
- Follow SOLID prinviples.
- Apply Well known patterns like GOF, POSA, Implementation Patterns and PoEAA

## Current Shape

```text
src/cogneetree/protocol.py
src/cogneetree/store.py
src/cogneetree/leader.py
src/cogneetree/cli.py
tests/test_decision_protocol.py
docs/DISTRIBUTED_IMPLEMENTATION.md
```

## Python Style

- Use dataclasses for protocol records.
- Use `StrEnum` for statuses.
- Use `pathlib.Path` for files.
- Use explicit names.
- Use early returns.
- Avoid clever abstractions.

## Testing

Run:

```bash
poetry run pytest
```

The important behaviors:

- first proposal for an area is accepted
- second proposal for the same area is rejected as stale
- different areas are accepted independently
- rejected proposals do not alter accepted Markdown
- events and rejection audit files are written


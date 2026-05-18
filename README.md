# Cogneetree

Distributed decision memory for agentic programming.

Cogneetree is restarting around one simple protocol:

> Agents propose decisions. A scoped leader admits one active decision per area.
> Accepted decisions become Markdown. Competing proposals are rejected as stale
> and the agent must re-evaluate from the latest state.

This repo intentionally keeps the implementation small. Markdown is the readable
source of truth. JSONL is the event log. Python is the reference
implementation.

## Why

Distributed agents do not need a magical conflict resolver first. They need a
simple way to avoid polluting shared memory:

- one decision area
- one active accepted decision
- one leader admitting writes
- stale proposals rejected with latest state
- accepted decisions stored as human-readable Markdown

## Example

```bash
cogneetree --memory memory init

cogneetree --memory memory propose-decision auth/session-storage \
  --content "Use Redis for session storage." \
  --rationale "Redis supports TTLs and fast lookup." \
  --agent backend-agent

cogneetree --memory memory decisions show auth/session-storage
```

If another agent proposes a different decision for the same area, it gets:

```text
REJECTED_STALE auth/session-storage

An active decision already exists for this area.
Re-evaluate using the latest accepted state.
```

## Repository Shape

```text
src/cogneetree/
  protocol.py   # dataclasses and statuses
  store.py      # Markdown and JSONL persistence
  leader.py     # one-active-decision admission logic
  cli.py        # minimal CLI

docs/
  DISTRIBUTED_IMPLEMENTATION.md

tests/
  test_decision_protocol.py
```

## Development

```bash
poetry install
poetry run pytest
```

## Current Status

This is a fresh, small reference implementation. Raft/lease-based leader
election is documented but not implemented yet. The first production-grade
coordination target should be an external coordinator such as etcd, Consul, or a
Kubernetes Lease adapter before building custom Raft internals.

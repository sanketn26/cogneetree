# Distributed Implementation

Keep Cogneetree small.

## Rule

```text
one decision area
one active accepted decision
one leader admits writes
Markdown is accepted state
JSONL is audit history
stale proposals retry with latest state
```

## Flow

```text
agent proposes decision
leader checks area

if area has no decision:
  accept
  write memory/decisions/<area>.md
  append accepted event

if area already has a decision:
  reject_stale
  append rejected event
  store audit JSON
  return latest Markdown to agent
```

Different areas do not conflict.

## Files

```text
memory/
  decisions/**/*.md
  events/decisions.jsonl
  audit/rejected/*.json
```

## Python Core

```python
from cogneetree import DecisionFileStore, DecisionProposal, MemoryLeader

store = DecisionFileStore("memory")
leader = MemoryLeader("node-a", store)

resolution = leader.review(
    DecisionProposal(
        area="auth/session-storage",
        content="Use Redis for session storage.",
        rationale="TTL support and fast lookup.",
        agent_id="backend-agent",
    )
)
```

## CLI

```bash
cogneetree --memory memory init

cogneetree --memory memory propose-decision auth/session-storage \
  --content "Use Redis for session storage." \
  --rationale "TTL support and fast lookup." \
  --agent backend-agent

cogneetree --memory memory decisions show auth/session-storage
```

## Raft / Consensus

Do not put memory logic inside Raft.

Raft should only replicate commands:

```python
@dataclass(frozen=True)
class RaftCommand:
    command_id: str
    command_type: str  # "propose_decision"
    payload: dict
    submitted_by: str
```

Committed command:

```text
Raft log commits command
  -> MemoryLeader.review(proposal)
  -> Markdown/event files updated
```

For v1, use a single local leader. For distributed v1, prefer etcd, Consul, or
Kubernetes Lease for leader election. Build custom Raft only after the protocol
is proven.

## Next Steps

1. Keep this protocol stable.
2. Add MCP tools over the same API.
3. Add exact-area and Markdown lexical recall.
4. Add an external leader-election adapter.
5. Add explicit `supersede_decision` when changing an existing area is needed.

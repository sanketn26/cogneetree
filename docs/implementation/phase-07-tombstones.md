# Phase 7: Tombstones and Lifecycle

Goal: deletion is explicit. Retired nodes remain addressable, are excluded
from default reads, and carry a reason plus optional superseded-by links.

## Deliverables

- retire proposals fully supported end-to-end
- active reads exclude retired nodes by default
- explicit history reads include retired nodes
- `superseded_by` edges recorded for replacement scenarios
- recreate-after-retire requires an explicit kind

## Schema Changes

None. `accepted_versions.status` already supports `retired` (Phase 2).
Frontmatter `retired_reason` and `superseded_by` were added in Phase 6.

## Package: `pkg/protocol` (Extended)

```go
type RetireProposal struct {
    Identity        Identity
    ExpectedVersion int
    Reason          string
    RetiredReason   string
    SupersededBy    []NodeRef
    ProposedBy      string
    SourceRefs      []string
}

func NewRetireProposal(r RetireProposal) Proposal
```

Add a new kind to `ProposalKind`:

```go
const KindRecreate ProposalKind = "recreate"
```

`Recreate` is required to bring a retired node back. Agents cannot use
`Create` against an identity that has a retired current version; the leader
rejects with `ErrRecreateRequired`. This forces a deliberate decision.

## Package: `internal/leader` (Extended)

### Resolve Algorithm (Phase 7)

```text
case Retire:
    current = ReadCurrentForUpdate(p.Identity)
    if current == nil: return RejectMissingCurrent(p)
    if current.Version != p.ExpectedVersion: return RejectStale(p)
    if current.Status == Retired: return RejectInvalid(p, "already retired")
    if p.Frontmatter.RetiredReason == "": return RejectInvalid(p, "retired_reason required")
    priorVersion = MarkPriorNotCurrent(p.Identity)
    InsertAccepted(version=priorVersion+1, from_version=priorVersion, status=retired)
    append audit "proposal_accepted" with kind=retire

case Recreate:
    current = ReadCurrentForUpdate(p.Identity)
    if current == nil: return RejectInvalid(p, "no prior version to recreate")
    if current.Status != Retired: return RejectInvalid(p, "current version is active, use Update")
    if current.Version != p.ExpectedVersion: return RejectStale(p)
    priorVersion = MarkPriorNotCurrent(p.Identity)
    InsertAccepted(version=priorVersion+1, from_version=priorVersion, status=active)
    append audit "proposal_accepted" with kind=recreate

case Create:
    current = ReadCurrentForUpdate(p.Identity)
    if current != nil:
        if current.Status == Retired:
            return RejectInvalid(p, "use Recreate to bring back a retired node")
        return RejectStale(p, current)
    InsertAccepted(version=1, from_version=0, status=active)
```

### Read Filtering

```go
type ReadOptions struct {
    IncludeRetired bool // default false
}

func (s *Store) ReadCurrent(
    ctx context.Context,
    id protocol.Identity,
    opts ...ReadOption,
) (*protocol.AcceptedVersion, error)

func WithRetired() ReadOption { ... }
```

The default `ReadCurrent` returns `nil` if the current version is retired.
`ReadCurrent(..., WithRetired())` returns the retired version. History reads
(`ListVersions`, `ReadVersion`) always return retired versions; retirement
is part of the history.

```sql
-- default
SELECT * FROM accepted_versions
 WHERE org_id=$1 AND area_id=$2 AND node_id=$3
   AND is_current=true AND status='active';

-- with retired
SELECT * FROM accepted_versions
 WHERE org_id=$1 AND area_id=$2 AND node_id=$3
   AND is_current=true;
```

### Listing APIs

```go
// ListActive returns all current, non-retired nodes in an area.
func (s *Store) ListActive(
    ctx context.Context,
    org protocol.OrgID,
    area protocol.AreaID,
) ([]protocol.AcceptedVersion, error)

// ListRetired returns all current, retired nodes in an area.
func (s *Store) ListRetired(
    ctx context.Context,
    org protocol.OrgID,
    area protocol.AreaID,
) ([]protocol.AcceptedVersion, error)
```

## Audit Events Added

| Type | Emitted When |
|---|---|
| `node_retired` | accepted retire proposal commits |
| `node_recreated` | accepted recreate proposal commits |

Each event payload includes `retired_reason` and (for recreate) the prior
retired version number.

## Tombstone Rendering

When `Render` emits Markdown for a retired node, prepend `Retired: ` to the
H1 title and include a notice block:

```markdown
---
id: rbac.legacy_acl_policy
status: retired
retired_reason: Replaced by rbac.acls.
superseded_by:
  - area: rbac
    node: rbac.acls
---

# Retired: Legacy ACL Policy

This node was retired on 2026-04-10 by rbac-leader.

Reason: Replaced by rbac.acls.

See: rbac.acls.
```

The renderer is a pure function over the frontmatter and body; no I/O.

## Tests Required

In `internal/leader`:

| Test | What It Asserts |
|---|---|
| `TestRetireRequiresReason` | retire proposal without retired_reason → rejected_invalid |
| `TestRetireSucceedsAndIncrementsVersion` | accepted retire writes status=retired, version+1 |
| `TestRetireRejectsAlreadyRetired` | retire on retired node → rejected_invalid |
| `TestActiveReadExcludesRetired` | ReadCurrent default returns nil for retired node |
| `TestExplicitReadIncludesRetired` | ReadCurrent(WithRetired()) returns retired version |
| `TestHistoryReadIncludesRetired` | ListVersions includes the retire transition |
| `TestCreateAfterRetireRequiresRecreate` | Create on retired identity → rejected_invalid |
| `TestRecreateRevivesNode` | Recreate writes status=active, version+1 |
| `TestRecreateRejectsActiveNode` | Recreate on active node → rejected_invalid |

In `internal/frontmatter`:

| Test | What It Asserts |
|---|---|
| `TestRenderRetiredAddsNotice` | retired frontmatter produces "Retired:" prefix and notice |
| `TestRenderRetiredPreservesSupersededBy` | superseded_by list survives round-trip |

## Exit Criteria

- retirement is an accepted state with reason and version increment
- active reads exclude retired nodes; explicit reads include them
- recreate is required to revive a retired node
- audit events distinguish retire from recreate from supersede
- tombstone Markdown rendering is deterministic
- all tests above pass

Once these are green, proceed to
[Phase 8: Graph Edges](phase-08-graph-edges.md).

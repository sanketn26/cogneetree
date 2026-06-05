# Phase 4: Supersede and Update

Goal: accepted memory can change. Update proposals carry an expected current
version. Matching versions increment; mismatched versions reject without
mutation.

## Deliverables

- update and retire proposal kinds processed by the leader
- compare-and-set semantics enforced by `(org, area, node, version)` PK plus
  the partial unique index on `is_current`
- `from_version` correctly recorded on every accepted update
- audit `proposal_superseded` events emitted
- supersede tests cover missing, stale, and accepted-update paths

## Schema Changes

None. Phase 2 already created `accepted_versions` with `version` and
`from_version`. Phase 4 makes use of them.

## Package: `internal/store` (Extended)

```go
// MarkPriorNotCurrent flips is_current=false on the row that is currently
// is_current=true for the given identity, inside the leader transaction.
// Returns the prior version number, or 0 if no prior version exists.
func (s *Store) MarkPriorNotCurrent(
    ctx context.Context,
    tx pgx.Tx,
    id protocol.Identity,
) (int, error)
```

### SQL: Mark Prior Not Current

```sql
UPDATE accepted_versions
   SET is_current = false
 WHERE org_id = $1
   AND area_id = $2
   AND node_id = $3
   AND is_current = true
RETURNING version;
```

If no rows match, the node does not exist; an update proposal must reject
with `rejected_missing_current`. A create proposal proceeds normally.

### SQL: Insert Accepted (Updated)

`InsertAccepted` now writes `from_version` based on the prior current row.
The PK on `(org, area, node, version)` prevents two leaders from inserting
the same version number.

```sql
INSERT INTO accepted_versions (
    org_id, area_id, node_id, version, is_current, status,
    frontmatter, body, proposal_id, from_version,
    accepted_by, accepted_at, reason, source_refs
) VALUES (
    $1, $2, $3, $4, true, $5,
    $6, $7, $8, $9,
    $10, now(), $11, $12
);
```

The leader computes the next version as `prior + 1`. If two leaders race to
insert version `N`, exactly one INSERT succeeds. The loser sees `23505`,
re-reads the current version, and converts to a stale rejection.

## Package: `internal/leader` (Extended)

### Resolve Algorithm (Phase 4)

```text
begin tx
  current = ReadCurrentForUpdate(p.Identity)   // SELECT ... FOR UPDATE
  switch p.Kind:
    case Create:
        if current != nil:
            return RejectStale(p, current)
        InsertAccepted(version=1, from_version=0, status=active)

    case Update:
        if current == nil:
            return RejectMissingCurrent(p)
        if current.Version != p.ExpectedVersion:
            return RejectStale(p, current)
        if current.Status == Retired:
            return RejectInvalid(p, "cannot update retired node")
        priorVersion = MarkPriorNotCurrent(p.Identity)
        InsertAccepted(version=priorVersion+1, from_version=priorVersion, status=active)
        append audit "proposal_superseded" linking prior + new version

    case Retire:
        if current == nil:
            return RejectMissingCurrent(p)
        if current.Version != p.ExpectedVersion:
            return RejectStale(p, current)
        // Phase 7 expands retirement; for Phase 4 we only need the basic
        // status transition to exist as a path.
        priorVersion = MarkPriorNotCurrent(p.Identity)
        InsertAccepted(version=priorVersion+1, from_version=priorVersion, status=retired)
commit
```

### Compare-And-Set Discipline

The leader uses `SELECT ... FOR UPDATE` on the current row to serialize
concurrent updates of the same node. Two principles:

- `FOR UPDATE` provides ordering within one leader process.
- The `(org, area, node, version)` PK provides correctness across leader
  processes. If `FOR UPDATE` is bypassed somehow, the PK still prevents
  duplicate version numbers.

### Stale Rejection Messages

```go
const (
    ReasonStaleVersion    = "expected version does not match current accepted version"
    ReasonMissingCurrent  = "no current accepted version exists for this node"
    ReasonRetiredTarget   = "cannot update a retired node; propose a recreate instead"
)
```

## Audit Events Added

| Event Type | Emitted When |
|---|---|
| `proposal_superseded` | accepted update where `from_version > 0` |
| `proposal_rejected_missing_current` | update/retire against a node that does not exist |

The `proposal_superseded` payload includes `from_version`, `to_version`,
`proposal_id`, and `reason`. Phase 5 enriches it with source_refs.

## Tests Required

In `internal/leader`:

| Test | What It Asserts |
|---|---|
| `TestUpdateMatchingVersionAccepted` | expected_version matches current → accept, version +1 |
| `TestUpdateStaleVersionRejected` | expected_version does not match → reject, no mutation |
| `TestUpdateMissingNodeRejected` | update on non-existent node → rejected_missing_current |
| `TestRetireMatchingVersionAccepted` | retire sets status=retired on new version |
| `TestCannotUpdateRetiredNode` | update on retired node → rejected_invalid |
| `TestConcurrentUpdatesOneWins` | two leaders race on same version; one accepts, one stale |
| `TestVersionMonotonic` | accepted versions strictly increase by 1 |
| `TestFromVersionRecorded` | accepted_versions.from_version equals prior version number |
| `TestPriorSnapshotReadable` | reading the prior version still returns its Markdown |

The "prior snapshot readable" test is the foundation that Phase 5 will turn
into snapshot-tier reads. In Phase 4 it just exercises that prior rows are
still queryable.

## Read APIs

```go
// ReadVersion returns a specific accepted version, current or historical.
// Used by audit reconstruction and by tests.
func (s *Store) ReadVersion(
    ctx context.Context,
    id protocol.Identity,
    version int,
) (*protocol.AcceptedVersion, error)

// ListVersions returns version metadata (no body) for a node, newest first.
func (s *Store) ListVersions(
    ctx context.Context,
    id protocol.Identity,
) ([]VersionSummary, error)

type VersionSummary struct {
    Version     int
    Status      protocol.NodeStatus
    AcceptedBy  string
    AcceptedAt  time.Time
    ProposalID  string
    Reason      string
    FromVersion int
}
```

## Exit Criteria

- update with matching expected version is accepted, version increments
- update with stale expected version is rejected without mutation
- update against missing node is rejected as missing-current
- retire transitions status to retired on a new version
- prior version remains readable via `ReadVersion`
- concurrent updates on the same node yield exactly one accepted, the rest stale
- audit `proposal_superseded` events written for every accepted update
- all tests above pass

Once these are green, proceed to
[Phase 5: Provenance, Snapshots, Change Sets](phase-05-provenance-snapshots.md).

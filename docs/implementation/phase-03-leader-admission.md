# Phase 3: Leader Admission

Goal: the first proposal for a node is accepted, the second is rejected as
stale. End-to-end, against real Postgres, with the partial unique index on
`is_current` enforcing the invariant at the database level.

This phase delivers the minimum viable write path. Updates and supersession
come in Phase 4.

## Deliverables

- `internal/store` write methods for proposals, accepted versions, audit
- `internal/leader` admission logic with `SKIP LOCKED` claiming
- `internal/audit` write path (DB rows + JSONL tail)
- `cogneetree.MemoryAPI` exposed via `pkg/cogneetreeclient`
- end-to-end tests proving stale rejection

## Package: `internal/store` (Extended)

```go
package store

// SubmitProposal inserts a pending row. Returns proposal ID.
func (s *Store) SubmitProposal(ctx context.Context, p protocol.Proposal) error

// ClaimNextPending pulls one pending proposal for the given org+area,
// marks it claimed, and returns it. Uses SELECT ... FOR UPDATE SKIP LOCKED
// so multiple leaders for the same area do not collide.
func (s *Store) ClaimNextPending(
    ctx context.Context,
    org protocol.OrgID,
    area protocol.AreaID,
    leaderID string,
) (*protocol.Proposal, error)

// ReadCurrent returns the latest current accepted version for an identity,
// or nil if none exists.
func (s *Store) ReadCurrent(
    ctx context.Context,
    id protocol.Identity,
) (*protocol.AcceptedVersion, error)

// InsertAccepted writes a new accepted version row inside a transaction,
// flipping is_current on any prior current row of the same node.
// Phase 3 only handles the "no prior version" case; Phase 4 adds the flip.
func (s *Store) InsertAccepted(
    ctx context.Context,
    tx pgx.Tx,
    av protocol.AcceptedVersion,
) error

// CompleteProposal writes the resolution back to the pending row.
func (s *Store) CompleteProposal(
    ctx context.Context,
    tx pgx.Tx,
    res protocol.Resolution,
) error

// WithTx runs fn in a single transaction, rolling back on error.
func (s *Store) WithTx(
    ctx context.Context,
    fn func(pgx.Tx) error,
) error
```

### SQL: Claim Next Pending

```sql
SELECT proposal_id, org_id, area_id, node_id, kind, expected_version,
       body, frontmatter, reason, source_refs, proposed_by, submitted_at,
       change_set_id
  FROM pending_proposals
 WHERE org_id = $1
   AND area_id = $2
   AND status = 'pending'
 ORDER BY submitted_at, proposal_id
 LIMIT 1
 FOR UPDATE SKIP LOCKED;
```

Then:

```sql
UPDATE pending_proposals
   SET status = 'claimed',
       claimed_by = $1,
       claimed_at = now()
 WHERE proposal_id = $2;
```

Both run inside the same transaction. The claim transaction stays open until
the leader resolves the proposal, so duplicate work is impossible.

### SQL: Insert Accepted (Phase 3, Create-Only)

```sql
INSERT INTO accepted_versions (
    org_id, area_id, node_id, version, is_current, status,
    frontmatter, body, proposal_id, from_version,
    accepted_by, accepted_at, reason, source_refs
) VALUES (
    $1, $2, $3, $4, true, 'active',
    $5, $6, $7, 0,
    $8, now(), $9, $10
);
```

The partial unique index `accepted_one_current` raises a `unique_violation`
(SQLSTATE `23505`) if a current version already exists. The leader treats this
as a stale rejection.

## Package: `internal/leader`

```go
package leader

type Leader struct {
    id     string
    store  *store.Store
    audit  audit.Writer
    clock  func() time.Time
}

func New(id string, st *store.Store, w audit.Writer) *Leader

// ProcessNext claims and resolves a single pending proposal for the given
// scope. Returns the resolution or nil if no pending proposal exists.
func (l *Leader) ProcessNext(
    ctx context.Context,
    org protocol.OrgID,
    area protocol.AreaID,
) (*protocol.Resolution, error)

// Resolve handles one already-claimed proposal. Exposed for tests and for
// the worker loop in Phase 10.
func (l *Leader) Resolve(
    ctx context.Context,
    p protocol.Proposal,
) (*protocol.Resolution, error)
```

### Admission Algorithm (Phase 3)

```text
begin tx
  current = ReadCurrent(p.Identity)
  if p.Kind == Create:
      if current != nil:
          return RejectStale(p, current)
      insert new accepted version (version=1)
      complete proposal as accepted
      append audit "proposal_accepted"
  else:
      // Updates and retires are Phase 4 territory.
      return RejectInvalid(p, "kind not yet supported")
commit
```

Stale handling:

```go
func (l *Leader) rejectStale(
    p protocol.Proposal,
    current *protocol.AcceptedVersion,
) *protocol.Resolution {
    return &protocol.Resolution{
        ProposalID:    p.ProposalID,
        Status:        protocol.ProposalRejectedStale,
        Identity:      p.Identity,
        LatestVersion: current.Version,
        Reason:        "an active accepted version already exists",
        DecidedAt:     l.clock(),
        DecidedBy:     l.id,
    }
}
```

The leader does **not** delete or roll back the pending row on rejection.
`CompleteProposal` writes the resolution into `pending_proposals.resolution`
and flips `status='resolved'`. The row stays for audit reconstruction.

### Concurrency Notes

- The leader does not hold a global lease in Phase 3. Per-area concurrency is
  controlled by `SKIP LOCKED` plus the partial unique index.
- Multiple leader processes can run for the same area; the first to claim a
  given proposal wins, the others move on.
- The unique-constraint CAS is the safety net. If two leaders race to insert
  the first version of the same node, exactly one INSERT succeeds; the loser
  gets `23505` and treats it as a stale rejection.

## Package: `internal/audit`

```go
package audit

type Writer interface {
    Append(ctx context.Context, ev Event) error
}

type Event struct {
    OrgID      protocol.OrgID
    AreaID     protocol.AreaID
    NodeID     protocol.NodeID
    Version    int
    Type       string // proposal_submitted, proposal_accepted, ...
    ProposalID string
    Payload    map[string]any
    At         time.Time
}

// DBWriter writes to audit_events. Used inside the leader transaction.
type DBWriter struct{ ... }

// JSONLWriter tails a per-org JSONL file as a redundant authoritative log.
// Phase 5 makes this canonical.
type JSONLWriter struct{ ... }

// Fanout writes to multiple writers; errors from any writer are returned.
type Fanout struct{ writers []Writer }
```

In Phase 3 the leader uses `Fanout{DBWriter, JSONLWriter}`. The DB writer
participates in the leader transaction; the JSONL writer appends after commit
to avoid two-phase coordination.

## Package: `pkg/cogneetreeclient`

The public Go API used by callers (HTTP adapter, CLI, external code).

```go
package cogneetreeclient

type MemoryAPI interface {
    Submit(ctx context.Context, p protocol.Proposal) (string, error)
    GetResolution(ctx context.Context, proposalID string) (*protocol.Resolution, error)
    ReadCurrent(ctx context.Context, id protocol.Identity) (*protocol.AcceptedVersion, error)
}

type DirectAPI struct { /* wraps store + leader */ }
```

Phase 3 ships `DirectAPI` only. An `HTTPClient` implementation lands in
Phase 11.

## Tests Required

In `internal/leader`:

| Test | What It Asserts |
|---|---|
| `TestFirstCreateAccepted` | first proposal for a missing node is accepted |
| `TestSecondCreateRejectedStale` | second create for same node is rejected stale |
| `TestDifferentNodesIndependent` | parallel creates for different nodes both accepted |
| `TestRejectionDoesNotMutate` | rejected proposal does not alter accepted Markdown |
| `TestRejectionEventRecordsCurrentVersion` | rejection audit names current version |
| `TestConcurrentCreatesOnlyOneWins` | 50 goroutines submit + leader processes; one accept, 49 stale |
| `TestSkipLockedDoesNotBlock` | two leaders claim different pending rows in parallel |

The concurrency test is the load-bearing one — it proves the partial unique
index plus SKIP LOCKED actually enforce the invariant under contention.

## Manual Verification

```bash
# Start postgres, run migrations, start a leader-process equivalent in a test
go test -run TestConcurrentCreatesOnlyOneWins ./internal/leader -v
```

The test should print one accepted resolution and many stale rejections,
exit zero, and leave exactly one row with `is_current = true` for the node.

## Exit Criteria

- `internal/store` exposes Submit, ClaimNext, ReadCurrent, InsertAccepted,
  CompleteProposal, WithTx
- `internal/leader` resolves create proposals end-to-end against Postgres
- `internal/audit` writes both DB and JSONL events
- `pkg/cogneetreeclient.DirectAPI` wires the three packages together
- all tests above pass
- `golangci-lint run ./...` exits 0

Once these are green, proceed to
[Phase 4: Supersede and Update](phase-04-supersede-update.md).

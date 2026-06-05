# Phase 5: Provenance, Snapshots, Change Sets

Goal: every accepted update is fully attributable, every prior version is
reconstructable, and multi-node updates are atomic.

## Deliverables

- `proposed_by`, `accepted_by`, `reason`, `source_refs` enforced on every
  accepted version
- snapshot storage that reconstructs any historical version
- `change_set_id` correlation across multi-node updates
- all-or-nothing change set admission
- JSONL audit becomes co-authoritative with the DB

## Schema Changes

### `0008_change_sets.sql`

```sql
-- +goose Up
CREATE TABLE change_sets (
  change_set_id text         PRIMARY KEY,
  org_id        text         NOT NULL,
  reason        text         NOT NULL,
  proposed_by   text         NOT NULL,
  submitted_at  timestamptz  NOT NULL DEFAULT now(),
  status        text         NOT NULL
                  CHECK (status IN ('pending','accepted','rejected')),
  resolved_at   timestamptz,
  resolution    jsonb
);

CREATE INDEX change_sets_status
  ON change_sets (org_id, status, submitted_at);

-- +goose Down
DROP TABLE change_sets;
```

### `0009_snapshots.sql`

Snapshots live in the same table as accepted versions — they're literally the
rows. The "snapshot" view is just a `SELECT` against `accepted_versions`. We
do not write loose snapshot files: the earlier per-file plan (`v0001.md`,
`v0002.md`) does not scale past ~100k versions per node-tree.

For optional human-browsable export, Phase 5 adds a maintenance task that
materializes snapshots to disk. The DB stays authoritative.

```sql
-- +goose Up
CREATE OR REPLACE VIEW node_history AS
  SELECT org_id, area_id, node_id, version, status, body, frontmatter,
         from_version, accepted_by, accepted_at, reason, source_refs,
         change_set_id, proposal_id
    FROM accepted_versions
   ORDER BY org_id, area_id, node_id, version;

-- +goose Down
DROP VIEW IF EXISTS node_history;
```

## Package: `internal/snapshot`

```go
package snapshot

type Store struct {
    db *store.Store
}

func New(db *store.Store) *Store

// Get returns the accepted version row at the requested version.
// Returns ErrNotFound if the version does not exist.
func (s *Store) Get(
    ctx context.Context,
    id protocol.Identity,
    version int,
) (*protocol.AcceptedVersion, error)

// Diff returns a unified diff between two versions of the same node.
// Phase 5 ships a simple line-oriented diff; richer Markdown-aware
// diffing can come later.
func (s *Store) Diff(
    ctx context.Context,
    id protocol.Identity,
    from, to int,
) ([]DiffLine, error)

// Export writes a snapshot to disk for human browsing.
// Format: <root>/<org>/<area>/<node>/v<NNNN>.md
func (s *Store) Export(
    ctx context.Context,
    root string,
    id protocol.Identity,
    version int,
) error

type DiffLine struct {
    Kind string // " ", "-", "+"
    Text string
}
```

## Package: `internal/leader` (Extended)

### Change Set Admission

```go
// ProcessChangeSet resolves all proposals in a change set atomically.
// If any proposal fails admission, the entire change set is rejected and
// no accepted_versions rows are written.
func (l *Leader) ProcessChangeSet(
    ctx context.Context,
    cs ChangeSetSubmission,
) (*ChangeSetResolution, error)

type ChangeSetSubmission struct {
    ChangeSetID string
    Reason      string
    ProposedBy  string
    Proposals   []protocol.Proposal // share ChangeSetID
}

type ChangeSetResolution struct {
    ChangeSetID string
    Status      protocol.ProposalStatus // accepted or rejected
    Reason      string
    Resolutions []protocol.Resolution
    DecidedAt   time.Time
    DecidedBy   string
}
```

### Algorithm

```text
begin tx
  validate every proposal individually (Phase 1 rules)
  for each proposal:
    current = ReadCurrentForUpdate(p.Identity)
    classify(p, current) into admission_plan
    if classification is reject:
        rollback
        return rejected change_set with per-node reasons

  // all proposals valid; commit them in one transaction
  for each proposal:
    apply admission_plan (mark prior not current, insert new version)
    append audit event with change_set_id

commit
append audit "change_set_accepted"
```

The change set is admitted only if **every** included proposal admits
individually. One stale rejection rolls the whole set back.

### Audit Correlation

Every audit event for a proposal in a change set carries the same
`change_set_id`. This lets later phases reconstruct "everything that happened
because of decision X" with one query.

## Package: `internal/audit` (Extended)

```go
type Event struct {
    // ... existing fields ...
    ChangeSetID string
    SourceRefs  []string
    Reason      string
    ProposedBy  string
}
```

Add event types:

| Type | When |
|---|---|
| `change_set_submitted` | a change set arrives |
| `change_set_accepted` | all proposals in a change set admitted |
| `change_set_rejected` | any proposal rejected; whole set rolled back |

### JSONL Becomes Co-Authoritative

Phase 3 wrote JSONL as a redundant log. Phase 5 makes it canonical: the
operator can rebuild the entire DB from `wiki/*.md` + `audit/*.jsonl`. This
is the "indexes are derived caches" invariant made real.

```go
package audit

// Rebuilder reads JSONL audit files plus accepted Markdown and reconstructs
// the SQL state. Used for disaster recovery and schema migrations.
type Rebuilder struct {
    Audit  fs.FS
    Wiki   fs.FS
    Target *store.Store
}

func (r *Rebuilder) Rebuild(ctx context.Context) error
```

JSONL rotation policy: one file per `org/area` per month. A manifest file
under `audit/<org>/manifest.json` lists rotated files in order.

## Provenance Required Fields

Phase 5 enforces these at admission time (return `ErrMissingProvenance` if
absent):

```go
type Provenance struct {
    ProposedBy  string   // required
    Reason      string   // required; >= 8 chars, free text
    SourceRefs  []string // required; each is a URL, doc ID, or audit event ID
}
```

`SourceRefs` may be empty for a proposal whose source is "agent observation
in conversation X" — in that case the agent must include the conversation ID
as a single source ref. The principle is *something must justify the change*.

## Tests Required

In `internal/leader`:

| Test | What It Asserts |
|---|---|
| `TestChangeSetAllAcceptedAtomic` | 3-node change set, all valid → all accepted in one tx |
| `TestChangeSetOneStaleRejectsAll` | 3-node change set, one stale → none mutated |
| `TestChangeSetReasonRecordedEveryNode` | every accepted row in the set shares reason |
| `TestProvenanceMissingProposedByRejects` | reject when ProposedBy empty |
| `TestProvenanceMissingReasonRejects` | reject when Reason empty |

In `internal/snapshot`:

| Test | What It Asserts |
|---|---|
| `TestSnapshotReconstructsPriorVersion` | Get returns prior version body byte-for-byte |
| `TestSnapshotMissingVersionReturnsNotFound` | unknown version → ErrNotFound |
| `TestDiffBetweenVersions` | Diff returns expected hunks for a known pair |
| `TestExportWritesFile` | Export writes `v0003.md` with correct content |

In `internal/audit`:

| Test | What It Asserts |
|---|---|
| `TestRebuildEmptyTargetFromJSONL` | rebuild from JSONL + Markdown reproduces accepted_versions |
| `TestRebuildIdempotent` | running rebuild twice does not duplicate rows |
| `TestRotationManifestOrdered` | manifest lists rotated files in chronological order |

## Maintenance: Monthly Audit Partition

Phase 2 partitioned `audit_events` by month but did not create future
partitions. Add a maintenance function:

```go
// EnsureAuditPartition creates the partition for the given month if missing.
// Call from a cron or a worker tick once per day.
func (s *Store) EnsureAuditPartition(ctx context.Context, month time.Time) error
```

## Exit Criteria

- every accepted row has `proposed_by`, `accepted_by`, `reason`, `source_refs`
- snapshots return historical versions byte-for-byte
- change sets are admitted all-or-nothing
- rejected change sets do not mutate any accepted Markdown
- audit JSONL plus accepted Markdown reconstruct the SQL state via Rebuilder
- monthly audit partitions can be created ahead of time
- all tests above pass

Once these are green, proceed to
[Phase 6: Node Metadata and Validation](phase-06-metadata-validation.md).

# Phase 8: Graph Edges

Goal: relationships between nodes are queryable. Edges are derived from
accepted frontmatter and admitted only by the leader. The tree locates
memory; edges describe meaning.

## Deliverables

- `edges` table with version-pinning rules
- edge admission alongside node acceptance
- neighborhood and dependency queries
- conflict detection visible to proposal validation

## Schema Changes

### `0010_edges.sql`

```sql
-- +goose Up
CREATE TABLE edges (
  org_id         text        NOT NULL,
  from_area      text        NOT NULL,
  from_node      text        NOT NULL,
  from_version   int,         -- NULL means "latest"
  edge_type      text        NOT NULL
                   CHECK (edge_type IN ('depends_on','related_to','supersedes','conflicts_with')),
  to_area        text        NOT NULL,
  to_node        text        NOT NULL,
  to_version     int,
  created_at     timestamptz NOT NULL DEFAULT now(),
  proposal_id    text        NOT NULL,
  PRIMARY KEY (org_id, from_area, from_node, from_version, edge_type, to_area, to_node, to_version)
) PARTITION BY LIST (org_id);

CREATE INDEX edges_from
  ON edges (org_id, from_area, from_node);

CREATE INDEX edges_to
  ON edges (org_id, to_area, to_node);

CREATE INDEX edges_type_to
  ON edges (org_id, edge_type, to_area, to_node);

ALTER TABLE edges ENABLE ROW LEVEL SECURITY;
CREATE POLICY org_isolation_edges ON edges
  USING (org_id = current_setting('cogneetree.org_id', true));

-- +goose Down
DROP TABLE edges;
```

### Version-Pinning Rules

```text
edge_type         from_version    to_version    reason
---------         ------------    ----------    ------
depends_on        latest          latest        dependencies follow current state
related_to        latest          latest        relatedness is structural, not version-specific
supersedes        specific        specific      "X replaced Y" pins to the exact moment
conflicts_with    specific        specific      conflicts are between specific accepted states
```

Encoded in code:

```go
type EdgePinning int
const (
    PinLatest   EdgePinning = iota // store NULL in from_version / to_version
    PinSpecific                    // store the integer version
)

func PinningFor(t EdgeType) (from, to EdgePinning)
```

## Package: `internal/edges`

```go
package edges

type EdgeType string

const (
    EdgeDependsOn     EdgeType = "depends_on"
    EdgeRelatedTo     EdgeType = "related_to"
    EdgeSupersedes    EdgeType = "supersedes"
    EdgeConflictsWith EdgeType = "conflicts_with"
)

type Edge struct {
    Org         protocol.OrgID
    FromArea    protocol.AreaID
    FromNode    protocol.NodeID
    FromVersion int   // 0 means latest (stored as NULL)
    Type        EdgeType
    ToArea      protocol.AreaID
    ToNode      protocol.NodeID
    ToVersion   int   // 0 means latest
    CreatedAt   time.Time
    ProposalID  string
}

type Store struct { db *store.Store }

func New(db *store.Store) *Store

// Upsert is called by the leader inside the admission transaction.
func (s *Store) Upsert(ctx context.Context, tx pgx.Tx, e Edge) error

// Delete removes an edge inside the admission transaction.
func (s *Store) Delete(ctx context.Context, tx pgx.Tx, e Edge) error

// ListFrom returns all edges originating at a node.
func (s *Store) ListFrom(ctx context.Context, id protocol.Identity) ([]Edge, error)

// ListTo returns all edges pointing at a node.
func (s *Store) ListTo(ctx context.Context, id protocol.Identity) ([]Edge, error)

// ListByType returns all edges of a given type within an org.
func (s *Store) ListByType(
    ctx context.Context, org protocol.OrgID, t EdgeType,
) ([]Edge, error)
```

### Neighborhood

```go
type Neighborhood struct {
    Node       protocol.AcceptedVersion
    Parents    []protocol.AcceptedVersion  // ancestors in the tree
    Children   []protocol.AcceptedVersion  // direct children in the tree
    DependsOn  []protocol.AcceptedVersion  // outgoing depends_on
    Dependents []protocol.AcceptedVersion  // incoming depends_on
    Related    []protocol.AcceptedVersion
    Conflicts  []protocol.AcceptedVersion
    Supersedes []protocol.AcceptedVersion
    SupersededBy []protocol.AcceptedVersion
}

func (s *Store) Neighborhood(
    ctx context.Context, id protocol.Identity, depth int,
) (*Neighborhood, error)
```

### Closure / Dependency Walking

Recursive CTE for transitive dependencies:

```sql
WITH RECURSIVE deps AS (
    SELECT to_area, to_node, to_version, 1 AS depth
      FROM edges
     WHERE org_id = $1
       AND from_area = $2
       AND from_node = $3
       AND edge_type = 'depends_on'
  UNION ALL
    SELECT e.to_area, e.to_node, e.to_version, deps.depth + 1
      FROM edges e
      JOIN deps ON e.from_area = deps.to_area
               AND e.from_node = deps.to_node
     WHERE e.org_id = $1
       AND e.edge_type = 'depends_on'
       AND deps.depth < $4
)
SELECT DISTINCT to_area, to_node, to_version FROM deps;
```

Always cap with a `depth` parameter. The default depth is 5. Anything beyond
that signals a graph-design issue.

## Package: `internal/leader` (Extended)

### Edge Derivation

Frontmatter declares edges. The leader derives the edge rows on admission.

```text
on accepted Create/Update/Recreate:
  proposed_edges = derive_edges_from(frontmatter)
  current_edges  = ListFrom(identity)
  diff = current_edges vs proposed_edges
  for added: Upsert
  for removed: Delete
on accepted Retire:
  keep edges as-is (history)
```

```go
// DeriveEdges reads frontmatter.{depends_on, related_to, supersedes,
// conflicts_with} and emits Edge rows pinned per the rules above.
func DeriveEdges(av protocol.AcceptedVersion) []Edge
```

Edge changes are part of the admission transaction. A failure to upsert an
edge fails the entire admission.

### Conflict Detection

When the leader processes an Update/Create:

```go
conflicts := s.edges.ListConflictsTouching(ctx, p.Identity)
for _, c := range conflicts {
    if c is unresolved and p does not acknowledge it:
        return RejectInvalid(p, "unacknowledged conflict with X")
}
```

Acknowledgement is a frontmatter field `conflict_acks: [node_ref...]` that
explicitly names the conflicts the proposer is overriding. Forces the agent
to be deliberate.

## Edge Integrity Rules

Enforced at admission, not just by foreign keys (which we don't use across
partitions):

```text
- target node must exist as accepted_versions row (current or historical for pinned)
- if to_version is pinned, that exact version must exist
- supersedes target must be retired or about to be retired (in same change set)
- conflicts_with target must be active
```

Implemented in `internal/edges.ValidateTarget`:

```go
func (s *Store) ValidateTarget(
    ctx context.Context, e Edge,
) error
```

## Read APIs Added

```go
// In edges.Store
func (s *Store) ListConflictsTouching(
    ctx context.Context, id protocol.Identity,
) ([]Edge, error)

func (s *Store) Dependencies(
    ctx context.Context, id protocol.Identity, maxDepth int,
) ([]protocol.AcceptedVersion, error)

func (s *Store) Dependents(
    ctx context.Context, id protocol.Identity, maxDepth int,
) ([]protocol.AcceptedVersion, error)
```

## Tests Required

In `internal/edges`:

| Test | What It Asserts |
|---|---|
| `TestUpsertEdgeWritesRow` | Upsert produces a queryable row |
| `TestUpsertIdempotent` | inserting the same edge twice does not duplicate |
| `TestPinningRulesEnforced` | depends_on/related_to store NULL versions; supersedes/conflicts_with store integers |
| `TestEdgeToMissingNodeRejected` | ValidateTarget fails when target version missing |
| `TestSupersedesRequiresRetired` | supersedes target must be retired or co-retired |
| `TestListFromReturnsAll` | ListFrom returns every outgoing edge |
| `TestListToReturnsAll` | ListTo returns every incoming edge |
| `TestDependencyClosureCapped` | depth cap prevents infinite traversal |
| `TestNeighborhoodReturnsExpectedShape` | neighborhood has ancestors, children, edges, conflicts |

In `internal/leader`:

| Test | What It Asserts |
|---|---|
| `TestAdmissionWritesDerivedEdges` | accepting an update writes edge rows from frontmatter |
| `TestAdmissionRemovesObsoleteEdges` | removing a depends_on in frontmatter deletes the edge |
| `TestUnacknowledgedConflictRejects` | proposal against node with active conflicts rejected unless acknowledged |
| `TestRetireDoesNotDeleteEdges` | edges remain after retire for history |
| `TestEdgeFailureRollsBackAdmission` | edge upsert failure leaves no accepted_versions row |

## Exit Criteria

- frontmatter edges land in the `edges` table on accept
- version-pinning rules enforced per edge type
- edges to missing nodes rejected
- neighborhood query returns node, ancestors, children, edges
- dependency and dependent closures capped by depth
- conflicts surfaced during admission validation
- all tests above pass

Once these are green, proceed to
[Phase 9: Search and Derived Indexes](phase-09-search-indexes.md).

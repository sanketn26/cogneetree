# Phase 1: Protocol Core

Goal: model identity, proposals, accepted versions, and resolutions as pure Go
types with no Postgres dependency. Everything in this phase is unit-testable
without a database.

## Package Location: Why `pkg/protocol`, not `internal/protocol`

These types appear in the public SDK's method signatures
(`pkg/cogneetreeclient`). Go does not allow importing `internal/...` from
outside the module, so any package whose types leak into a public SDK must
itself live under `pkg/`. The protocol package is the canonical example —
`Identity`, `Proposal`, `Resolution`, and `AcceptedVersion` are the wire
vocabulary, not internal scaffolding.

## Deliverables

- `pkg/protocol` package with identity types, records, statuses, errors
- validators for IDs and paths
- table-driven unit tests covering every validation rule
- public test fixtures for use in later phases

## Package: `pkg/protocol`

### Identity Types

```go
package protocol

type OrgID  string
type AreaID string
type NodeID string

// Identity is the canonical key for an accepted memory node.
// org + area + node together identify a node uniquely.
type Identity struct {
    Org  OrgID
    Area AreaID
    Node NodeID
}

func (i Identity) String() string // returns "org/area/node"
func (i Identity) Validate() error
```

### Statuses

```go
type ProposalStatus string

const (
    ProposalPending           ProposalStatus = "pending"
    ProposalClaimed           ProposalStatus = "claimed"
    ProposalAccepted          ProposalStatus = "accepted"
    ProposalRejectedStale     ProposalStatus = "rejected_stale"
    ProposalRejectedMissing   ProposalStatus = "rejected_missing_current"
    ProposalRejectedInvalid   ProposalStatus = "rejected_invalid"
)

type NodeStatus string

const (
    NodeActive  NodeStatus = "active"
    NodeRetired NodeStatus = "retired"
)

type ProposalKind string

const (
    KindCreate    ProposalKind = "create"
    KindUpdate    ProposalKind = "update"
    KindRetire    ProposalKind = "retire"
)
```

### Records

```go
type Proposal struct {
    ProposalID      string
    Identity        Identity
    Kind            ProposalKind
    ExpectedVersion int            // 0 for create; >=1 for update/retire
    Body            string         // raw Markdown body
    Frontmatter    Frontmatter     // typed metadata (see Phase 6 for full schema)
    Reason          string
    SourceRefs      []string
    ProposedBy      string
    SubmittedAt     time.Time
    ChangeSetID     string         // empty unless part of a multi-node change set
}

type AcceptedVersion struct {
    Identity     Identity
    Version      int
    Status       NodeStatus
    Body         string
    Frontmatter  Frontmatter
    AcceptedBy   string
    AcceptedAt   time.Time
    FromVersion  int    // 0 for first version
    ProposalID   string
    Reason       string
    SourceRefs   []string
}

type Resolution struct {
    ProposalID      string
    Status          ProposalStatus
    Identity        Identity
    LatestVersion   int             // current accepted version after resolution
    AcceptedVersion *AcceptedVersion // nil unless Status == Accepted
    Reason          string
    DecidedAt       time.Time
    DecidedBy       string          // leader node ID
}
```

### Frontmatter (Phase-1 Skeleton)

Phase 6 extends this with node_type, authority, aliases, tags. Keep it minimal
here so later phases add fields without restructuring.

```go
type Frontmatter struct {
    Title       string
    NodeType    string  // validated in Phase 6
    Authority   string  // validated in Phase 6
    Aliases     []string
    Tags        []string
    ReviewAfter *time.Time
    Extra       map[string]any // unknown fields preserved verbatim
}
```

### Errors

Sentinel errors for protocol failures; use `errors.Is` for matching.

```go
var (
    ErrInvalidOrgID         = errors.New("invalid org id")
    ErrInvalidAreaID        = errors.New("invalid area id")
    ErrInvalidNodeID        = errors.New("invalid node id")
    ErrPathTraversal        = errors.New("path traversal denied")
    ErrEmptyBody            = errors.New("body is empty for active node")
    ErrMissingFrontmatter   = errors.New("required frontmatter missing")
    ErrUnknownNodeType      = errors.New("unknown node type")
    ErrUnknownAuthority     = errors.New("unknown authority")
    ErrExpectedVersionUnset = errors.New("update requires expected version")
    ErrIdentityMismatch     = errors.New("frontmatter identity does not match proposal identity")
)
```

Typed errors for rich rejection context:

```go
type StaleError struct {
    Identity      Identity
    Expected      int
    Current       int
}

func (e *StaleError) Error() string
```

### Validators

Pure functions, no I/O. Each returns the first error encountered.

```go
func ValidateOrgID(s string) error      // [a-z0-9][a-z0-9-]{1,63}
func ValidateAreaID(s string) error     // [a-z][a-z0-9._-]{0,63}
func ValidateNodeID(s string) error     // [a-z][a-z0-9._-]{0,127}; no '..'
func ValidatePathComponent(s string) error
func ValidateProposal(p Proposal) error // composes the above
func ValidateFrontmatter(f Frontmatter) error
```

Rules to encode:

- IDs are lowercase ASCII; underscores not permitted (use `.` or `-`)
- no segment may be `.`, `..`, or empty
- node ID may not contain `/`
- create proposals must have `ExpectedVersion == 0`
- update/retire proposals must have `ExpectedVersion >= 1`
- body must be non-empty when status is active
- frontmatter `id`, `org_id`, `area` must match the proposal's `Identity`

### ID Generation

```go
func NewProposalID() string  // "prop_" + 22-char base32 ULID
func NewVersionID() string   // "ver_" + 22-char base32 ULID
```

Use `github.com/oklog/ulid/v2` or a similar tiny dependency. IDs are
lexically sortable so they double as ordering keys.

### Time

```go
func Now() time.Time // returns time.Now().UTC()
```

Use this everywhere. Tests can swap it out via build tags or an injected
clock interface later if needed.

## Package Layout

```text
pkg/protocol/
  doc.go            # package comment, links to PROTOCOL.md
  identity.go       # OrgID, AreaID, NodeID, Identity
  status.go         # all enum-like consts
  proposal.go       # Proposal, ProposalKind
  accepted.go       # AcceptedVersion, NodeStatus
  resolution.go     # Resolution
  frontmatter.go    # Frontmatter
  errors.go         # sentinels + StaleError
  validate.go       # validation functions
  id.go             # ID generators
  clock.go          # Now()
  fixtures.go       # canonical fixtures (Test* prefix, exported for cross-package reuse)
  *_test.go         # table-driven tests per file
```

The fixtures live in a non-`_test.go` file so other packages' tests can
import them. Each fixture name starts with `Fixture` so production code
that accidentally imports them is easy to spot in review.

## Tests Required

In `pkg/protocol`:

| Test File | What It Asserts |
|---|---|
| `identity_test.go` | identity string format, equality, validation |
| `validate_test.go` | every validation rule and rejection path |
| `proposal_test.go` | create/update/retire field rules |
| `frontmatter_test.go` | required fields, identity match, extras preserved |
| `errors_test.go` | sentinel error matching via `errors.Is` |

Tests must be table-driven. Each rule from [PROTOCOL.md](../PROTOCOL.md)
"Identity Tests" and "Markdown Validation Tests" sections gets at least one
row.

## Canonical Fixtures

Expose canonical org/area/node fixtures for use in later phase tests.
Because the file is not `_test.go`, other packages' tests can import them.

```go
// pkg/protocol/fixtures.go
package protocol

var (
    FixtureOrgAcme            = OrgID("acme")
    FixtureAreaRBAC           = AreaID("rbac")
    FixtureNodeACLs           = NodeID("rbac.acls")
    FixtureIdentityRBACACLs   = Identity{
        Org: FixtureOrgAcme, Area: FixtureAreaRBAC, Node: FixtureNodeACLs,
    }
)
```

## Exit Criteria

- `go test ./pkg/protocol/...` exits 0
- every validation rule has a failing-first test that passes after implementation
- no package outside `pkg/protocol` is imported (zero deps within module)
- `golangci-lint run ./pkg/protocol/...` exits 0
- public API documented with package-level Go doc

Once these are green, proceed to
[Phase 2: Postgres Schema and Migrations](phase-02-postgres-schema.md).

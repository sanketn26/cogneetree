# Phase 6: Node Metadata and Validation

Goal: agents can interpret nodes by type and authority. Invalid frontmatter is
rejected before acceptance, not at read time.

## Deliverables

- `node_type` enum enforced
- `authority` enum enforced
- aliases and tags validated and indexed in frontmatter
- review/expiration metadata supported
- content size limits enforced
- frontmatter-identity match verified

## Package: `pkg/protocol` (Extended)

```go
type NodeType string

const (
    NodeTypePolicy            NodeType = "policy"
    NodeTypeDecision          NodeType = "decision"
    NodeTypeProcedure         NodeType = "procedure"
    NodeTypeFact              NodeType = "fact"
    NodeTypeRunbook           NodeType = "runbook"
    NodeTypeLesson            NodeType = "lesson"
    NodeTypeInterfaceContract NodeType = "interface_contract"
    NodeTypeOpenQuestion      NodeType = "open_question"
)

type Authority string

const (
    AuthorityNormative     Authority = "normative"
    AuthorityAdvisory      Authority = "advisory"
    AuthorityObservational Authority = "observational"
)

// AllNodeTypes and AllAuthorities are exposed for validators and UI.
func AllNodeTypes() []NodeType
func AllAuthorities() []Authority
```

### Frontmatter (Full Schema)

```go
type Frontmatter struct {
    ID          NodeID              `json:"id"`
    OrgID       OrgID               `json:"org_id"`
    Area        AreaID              `json:"area"`
    Title       string              `json:"title"`
    NodeType    NodeType            `json:"node_type"`
    Authority   Authority           `json:"authority"`
    Status      NodeStatus          `json:"status"`
    Version     int                 `json:"version"`
    UpdatedAt   time.Time           `json:"updated_at"`
    UpdatedBy   string              `json:"updated_by"`
    AcceptedBy  string              `json:"accepted_by"`
    Reason      string              `json:"change_reason"`
    ReviewAfter *time.Time          `json:"review_after,omitempty"`
    ExpireAfter *time.Time          `json:"expire_after,omitempty"`
    Aliases     []string            `json:"aliases,omitempty"`
    Tags        []string            `json:"tags,omitempty"`
    RetiredReason string            `json:"retired_reason,omitempty"`
    SupersededBy []NodeRef          `json:"superseded_by,omitempty"`
    ConflictsWith []NodeRef         `json:"conflicts_with,omitempty"`
    Extra       map[string]any      `json:"-"`
}

type NodeRef struct {
    Area    AreaID
    Node    NodeID
    Version int // 0 means "latest"
}
```

### Validation Rules

```go
// ValidateFrontmatter returns the first failure encountered.
func ValidateFrontmatter(f Frontmatter, id Identity) error
```

Encoded rules:

| Rule | Error |
|---|---|
| `f.OrgID == id.Org` | `ErrIdentityMismatch` |
| `f.Area == id.Area` | `ErrIdentityMismatch` |
| `f.ID == id.Node` | `ErrIdentityMismatch` |
| `f.Title != ""` | `ErrMissingTitle` |
| `f.NodeType` is a known constant | `ErrUnknownNodeType` |
| `f.Authority` is a known constant | `ErrUnknownAuthority` |
| `f.Status` is `active` or `retired` | `ErrUnknownStatus` |
| If `f.Status == retired`, `f.RetiredReason != ""` | `ErrRetirementReasonMissing` |
| `f.Version >= 1` | `ErrVersionNotPositive` |
| `len(f.Aliases) <= 16` and each alias <= 64 chars | `ErrAliasLimit` |
| `len(f.Tags) <= 24` and each tag matches `^[a-z][a-z0-9-]{0,31}$` | `ErrTagFormat` |
| `f.ReviewAfter` is in the future when set | `ErrReviewAfterPast` |
| `f.ExpireAfter > f.ReviewAfter` when both set | `ErrExpireBeforeReview` |

### Body Validation

```go
const (
    MaxBodyBytes       = 256 * 1024  // 256 KB
    MinActiveBodyBytes = 1
)

func ValidateBody(status NodeStatus, body string) error
```

Rules:

- active nodes have non-empty body
- retired nodes may have empty body
- body must be valid UTF-8
- body must not exceed `MaxBodyBytes`
- body must not contain control characters except `\n` and `\t`

## Package: `internal/leader` (Extended)

Validation runs before any database state changes:

```text
begin tx
  ValidateFrontmatter(p.Frontmatter, p.Identity)
  ValidateBody(implied_status_from_kind, p.Body)
  if any validation fails:
      rollback, return RejectInvalid(p, validation_error)
  ... existing admission ...
commit
```

`RejectInvalid` is now a first-class status, not a placeholder. Audit event
`proposal_rejected_invalid` carries the validation error code.

## Frontmatter Serialization

```go
package frontmatter // internal/frontmatter

// Render produces YAML frontmatter plus a Markdown body.
func Render(f protocol.Frontmatter, body string) (string, error)

// Parse extracts frontmatter and body from a Markdown document.
func Parse(markdown string) (protocol.Frontmatter, string, error)
```

Use `gopkg.in/yaml.v3` (or `goccy/go-yaml`) for marshaling. Preserve unknown
fields in `Frontmatter.Extra` so older agents writing newer fields don't lose
data on round-trip.

## Tests Required

In `pkg/protocol`:

| Test | What It Asserts |
|---|---|
| `TestValidateFrontmatterRequiresIdentity` | mismatched org/area/id → ErrIdentityMismatch |
| `TestValidateFrontmatterRejectsUnknownNodeType` | invalid node_type → error |
| `TestValidateFrontmatterRejectsUnknownAuthority` | invalid authority → error |
| `TestValidateFrontmatterRetiredRequiresReason` | retired without retired_reason → error |
| `TestValidateAliasLimits` | >16 aliases or alias >64 chars → error |
| `TestValidateTagFormat` | invalid tag pattern → error |
| `TestValidateReviewAfterFuture` | review_after in past → error |
| `TestValidateExpireBeforeReview` | expire_after <= review_after → error |
| `TestValidateBodyEmptyActiveRejected` | active node with empty body → error |
| `TestValidateBodyTooLargeRejected` | body > 256 KB → error |
| `TestValidateBodyInvalidUTF8Rejected` | invalid UTF-8 → error |

In `internal/frontmatter`:

| Test | What It Asserts |
|---|---|
| `TestRenderParseRoundTrip` | Render then Parse returns equal frontmatter |
| `TestParsePreservesUnknownFields` | unknown fields land in Extra |
| `TestRenderProducesYAMLBlock` | output starts with `---\n` and has body after second `---` |

In `internal/leader`:

| Test | What It Asserts |
|---|---|
| `TestRejectInvalidBeforeAcceptance` | proposal with bad frontmatter never writes accepted row |
| `TestRejectInvalidAuditEvent` | rejected_invalid event includes the validation error code |
| `TestExpiredNodeRemainsReadable` | a node past expire_after is still returned by ReadCurrent (Phase 6 only flags it; later phases may filter) |

## Read APIs (Extended)

```go
// ReadCurrent returns the current accepted version even if expired.
// Callers filter on Frontmatter.ExpireAfter as needed.
func (s *Store) ReadCurrent(
    ctx context.Context,
    id protocol.Identity,
) (*protocol.AcceptedVersion, error)

// ListDueForReview returns nodes whose review_after is past.
func (s *Store) ListDueForReview(
    ctx context.Context,
    org protocol.OrgID,
    now time.Time,
) ([]protocol.AcceptedVersion, error)
```

Backing SQL for review listing:

```sql
SELECT * FROM accepted_versions
 WHERE org_id = $1
   AND is_current = true
   AND status = 'active'
   AND (frontmatter ->> 'review_after')::timestamptz < $2
 ORDER BY (frontmatter ->> 'review_after')::timestamptz;
```

## Exit Criteria

- invalid frontmatter rejected before any accepted row exists
- all node types and authorities enforced from the protocol constants
- `Render` / `Parse` round-trip preserves frontmatter, unknown fields included
- review-due nodes queryable
- all tests above pass

Once these are green, proceed to
[Phase 7: Tombstones and Lifecycle](phase-07-tombstones.md).

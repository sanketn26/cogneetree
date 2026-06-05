# Phase 11: HTTP Adapter

Goal: expose the protocol through HTTP so agents (and tooling) talk to
Cogneetree over the wire. Thin adapter — no admission logic, no validation
beyond request parsing. The leader package remains authoritative.

## Deliverables

- REST endpoints matching the validation contract
- per-request RLS via Postgres GUC
- bearer-token authentication with scoped permissions
- structured request logging
- OpenAPI document generated from handlers (optional, recommended)

## Package: `internal/httpadapter`

```go
package httpadapter

type Server struct {
    cfg     Config
    api     cogneetreeclient.MemoryAPI
    store   *store.Store
    auth    Authenticator
    log     *slog.Logger
}

type Config struct {
    Addr            string
    ReadTimeout     time.Duration
    WriteTimeout    time.Duration
    IdleTimeout     time.Duration
    MaxBodyBytes    int64 // default 1 MiB
}

func New(
    cfg Config,
    api cogneetreeclient.MemoryAPI,
    db *store.Store,
    auth Authenticator,
    log *slog.Logger,
) *Server

// Run blocks until ctx is cancelled. Returns nil on graceful shutdown.
func (s *Server) Run(ctx context.Context) error
```

### Routing

Stdlib `http.ServeMux` (Go 1.22+ pattern syntax):

```go
mux.HandleFunc("POST   /v1/orgs/{org}/areas/{area}/nodes/{node}/proposals",  s.submitProposal)
mux.HandleFunc("GET    /v1/orgs/{org}/areas/{area}/nodes/{node}",            s.readNode)
mux.HandleFunc("GET    /v1/orgs/{org}/areas/{area}/nodes/{node}/versions",   s.listVersions)
mux.HandleFunc("GET    /v1/orgs/{org}/areas/{area}/nodes/{node}/versions/{version}", s.readVersion)
mux.HandleFunc("GET    /v1/orgs/{org}/areas/{area}/nodes/{node}/history",    s.nodeHistory)
mux.HandleFunc("GET    /v1/orgs/{org}/areas/{area}/nodes/{node}/edges",      s.listEdges)
mux.HandleFunc("GET    /v1/orgs/{org}/areas/{area}/nodes/{node}/neighborhood", s.neighborhood)
mux.HandleFunc("GET    /v1/orgs/{org}/areas/{area}/tree",                    s.areaTree)
mux.HandleFunc("GET    /v1/orgs/{org}/search",                               s.search)
mux.HandleFunc("GET    /v1/orgs/{org}/manifest",                             s.manifest)

mux.HandleFunc("GET    /v1/proposals/{proposal_id}",                         s.getResolution)
mux.HandleFunc("POST   /v1/orgs/{org}/change-sets",                          s.submitChangeSet)
mux.HandleFunc("GET    /v1/change-sets/{change_set_id}",                     s.getChangeSet)

mux.HandleFunc("POST   /internal/worker/process-next",                       s.workerProcessNext)
mux.HandleFunc("GET    /healthz",                                            s.health)
mux.HandleFunc("GET    /readyz",                                             s.ready)
```

Internal endpoints (`/internal/...`) require the leader role and are not
exposed to agents.

## Authentication and Authorization

```go
type Authenticator interface {
    Authenticate(r *http.Request) (Principal, error)
}

type Principal struct {
    ID         string
    Scopes     []Scope
    OrgAccess  map[protocol.OrgID][]Scope // per-org scope overrides
}

type Scope string

const (
    ScopeRead    Scope = "read"
    ScopePropose Scope = "propose"
    ScopeAccept  Scope = "accept" // leader-only
    ScopeAdmin   Scope = "admin"
)
```

The default implementation reads a bearer token from `Authorization: Bearer`
and looks up scopes in a `principals` table. Phase 11 ships with a static
token implementation for development; production deployments swap in OAuth
or JWT.

### Authorization Middleware

```go
func requireScope(s Scope) func(http.Handler) http.Handler

// example
mux.Handle("POST /v1/orgs/{org}/areas/{area}/nodes/{node}/proposals",
    requireScope(ScopePropose)(http.HandlerFunc(s.submitProposal)))
```

The org is parsed from the path; if the principal's `OrgAccess` does not
include the scope, 403.

## RLS GUC Per Request

`set_config(..., is_local=true)` is **transaction-local**. The setting only
applies within an explicit transaction; outside a transaction it lasts for
the duration of the `SELECT set_config` statement and nothing more. The
helper must therefore wrap the GUC and every scoped query in one
transaction.

```go
func (s *Server) withOrgGUC(
    ctx context.Context,
    org protocol.OrgID,
    fn func(ctx context.Context, tx pgx.Tx) error,
) error {
    return s.store.Pool().BeginFunc(ctx, func(tx pgx.Tx) error {
        // is_local=true scopes the GUC to this transaction; it is
        // discarded on commit/rollback and never leaks across pool reuse.
        if _, err := tx.Exec(
            ctx,
            "SELECT set_config('cogneetree.org_id', $1, true)",
            string(org),
        ); err != nil {
            return fmt.Errorf("set rls org guc: %w", err)
        }
        return fn(ctx, tx)
    })
}
```

All scoped reads inside `fn` run on `tx`, not a fresh connection. Handlers
that need RLS-scoped access use `withOrgGUC`; the leader path runs as the
bypass-RLS role and does not.

The alternative — `set_config(..., false)` outside a transaction — is
**rejected**. It would set the GUC on the pooled connection and leak the
org binding to whoever next acquired the connection. A bug there would
silently let one tenant read another's rows.

## Request and Response Shapes

### Submit Proposal

```text
POST /v1/orgs/{org}/areas/{area}/nodes/{node}/proposals
Content-Type: application/json
Authorization: Bearer <token>

{
  "kind": "update",
  "expected_version": 4,
  "body": "...",
  "frontmatter": { ... },
  "reason": "Contractor access reduced after policy update.",
  "source_refs": ["conversation:abc123", "https://internal.wiki/policy/q2"],
  "change_set_id": null
}

202 Accepted
{
  "proposal_id": "prop_01HZ...",
  "status": "pending",
  "submitted_at": "2026-06-05T22:00:00Z"
}
```

The response is 202 because admission is asynchronous via the worker. Clients
poll `GET /v1/proposals/{id}` or rely on a future webhook (out of scope).

### Get Resolution

```text
GET /v1/proposals/prop_01HZ...

200 OK
{
  "proposal_id": "prop_01HZ...",
  "status": "accepted",
  "identity": {"org":"acme","area":"rbac","node":"rbac.acls"},
  "latest_version": 5,
  "accepted_version": { ...AcceptedVersion... },
  "decided_by": "leader-1",
  "decided_at": "2026-06-05T22:00:01Z"
}
```

Statuses match `protocol.ProposalStatus` values.

### Read Node

```text
GET /v1/orgs/acme/areas/rbac/nodes/rbac.acls

200 OK
{
  "identity": {"org":"acme","area":"rbac","node":"rbac.acls"},
  "version": 5,
  "status": "active",
  "frontmatter": { ... },
  "body": "...",
  "accepted_by": "rbac-leader",
  "accepted_at": "2026-06-05T22:00:01Z"
}
```

Headers:
- `ETag: "<version>"` — the version number doubles as the ETag
- `Cache-Control: private, max-age=0, must-revalidate`

`If-None-Match` is honored: if the version matches, 304.

### Search

```text
GET /v1/orgs/acme/search?q=acl+contractor&area=rbac&tag=access&limit=10

200 OK
{
  "hits": [
    {
      "identity": {"org":"acme","area":"rbac","node":"rbac.acls"},
      "title": "ACLs",
      "version": 5,
      "score": 0.84,
      "snippet": "«ACL» policy for «contractor» access ...",
      "node_type": "policy",
      "authority": "normative",
      "updated_at": "2026-06-05T22:00:01Z"
    }
  ],
  "total": 1
}
```

### Errors

```text
HTTP/1.1 409 Conflict
Content-Type: application/json

{
  "error": "stale",
  "message": "expected version does not match current accepted version",
  "current_version": 6,
  "expected_version": 4
}
```

Error codes map to `protocol.ProposalStatus`:

| HTTP | error code | source |
|---|---|---|
| 400 | invalid | validation errors |
| 401 | unauthenticated | missing or bad token |
| 403 | unauthorized | scope mismatch |
| 404 | not_found | unknown node or version |
| 409 | stale | version CAS failed |
| 422 | conflict_unacknowledged | unacknowledged conflict edge |
| 500 | internal | unexpected |

## Middleware Stack

```text
panic-recovery
  -> request-id
    -> structured logger
      -> bearer-auth
        -> scope-check
          -> RLS GUC binding
            -> handler
```

Each middleware is a `func(http.Handler) http.Handler`. Order matters; do not
reorder without a test.

## OpenAPI

Optional but recommended. Generate via `kin-openapi` from typed request and
response structs annotated with godoc comments. Serve at `GET /openapi.json`.

If skipped in Phase 11, document the endpoints in
`docs/implementation/phase-11-http-adapter.md` (this file) as the contract.

## Tests Required

In `internal/httpadapter`:

| Test | What It Asserts |
|---|---|
| `TestSubmitProposalReturns202` | POST returns 202 with proposal_id |
| `TestSubmitProposalRequiresProposeScope` | missing scope → 403 |
| `TestReadNodeReturns200WithETag` | GET includes ETag header |
| `TestReadNode304WhenETagMatches` | If-None-Match returns 304 |
| `TestReadNodeRLSBlocksOtherOrgs` | token scoped to org A reading org B → 404 (not 403) |
| `TestStaleRejectionReturns409` | stale update returns 409 with current_version |
| `TestInvalidFrontmatterReturns400` | malformed body returns 400 |
| `TestSearchReturnsRankedHits` | search results sorted by score |
| `TestHealthzReturns200` | always responds 200 when DB pool is open |
| `TestReadyzFailsBeforeMigrations` | returns 503 if migrations not applied |
| `TestGracefulShutdownDrainsRequests` | in-flight requests complete before exit |
| `TestRequestBodyLimitEnforced` | bodies > MaxBodyBytes return 413 |

## CLI Extension

```text
cogneetree serve         # starts HTTP + worker in one process (dev mode)
cogneetree serve --http  # HTTP only (production usually splits)
cogneetree migrate up
cogneetree migrate status
cogneetree index rebuild --org acme
```

## Operational Notes

- Run HTTP and worker as separate processes in production. They share the
  database. The worker uses the leader role; the HTTP server uses the app role.
- Behind a reverse proxy that terminates TLS. Cogneetree does not handle TLS
  itself.
- Configure connection pool sizes per process: HTTP gets `max_open` based on
  concurrent requests; worker gets ~2× `MaxAreaGoroutines`.

## Exit Criteria

- every public API in PROTOCOL.md is reachable over HTTP
- per-request RLS verified end-to-end (cross-org access denied)
- bearer auth and scope checks enforced on every mutation endpoint
- 202 / 409 / 400 / 403 / 404 / 422 / 500 codes returned per the table above
- graceful shutdown drains in-flight requests
- all tests above pass

Once these are green, proceed to
[Phase 12: Observability](phase-12-observability.md).

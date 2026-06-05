# Phase 2: Postgres Schema and Migrations

Goal: stand up the durable storage layer. Define the tables, partitions, RLS
policies, and migration tooling that every subsequent phase writes against.

## Deliverables

- migration tool (`goose`) wired in and runnable via `make migrate`
- migrations create the full schema on a fresh database
- partitioned tables for org isolation
- row-level security enabled with org-scoping policies
- helper to register a new org (creates partitions)
- integration tests against a real Postgres via testcontainers

## Package: `internal/store`

The Phase 2 store is intentionally minimal — it exposes the DB pool and the
org registration helper. Phase 3 adds CRUD methods.

```go
package store

type Store struct {
    pool *pgxpool.Pool
}

func Open(ctx context.Context, dsn string) (*Store, error)
func (s *Store) Close()
func (s *Store) Ping(ctx context.Context) error
func (s *Store) Pool() *pgxpool.Pool

// RegisterOrg creates partitions for a new org. Idempotent.
func (s *Store) RegisterOrg(ctx context.Context, org protocol.OrgID) error
```

## Migration Files

Migrations live under `/migrations` as `NNNN_name.sql` pairs. Each file has
`-- +goose Up` and `-- +goose Down` sections.

### `0001_extensions.sql`

```sql
-- +goose Up
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- +goose Down
DROP EXTENSION IF EXISTS btree_gin;
DROP EXTENSION IF EXISTS pg_trgm;
```

### `0002_organizations.sql`

```sql
-- +goose Up
CREATE TABLE organizations (
  org_id       text PRIMARY KEY,
  created_at   timestamptz NOT NULL DEFAULT now(),
  display_name text NOT NULL,
  CONSTRAINT org_id_format CHECK (org_id ~ '^[a-z0-9][a-z0-9-]{0,63}$')
);

-- +goose Down
DROP TABLE organizations;
```

### `0003_accepted_versions.sql`

Partitioned by `org_id` so each tenant gets its own physical table. New orgs
register a partition at runtime via `Store.RegisterOrg`.

```sql
-- +goose Up
CREATE TABLE accepted_versions (
  org_id        text        NOT NULL,
  area_id       text        NOT NULL,
  node_id       text        NOT NULL,
  version       int         NOT NULL,
  is_current    boolean     NOT NULL DEFAULT false,
  status        text        NOT NULL CHECK (status IN ('active','retired')),
  frontmatter   jsonb       NOT NULL,
  body          text        NOT NULL,
  body_tsv      tsvector,   -- populated in Phase 9
  proposal_id   text        NOT NULL,
  from_version  int         NOT NULL DEFAULT 0,
  accepted_by   text        NOT NULL,
  accepted_at   timestamptz NOT NULL DEFAULT now(),
  reason        text        NOT NULL DEFAULT '',
  source_refs   jsonb       NOT NULL DEFAULT '[]'::jsonb,
  change_set_id text,
  PRIMARY KEY (org_id, area_id, node_id, version)
) PARTITION BY LIST (org_id);

CREATE UNIQUE INDEX accepted_one_current
  ON accepted_versions (org_id, area_id, node_id)
  WHERE is_current;

CREATE INDEX accepted_by_area
  ON accepted_versions (org_id, area_id)
  WHERE is_current;

-- +goose Down
DROP TABLE accepted_versions;
```

The `accepted_one_current` partial unique index is load-bearing: it enforces
"one active accepted state per (org, area, node)" at the database level. No
application code can violate this without raising a unique violation.

### `0004_pending_proposals.sql`

```sql
-- +goose Up
CREATE TABLE pending_proposals (
  org_id           text        NOT NULL,
  proposal_id      text        NOT NULL,
  area_id          text        NOT NULL,
  node_id          text        NOT NULL,
  kind             text        NOT NULL CHECK (kind IN ('create','update','retire')),
  expected_version int         NOT NULL,
  body             text        NOT NULL,
  frontmatter      jsonb       NOT NULL,
  reason           text        NOT NULL DEFAULT '',
  source_refs      jsonb       NOT NULL DEFAULT '[]'::jsonb,
  proposed_by      text        NOT NULL,
  submitted_at     timestamptz NOT NULL DEFAULT now(),
  change_set_id    text,
  status           text        NOT NULL
                     CHECK (status IN ('pending','claimed','resolved')),
  claimed_by       text,
  claimed_at       timestamptz,
  resolution       jsonb,
  PRIMARY KEY (org_id, proposal_id)
) PARTITION BY LIST (org_id);

CREATE INDEX pending_ready
  ON pending_proposals (org_id, area_id, submitted_at)
  WHERE status = 'pending';

-- +goose Down
DROP TABLE pending_proposals;
```

Postgres requires that the primary key of a partitioned table include the
partition key, so `org_id` leads the composite PK. Proposal IDs are ULIDs
and remain globally unique in practice; the composite PK is just a
Postgres constraint requirement.

### `0005_audit_events.sql`

Partitioned by month, range-partitioned on `created_at`. A maintenance task
(Phase 5) creates monthly partitions ahead of time.

```sql
-- +goose Up
CREATE TABLE audit_events (
  event_id    bigint      GENERATED ALWAYS AS IDENTITY,
  org_id      text        NOT NULL,
  area_id     text,
  node_id     text,
  version     int,
  event_type  text        NOT NULL,
  proposal_id text,
  payload     jsonb       NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (created_at, event_id)
) PARTITION BY RANGE (created_at);

CREATE INDEX audit_by_node
  ON audit_events (org_id, area_id, node_id, created_at);

-- Default partition catches any row whose timestamp falls outside the
-- explicit monthly partitions created by EnsureAuditPartition (Phase 5).
-- Without this, Phase 3 inserts would fail before Phase 5 lands.
CREATE TABLE audit_events_default
  PARTITION OF audit_events DEFAULT;

-- +goose Down
DROP TABLE audit_events;
```

The default partition is a permanent backstop. Phase 5 adds monthly
partitions via `EnsureAuditPartition`; rows in those months land in the
monthly partition instead of the default. The default partition is allowed
to accumulate rows; the operator periodically detaches and migrates them
into monthly partitions as part of routine maintenance.

### `0006_rls.sql`

Row-level security so a session that sets `cogneetree.org_id` cannot read or
write rows belonging to another org. The HTTP adapter (Phase 11) sets this
GUC at the start of every request.

```sql
-- +goose Up
ALTER TABLE accepted_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE pending_proposals ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_events       ENABLE ROW LEVEL SECURITY;

CREATE POLICY org_isolation_accepted ON accepted_versions
  USING (org_id = current_setting('cogneetree.org_id', true));

CREATE POLICY org_isolation_pending ON pending_proposals
  USING (org_id = current_setting('cogneetree.org_id', true));

CREATE POLICY org_isolation_audit ON audit_events
  USING (org_id = current_setting('cogneetree.org_id', true));

-- +goose Down
DROP POLICY IF EXISTS org_isolation_accepted ON accepted_versions;
DROP POLICY IF EXISTS org_isolation_pending  ON pending_proposals;
DROP POLICY IF EXISTS org_isolation_audit    ON audit_events;
ALTER TABLE accepted_versions DISABLE ROW LEVEL SECURITY;
ALTER TABLE pending_proposals DISABLE ROW LEVEL SECURITY;
ALTER TABLE audit_events       DISABLE ROW LEVEL SECURITY;
```

The leader process runs with a `BYPASSRLS` role so it can resolve proposals
across orgs without juggling GUCs. Application connections run with a
restricted role that cannot bypass RLS.

## Org Registration

`Store.RegisterOrg` is idempotent and runs three statements in a single
transaction:

```sql
INSERT INTO organizations (org_id, display_name) VALUES ($1, $1)
  ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS accepted_versions_{org}
  PARTITION OF accepted_versions FOR VALUES IN ('{org}');

CREATE TABLE IF NOT EXISTS pending_proposals_{org}
  PARTITION OF pending_proposals FOR VALUES IN ('{org}');
```

The `{org}` substitution must be sanitized through `pgx`'s `Identifier.Sanitize`
to avoid SQL injection. The `protocol.ValidateOrgID` guard is the first line of
defense; identifier sanitization is the second.

## Roles

Two database roles, created in `0007_roles.sql`:

```sql
-- +goose Up
CREATE ROLE cogneetree_app    NOLOGIN;
CREATE ROLE cogneetree_leader NOLOGIN BYPASSRLS;

GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO cogneetree_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO cogneetree_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO cogneetree_leader;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO cogneetree_leader;

-- +goose Down
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM cogneetree_app;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM cogneetree_leader;
DROP ROLE IF EXISTS cogneetree_app;
DROP ROLE IF EXISTS cogneetree_leader;
```

Operators grant these roles to login users out of band; the migration only
creates the role shapes.

## Package: `internal/store/migrate`

```go
package migrate

//go:embed all:*.sql
var FS embed.FS

func Up(ctx context.Context, dsn string) error
func Down(ctx context.Context, dsn string) error
func Status(ctx context.Context, dsn string) ([]Status, error)
```

The CLI exposes `cogneetree migrate up` / `cogneetree migrate status` in
Phase 11.

## Tests Required

In `internal/store`:

| Test | What It Asserts |
|---|---|
| `TestMigrationsRunOnEmptyDB` | up + down cycle leaves no leftover tables |
| `TestMigrationsIdempotent` | running up twice does not fail |
| `TestRegisterOrgIdempotent` | registering same org twice does not fail |
| `TestRegisterOrgCreatesPartitions` | partitions exist after registration |
| `TestPartialUniqueOnCurrent` | inserting two `is_current=true` rows fails |
| `TestRLSBlocksCrossOrgRead` | session set to org A cannot see org B rows |
| `TestRLSLeaderRoleBypasses` | leader role sees all orgs |

Use `testcontainers-go/modules/postgres` to spin up a real Postgres per test
package. Share the container across tests within a package; isolate via
`TRUNCATE` between tests.

## Exit Criteria

- `make migrate` runs cleanly on a fresh database
- `make migrate` is idempotent (repeated runs do nothing)
- a new org can be registered and its partitions appear
- the partial unique index prevents two current versions of a node
- RLS blocks cross-org reads on the app role
- RLS allows cross-org access on the leader role
- `go test ./internal/store/...` exits 0 with at least the tests above

Once these are green, proceed to
[Phase 3: Leader Admission](phase-03-leader-admission.md).

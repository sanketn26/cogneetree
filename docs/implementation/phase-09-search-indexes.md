# Phase 9: Search and Derived Indexes

Goal: agents can find nodes by lexical match, alias, or tag. Search is fast,
ranked, and limited to the hot tier (current accepted versions). Indexes are
rebuildable caches; deleting them never loses accepted knowledge.

## Deliverables

- `body_tsv` populated on every accepted row (current and historical)
- partial GIN index on `body_tsv` restricted to `is_current`
- `pg_trgm` GIN index on title and aliases (current only)
- alias and tag lookup tables (derived)
- a manifest rebuild command
- search APIs with BM25-style ranking

Non-goal: semantic / vector search. Cogneetree is a wiki, not a vector store.
See [PROTOCOL.md](../PROTOCOL.md) for the explicit non-goal.

## Schema Changes

### `0011_search_indexes.sql`

```sql
-- +goose Up
-- Populate tsvector with weighted columns:
--   A = title (highest)
--   B = aliases + tags
--   C = body
UPDATE accepted_versions
   SET body_tsv =
       setweight(to_tsvector('english', coalesce(frontmatter->>'title','')), 'A') ||
       setweight(to_tsvector('english', coalesce(frontmatter->>'aliases',''), 'B'), 'B') ||
       setweight(to_tsvector('english', body), 'C')
 WHERE body_tsv IS NULL;

-- Make body_tsv a generated column going forward.
ALTER TABLE accepted_versions
  DROP COLUMN body_tsv,
  ADD COLUMN body_tsv tsvector
    GENERATED ALWAYS AS (
      setweight(to_tsvector('english', coalesce(frontmatter->>'title','')), 'A') ||
      setweight(to_tsvector('english', coalesce(jsonb_path_query_array(frontmatter,'$.aliases[*]')::text,'')), 'B') ||
      setweight(to_tsvector('english', body), 'C')
    ) STORED;

-- Partial GIN index: only current versions are searchable by default.
CREATE INDEX accepted_current_fts
  ON accepted_versions USING GIN (body_tsv)
  WHERE is_current AND status = 'active';

-- Trigram fuzzy match on title (current only).
CREATE INDEX accepted_current_title_trgm
  ON accepted_versions
  USING GIN ((frontmatter->>'title') gin_trgm_ops)
  WHERE is_current AND status = 'active';

-- +goose Down
DROP INDEX IF EXISTS accepted_current_title_trgm;
DROP INDEX IF EXISTS accepted_current_fts;
ALTER TABLE accepted_versions DROP COLUMN body_tsv;
ALTER TABLE accepted_versions ADD COLUMN body_tsv tsvector;
```

### `0012_alias_index.sql`

Aliases are repeated across nodes; a dedicated lookup table makes `find by
alias` an indexed equality scan rather than a JSONB array probe.

```sql
-- +goose Up
CREATE TABLE alias_index (
  org_id     text NOT NULL,
  alias      text NOT NULL,
  area_id    text NOT NULL,
  node_id    text NOT NULL,
  PRIMARY KEY (org_id, alias, area_id, node_id)
) PARTITION BY LIST (org_id);

CREATE INDEX alias_index_by_alias ON alias_index (org_id, alias);

ALTER TABLE alias_index ENABLE ROW LEVEL SECURITY;
CREATE POLICY org_isolation_alias ON alias_index
  USING (org_id = current_setting('cogneetree.org_id', true));

-- +goose Down
DROP TABLE alias_index;
```

### `0013_tag_index.sql`

```sql
-- +goose Up
CREATE TABLE tag_index (
  org_id  text NOT NULL,
  tag     text NOT NULL,
  area_id text NOT NULL,
  node_id text NOT NULL,
  PRIMARY KEY (org_id, tag, area_id, node_id)
) PARTITION BY LIST (org_id);

CREATE INDEX tag_index_by_tag ON tag_index (org_id, tag);

ALTER TABLE tag_index ENABLE ROW LEVEL SECURITY;
CREATE POLICY org_isolation_tag ON tag_index
  USING (org_id = current_setting('cogneetree.org_id', true));

-- +goose Down
DROP TABLE tag_index;
```

## Package: `internal/search`

```go
package search

type Engine struct {
    db *store.Store
}

func New(db *store.Store) *Engine

type Query struct {
    Org        protocol.OrgID
    Text       string         // free-text; empty allowed if filters set
    Areas      []protocol.AreaID
    Tags       []string
    NodeTypes  []protocol.NodeType
    Authority  []protocol.Authority
    Limit      int            // default 20, max 100
    IncludeRetired bool
}

type Hit struct {
    Identity    protocol.Identity
    Title       string
    Version     int
    Score       float32
    Snippet     string  // ts_headline output
    UpdatedAt   time.Time
    NodeType    protocol.NodeType
    Authority   protocol.Authority
}

func (e *Engine) Search(ctx context.Context, q Query) ([]Hit, error)
func (e *Engine) ByAlias(ctx context.Context, org protocol.OrgID, alias string) ([]Hit, error)
func (e *Engine) ByTag(ctx context.Context, org protocol.OrgID, tag string) ([]Hit, error)
```

### Ranked Search SQL

```sql
WITH q AS (SELECT plainto_tsquery('english', $1) AS tsq)
SELECT org_id, area_id, node_id, version,
       frontmatter->>'title' AS title,
       ts_rank_cd(body_tsv, q.tsq, 32) AS score,
       ts_headline('english', body, q.tsq,
                   'StartSel=«,StopSel=»,MaxFragments=2,MaxWords=24,MinWords=8') AS snippet,
       accepted_at,
       frontmatter->>'node_type'  AS node_type,
       frontmatter->>'authority'  AS authority
  FROM accepted_versions, q
 WHERE org_id = $2
   AND is_current = true
   AND status = 'active'
   AND body_tsv @@ q.tsq
 ORDER BY score DESC, accepted_at DESC
 LIMIT $3;
```

Filters (`area_id ANY($4)`, tags, node_types) are appended as `AND` clauses
in Go using `pgx` named args.

### Fuzzy Title Lookup

```sql
SELECT org_id, area_id, node_id, version,
       frontmatter->>'title' AS title,
       similarity(frontmatter->>'title', $1) AS score
  FROM accepted_versions
 WHERE org_id = $2
   AND is_current = true
   AND status = 'active'
   AND frontmatter->>'title' % $1
 ORDER BY score DESC
 LIMIT $3;
```

Use this for "did you mean…" lookups. Threshold via `set_limit(0.3)`.

## Package: `internal/leader` (Extended)

On accept, the leader maintains the derived indexes inside the admission
transaction:

```go
func (l *Leader) refreshDerivedIndexes(
    ctx context.Context, tx pgx.Tx, av protocol.AcceptedVersion,
) error {
    // Remove old alias/tag rows for the prior current version of this node.
    if err := l.aliases.DeleteForNode(ctx, tx, av.Identity); err != nil {...}
    if err := l.tags.DeleteForNode(ctx, tx, av.Identity); err != nil {...}
    // Insert new ones.
    for _, a := range av.Frontmatter.Aliases {
        if err := l.aliases.Insert(ctx, tx, av.Identity, a); err != nil {...}
    }
    for _, t := range av.Frontmatter.Tags {
        if err := l.tags.Insert(ctx, tx, av.Identity, t); err != nil {...}
    }
    return nil
}
```

`body_tsv` is a generated column; no application action needed.

## Manifest and Rebuild

```go
package search

// Manifest is a derived navigation cache: the list of (org, area, node) with
// titles, types, and update times. Useful for tree rendering and sitemaps.
type Manifest struct {
    GeneratedAt time.Time
    Entries     []ManifestEntry
}

type ManifestEntry struct {
    Identity  protocol.Identity
    Title     string
    NodeType  protocol.NodeType
    Authority protocol.Authority
    Version   int
    UpdatedAt time.Time
    Retired   bool
}

func (e *Engine) BuildManifest(
    ctx context.Context, org protocol.OrgID,
) (*Manifest, error)

// RebuildIndexes drops and recreates alias_index and tag_index for an org
// from accepted_versions. Used after schema migrations or corruption recovery.
func (e *Engine) RebuildIndexes(ctx context.Context, org protocol.OrgID) error
```

CLI exposure (Phase 11):

```text
cogneetree index rebuild --org acme
cogneetree manifest dump  --org acme > acme-manifest.json
```

## Search Scope: Hot Tier Only

The partial GIN index on `is_current AND status = 'active'` makes searching
historical versions impossible by default. This is intentional. To search
history, an operator runs:

```text
cogneetree index rebuild --org acme --include-history
```

That command creates a secondary index `accepted_history_fts` covering all
versions. It is created on demand, not by default, because it can be many
gigabytes.

## Tests Required

In `internal/search`:

| Test | What It Asserts |
|---|---|
| `TestSearchRanksTitleAboveBody` | a hit in title scores higher than a hit in body alone |
| `TestSearchExcludesRetired` | default search omits retired nodes |
| `TestSearchIncludesRetiredWhenFlagSet` | IncludeRetired returns retired hits |
| `TestSearchAreaFilter` | only nodes in given areas returned |
| `TestSearchTagFilter` | only nodes with given tag returned |
| `TestByAliasReturnsExactMatch` | exact alias lookup returns the node |
| `TestFuzzyTitleSuggestsCloseMatches` | typo in query still returns the right node |
| `TestSnippetHighlights` | snippet wraps matched terms with markers |

In `internal/leader`:

| Test | What It Asserts |
|---|---|
| `TestAcceptRefreshesAliasIndex` | alias added on update appears in alias_index |
| `TestAcceptRemovesObsoleteAliases` | alias removed on update disappears from index |
| `TestAcceptRefreshesTagIndex` | tag changes reflected in tag_index |

In `internal/search` (rebuild):

| Test | What It Asserts |
|---|---|
| `TestRebuildIndexesFromAccepted` | dropping alias_index and rebuilding restores rows |
| `TestRebuildIdempotent` | running rebuild twice does not duplicate |
| `TestBuildManifestReturnsEveryActiveNode` | manifest entry count equals active node count |

## Performance Targets

Asserted via benchmarks in `internal/search/bench_test.go`:

```text
search p50 < 5ms   at 10k current nodes
search p99 < 25ms  at 10k current nodes
search p50 < 20ms  at 100k current nodes
search p99 < 80ms  at 100k current nodes
```

If targets aren't met, the index is wrong. Do not reach for Sonic /
Meilisearch / Typesense; see [PROTOCOL.md](../PROTOCOL.md) for the
non-goal statement.

## Exit Criteria

- `body_tsv` populated as a stored generated column with weighted sections
- partial GIN index limits search to current active nodes
- `pg_trgm` index enables fuzzy title lookup
- alias and tag lookup tables stay consistent with accepted state
- manifest and indexes rebuild from accepted_versions
- benchmark targets met on a populated DB
- all tests above pass

Once these are green, proceed to
[Phase 10: Long-Running Worker Loop](phase-10-worker-loop.md).

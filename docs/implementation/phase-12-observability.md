# Phase 12: Observability

Goal: production-grade visibility. Metrics, structured logs, and distributed
traces for every hot path. The operator can answer "is it slow? is it
broken? why?" without attaching a debugger.

## Deliverables

- Prometheus metrics for proposals, admissions, search, and worker health
- OpenTelemetry traces across HTTP → leader → store → Postgres
- structured slog output with consistent field names
- `/metrics` endpoint on a separate port from the API

## Stack

```text
metrics      prometheus/client_golang
tracing      go.opentelemetry.io/otel (OTLP exporter)
logs         log/slog with slog-formatter for JSON output
dashboards   sample Grafana JSON committed to /observability/dashboards
```

## Package: `internal/telemetry`

```go
package telemetry

type Provider struct {
    Meter   metric.Meter
    Tracer  trace.Tracer
    Logger  *slog.Logger
}

type Config struct {
    ServiceName     string
    ServiceVersion  string
    OTLPEndpoint    string
    PrometheusAddr  string
    LogLevel        string
    LogFormat       string // "json" or "text"
}

func Setup(ctx context.Context, cfg Config) (*Provider, func(context.Context) error, error)
```

`Setup` returns the provider and a shutdown function that flushes spans and
metrics. Call shutdown in `main` before exit.

## Metrics Catalog

All metrics use the `cogneetree_` prefix. Labels stay low-cardinality: never
include `node_id` or `proposal_id` as a label.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `proposals_submitted_total` | counter | org, area, kind | proposals received |
| `proposals_resolved_total` | counter | org, area, status | resolutions by outcome |
| `proposal_resolution_duration_seconds` | histogram | org, area, status | time from claim to resolution |
| `admission_db_duration_seconds` | histogram | op | per-statement timing inside leader tx |
| `worker_queue_depth` | gauge | org, area | pending count per area, sampled |
| `worker_areas_active` | gauge | — | number of areaWorker goroutines |
| `worker_listener_reconnects_total` | counter | — | LISTEN connection drops |
| `worker_claim_skip_locked_total` | counter | org, area | times SKIP LOCKED returned nothing |
| `search_requests_total` | counter | org, has_text | search calls |
| `search_duration_seconds` | histogram | org | search latency |
| `search_hits_returned` | histogram | org | hit count per query |
| `http_requests_total` | counter | route, method, status | HTTP request count |
| `http_request_duration_seconds` | histogram | route, method, status | HTTP latency |
| `db_pool_in_use` | gauge | role | active connections per pool |
| `db_pool_idle` | gauge | role | idle connections |
| `rls_set_config_failures_total` | counter | org | failures to set the GUC |
| `audit_jsonl_write_failures_total` | counter | org | JSONL append errors |

### Histogram Buckets

Latency buckets (seconds):

```text
[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
```

Search hit-count buckets:

```text
[0, 1, 5, 10, 25, 50, 100]
```

## Tracing

Every HTTP request opens a root span. Spans propagate through:

```text
http.handler
  -> auth.principal_resolve
  -> rls.set_config
  -> client.MemoryAPI.{Submit|ReadCurrent|...}
    -> leader.Resolve
      -> store.<operation>
        -> pgx.Exec/Query    (auto-instrumented via otelpgx)
```

Required span attributes:

| Attribute | Where Set |
|---|---|
| `cogneetree.org_id` | http middleware |
| `cogneetree.area_id` | http handler when path-bound |
| `cogneetree.node_id` | leader.Resolve |
| `cogneetree.proposal_id` | leader.Resolve |
| `cogneetree.kind` | leader.Resolve |
| `cogneetree.expected_version` | leader.Resolve |
| `cogneetree.outcome` | resolution write |

Do not put body or frontmatter into span attributes. Bodies can be large or
contain sensitive data.

## Logging

`log/slog` with JSON output. One logger per process, child loggers per
request and per claim.

Standard fields:

```text
service       cogneetree
env           prod | stage | dev
component     http | worker | leader | store
org_id        acme
area_id       rbac
node_id       rbac.acls
proposal_id   prop_01HZ...
trace_id      <propagated from OTel>
request_id    <from header or generated>
```

Required log lines (with levels):

```text
INFO  http.request.start    route method
INFO  http.request.end      route method status duration_ms
INFO  proposal.submitted    proposal_id org_id area_id node_id kind
INFO  proposal.claimed      proposal_id leader_id
INFO  proposal.resolved     proposal_id status duration_ms
WARN  proposal.rejected     proposal_id reason current_version expected_version
ERROR admission.failed      proposal_id err
INFO  worker.area.started   org_id area_id
INFO  worker.area.idle      org_id area_id idle_for_ms
WARN  worker.area.evicted   org_id area_id reason
ERROR listener.disconnect   err
INFO  listener.reconnected  attempts duration_ms
INFO  shutdown.started      signal
INFO  shutdown.complete     duration_ms
```

No `printf`-style messages. Every log entry is a structured event with a
stable name field.

## Health and Readiness

`/healthz` returns 200 if the process is alive (no DB check). `/readyz`
returns 200 only when:

- DB pool can `SELECT 1`
- the latest expected migration is applied
- the worker (if in-process) has claimed leadership at least once or
  successfully listened for 5 seconds without error

These distinctions matter for Kubernetes-style orchestration: liveness restart
on `/healthz` failure, traffic-shift on `/readyz` failure.

```go
type Health struct {
    db       *store.Store
    migrator *migrate.Migrator
    listener *worker.Listener
}

func (h *Health) Healthz(w http.ResponseWriter, r *http.Request)
func (h *Health) Readyz(w http.ResponseWriter, r *http.Request)
```

## Sampling and Cardinality Discipline

- traces: parent-based sampling at 5% by default; head-sample at 100% for
  errors and slow paths
- metrics: never use `node_id`, `proposal_id`, `alias`, or `tag` as a label
- logs: high-cardinality fields go in the *body*, not as labels

A metric with `node_id` as a label will create one time-series per node and
melt Prometheus. Resist the urge.

## Dashboards

Sample Grafana dashboards committed to `observability/dashboards/`:

```text
overview.json      RPS, p99 latency, error rate, queue depth
proposals.json     submitted, accepted, rejected, supersede rate
search.json        QPS, p99 latency, zero-hit rate
worker.json        per-area drain rate, listener health, claim skip rate
db.json            pool utilization, slow queries, lock waits
```

Dashboards are committed as source so they version-control with the schema
they query against.

## Alerting (Suggested, Not Required for Phase 12 Exit)

```text
- p99 search latency > 200ms for 10 minutes
- proposal_rejected_stale rate > 50% for 10 minutes  → contention spike
- worker_listener_reconnects_total > 5/min          → network or DB issue
- /readyz failing for 2 minutes
- pending queue depth > 1000 for any area for 5 minutes
```

These ship as a `prometheus-rules.yaml` alongside the dashboards.

## Tests Required

In `internal/telemetry`:

| Test | What It Asserts |
|---|---|
| `TestMetricsRegisteredOnSetup` | every metric in the catalog is registered |
| `TestNoNodeIDLabel` | metric label sets are validated; node_id absent everywhere |
| `TestLoggerEmitsJSON` | default config produces parseable JSON |
| `TestTraceContextPropagates` | a span started in HTTP appears as parent of leader span |

In `internal/httpadapter`:

| Test | What It Asserts |
|---|---|
| `TestHTTPRequestMetricRecorded` | a request increments http_requests_total |
| `TestHTTPRequestDurationRecorded` | histogram observes the right bucket |
| `TestReadyzFailsWhenMigrationsMissing` | readyz returns 503 |

In `internal/worker`:

| Test | What It Asserts |
|---|---|
| `TestWorkerEmitsResolvedMetric` | proposals_resolved_total increments per resolution |
| `TestListenerReconnectMetricBumps` | reconnect counter rises after simulated drop |

## Performance Budget

The observability path must not become the bottleneck.

```text
metrics overhead       < 1% of request CPU at 1k RPS
tracing overhead       < 2% of request CPU at 5% sample rate
log throughput         > 50k events/sec to stderr without backpressure
```

Benchmarks live in `internal/telemetry/bench_test.go`.

## Exit Criteria

- every metric in the catalog is exported on `/metrics`
- traces span HTTP → leader → store → pgx end to end
- structured logs use the standard field set
- `/readyz` correctly differentiates healthy from ready
- sample dashboards committed
- cardinality discipline enforced via tests
- all tests above pass

This is the last phase in the current plan. Subsequent phases (deferred)
include:

- gRPC adapter
- MCP adapter (only after HTTP is stable per [PROTOCOL.md](../PROTOCOL.md))
- semantic search (out of scope per protocol non-goals)
- multi-region replication

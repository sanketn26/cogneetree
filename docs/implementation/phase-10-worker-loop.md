# Phase 10: Long-Running Worker Loop

Goal: a daemon process that listens for proposals, processes them through
the leader, and shuts down cleanly. Uses Postgres `LISTEN/NOTIFY` so there is
no polling loop and no idle wakeup tax.

## Deliverables

- `internal/worker` package with a runnable daemon
- LISTEN/NOTIFY wired into `pending_proposals` inserts
- per-(org, area) worker fan-out
- graceful shutdown on SIGTERM and SIGINT
- structured logging of every claim and resolution

## Schema Changes

### `0014_notify_trigger.sql`

```sql
-- +goose Up
CREATE OR REPLACE FUNCTION notify_proposal_pending() RETURNS trigger AS $$
BEGIN
  PERFORM pg_notify(
    'cogneetree_proposal_pending',
    NEW.org_id || ':' || NEW.area_id
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER pending_proposals_notify
AFTER INSERT ON pending_proposals
FOR EACH ROW EXECUTE FUNCTION notify_proposal_pending();

-- +goose Down
DROP TRIGGER IF EXISTS pending_proposals_notify ON pending_proposals;
DROP FUNCTION IF EXISTS notify_proposal_pending();
```

The trigger fires on every insert. Payload is `org_id:area_id` so the worker
knows which area to wake.

## Package: `internal/worker`

```go
package worker

type Worker struct {
    id         string
    db         *store.Store
    leader     *leader.Leader
    log        *slog.Logger
    listener   *Listener
    pool       *AreaPool
    stop       chan struct{}
    stopped    chan struct{}
}

type Config struct {
    ID                string
    MaxAreaGoroutines int           // default 64
    IdleTimeout       time.Duration // default 30s
    DrainTimeout      time.Duration // default 30s on shutdown
}

func New(cfg Config, db *store.Store, ld *leader.Leader, log *slog.Logger) *Worker

// Run blocks until ctx is cancelled or Stop is called.
func (w *Worker) Run(ctx context.Context) error

// Stop signals graceful shutdown.
func (w *Worker) Stop()
```

### Architecture

```text
       Postgres
       LISTEN cogneetree_proposal_pending
              |
              v
       internal/worker/Listener  (one connection, demultiplexes)
              |
              v
       AreaPool { org:area -> areaWorker goroutine }
              |
              v
       leader.Leader.ProcessNext(org, area)
              |
              v
       store + audit
```

A single `LISTEN` connection feeds many area workers. Each area worker drains
its area's pending queue until empty, then idles until the next NOTIFY for
that area.

### Listener

```go
type Listener struct {
    conn *pgx.Conn
    out  chan<- AreaKey
    log  *slog.Logger
}

type AreaKey struct {
    Org  protocol.OrgID
    Area protocol.AreaID
}

func NewListener(ctx context.Context, dsn string, out chan<- AreaKey) (*Listener, error)

// Run blocks until ctx is cancelled. Restarts on transient errors with
// exponential backoff capped at 30 seconds.
func (l *Listener) Run(ctx context.Context) error
```

The listener uses a dedicated single-purpose connection because `LISTEN` is
not pool-friendly.

### AreaPool

```go
type AreaPool struct {
    mu        sync.Mutex
    workers   map[AreaKey]*areaWorker
    max       int
    leader    *leader.Leader
    db        *store.Store
    log       *slog.Logger
    idle      time.Duration
    wg        sync.WaitGroup
}

func (p *AreaPool) Wake(k AreaKey)         // start or notify the area worker
func (p *AreaPool) StopAll(ctx context.Context) error
```

```go
type areaWorker struct {
    key    AreaKey
    wakeup chan struct{}
    quit   chan struct{}
    leader *leader.Leader
    log    *slog.Logger
}

func (a *areaWorker) run() {
    for {
        select {
        case <-a.quit:
            return
        case <-a.wakeup:
            a.drain()
        }
    }
}

func (a *areaWorker) drain() {
    for {
        res, err := a.leader.ProcessNext(ctx, a.key.Org, a.key.Area)
        if err != nil {
            a.log.Error("process_next_failed", "err", err)
            return // back off; next NOTIFY will retry
        }
        if res == nil {
            return // queue empty
        }
        a.log.Info("resolution", "proposal", res.ProposalID, "status", res.Status)
    }
}
```

Capacity policy: if `len(workers) >= max`, evict the least-recently-used
idle worker. Eviction does not lose proposals — the next NOTIFY recreates the
worker.

### Graceful Shutdown

```text
SIGTERM | SIGINT received
  -> Worker.Stop()
  -> Listener.Run returns
  -> AreaPool.StopAll(drainTimeout)
     -> close(quit) on every areaWorker
     -> wait for in-flight ProcessNext to finish or drainTimeout to elapse
  -> Worker.Run returns nil
  -> main exits 0
```

Workers do not start new claims after Stop is signalled. In-flight claims
complete or roll back via context cancellation propagated into pgx.

### Crash Recovery

If a worker crashes mid-claim, the `pending_proposals.status='claimed'` row
is left behind. On startup, the worker runs:

```sql
UPDATE pending_proposals
   SET status = 'pending',
       claimed_by = NULL,
       claimed_at = NULL
 WHERE status = 'claimed'
   AND claimed_at < now() - interval '5 minutes';
```

This requeues abandoned claims. The 5-minute threshold can be configured.
Resolved proposals (`status='resolved'`) are never requeued.

## CLI

Extend `cmd/cogneetree/main.go`:

```go
case "worker":
    runWorker(ctx, cfg)
```

`runWorker` constructs `store`, `leader`, `worker`, attaches a signal handler,
and calls `Run`.

## Tests Required

In `internal/worker`:

| Test | What It Asserts |
|---|---|
| `TestWorkerDrainsPendingOnStart` | worker started against a populated queue empties it |
| `TestWorkerProcessesNewProposalsWithoutPolling` | inserting a proposal triggers processing without timer ticks |
| `TestWorkerHandlesManyAreasInParallel` | proposals across 10 areas processed concurrently |
| `TestGracefulShutdownCompletesInFlight` | SIGTERM during processing finishes the current resolution before exit |
| `TestShutdownTimeoutAborts` | a stuck claim is abandoned after DrainTimeout |
| `TestCrashedClaimRequeued` | claimed row older than 5 min becomes pending again on startup |
| `TestListenerReconnectOnDrop` | killing the connection reconnects within 5 seconds |
| `TestNoIdlePolling` | with no proposals, worker makes zero DB queries for 10 seconds |

The "no idle polling" test is the load-bearing one — it proves LISTEN/NOTIFY
is actually being used.

## Metrics (Hooks for Phase 12)

Even before Phase 12 adds the metrics package, leave hooks in place:

```go
type Metrics interface {
    ProposalsProcessed(area AreaKey, status protocol.ProposalStatus)
    ProcessDuration(area AreaKey, d time.Duration)
    QueueDrained(area AreaKey)
    ListenerReconnect()
}

type noopMetrics struct{}
```

`New` accepts a `Metrics` (default `noopMetrics{}`). Phase 12 swaps in a real
implementation.

## Exit Criteria

- `cogneetree worker` runs as a daemon
- NOTIFY-driven; no timer-based polling
- multiple areas processed in parallel up to `MaxAreaGoroutines`
- SIGTERM and SIGINT trigger graceful drain within `DrainTimeout`
- abandoned claims requeued on startup
- listener reconnects on transient failures
- all tests above pass

Once these are green, proceed to
[Phase 11: HTTP Adapter](phase-11-http-adapter.md).

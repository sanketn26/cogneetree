# Phase 0: Foundations

Goal: bootstrap the repository, toolchain, and local development loop so every
later phase has a place to land code and a way to run tests against a real
Postgres.

## Deliverables

- working Go module
- Postgres running locally (Docker)
- migration tool selected and wired
- `go test ./...` runs (even if no tests exist yet)
- CI workflow file that runs lint + test
- a working `cogneetree` binary that prints version and exits

## Tooling Decisions

```text
Go toolchain        1.23 or newer
Postgres            16 (via docker compose for local dev)
pgx                 v5
migrations          pressly/goose (single binary, embeds SQL)
config              caarlos0/env/v11 + stdlib flag
logging             log/slog (stdlib)
linter              golangci-lint
test container      testcontainers-go/postgres for integration tests
```

## Module Initialization

```bash
go mod init github.com/sanketn26/cogneetree
go get github.com/jackc/pgx/v5
go get github.com/pressly/goose/v3
go get github.com/caarlos0/env/v11
go get github.com/testcontainers/testcontainers-go/modules/postgres
```

## Repository Layout

Create empty directories with `.gitkeep` files so the structure is committed
before any code lands.

```text
cmd/cogneetree/main.go
internal/
  config/
  protocol/
  store/
  leader/
  audit/
  snapshot/
  edges/
  search/
  worker/
  httpadapter/
pkg/cogneetreeclient/
migrations/
docs/implementation/
.github/workflows/
docker/
```

## Files To Create

### `cmd/cogneetree/main.go`

Minimal entry point. Prints version, exits cleanly. Phases 10 and 11 expand
this into a daemon.

```go
package main

import (
    "fmt"
    "os"
)

const version = "0.0.0-dev"

func main() {
    if len(os.Args) > 1 && os.Args[1] == "version" {
        fmt.Println(version)
        return
    }
    fmt.Fprintln(os.Stderr, "cogneetree: no command. try 'cogneetree version'.")
    os.Exit(2)
}
```

### `internal/config/config.go`

```go
package config

type Config struct {
    DatabaseURL string `env:"COGNEETREE_DATABASE_URL,required"`
    HTTPAddr    string `env:"COGNEETREE_HTTP_ADDR" envDefault:":8080"`
    LogLevel    string `env:"COGNEETREE_LOG_LEVEL" envDefault:"info"`
}

func Load() (Config, error) { /* env.Parse */ }
```

Function list:

- `Load() (Config, error)` — reads from env, returns typed config or error.

### `docker/docker-compose.yml`

Local Postgres 16 with sane defaults and a named volume.

```yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: cogneetree
      POSTGRES_PASSWORD: cogneetree
      POSTGRES_DB: cogneetree
    ports:
      - "5432:5432"
    volumes:
      - cogneetree_pg:/var/lib/postgresql/data

volumes:
  cogneetree_pg:
```

### `Makefile`

Thin convenience wrappers. Do not embed business logic.

```make
.PHONY: up down test lint migrate build

up:
	docker compose -f docker/docker-compose.yml up -d

down:
	docker compose -f docker/docker-compose.yml down

test:
	go test ./...

lint:
	golangci-lint run

build:
	go build -o bin/cogneetree ./cmd/cogneetree

migrate:
	goose -dir migrations postgres "$$COGNEETREE_DATABASE_URL" up
```

### `.golangci.yml`

```yaml
run:
  timeout: 5m
linters:
  enable:
    - errcheck
    - govet
    - ineffassign
    - staticcheck
    - unused
    - gofmt
    - goimports
    - misspell
    - revive
```

### `.github/workflows/ci.yml`

Lint + test on push and pull request. Use the official `setup-go` action and
`golangci-lint-action`. Postgres is launched via service container for tests
that need it (none yet in Phase 0).

## Verification Steps

1. `make up` — Postgres starts and accepts a connection on 5432.
2. `make build` — `bin/cogneetree` produced.
3. `bin/cogneetree version` — prints `0.0.0-dev`.
4. `make test` — exits 0 with `no test files`.
5. `make lint` — exits 0.

## Exit Criteria

- repository builds with `go build ./...`
- `go test ./...` exits 0
- `golangci-lint run` exits 0
- Postgres reachable locally via `make up`
- CI passes on a fresh clone
- module path is `github.com/sanketn26/cogneetree`
- no application logic exists yet beyond `version`

Once these are green, proceed to
[Phase 1: Protocol Core](phase-01-protocol-core.md).

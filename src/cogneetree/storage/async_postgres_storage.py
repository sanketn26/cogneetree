"""Async PostgreSQL storage backend using asyncpg."""

from datetime import datetime
from typing import List, Optional, Any, Dict

from cogneetree.core.interfaces import AsyncContextStorageABC
from cogneetree.core.models import (
    Session,
    Activity,
    Task,
    ContextItem,
    ContextCategory,
    ImportanceTier,
)

try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None  # type: ignore[assignment]
    ASYNCPG_AVAILABLE = False


class AsyncPostgresStorage(AsyncContextStorageABC):
    """
    PostgreSQL storage backend using an asyncpg connection pool.

    Do not instantiate directly — use the async factory::

        storage = await AsyncPostgresStorage.create("postgresql://user:pw@host/db")
        # ... use storage ...
        await storage.close()

    Connection pool size defaults to asyncpg's built-in defaults (min 10, max 10).
    Override by passing ``min_size`` / ``max_size`` kwargs to :meth:`create`.
    """

    def __init__(self, pool: "asyncpg.Pool") -> None:
        if not ASYNCPG_AVAILABLE:
            raise ImportError("asyncpg is required: pip install asyncpg")
        self._pool = pool
        self.current_session_id: Optional[str] = None
        self.current_activity_id: Optional[str] = None
        self.current_task_id: Optional[str] = None

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    async def create(cls, dsn: str, **pool_kwargs: Any) -> "AsyncPostgresStorage":
        """Create a pool, initialise the schema, and return a ready instance."""
        if not ASYNCPG_AVAILABLE:
            raise ImportError("asyncpg is required: pip install asyncpg")
        pool = await asyncpg.create_pool(dsn, **pool_kwargs)
        instance = cls(pool)
        await instance._init_db()
        return instance

    async def _init_db(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id      TEXT PRIMARY KEY,
                    original_ask    TEXT,
                    high_level_plan TEXT,
                    created_at      TIMESTAMP
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS activities (
                    activity_id      TEXT PRIMARY KEY,
                    session_id       TEXT,
                    description      TEXT,
                    tags             JSONB,
                    mode             TEXT,
                    component        TEXT,
                    planner_analysis TEXT
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id     TEXT PRIMARY KEY,
                    activity_id TEXT,
                    description TEXT,
                    tags        JSONB,
                    result      TEXT
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS context_items (
                    id        SERIAL PRIMARY KEY,
                    content   TEXT,
                    category  TEXT,
                    tags      JSONB,
                    parent_id TEXT,
                    timestamp TIMESTAMP,
                    metadata  JSONB
                )
            """)

    async def close(self) -> None:
        """Close the connection pool."""
        await self._pool.close()

    # ── Context creation ──────────────────────────────────────────────────────

    async def create_session(
        self, session_id: str, original_ask: str, high_level_plan: str
    ) -> Session:
        now = datetime.now()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO sessions VALUES ($1, $2, $3, $4) "
                "ON CONFLICT (session_id) DO UPDATE SET original_ask = EXCLUDED.original_ask",
                session_id,
                original_ask,
                high_level_plan,
                now,
            )
        self.current_session_id = session_id
        return Session(session_id, original_ask, high_level_plan, created_at=now)

    async def create_activity(
        self,
        activity_id: str,
        session_id: str,
        description: str,
        tags: List[str],
        mode: str,
        component: str,
        planner_analysis: str,
    ) -> Activity:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO activities VALUES ($1,$2,$3,$4,$5,$6,$7) "
                "ON CONFLICT (activity_id) DO NOTHING",
                activity_id,
                session_id,
                description,
                tags,
                mode,
                component,
                planner_analysis,
            )
        self.current_activity_id = activity_id
        return Activity(activity_id, session_id, description, tags, mode, component, planner_analysis)

    async def create_task(
        self, task_id: str, activity_id: str, description: str, tags: List[str]
    ) -> Task:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO tasks VALUES ($1,$2,$3,$4,$5) ON CONFLICT (task_id) DO NOTHING",
                task_id,
                activity_id,
                description,
                tags,
                None,
            )
        self.current_task_id = task_id
        return Task(task_id, activity_id, description, tags)

    async def complete_task(self, task_id: str, result: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE tasks SET result = $1 WHERE task_id = $2", result, task_id
            )

    async def add_item(
        self,
        content: str,
        category: ContextCategory,
        tags: List[str],
        parent_id: Optional[str] = None,
        embedding: Optional[Any] = None,
        tier: Optional[ImportanceTier] = None,
    ) -> ContextItem:
        normalized_tier = tier or ImportanceTier.MINOR
        metadata: Dict[str, Any] = {"tier": normalized_tier.value}
        now = datetime.now()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO context_items (content, category, tags, parent_id, timestamp, metadata) "
                "VALUES ($1,$2,$3,$4,$5,$6)",
                content,
                category.value,
                tags,
                parent_id,
                now,
                metadata,
            )
        return ContextItem(
            content=content,
            category=category,
            tags=tags,
            parent_id=parent_id,
            timestamp=now,
            metadata=metadata,
            tier=normalized_tier,
        )

    # ── Entity lookups ────────────────────────────────────────────────────────

    async def get_session(self, session_id: str) -> Optional[Session]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM sessions WHERE session_id = $1", session_id
            )
        if row:
            return Session(
                row["session_id"], row["original_ask"], row["high_level_plan"], row["created_at"]
            )
        return None

    async def get_activity(self, activity_id: str) -> Optional[Activity]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM activities WHERE activity_id = $1", activity_id
            )
        if row:
            return Activity(
                row["activity_id"],
                row["session_id"],
                row["description"],
                list(row["tags"]),
                row["mode"],
                row["component"],
                row["planner_analysis"],
            )
        return None

    async def get_task(self, task_id: str) -> Optional[Task]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM tasks WHERE task_id = $1", task_id)
        if row:
            return Task(
                row["task_id"], row["activity_id"], row["description"], list(row["tags"]), row["result"]
            )
        return None

    async def get_current_session(self) -> Optional[Session]:
        return await self.get_session(self.current_session_id) if self.current_session_id else None

    async def get_current_activity(self) -> Optional[Activity]:
        return await self.get_activity(self.current_activity_id) if self.current_activity_id else None

    async def get_current_task(self) -> Optional[Task]:
        return await self.get_task(self.current_task_id) if self.current_task_id else None

    # ── Item queries ──────────────────────────────────────────────────────────

    def _record_to_item(self, row: "asyncpg.Record") -> ContextItem:
        metadata = dict(row["metadata"]) if row["metadata"] else {}
        try:
            tier = ImportanceTier(metadata.get("tier", ImportanceTier.MINOR.value))
        except ValueError:
            tier = ImportanceTier.MINOR
        return ContextItem(
            content=row["content"],
            category=ContextCategory(row["category"]),
            tags=list(row["tags"]) if row["tags"] else [],
            parent_id=row["parent_id"],
            timestamp=row["timestamp"],
            metadata=metadata,
            tier=tier,
        )

    async def get_items_by_tags(self, tags: List[str]) -> List[ContextItem]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM context_items
                WHERE EXISTS (
                    SELECT 1
                    FROM   jsonb_array_elements_text(tags) elem
                    WHERE  elem = ANY($1::text[])
                )
                """,
                tags,
            )
        return [self._record_to_item(r) for r in rows]

    async def get_items_by_category(self, category: ContextCategory) -> List[ContextItem]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM context_items WHERE category = $1", category.value
            )
        return [self._record_to_item(r) for r in rows]

    async def get_items_by_task(self, task_id: str) -> List[ContextItem]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM context_items WHERE parent_id = $1", task_id
            )
        return [self._record_to_item(r) for r in rows]

    async def get_items_by_activity(self, activity_id: str) -> List[ContextItem]:
        async with self._pool.acquire() as conn:
            task_ids = [
                r["task_id"]
                for r in await conn.fetch(
                    "SELECT task_id FROM tasks WHERE activity_id = $1", activity_id
                )
            ]
            if not task_ids:
                return []
            rows = await conn.fetch(
                "SELECT * FROM context_items WHERE parent_id = ANY($1::text[])", task_ids
            )
        return [self._record_to_item(r) for r in rows]

    async def get_items_by_session(self, session_id: str) -> List[ContextItem]:
        async with self._pool.acquire() as conn:
            activity_ids = [
                r["activity_id"]
                for r in await conn.fetch(
                    "SELECT activity_id FROM activities WHERE session_id = $1", session_id
                )
            ]
            if not activity_ids:
                return []
            task_ids = [
                r["task_id"]
                for r in await conn.fetch(
                    "SELECT task_id FROM tasks WHERE activity_id = ANY($1::text[])", activity_ids
                )
            ]
            if not task_ids:
                return []
            rows = await conn.fetch(
                "SELECT * FROM context_items WHERE parent_id = ANY($1::text[])", task_ids
            )
        return [self._record_to_item(r) for r in rows]

    async def get_all_sessions(self) -> List[Session]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM sessions ORDER BY created_at")
        return [
            Session(r["session_id"], r["original_ask"], r["high_level_plan"], r["created_at"])
            for r in rows
        ]

    async def get_activity_tasks(self, activity_id: str) -> List[Task]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM tasks WHERE activity_id = $1", activity_id
            )
        return [
            Task(r["task_id"], r["activity_id"], r["description"], list(r["tags"]), r["result"])
            for r in rows
        ]

    async def get_stats(self) -> Dict[str, int]:
        async with self._pool.acquire() as conn:
            sessions = await conn.fetchval("SELECT COUNT(*) FROM sessions")
            activities = await conn.fetchval("SELECT COUNT(*) FROM activities")
            tasks = await conn.fetchval("SELECT COUNT(*) FROM tasks")
            items = await conn.fetchval("SELECT COUNT(*) FROM context_items")
        return {
            "sessions": int(sessions),
            "activities": int(activities),
            "tasks": int(tasks),
            "items": int(items),
        }

    async def clear(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("TRUNCATE sessions, activities, tasks, context_items")
        self.current_session_id = None
        self.current_activity_id = None
        self.current_task_id = None

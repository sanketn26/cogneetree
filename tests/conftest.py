"""Shared pytest fixtures for storage contract tests."""

import os

import pytest
import pytest_asyncio

from cogneetree.storage.in_memory_storage import InMemoryStorage
from cogneetree.storage.sql_storage import SQLiteStorage

try:
    from cogneetree.storage.async_postgres_storage import AsyncPostgresStorage

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False


class _SyncToAsyncAdapter:
    """
    Wraps a sync ContextStorageABC as an async interface.

    Used in contract tests so the same async test body runs against
    InMemoryStorage and SQLiteStorage without duplication.
    """

    def __init__(self, storage):
        self._s = storage

    # Context stack exposed as properties (sync storages use plain attrs)
    @property
    def current_session_id(self):
        return self._s.current_session_id

    @property
    def current_activity_id(self):
        return self._s.current_activity_id

    @property
    def current_task_id(self):
        return self._s.current_task_id

    # All delegating coroutines — sync calls are fine inside async functions
    async def create_session(self, *a, **kw):
        return self._s.create_session(*a, **kw)

    async def create_activity(self, *a, **kw):
        return self._s.create_activity(*a, **kw)

    async def create_task(self, *a, **kw):
        return self._s.create_task(*a, **kw)

    async def complete_task(self, *a, **kw):
        return self._s.complete_task(*a, **kw)

    async def add_item(self, *a, **kw):
        return self._s.add_item(*a, **kw)

    async def get_session(self, *a, **kw):
        return self._s.get_session(*a, **kw)

    async def get_activity(self, *a, **kw):
        return self._s.get_activity(*a, **kw)

    async def get_task(self, *a, **kw):
        return self._s.get_task(*a, **kw)

    async def get_current_session(self):
        return self._s.get_current_session()

    async def get_current_activity(self):
        return self._s.get_current_activity()

    async def get_current_task(self):
        return self._s.get_current_task()

    async def get_items_by_tags(self, *a, **kw):
        return self._s.get_items_by_tags(*a, **kw)

    async def get_items_by_category(self, *a, **kw):
        return self._s.get_items_by_category(*a, **kw)

    async def get_items_by_task(self, *a, **kw):
        return self._s.get_items_by_task(*a, **kw)

    async def get_items_by_activity(self, *a, **kw):
        return self._s.get_items_by_activity(*a, **kw)

    async def get_items_by_session(self, *a, **kw):
        return self._s.get_items_by_session(*a, **kw)

    async def get_all_sessions(self):
        return self._s.get_all_sessions()

    async def get_activity_tasks(self, *a, **kw):
        return self._s.get_activity_tasks(*a, **kw)

    async def get_stats(self):
        return self._s.get_stats()

    async def clear(self):
        return self._s.clear()

    async def close(self):
        pass  # no-op — sync backends need no teardown


@pytest_asyncio.fixture(params=["memory", "sqlite", "postgres"])
async def any_storage(request, tmp_path):
    """
    Async storage fixture parametrised across all three backends.

    * memory  — InMemoryStorage wrapped in _SyncToAsyncAdapter
    * sqlite  — SQLiteStorage (tmp file) wrapped in _SyncToAsyncAdapter
    * postgres — AsyncPostgresStorage; skipped unless POSTGRES_TEST_DSN is set

    Set POSTGRES_TEST_DSN to a valid libpq DSN to run the postgres variant, e.g.::

        export POSTGRES_TEST_DSN="postgresql://user:pw@localhost/cogneetree_test"
    """
    if request.param == "memory":
        yield _SyncToAsyncAdapter(InMemoryStorage())

    elif request.param == "sqlite":
        s = SQLiteStorage(str(tmp_path / "test.db"))
        yield _SyncToAsyncAdapter(s)

    elif request.param == "postgres":
        if not ASYNCPG_AVAILABLE:
            pytest.skip("asyncpg not installed")
        dsn = os.environ.get("POSTGRES_TEST_DSN")
        if not dsn:
            pytest.skip("POSTGRES_TEST_DSN not set")
        try:
            s = await AsyncPostgresStorage.create(dsn)
        except Exception as exc:
            pytest.skip(f"PostgreSQL unavailable: {exc}")
        await s.clear()
        yield s
        await s.clear()
        await s.close()

"""
Storage contract tests.

Every test in this module runs against all three backends:
  - InMemoryStorage   (via _SyncToAsyncAdapter)
  - SQLiteStorage     (via _SyncToAsyncAdapter)
  - AsyncPostgresStorage  (skipped unless POSTGRES_TEST_DSN is set)

The ``any_storage`` fixture is defined in conftest.py.
"""

import asyncio

import pytest

from cogneetree.core.models import ContextCategory, ImportanceTier


@pytest.mark.asyncio
class TestStorageContract:
    """ContextStorageABC contract — every compliant backend must pass all tests."""

    # ── Context creation & current-id tracking ────────────────────────────────

    async def test_create_session_sets_current(self, any_storage):
        s = await any_storage.create_session("s1", "Build auth", "JWT plan")
        assert s.session_id == "s1"
        assert s.original_ask == "Build auth"
        assert any_storage.current_session_id == "s1"

    async def test_create_activity_sets_current(self, any_storage):
        await any_storage.create_session("s1", "Ask", "Plan")
        a = await any_storage.create_activity(
            "a1", "s1", "Token handling", ["jwt"], "coder", "auth", "analysis"
        )
        assert a.activity_id == "a1"
        assert any_storage.current_activity_id == "a1"

    async def test_create_task_sets_current(self, any_storage):
        await any_storage.create_session("s1", "Ask", "Plan")
        await any_storage.create_activity("a1", "s1", "Desc", ["tag"], "coder", "comp", "analysis")
        t = await any_storage.create_task("t1", "a1", "Sign tokens", ["jwt"])
        assert t.task_id == "t1"
        assert any_storage.current_task_id == "t1"

    # ── get_current_* ─────────────────────────────────────────────────────────

    async def test_get_current_session(self, any_storage):
        await any_storage.create_session("s1", "Ask", "Plan")
        session = await any_storage.get_current_session()
        assert session is not None
        assert session.session_id == "s1"

    async def test_get_current_activity(self, any_storage):
        await any_storage.create_session("s1", "Ask", "Plan")
        await any_storage.create_activity("a1", "s1", "Desc", ["tag"], "coder", "comp", "analysis")
        activity = await any_storage.get_current_activity()
        assert activity is not None
        assert activity.activity_id == "a1"

    async def test_get_current_task(self, any_storage):
        await any_storage.create_session("s1", "Ask", "Plan")
        await any_storage.create_activity("a1", "s1", "Desc", ["tag"], "coder", "comp", "analysis")
        await any_storage.create_task("t1", "a1", "Task", ["tag"])
        task = await any_storage.get_current_task()
        assert task is not None
        assert task.task_id == "t1"

    # ── complete_task ─────────────────────────────────────────────────────────

    async def test_complete_task_updates_result(self, any_storage):
        await any_storage.create_session("s1", "Ask", "Plan")
        await any_storage.create_activity("a1", "s1", "Desc", ["tag"], "coder", "comp", "analysis")
        await any_storage.create_task("t1", "a1", "Task", ["tag"])
        await any_storage.complete_task("t1", "Done")
        task = await any_storage.get_task("t1")
        assert task.result == "Done"

    # ── add_item / tier persistence ───────────────────────────────────────────

    async def test_add_item_default_tier(self, any_storage):
        item = await any_storage.add_item("Content", ContextCategory.ACTION, ["tag"])
        assert item.tier == ImportanceTier.MINOR

    async def test_add_item_explicit_tier_persists(self, any_storage):
        await any_storage.create_session("s1", "Ask", "Plan")
        await any_storage.create_activity("a1", "s1", "Desc", ["tag"], "coder", "comp", "analysis")
        await any_storage.create_task("t1", "a1", "Task", ["tag"])
        item = await any_storage.add_item(
            "Critical note",
            ContextCategory.DECISION,
            ["arch"],
            parent_id="t1",
            tier=ImportanceTier.CRITICAL,
        )
        assert item.tier == ImportanceTier.CRITICAL
        # Round-trip: retrieve and check tier survives serialisation
        items = await any_storage.get_items_by_task("t1")
        assert items[0].tier == ImportanceTier.CRITICAL

    # ── get_items_by_tags ─────────────────────────────────────────────────────

    async def test_get_items_by_tags_any_match(self, any_storage):
        await any_storage.add_item("Item 1", ContextCategory.ACTION, ["api", "auth"])
        await any_storage.add_item("Item 2", ContextCategory.ACTION, ["api"])
        await any_storage.add_item("Item 3", ContextCategory.DECISION, ["auth"])

        api_items = await any_storage.get_items_by_tags(["api"])
        assert len(api_items) == 2

        auth_items = await any_storage.get_items_by_tags(["auth"])
        assert len(auth_items) == 2

    async def test_get_items_by_tags_no_match_returns_empty(self, any_storage):
        await any_storage.add_item("Item", ContextCategory.ACTION, ["jwt"])
        result = await any_storage.get_items_by_tags(["nonexistent"])
        assert result == []

    # ── get_items_by_category ─────────────────────────────────────────────────

    async def test_get_items_by_category(self, any_storage):
        await any_storage.add_item("Action 1", ContextCategory.ACTION, ["tag"])
        await any_storage.add_item("Action 2", ContextCategory.ACTION, ["tag"])
        await any_storage.add_item("Decision", ContextCategory.DECISION, ["tag"])

        actions = await any_storage.get_items_by_category(ContextCategory.ACTION)
        assert len(actions) == 2
        assert all(i.category == ContextCategory.ACTION for i in actions)

    # ── get_items_by_task ─────────────────────────────────────────────────────

    async def test_get_items_by_task_filters_by_parent(self, any_storage):
        await any_storage.create_session("s1", "Ask", "Plan")
        await any_storage.create_activity("a1", "s1", "Desc", ["tag"], "coder", "comp", "analysis")
        await any_storage.create_task("t1", "a1", "Task 1", ["tag"])
        await any_storage.create_task("t2", "a1", "Task 2", ["tag"])

        await any_storage.add_item("For t1", ContextCategory.ACTION, ["tag"], parent_id="t1")
        await any_storage.add_item("Also t1", ContextCategory.ACTION, ["tag"], parent_id="t1")
        await any_storage.add_item("For t2", ContextCategory.ACTION, ["tag"], parent_id="t2")

        t1_items = await any_storage.get_items_by_task("t1")
        assert len(t1_items) == 2
        assert all(i.parent_id == "t1" for i in t1_items)

    # ── get_items_by_activity ─────────────────────────────────────────────────

    async def test_get_items_by_activity_spans_tasks(self, any_storage):
        await any_storage.create_session("s1", "Ask", "Plan")
        await any_storage.create_activity("a1", "s1", "Desc", ["tag"], "coder", "comp", "analysis")
        await any_storage.create_task("t1", "a1", "Task 1", ["tag"])
        await any_storage.create_task("t2", "a1", "Task 2", ["tag"])

        await any_storage.add_item("For t1", ContextCategory.ACTION, ["tag"], parent_id="t1")
        await any_storage.add_item("For t2", ContextCategory.ACTION, ["tag"], parent_id="t2")

        items = await any_storage.get_items_by_activity("a1")
        assert len(items) == 2

    async def test_get_items_by_activity_empty_when_no_tasks(self, any_storage):
        await any_storage.create_session("s1", "Ask", "Plan")
        await any_storage.create_activity("a1", "s1", "Desc", ["tag"], "coder", "comp", "analysis")
        items = await any_storage.get_items_by_activity("a1")
        assert items == []

    # ── get_items_by_session ──────────────────────────────────────────────────

    async def test_get_items_by_session_spans_activities(self, any_storage):
        await any_storage.create_session("s1", "Ask", "Plan")
        await any_storage.create_activity("a1", "s1", "Act 1", ["tag"], "coder", "comp", "analysis")
        await any_storage.create_activity("a2", "s1", "Act 2", ["tag"], "coder", "comp", "analysis")
        await any_storage.create_task("t1", "a1", "Task 1", ["tag"])
        await any_storage.create_task("t2", "a2", "Task 2", ["tag"])

        await any_storage.add_item("Item 1", ContextCategory.ACTION, ["tag"], parent_id="t1")
        await any_storage.add_item("Item 2", ContextCategory.ACTION, ["tag"], parent_id="t2")

        items = await any_storage.get_items_by_session("s1")
        assert len(items) == 2

    # ── get_all_sessions ──────────────────────────────────────────────────────

    async def test_get_all_sessions_chronological_order(self, any_storage):
        await any_storage.create_session("s1", "First", "Plan 1")
        await asyncio.sleep(0.01)
        await any_storage.create_session("s2", "Second", "Plan 2")

        sessions = await any_storage.get_all_sessions()
        assert len(sessions) == 2
        assert sessions[0].session_id == "s1"
        assert sessions[1].session_id == "s2"

    # ── get_activity_tasks ────────────────────────────────────────────────────

    async def test_get_activity_tasks(self, any_storage):
        await any_storage.create_session("s1", "Ask", "Plan")
        await any_storage.create_activity("a1", "s1", "Desc", ["tag"], "coder", "comp", "analysis")
        await any_storage.create_task("t1", "a1", "Task 1", ["jwt"])
        await any_storage.create_task("t2", "a1", "Task 2", ["auth"])

        tasks = await any_storage.get_activity_tasks("a1")
        assert len(tasks) == 2
        assert {t.task_id for t in tasks} == {"t1", "t2"}

    # ── get_stats ─────────────────────────────────────────────────────────────

    async def test_get_stats_counts_entities(self, any_storage):
        await any_storage.create_session("s1", "Ask", "Plan")
        await any_storage.create_activity("a1", "s1", "Desc", ["tag"], "coder", "comp", "analysis")
        await any_storage.create_task("t1", "a1", "Task", ["tag"])
        await any_storage.add_item("Item", ContextCategory.ACTION, ["tag"])

        stats = await any_storage.get_stats()
        assert stats["sessions"] == 1
        assert stats["activities"] == 1
        assert stats["tasks"] == 1
        assert stats["items"] == 1

    # ── clear ─────────────────────────────────────────────────────────────────

    async def test_clear_removes_all_data_and_resets_ids(self, any_storage):
        await any_storage.create_session("s1", "Ask", "Plan")
        await any_storage.add_item("Item", ContextCategory.ACTION, ["tag"])

        await any_storage.clear()

        stats = await any_storage.get_stats()
        assert all(v == 0 for v in stats.values())
        assert any_storage.current_session_id is None
        assert any_storage.current_activity_id is None
        assert any_storage.current_task_id is None

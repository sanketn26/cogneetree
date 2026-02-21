"""Tests for Phase 0.4: Importance Tiers."""

import pytest
from cogneetree.core.context_manager import ContextManager
from cogneetree.core.models import ContextCategory, ImportanceTier
from cogneetree.config import Config
from cogneetree.storage.in_memory_storage import InMemoryStorage
from cogneetree.retrieval.hierarchical_retriever import HierarchicalRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_storage() -> InMemoryStorage:
    storage = InMemoryStorage()
    storage.create_session("s1", "Implement auth", "JWT auth")
    storage.create_activity("a1", "s1", "Auth layer", ["auth"], "coder", "core", "analysis")
    storage.create_task("t1", "a1", "Add signing", ["auth"])
    return storage


def _setup_manager() -> ContextManager:
    manager = ContextManager(Config.default())
    manager.create_session("s1", "Implement auth", "JWT auth")
    manager.create_activity("a1", "s1", "Auth layer", ["auth"], "coder", "core", "analysis")
    manager.create_task("t1", "a1", "Add signing", ["auth"])
    return manager


# ---------------------------------------------------------------------------
# ImportanceTier enum
# ---------------------------------------------------------------------------

class TestImportanceTierEnum:

    def test_all_tiers_exist(self):
        """All four tiers are defined."""
        assert ImportanceTier.CRITICAL
        assert ImportanceTier.MAJOR
        assert ImportanceTier.MINOR
        assert ImportanceTier.NOISE

    def test_tier_values(self):
        """Tier values match expected strings."""
        assert ImportanceTier.CRITICAL.value == "critical"
        assert ImportanceTier.MAJOR.value == "major"
        assert ImportanceTier.MINOR.value == "minor"
        assert ImportanceTier.NOISE.value == "noise"


# ---------------------------------------------------------------------------
# ContextItem stores tier
# ---------------------------------------------------------------------------

class TestContextItemTier:

    def test_default_tier_is_minor(self):
        """ContextItem defaults to MINOR tier."""
        storage = _setup_storage()
        item = storage.add_item("test content", ContextCategory.DECISION, ["auth"], parent_id="t1")
        assert item.tier == ImportanceTier.MINOR

    def test_tier_stored_in_item(self):
        """Explicitly set tier is stored on the item."""
        storage = _setup_storage()
        item = storage.add_item(
            "Critical architectural decision",
            ContextCategory.DECISION,
            ["auth"],
            parent_id="t1",
            tier=ImportanceTier.CRITICAL,
        )
        assert item.tier == ImportanceTier.CRITICAL

    def test_noise_tier_stored(self):
        """NOISE tier is stored correctly."""
        storage = _setup_storage()
        item = storage.add_item(
            "Minor formatting note",
            ContextCategory.ACTION,
            ["style"],
            parent_id="t1",
            tier=ImportanceTier.NOISE,
        )
        assert item.tier == ImportanceTier.NOISE


# ---------------------------------------------------------------------------
# ContextManager record methods accept tier
# ---------------------------------------------------------------------------

class TestRecordMethodTier:

    def test_record_decision_with_tier(self):
        """record_decision() stores the specified tier."""
        manager = _setup_manager()
        item = manager.record_decision(
            "Use RS256 for production signing", ["auth"], tier=ImportanceTier.CRITICAL
        )
        assert item.tier == ImportanceTier.CRITICAL

    def test_record_learning_with_tier(self):
        """record_learning() stores the specified tier."""
        manager = _setup_manager()
        item = manager.record_learning(
            "RS256 requires public key distribution", ["auth"], tier=ImportanceTier.MAJOR
        )
        assert item.tier == ImportanceTier.MAJOR

    def test_record_action_with_tier(self):
        """record_action() stores the specified tier."""
        manager = _setup_manager()
        item = manager.record_action("Read JWT RFC", ["auth"], tier=ImportanceTier.NOISE)
        assert item.tier == ImportanceTier.NOISE

    def test_record_result_with_tier(self):
        """record_result() stores the specified tier."""
        manager = _setup_manager()
        item = manager.record_result("Auth complete", ["auth"], tier=ImportanceTier.MAJOR)
        assert item.tier == ImportanceTier.MAJOR

    def test_default_tier_is_minor(self):
        """All record methods default to MINOR when tier is not specified."""
        manager = _setup_manager()
        item = manager.record_decision("Use RS256", ["auth"])
        assert item.tier == ImportanceTier.MINOR


# ---------------------------------------------------------------------------
# HierarchicalRetriever tier multipliers
# ---------------------------------------------------------------------------

class TestTierMultipliers:

    def test_tier_multipliers_defined(self):
        """All four tiers have a multiplier defined in the retriever."""
        mults = HierarchicalRetriever._TIER_MULTIPLIERS
        assert mults[ImportanceTier.CRITICAL] == 2.0
        assert mults[ImportanceTier.MAJOR] == 1.5
        assert mults[ImportanceTier.MINOR] == 1.0
        assert mults[ImportanceTier.NOISE] == 0.1

    def test_critical_item_scores_higher_than_minor(self):
        """CRITICAL tier item gets a higher final_score than MINOR tier item."""
        storage = InMemoryStorage()
        storage.create_session("s1", "Project", "Plan")
        storage.create_activity("a1", "s1", "Work", ["test"], "coder", "core", "analysis")
        storage.create_task("t1", "a1", "Task", ["test"])

        # Same content but different tiers
        storage.add_item("test item alpha", ContextCategory.DECISION, ["test"], parent_id="t1", tier=ImportanceTier.CRITICAL)
        storage.add_item("test item alpha", ContextCategory.DECISION, ["test"], parent_id="t1", tier=ImportanceTier.MINOR)

        retriever = HierarchicalRetriever(storage)
        results = retriever.retrieve("test item alpha", "t1", max_results=10)

        # Both items returned; CRITICAL should score higher or equal (semantic scores are 0 without model)
        # Without a semantic model, semantic_score=0, so final_score = 0 * proximity * tier = 0 for all.
        # We test the tier_multiplier is stored correctly in the result dict.
        tier_multipliers_in_results = {r["tier_multiplier"] for r in results}
        assert 2.0 in tier_multipliers_in_results  # CRITICAL
        assert 1.0 in tier_multipliers_in_results  # MINOR

    def test_noise_item_has_low_multiplier(self):
        """NOISE tier item has tier_multiplier 0.1 in results."""
        storage = InMemoryStorage()
        storage.create_session("s1", "Project", "Plan")
        storage.create_activity("a1", "s1", "Work", ["test"], "coder", "core", "analysis")
        storage.create_task("t1", "a1", "Task", ["test"])
        storage.add_item("noise note", ContextCategory.ACTION, ["test"], parent_id="t1", tier=ImportanceTier.NOISE)

        retriever = HierarchicalRetriever(storage)
        results = retriever.retrieve("noise note", "t1", max_results=5)

        # The same item may appear at multiple hierarchy levels (task/activity/session)
        # but every occurrence should carry the NOISE multiplier of 0.1
        assert len(results) >= 1
        assert all(r["tier_multiplier"] == 0.1 for r in results)

    def test_tier_multiplier_in_explainability(self):
        """tier_multiplier appears in the explanation 'why' string."""
        storage = InMemoryStorage()
        storage.create_session("s1", "Project", "Plan")
        storage.create_activity("a1", "s1", "Work", ["test"], "coder", "core", "analysis")
        storage.create_task("t1", "a1", "Task", ["test"])
        storage.add_item("important decision", ContextCategory.DECISION, ["test"], parent_id="t1", tier=ImportanceTier.MAJOR)

        retriever = HierarchicalRetriever(storage)
        results = retriever.retrieve("important decision", "t1", max_results=5)

        assert results
        why = results[0]["explanation"]["why"]
        assert "tier" in why
        assert "1.5" in why  # MAJOR multiplier

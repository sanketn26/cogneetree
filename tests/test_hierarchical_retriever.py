"""Tests for hierarchical retriever with user control over history."""

import pytest
from datetime import datetime, timedelta

from cogneetree.storage.in_memory_storage import InMemoryStorage
from cogneetree.core.models import ContextCategory
from cogneetree.retrieval.hierarchical_retriever import (
    HierarchicalRetriever,
    RetrievalConfig,
    HistoryMode,
    RETRIEVAL_PRESETS,
)


@pytest.fixture
def storage_with_history():
    """Create storage with multiple sessions and history."""
    storage = InMemoryStorage()

    # Session 1: OAuth2 (6 months ago)
    storage.create_session("s1", "Implement OAuth2", "Build auth system")
    storage.create_activity("a1", "s1", "Understand OAuth2", ["auth"], "learner", "core", "analysis")
    storage.create_task("t1", "a1", "Learn grant types", ["oauth2", "grants"])
    storage.add_item(
        "OAuth2 has 4 main grant types: auth code, implicit, client creds, resource owner",
        ContextCategory.LEARNING,
        ["oauth2"],
        parent_id="t1",
    )
    storage.add_item(
        "Auth code grant most secure for web apps",
        ContextCategory.DECISION,
        ["oauth2", "grants"],
        parent_id="t1",
    )

    storage.create_task("t2", "a1", "Understand JWT", ["jwt", "tokens"])
    storage.add_item(
        "JWT: header.payload.signature encoded in base64",
        ContextCategory.LEARNING,
        ["jwt"],
        parent_id="t2",
    )
    storage.add_item(
        "Use HS256 for symmetric signing",
        ContextCategory.DECISION,
        ["jwt", "signing"],
        parent_id="t2",
    )

    # Session 2: Microservices (3 months ago)
    storage.create_session("s2", "Build microservices", "Implement distributed auth")
    storage.create_activity(
        "a2", "s2", "Service authentication", ["auth", "microservices"], "coder", "core", "analysis"
    )
    storage.create_task("t3", "a2", "Inter-service JWT validation", ["jwt", "services"])
    storage.add_item(
        "Services validate each other's JWT tokens using shared public keys",
        ContextCategory.LEARNING,
        ["jwt", "services"],
        parent_id="t3",
    )
    storage.add_item(
        "Store JWKS endpoint for key rotation",
        ContextCategory.DECISION,
        ["jwt", "keys"],
        parent_id="t3",
    )

    # Session 3: Current project (today)
    storage.create_session("s3", "New project setup", "Build authentication")
    storage.create_activity("a3", "s3", "Setup authentication", ["auth"], "coder", "core", "analysis")
    storage.create_task("t4", "a3", "Add JWT validation middleware", ["jwt", "middleware"])
    storage.add_item(
        "Middleware validates token signature before processing request",
        ContextCategory.ACTION,
        ["jwt", "validation"],
        parent_id="t4",
    )

    return storage


class TestHistoryModes:
    """Test different history retrieval modes."""

    def test_current_only_mode(self, storage_with_history):
        """CURRENT_ONLY: Only search current session."""
        retriever = HierarchicalRetriever(storage_with_history)
        config = RetrievalConfig(history_mode=HistoryMode.CURRENT_ONLY)

        results = retriever.retrieve(
            query="JWT validation",
            current_task_id="t4",
            max_results=10,
            config=config,
        )

        # Should only find items from current session (s3)
        for result in results:
            assert result["explanation"]["source_session_id"] == "s3"

        # Should find current task item
        assert any("Middleware validates" in result["item"].content for result in results)

        # Should NOT find items from previous sessions
        assert not any("Grant types" in result["item"].content for result in results)

    def test_current_weighted_mode(self, storage_with_history):
        """CURRENT_WEIGHTED: Prioritize current, but include history."""
        retriever = HierarchicalRetriever(storage_with_history)
        config = RetrievalConfig(
            history_mode=HistoryMode.CURRENT_WEIGHTED,
            return_explainability=True,
        )

        results = retriever.retrieve(
            query="JWT validation",
            current_task_id="t4",
            max_results=10,
            config=config,
        )

        # Should have results from both current and history
        session_ids = {result["explanation"]["source_session_id"] for result in results}
        assert "s3" in session_ids  # Current session
        assert len(session_ids) > 1  # Plus historical sessions

        # Current session items should have higher weight (appear first)
        current_results = [r for r in results if r["explanation"]["source_session_id"] == "s3"]
        if current_results and len(results) > 1:
            # Current weighted higher, should rank higher
            assert current_results[0]["final_score"] >= results[-1]["final_score"]

    def test_all_sessions_mode(self, storage_with_history):
        """ALL_SESSIONS: Search all sessions equally (except current task boost)."""
        retriever = HierarchicalRetriever(storage_with_history)
        config = RetrievalConfig(
            history_mode=HistoryMode.ALL_SESSIONS,
            return_explainability=True,
        )

        results = retriever.retrieve(
            query="JWT",
            current_task_id="t4",
            max_results=10,
            config=config,
        )

        # Should have results from all sessions with JWT content
        session_ids = {result["explanation"]["source_session_id"] for result in results}
        assert len(session_ids) >= 2  # Should find JWT content from s1, s2, s3

        # Verify JWT-related items are found
        jwt_items = [r for r in results if "JWT" in r["item"].content or "jwt" in str(r["item"].tags)]
        assert len(jwt_items) > 0

    def test_selected_sessions_mode(self, storage_with_history):
        """SELECTED_SESSIONS: Only search specific sessions."""
        retriever = HierarchicalRetriever(storage_with_history)
        config = RetrievalConfig(
            history_mode=HistoryMode.SELECTED_SESSIONS,
            include_session_ids=["s1", "s3"],  # Only first and current
            return_explainability=True,
        )

        results = retriever.retrieve(
            query="JWT",
            current_task_id="t4",
            max_results=10,
            config=config,
        )

        # Should only have results from s1 and s3
        session_ids = {result["explanation"]["source_session_id"] for result in results}
        assert session_ids.issubset({"s1", "s3"})
        assert "s2" not in session_ids  # Not in selected list

    def test_presets(self, storage_with_history):
        """Test predefined retrieval presets."""
        retriever = HierarchicalRetriever(storage_with_history)

        # Fresh preset: current only
        fresh = retriever.retrieve(
            query="JWT", current_task_id="t4", config=RETRIEVAL_PRESETS["fresh"]
        )
        assert all(
            r["explanation"]["source_session_id"] == "s3"
            for r in fresh if "explanation" in r
        )

        # Balanced preset: weighted history
        balanced = retriever.retrieve(
            query="JWT", current_task_id="t4", config=RETRIEVAL_PRESETS["balanced"]
        )
        session_ids = {r["explanation"]["source_session_id"] for r in balanced if "explanation" in r}
        assert len(session_ids) > 1  # Should have multiple sessions

        # Learn from all: equal weight
        learn_all = retriever.retrieve(
            query="JWT", current_task_id="t4", config=RETRIEVAL_PRESETS["learn_from_all"]
        )
        assert len(learn_all) > 0


class TestUserControl:
    """Test user agency over historical context."""

    def test_weight_configuration(self, storage_with_history):
        """User can control weight multipliers."""
        retriever = HierarchicalRetriever(storage_with_history)

        # High history weight
        config_high = RetrievalConfig(
            history_mode=HistoryMode.CURRENT_WEIGHTED,
            history_weight=1.5,  # Boost history
            use_semantic=False,  # Disable semantic to test proximity weighting
        )
        results_high = retriever.retrieve(
            query="grant types", current_task_id="t4", config=config_high
        )

        # Low history weight
        config_low = RetrievalConfig(
            history_mode=HistoryMode.CURRENT_WEIGHTED,
            history_weight=0.1,  # Suppress history
            use_semantic=False,
        )
        results_low = retriever.retrieve(
            query="grant types", current_task_id="t4", config=config_low
        )

        # Results should have different proximity weights
        high_weights = [r["proximity_weight"] for r in results_high]
        low_weights = [r["proximity_weight"] for r in results_low]

        # High config should have items with weight 1.5, low config with 0.1
        assert any(w == 1.5 for w in high_weights)
        assert any(w == 0.1 for w in low_weights)
        assert max(high_weights) > max(low_weights)  # High config has higher weights

    def test_time_filtering(self, storage_with_history):
        """User can filter by time depth."""
        retriever = HierarchicalRetriever(storage_with_history)

        # Get items without time filtering
        config_all = RetrievalConfig(
            history_mode=HistoryMode.CURRENT_WEIGHTED,
            history_depth_days=None,  # All history
        )
        results_all = retriever.retrieve(
            query="JWT", current_task_id="t4", max_results=10, config=config_all
        )

        # Get items within last 1 day only
        config_recent = RetrievalConfig(
            history_mode=HistoryMode.CURRENT_WEIGHTED,
            history_depth_days=1,  # Only today
        )
        results_recent = retriever.retrieve(
            query="JWT", current_task_id="t4", max_results=10, config=config_recent
        )

        # Recent config should have fewer results (only current session)
        # since other sessions are older than 1 day
        assert len(results_recent) <= len(results_all)

    def test_explainability_on_off(self, storage_with_history):
        """User can control whether to see reasoning."""
        retriever = HierarchicalRetriever(storage_with_history)

        config_with_explain = RetrievalConfig(
            history_mode=HistoryMode.CURRENT_WEIGHTED,
            return_explainability=True,
        )
        results_with = retriever.retrieve(
            query="JWT", current_task_id="t4", config=config_with_explain
        )

        config_without = RetrievalConfig(
            history_mode=HistoryMode.CURRENT_WEIGHTED,
            return_explainability=False,
        )
        results_without = retriever.retrieve(
            query="JWT", current_task_id="t4", config=config_without
        )

        # With explainability
        assert all("explanation" in result for result in results_with)
        assert all("source_session_id" in result.get("explanation", {}) for result in results_with)

        # Without explainability
        assert all("explanation" not in result for result in results_without)


class TestHierarchicalOrganization:
    """Test that hierarchy is respected in retrieval."""

    def test_same_task_items_highest_priority(self, storage_with_history):
        """Items from current task should rank highest."""
        retriever = HierarchicalRetriever(storage_with_history)
        config = RetrievalConfig(
            history_mode=HistoryMode.CURRENT_WEIGHTED,
            return_explainability=True,
        )

        results = retriever.retrieve(
            query="JWT", current_task_id="t4", config=config
        )

        # First result should be from current task if it exists
        if results:
            first = results[0]
            proximity_weight = first["proximity_weight"]
            # Current task items have weight 1.0
            assert proximity_weight >= 0.7  # At least sibling weight

    def test_sibling_tasks_boost(self, storage_with_history):
        """Sibling tasks in same activity should get higher weight than other activities."""
        retriever = HierarchicalRetriever(storage_with_history)
        config = RetrievalConfig(
            history_mode=HistoryMode.CURRENT_ONLY,
            sibling_weight=0.9,
            activity_weight=0.5,
        )

        results = retriever.retrieve(
            query="any", current_task_id="t4", max_results=20, config=config
        )

        # Items from same activity should rank higher than other activities
        # (but lower than current task)
        weights = [result["proximity_weight"] for result in results]
        if weights:
            assert max(weights) == 1.0  # Current task weight
            assert min(weights) <= 0.5  # At least down to activity weight

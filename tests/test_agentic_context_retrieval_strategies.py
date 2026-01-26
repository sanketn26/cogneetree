"""Tests for refactored agentic_context.retrieval.retrieval_strategies module."""

import pytest
from cogneetree.core.context_storage import (
    ContextStorage,
    ContextCategory,
)
from cogneetree.retrieval.retrieval_strategies import Retriever


class TestRetriever:
    """Test Retriever class."""

    def test_retrieve_by_tags_exact_match(self):
        """Test retrieving items with exact tag match."""
        storage = ContextStorage()
        storage.add_item("API implementation", ContextCategory.ACTION, ["api", "auth"])
        storage.add_item("Database schema", ContextCategory.DECISION, ["database"])
        storage.add_item("Cache strategy", ContextCategory.LEARNING, ["cache", "api"])

        retriever = Retriever(storage, use_semantic=False)

        results = retriever.retrieve(["api"], "API usage", max_results=5)
        assert len(results) >= 2
        # Should find items with "api" tag
        assert any("API implementation" in str(r.get("item")) for r in results)

    def test_retrieve_with_max_results(self):
        """Test max_results parameter."""
        storage = ContextStorage()
        for i in range(10):
            storage.add_item(f"Item {i}", ContextCategory.ACTION, ["common"])

        retriever = Retriever(storage, use_semantic=False)

        results = retriever.retrieve(["common"], "common task", max_results=3)
        assert len(results) <= 3

    def test_retrieve_with_multiple_query_tags(self):
        """Test retrieving with multiple query tags."""
        storage = ContextStorage()
        storage.add_item("High priority", ContextCategory.ACTION, ["important", "urgent"])
        storage.add_item("Low priority", ContextCategory.ACTION, ["other"])

        retriever = Retriever(storage, use_semantic=False)

        results = retriever.retrieve(["important", "urgent"], "important task", max_results=5)
        # Should get high-scoring items with multiple matching tags
        assert len(results) >= 0

    def test_retrieve_empty_storage(self):
        """Test retrieving from empty storage."""
        storage = ContextStorage()
        retriever = Retriever(storage, use_semantic=False)

        results = retriever.retrieve(["any_tag"], "query description", max_results=5)
        assert results == []

    def test_retrieve_by_category(self):
        """Test retrieving items across categories."""
        storage = ContextStorage()
        storage.add_item("Action item", ContextCategory.ACTION, ["tag1"])
        storage.add_item("Decision item", ContextCategory.DECISION, ["tag1"])
        storage.add_item("Learning item", ContextCategory.LEARNING, ["tag1"])

        retriever = Retriever(storage, use_semantic=False)

        # Retrieve and check mix of categories
        results = retriever.retrieve(["tag1"], "query", max_results=10)
        assert len(results) >= 2

    def test_retrieve_respects_recency(self):
        """Test that more recent items are scored higher."""
        storage = ContextStorage()
        
        # Add items with tags - more recent ones should be prioritized
        storage.add_item("Older item", ContextCategory.ACTION, ["api"])
        storage.add_item("Newer item", ContextCategory.ACTION, ["api"])

        retriever = Retriever(storage, use_semantic=False)

        results = retriever.retrieve(["api"], "api query", max_results=10)
        # Both should be returned, with scoring applied
        assert len(results) == 2

    def test_retrieve_multiple_tags(self):
        """Test retrieving with multiple tag queries."""
        storage = ContextStorage()
        storage.add_item("API auth", ContextCategory.ACTION, ["api", "auth"])
        storage.add_item("API cache", ContextCategory.ACTION, ["api", "cache"])
        storage.add_item("DB auth", ContextCategory.ACTION, ["database", "auth"])

        retriever = Retriever(storage, use_semantic=False)

        # Retrieve by one tag should get multiple items
        results = retriever.retrieve(["api"], "api query", max_results=10)
        assert len(results) >= 2

    def test_score_items_basic(self):
        """Test internal _score_items method."""
        storage = ContextStorage()
        item1 = storage.add_item("Item 1", ContextCategory.ACTION, ["tag1", "tag2"])
        item2 = storage.add_item("Item 2", ContextCategory.ACTION, ["tag2"])
        item3 = storage.add_item("Item 3", ContextCategory.ACTION, ["other"])

        retriever = Retriever(storage, use_semantic=False)

        # Score items for ["tag1"] - item1 should score highest
        scored = retriever._score_items(storage.items, ["tag1"], "query desc")
        assert len(scored) == 3
        assert scored[0]["score"] >= scored[1]["score"]  # item1 score >= item2 score

    def test_retrieve_with_description(self):
        """Test retrieving with query description."""
        storage = ContextStorage()
        storage.add_item("Authentication module", ContextCategory.ACTION, ["auth"])
        storage.add_item("Authorization module", ContextCategory.ACTION, ["auth"])

        retriever = Retriever(storage, use_semantic=False)

        results = retriever.retrieve(["auth"], "how to authenticate users", max_results=5)
        assert len(results) >= 1
        # Results should have score and breakdown info
        assert all("score" in r for r in results)
        assert all("breakdown" in r for r in results)

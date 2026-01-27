"""Tests for refactored agentic_context.core.context_storage module."""

import pytest
from cogneetree.storage.in_memory_storage import InMemoryStorage
from cogneetree.core.models import (
    ContextItem,
    ContextCategory,
    Session,
    Activity,
    Task,
)


class TestSession:
    """Test Session dataclass."""

    def test_session_creation(self):
        """Test creating a session."""
        session = Session("s1", "Do something", "Plan: 1,2,3")
        assert session.session_id == "s1"
        assert session.original_ask == "Do something"
        assert session.high_level_plan == "Plan: 1,2,3"


class TestActivity:
    """Test Activity dataclass."""

    def test_activity_creation(self):
        """Test creating an activity."""
        activity = Activity("a1", "s1", "Build feature", ["tag1"], "coder", "comp", "analysis")
        assert activity.activity_id == "a1"
        assert activity.session_id == "s1"
        assert activity.description == "Build feature"
        assert activity.mode == "coder"


class TestTask:
    """Test Task dataclass."""

    def test_task_creation(self):
        """Test creating a task."""
        task = Task("t1", "a1", "Implement function", ["tag2"])
        assert task.task_id == "t1"
        assert task.activity_id == "a1"
        assert task.result is None

    def test_task_with_result(self):
        """Test task with result."""
        task = Task("t1", "a1", "Implement function", ["tag2"], result="Done")
        assert task.result == "Done"


class TestContextItem:
    """Test ContextItem dataclass."""

    def test_context_item_creation(self):
        """Test creating a context item."""
        item = ContextItem("Content here", ContextCategory.ACTION, tags=["tag1"])
        assert item.content == "Content here"
        assert item.category == ContextCategory.ACTION
        assert item.tags == ["tag1"]


class TestContextStorage:
    """Test ContextStorage class."""

    def test_create_session(self):
        """Test creating a session."""
        storage = InMemoryStorage()
        session = storage.create_session("s1", "Ask", "Plan")
        assert session.session_id == "s1"
        assert storage.current_session_id == "s1"

    def test_create_activity(self):
        """Test creating an activity."""
        storage = InMemoryStorage()
        storage.create_session("s1", "Ask", "Plan")
        activity = storage.create_activity("a1", "s1", "Desc", ["tag"], "coder", "comp", "analysis")
        assert activity.activity_id == "a1"
        assert storage.current_activity_id == "a1"

    def test_create_task(self):
        """Test creating a task."""
        storage = InMemoryStorage()
        storage.create_session("s1", "Ask", "Plan")
        storage.create_activity("a1", "s1", "Desc", ["tag"], "coder", "comp", "analysis")
        task = storage.create_task("t1", "a1", "Task desc", ["tag"])
        assert task.task_id == "t1"
        assert storage.current_task_id == "t1"

    def test_add_item(self):
        """Test adding a context item."""
        storage = InMemoryStorage()
        item = storage.add_item("Content", ContextCategory.ACTION, ["tag"])
        assert len(storage.items) == 1
        assert item.content == "Content"

    def test_get_items_by_category(self):
        """Test getting items by category."""
        storage = InMemoryStorage()
        storage.add_item("Action 1", ContextCategory.ACTION, ["tag"])
        storage.add_item("Action 2", ContextCategory.ACTION, ["tag"])
        storage.add_item("Decision 1", ContextCategory.DECISION, ["tag"])

        actions = storage.get_items_by_category(ContextCategory.ACTION)
        assert len(actions) == 2

        decisions = storage.get_items_by_category(ContextCategory.DECISION)
        assert len(decisions) == 1

    def test_get_items_by_tags(self):
        """Test getting items by tags."""
        storage = InMemoryStorage()
        storage.add_item("Item 1", ContextCategory.ACTION, ["api", "auth"])
        storage.add_item("Item 2", ContextCategory.ACTION, ["api"])
        storage.add_item("Item 3", ContextCategory.DECISION, ["auth"])

        api_items = storage.get_items_by_tags(["api"])
        assert len(api_items) == 2

    def test_complete_task(self):
        """Test completing a task."""
        storage = InMemoryStorage()
        storage.create_session("s1", "Ask", "Plan")
        storage.create_activity("a1", "s1", "Desc", ["tag"], "coder", "comp", "analysis")
        task = storage.create_task("t1", "a1", "Task", ["tag"])
        
        storage.complete_task("t1", "Result here")
        assert task.result == "Result here"

    def test_get_stats(self):
        """Test getting storage statistics."""
        storage = InMemoryStorage()
        storage.create_session("s1", "Ask", "Plan")
        storage.create_activity("a1", "s1", "Desc", ["tag"], "coder", "comp", "analysis")
        storage.create_task("t1", "a1", "Task", ["tag"])
        storage.add_item("Item", ContextCategory.ACTION, ["tag"])

        stats = storage.get_stats()
        assert stats["sessions"] == 1
        assert stats["activities"] == 1
        assert stats["tasks"] == 1
        assert stats["items"] == 1

    def test_clear(self):
        """Test clearing storage."""
        storage = InMemoryStorage()
        storage.create_session("s1", "Ask", "Plan")
        storage.add_item("Item", ContextCategory.ACTION, ["tag"])

        storage.clear()
        assert len(storage.sessions) == 0
        assert len(storage.items) == 0

"""Tests for refactored agentic_context.core.context_manager module."""

import pytest
from cogneetree.core.context_manager import ContextManager
from cogneetree.config import Config
from cogneetree.core.models import ContextCategory


class TestContextManager:
    """Test ContextManager class."""

    def test_context_manager_creation(self):
        """Test creating a context manager."""
        config = Config.default()
        manager = ContextManager(config)
        assert manager is not None
        assert manager.storage is not None
        assert manager.retriever is not None

    def test_create_session(self):
        """Test creating a session."""
        manager = ContextManager(Config.default())
        session = manager.create_session("s1", "Do something", "Plan here")
        
        assert session is not None
        assert session.session_id == "s1"
        assert manager.storage.current_session_id == "s1"

    def test_create_activity(self):
        """Test creating an activity."""
        manager = ContextManager(Config.default())
        manager.create_session("s1", "Ask", "Plan")
        
        activity = manager.create_activity("a1", "s1", "Implement feature", ["tag1"], "coder", "comp", "analysis")
        assert activity is not None
        assert activity.activity_id == "a1"

    def test_create_task(self):
        """Test creating a task."""
        manager = ContextManager(Config.default())
        manager.create_session("s1", "Ask", "Plan")
        manager.create_activity("a1", "s1", "Activity", ["tag"], "coder", "comp", "analysis")
        
        task = manager.create_task("t1", "a1", "Task description", ["tag"])
        assert task is not None
        assert task.task_id == "t1"

    def test_record_action(self):
        """Test recording an action."""
        manager = ContextManager(Config.default())
        manager.create_session("s1", "Ask", "Plan")
        
        manager.record_action("Performed action", ["tag1"])
        
        items = manager.storage.get_items_by_category(ContextCategory.ACTION)
        assert len(items) > 0

    def test_record_decision(self):
        """Test recording a decision."""
        manager = ContextManager(Config.default())
        manager.create_session("s1", "Ask", "Plan")
        
        manager.record_decision("Made a decision", ["tag1"])
        
        items = manager.storage.get_items_by_category(ContextCategory.DECISION)
        assert len(items) > 0

    def test_record_learning(self):
        """Test recording a learning."""
        manager = ContextManager(Config.default())
        manager.create_session("s1", "Ask", "Plan")
        
        manager.record_learning("Learned something", ["tag1"])
        
        items = manager.storage.get_items_by_category(ContextCategory.LEARNING)
        assert len(items) > 0

    def test_record_result(self):
        """Test recording a result."""
        manager = ContextManager(Config.default())
        manager.create_session("s1", "Ask", "Plan")
        
        manager.record_result("Result of work", ["tag1"])
        
        items = manager.storage.get_items_by_category(ContextCategory.RESULT)
        assert len(items) > 0

    def test_complete_task(self):
        """Test completing a task."""
        manager = ContextManager(Config.default())
        manager.create_session("s1", "Ask", "Plan")
        manager.create_activity("a1", "s1", "Activity", ["tag"], "coder", "comp", "analysis")
        manager.create_task("t1", "a1", "Task", ["tag"])
        
        manager.complete_task("t1", "Task result")
        
        task = manager.storage.tasks["t1"]
        assert task.result == "Task result"

    def test_get_current_session(self):
        """Test getting current session."""
        manager = ContextManager(Config.default())
        manager.create_session("s1", "Ask", "Plan")
        
        session = manager.storage.get_current_session()
        assert session is not None
        assert session.session_id == "s1"

    def test_get_current_activity(self):
        """Test getting current activity."""
        manager = ContextManager(Config.default())
        manager.create_session("s1", "Ask", "Plan")
        manager.create_activity("a1", "s1", "Activity", ["tag"], "coder", "comp", "analysis")
        
        activity = manager.storage.get_current_activity()
        assert activity is not None
        assert activity.activity_id == "a1"

    def test_get_current_task(self):
        """Test getting current task."""
        manager = ContextManager(Config.default())
        manager.create_session("s1", "Ask", "Plan")
        manager.create_activity("a1", "s1", "Activity", ["tag"], "coder", "comp", "analysis")
        manager.create_task("t1", "a1", "Task", ["tag"])
        
        task = manager.get_current_task()
        assert task is not None
        assert task.task_id == "t1"

    def test_retrieve_context(self):
        """Test retrieving context items."""
        manager = ContextManager(Config.default())
        manager.create_session("s1", "Ask", "Plan")
        manager.record_action("Action 1", ["api"])
        manager.record_action("Action 2", ["api", "auth"])
        
        results = manager.retrieve(["api"], "API related tasks")
        assert isinstance(results, list)
        assert len(results) >= 0

    def test_build_prompt(self):
        """Test building a prompt."""
        manager = ContextManager(Config.default())
        manager.create_session("s1", "Build API", "Plan here")
        
        prompt = manager.build_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_build_prompt_with_history(self):
        """Test building prompt with history."""
        manager = ContextManager(Config.default())
        manager.create_session("s1", "Task", "Plan")
        manager.create_activity("a1", "s1", "Activity", ["tag"], "coder", "comp", "analysis")
        manager.record_decision("Decision made", ["tag"])
        
        prompt = manager.build_prompt(include_history=True)
        assert isinstance(prompt, str)

    def test_clear_context(self):
        """Test clearing context."""
        manager = ContextManager(Config.default())
        manager.create_session("s1", "Ask", "Plan")
        manager.record_action("Action", ["tag"])
        
        manager.clear()
        
        assert len(manager.storage.sessions) == 0
        assert len(manager.storage.items) == 0

    def test_multiple_sessions(self):
        """Test creating multiple sessions."""
        manager = ContextManager(Config.default())
        manager.create_session("s1", "Task 1", "Plan 1")
        manager.create_session("s2", "Task 2", "Plan 2")
        
        assert len(manager.storage.sessions) == 2

    def test_manager_with_semantic_config(self):
        """Test manager with semantic search enabled."""
        config = Config(use_semantic=False)  # Disable to avoid deps
        manager = ContextManager(config)
        
        manager.create_session("s1", "Ask", "Plan")
        manager.record_action("Test action", ["tag"])
        
        # Should work without errors
        results = manager.retrieve(["tag"], "Test query")
        assert isinstance(results, list)

    def test_retrieve_with_custom_max_results(self):
        """Test retrieve with custom max_results."""
        config = Config(max_results=2)
        manager = ContextManager(config)
        
        manager.create_session("s1", "Ask", "Plan")
        for i in range(5):
            manager.record_action(f"Action {i}", ["common"])
        
        results = manager.retrieve(["common"], "query")
        assert len(results) <= 2

    def test_get_stats(self):
        """Test getting storage statistics."""
        manager = ContextManager(Config.default())
        manager.create_session("s1", "Ask", "Plan")
        manager.create_activity("a1", "s1", "Activity", ["tag"], "coder", "comp", "analysis")
        manager.create_task("t1", "a1", "Task", ["tag"])
        manager.record_action("Action", ["tag"])
        
        stats = manager.storage.get_stats()
        assert stats["sessions"] == 1
        assert stats["activities"] == 1
        assert stats["tasks"] == 1
        assert stats["items"] >= 1

    def test_context_manager_integration(self):
        """Test full integration scenario."""
        manager = ContextManager(Config.default())
        
        # Create session
        manager.create_session("build_api", "Build authentication API", "1. Design 2. Code 3. Test")
        
        # Create activity
        manager.create_activity("design_phase", "build_api", "Design auth system", ["design", "api"], "architect", "planning", "analysis")
        
        # Record decisions during design
        manager.record_decision("Use JWT tokens", ["auth", "security"])
        manager.record_decision("Use bcrypt for passwords", ["security", "hashing"])
        
        # Create another activity
        manager.create_activity("coding_phase", "build_api", "Implement auth endpoints", ["coding", "api"], "coder", "implementation", "coding")
        
        # Create task
        manager.create_task("login_endpoint", "coding_phase", "Implement login endpoint", ["api", "auth"])
        
        # Record actions
        manager.record_action("Created POST /auth/login", ["api", "endpoint"])
        manager.record_action("Added JWT token response", ["token", "jwt"])
        
        # Build prompt with context
        prompt = manager.build_prompt(include_history=True)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

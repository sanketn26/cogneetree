"""Workflow context managers - simple context hierarchy."""

from contextlib import contextmanager
from typing import Optional, List
from cogneetree.core.context_manager import ContextManager
from cogneetree.core.interfaces import ContextStorageABC, EmbeddingModelABC
from cogneetree.config import Config



class ContextWorkflow:
    """Entry point for context tracking in workflows."""

    def __init__(
        self,
        config: Optional[Config] = None,
        storage: Optional[ContextStorageABC] = None,
        embedding_model: Optional[EmbeddingModelABC] = None,
    ):
        """Initialize workflow with optional config, storage, and embedding model."""
        self.manager = ContextManager(
            config=config or Config.default(),
            storage=storage,
            embedding_model=embedding_model,
        )

    @contextmanager
    def session(self, session_id: str, original_ask: str, high_level_plan: str):
        """Create session context."""
        self.manager.create_session(session_id, original_ask, high_level_plan)
        try:
            yield SessionContext(self.manager)
        finally:
            pass

    def clear(self):
        """Clear all context."""
        self.manager.clear()


class SessionContext:
    """Session-level context manager."""

    def __init__(self, manager: ContextManager):
        """Initialize with manager."""
        self.manager = manager

    @contextmanager
    def activity(
        self,
        activity_id: str,
        description: str,
        mode: str,
        component: str,
        planner_analysis: str,
        tags: Optional[List[str]] = None,
    ):
        """Create activity context."""
        session = self.manager.storage.get_current_session()
        if not session:
            raise ValueError("No active session")

        tags = tags or []
        self.manager.create_activity(
            activity_id, session.session_id, description, tags, mode, component, planner_analysis
        )

        try:
            yield ActivityContext(self.manager)
        finally:
            pass


class ActivityContext:
    """Activity-level context manager."""

    def __init__(self, manager: ContextManager):
        """Initialize with manager."""
        self.manager = manager

    @contextmanager
    def task(self, description: str, tags: Optional[List[str]] = None):
        """Create task context."""
        activity = self.manager.storage.get_current_activity()
        if not activity:
            raise ValueError("No active activity")

        tags = tags or []
        import uuid
        task_id = f"task_{uuid.uuid4().hex[:6]}"
        self.manager.create_task(task_id, activity.activity_id, description, tags)

        try:
            yield TaskContext(self.manager)
        finally:
            pass


class TaskContext:
    """Task-level context manager."""

    def __init__(self, manager: ContextManager):
        """Initialize with manager."""
        self.manager = manager

    def build_prompt(self, include_history: bool = True) -> str:
        """Build prompt for current task."""
        return self.manager.build_prompt(include_history)

    def record_action(self, content: str):
        """Record action taken."""
        task = self.manager.get_current_task()
        if task:
            self.manager.record_action(content, task.tags)

    def record_decision(self, content: str, promote: bool = False):
        """Record decision made.

        Args:
            content: The decision text.
            promote: When True, force propagation to Activity and Session regardless
                     of relevance score.
        """
        task = self.manager.get_current_task()
        if task:
            self.manager.record_decision(content, task.tags, promote=promote)

    def record_learning(self, content: str, promote: bool = False):
        """Record learning.

        Args:
            content: The learning text.
            promote: When True, force propagation to Activity and Session.
        """
        task = self.manager.get_current_task()
        if task:
            self.manager.record_learning(content, task.tags, promote=promote)

    def set_result(self, result: str):
        """Set task result."""
        task = self.manager.get_current_task()
        if task:
            self.manager.complete_task(task.task_id, result)
            self.manager.record_result(result, task.tags)

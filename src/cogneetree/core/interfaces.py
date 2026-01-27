"""
Core interfaces for storage and embeddings.

IMPLEMENTING A CUSTOM STORAGE BACKEND:

To implement a custom storage backend (SQL, Redis, etc.):

1. Subclass ContextStorageABC
2. Maintain a context stack (current_session_id, current_activity_id, current_task_id)
   - These should be updated when create_session/activity/task() are called
   - Used by ContextManager.build_prompt() to get active context
3. Implement all abstract methods (see ContextStorageABC below)
4. For list methods (get_items_by_tags, get_items_by_category), support empty results
5. The clear() method must reset the context stack IDs to None

EXAMPLE:
    class RedisStorage(ContextStorageABC):
        def __init__(self, url: str):
            self.redis = redis.from_url(url)
            self.current_session_id = None  # Track current context
            # ... implement all abstract methods ...

        def create_session(self, session_id, original_ask, high_level_plan):
            session = Session(session_id, original_ask, high_level_plan)
            self.redis.hset(f"sessions:{session_id}", mapping={...})
            self.current_session_id = session_id  # UPDATE CONTEXT STACK
            return session
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from cogneetree.core.models import Session, Activity, Task, ContextItem, ContextCategory


class EmbeddingModelABC(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def encode(self, text: str) -> Any:
        """Encode text into an embedding."""
        pass

    @abstractmethod
    def similarity(self, emb1: Any, emb2: Any) -> float:
        """Calculate similarity between two embeddings."""
        pass


class ContextStorageABC(ABC):
    """
    Abstract base class for context storage.

    Implementations MUST maintain a context stack (current session/activity/task).
    When create_session/activity/task() is called, that entity becomes current.
    This allows build_prompt() and record_*() to always reference the active context.
    """

    @abstractmethod
    def create_session(self, session_id: str, original_ask: str, high_level_plan: str) -> Session:
        """Create and store a session. Sets this session as current."""
        pass

    @abstractmethod
    def create_activity(
        self,
        activity_id: str,
        session_id: str,
        description: str,
        tags: List[str],
        mode: str,
        component: str,
        planner_analysis: str,
    ) -> Activity:
        """Create and store an activity. Sets this activity as current."""
        pass

    @abstractmethod
    def create_task(self, task_id: str, activity_id: str, description: str, tags: List[str]) -> Task:
        """Create and store a task. Sets this task as current."""
        pass

    @abstractmethod
    def complete_task(self, task_id: str, result: str) -> None:
        """Mark a task as complete and update its result."""
        pass

    @abstractmethod
    def add_item(
        self,
        content: str,
        category: ContextCategory,
        tags: List[str],
        parent_id: Optional[str] = None,
        embedding: Optional[Any] = None,
    ) -> ContextItem:
        """Add a context item."""
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        pass

    @abstractmethod
    def get_activity(self, activity_id: str) -> Optional[Activity]:
        """Get an activity by ID."""
        pass

    @abstractmethod
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        pass

    @abstractmethod
    def get_current_session(self) -> Optional[Session]:
        """Get the current active session."""
        pass

    @abstractmethod
    def get_current_activity(self) -> Optional[Activity]:
        """Get the current active activity."""
        pass

    @abstractmethod
    def get_current_task(self) -> Optional[Task]:
        """Get the current active task."""
        pass

    @abstractmethod
    def get_items_by_tags(self, tags: List[str]) -> List[ContextItem]:
        """Get items matching *any* of the provided tags."""
        pass

    @abstractmethod
    def get_items_by_category(self, category: ContextCategory) -> List[ContextItem]:
        """Get items filtered by category type."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, int]:
        """Get storage statistics (session, activity, task, item counts)."""
        pass

    @abstractmethod
    def get_items_by_task(self, task_id: str) -> List[ContextItem]:
        """Get items recorded within a specific task."""
        pass

    @abstractmethod
    def get_items_by_activity(self, activity_id: str) -> List[ContextItem]:
        """Get items recorded within a specific activity (all its tasks)."""
        pass

    @abstractmethod
    def get_items_by_session(self, session_id: str) -> List[ContextItem]:
        """Get items recorded within a specific session (all its activities/tasks)."""
        pass

    @abstractmethod
    def get_all_sessions(self) -> List[Session]:
        """Get all sessions in chronological order."""
        pass

    @abstractmethod
    def get_activity_tasks(self, activity_id: str) -> List[Task]:
        """Get all tasks within an activity."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored data."""
        pass

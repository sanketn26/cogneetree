"""Context Storage - Simple flat storage with parent tracking."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class ContextCategory(Enum):
    """Context item categories."""

    SESSION = "session"
    ACTIVITY = "activity"
    TASK = "task"
    ACTION = "action"
    DECISION = "decision"
    LEARNING = "learning"
    RESULT = "result"


@dataclass
class ContextItem:
    """Single context item with metadata."""

    content: str
    category: ContextCategory
    tags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[Any] = None


@dataclass
class Session:
    """Session context."""

    session_id: str
    original_ask: str
    high_level_plan: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Activity:
    """Activity context."""

    activity_id: str
    session_id: str
    description: str
    tags: List[str]
    mode: str
    component: str
    planner_analysis: str


@dataclass
class Task:
    """Task context."""

    task_id: str
    activity_id: str
    description: str
    tags: List[str]
    result: Optional[str] = None


class ContextStorage:
    """Simple flat storage for context items."""

    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.activities: Dict[str, Activity] = {}
        self.tasks: Dict[str, Task] = {}
        self.items: List[ContextItem] = []

        self.current_session_id: Optional[str] = None
        self.current_activity_id: Optional[str] = None
        self.current_task_id: Optional[str] = None

    def create_session(self, session_id: str, original_ask: str, high_level_plan: str) -> Session:
        """Create session."""
        session = Session(session_id, original_ask, high_level_plan)
        self.sessions[session_id] = session
        self.current_session_id = session_id
        return session

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
        """Create activity."""
        activity = Activity(activity_id, session_id, description, tags, mode, component, planner_analysis)
        self.activities[activity_id] = activity
        self.current_activity_id = activity_id
        return activity

    def create_task(self, task_id: str, activity_id: str, description: str, tags: List[str]) -> Task:
        """Create task."""
        task = Task(task_id, activity_id, description, tags)
        self.tasks[task_id] = task
        self.current_task_id = task_id
        return task

    def complete_task(self, task_id: str, result: str):
        """Mark task complete."""
        if task_id in self.tasks:
            self.tasks[task_id].result = result

    def add_item(
        self,
        content: str,
        category: ContextCategory,
        tags: List[str],
        parent_id: Optional[str] = None,
        embedding: Optional[Any] = None,
    ) -> ContextItem:
        """Add context item."""
        item = ContextItem(content, category, tags, parent_id=parent_id, embedding=embedding)
        self.items.append(item)
        return item

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def get_activity(self, activity_id: str) -> Optional[Activity]:
        """Get activity by ID."""
        return self.activities.get(activity_id)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def get_current_session(self) -> Optional[Session]:
        """Get current session."""
        if self.current_session_id:
            return self.sessions.get(self.current_session_id)
        return None

    def get_current_activity(self) -> Optional[Activity]:
        """Get current activity."""
        if self.current_activity_id:
            return self.activities.get(self.current_activity_id)
        return None

    def get_current_task(self) -> Optional[Task]:
        """Get current task."""
        if self.current_task_id:
            return self.tasks.get(self.current_task_id)
        return None

    def get_items_by_category(self, category: ContextCategory) -> List[ContextItem]:
        """Get items by category."""
        return [item for item in self.items if item.category == category]

    def get_items_by_tags(self, tags: List[str]) -> List[ContextItem]:
        """Get items matching any tag."""
        return [item for item in self.items if any(tag in item.tags for tag in tags)]

    def get_items_for_parent(self, parent_id: str) -> List[ContextItem]:
        """Get items for session/activity/task."""
        return [item for item in self.items if item.parent_id == parent_id]

    def get_stats(self) -> Dict[str, int]:
        """Get storage statistics."""
        return {
            "sessions": len(self.sessions),
            "activities": len(self.activities),
            "tasks": len(self.tasks),
            "items": len(self.items),
        }

    def clear(self):
        """Clear all data."""
        self.sessions.clear()
        self.activities.clear()
        self.tasks.clear()
        self.items.clear()
        self.current_session_id = None
        self.current_activity_id = None
        self.current_task_id = None

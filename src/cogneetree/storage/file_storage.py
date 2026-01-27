"""File-based storage backends (JSON, Markdown)."""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from cogneetree.core.interfaces import ContextStorageABC
from cogneetree.storage.in_memory_storage import InMemoryStorage  # Use as base for caching
from cogneetree.core.models import Session, Activity, Task, ContextItem, ContextCategory


class JsonFileStorage(InMemoryStorage):
    """Storage that persists to a JSON file on every write."""

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self._load()

    def _persist(self):
        """Save current state to file."""
        data = {
            "sessions": {k: self._serialize_session(v) for k, v in self.sessions.items()},
            "activities": {k: self._serialize_activity(v) for k, v in self.activities.items()},
            "tasks": {k: self._serialize_task(v) for k, v in self.tasks.items()},
            "items": [self._serialize_item(i) for i in self.items],
            "current": {
                "session": self.current_session_id,
                "activity": self.current_activity_id,
                "task": self.current_task_id,
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self):
        """Load state from file."""
        if not os.path.exists(self.file_path):
            return

        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)

            # Reconstruct objects
            self.sessions = {k: self._deserialize_session(v) for k, v in data.get("sessions", {}).items()}
            self.activities = {k: self._deserialize_activity(v) for k, v in data.get("activities", {}).items()}
            self.tasks = {k: self._deserialize_task(v) for k, v in data.get("tasks", {}).items()}
            self.items = [self._deserialize_item(i) for i in data.get("items", [])]
            
            curr = data.get("current", {})
            self.current_session_id = curr.get("session")
            self.current_activity_id = curr.get("activity")
            self.current_task_id = curr.get("task")

        except Exception as e:
            print(f"Error loading JSON storage: {e}")

    # Helpers for serialization
    def _serialize_session(self, s: Session) -> dict:
        return {"session_id": s.session_id, "original_ask": s.original_ask, "high_level_plan": s.high_level_plan, "created_at": s.created_at.isoformat()}

    def _deserialize_session(self, d: dict) -> Session:
        return Session(d["session_id"], d["original_ask"], d["high_level_plan"], created_at=datetime.fromisoformat(d["created_at"]))

    def _serialize_activity(self, a: Activity) -> dict:
        return {
            "activity_id": a.activity_id, "session_id": a.session_id, "description": a.description,
            "tags": a.tags, "mode": a.mode, "component": a.component, "planner_analysis": a.planner_analysis
        }

    def _deserialize_activity(self, d: dict) -> Activity:
        return Activity(**d)

    def _serialize_task(self, t: Task) -> dict:
        return {"task_id": t.task_id, "activity_id": t.activity_id, "description": t.description, "tags": t.tags, "result": t.result}

    def _deserialize_task(self, d: dict) -> Task:
        return Task(**d)

    def _serialize_item(self, i: ContextItem) -> dict:
        return {
            "content": i.content, "category": i.category.value, "tags": i.tags,
            "timestamp": i.timestamp.isoformat(), "parent_id": i.parent_id, "metadata": i.metadata,
            # Skip embedding for JSON serialization simplicity
        }

    def _deserialize_item(self, d: dict) -> ContextItem:
        return ContextItem(
            content=d["content"],
            category=ContextCategory(d["category"]),
            tags=d["tags"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            parent_id=d.get("parent_id"),
            metadata=d.get("metadata", {})
        )

    # Overrides to trigger persistence
    def create_session(self, *args, **kwargs):
        res = super().create_session(*args, **kwargs)
        self._persist()
        return res

    def create_activity(self, *args, **kwargs):
        res = super().create_activity(*args, **kwargs)
        self._persist()
        return res

    def create_task(self, *args, **kwargs):
        res = super().create_task(*args, **kwargs)
        self._persist()
        return res

    def complete_task(self, *args, **kwargs):
        super().complete_task(*args, **kwargs)
        self._persist()

    def add_item(self, *args, **kwargs):
        res = super().add_item(*args, **kwargs)
        self._persist()
        return res

    def clear(self):
        super().clear()
        if os.path.exists(self.file_path):
            os.remove(self.file_path)


class MarkdownFileStorage(InMemoryStorage):
    """Write-only storage that appends to a human-readable Markdown file."""

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)

    def _append(self, text: str):
        """Append text to markdown file."""
        with open(self.file_path, "a") as f:
            f.write(text + "\n")

    def create_session(self, session_id: str, original_ask: str, high_level_plan: str):
        res = super().create_session(session_id, original_ask, high_level_plan)
        self._append(f"\n# Session: {session_id}\n")
        self._append(f"**Ask:** {original_ask}\n")
        self._append(f"**Plan:** {high_level_plan}\n")
        self._append(f"_{datetime.now().isoformat()}_\n")
        self._append("---\n")
        return res

    def create_activity(self, activity_id: str, session_id: str, description: str, tags: List[str], mode: str, component: str, planner_analysis: str):
        res = super().create_activity(activity_id, session_id, description, tags, mode, component, planner_analysis)
        self._append(f"\n## Activity: {description} (`{activity_id}`)\n")
        self._append(f"*Mode:* {mode} | *Component:* {component}\n")
        self._append(f"*Analysis:* {planner_analysis}\n")
        self._append(f"*Tags:* {', '.join(tags)}\n")
        return res

    def create_task(self, task_id: str, activity_id: str, description: str, tags: List[str]):
        res = super().create_task(task_id, activity_id, description, tags)
        self._append(f"\n### Task: {description} (`{task_id}`)\n")
        self._append(f"*Tags:* {', '.join(tags)}\n")
        return res

    def complete_task(self, task_id: str, result: str):
        super().complete_task(task_id, result)
        self._append(f"\n**[COMPLETED]** Task `{task_id}`\n")
        self._append(f"> Result: {result}\n")

    def add_item(self, content: str, category: ContextCategory, tags: List[str], parent_id: Optional[str] = None, embedding: Optional[Any] = None):
        res = super().add_item(content, category, tags, parent_id, embedding)
        icon = {
            ContextCategory.ACTION: "⚡",
            ContextCategory.DECISION: "🤔",
            ContextCategory.LEARNING: "💡",
            ContextCategory.RESULT: "✅",
        }.get(category, "•")
        
        self._append(f"- {icon} **{category.name}**: {content}")
        if tags:
            self._append(f"  - _Tags: {', '.join(tags)}_")
        return res

    def clear(self):
        super().clear()
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

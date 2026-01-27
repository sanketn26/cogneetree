"""Context manager - simple interface to storage and retrieval."""

from typing import List, Dict, Any, Optional
from cogneetree.core.interfaces import ContextStorageABC, EmbeddingModelABC
from cogneetree.storage.in_memory_storage import InMemoryStorage
from cogneetree.retrieval.retrieval_strategies import Retriever
from cogneetree.config import Config
from cogneetree.core.models import ContextCategory


class ContextManager:
    """Simple context manager - storage + retrieval."""

    def __init__(
        self,
        config: Optional[Config] = None,
        storage: Optional[ContextStorageABC] = None,
        embedding_model: Optional[EmbeddingModelABC] = None,
    ):
        """Initialize with optional config, storage, and embedding model."""
        self.config = config or Config.default()
        # Use provided storage or create default
        self.storage = storage or self._create_storage(self.config)
        self.retriever = Retriever(
            self.storage,
            use_semantic=self.config.use_semantic,
            embedding_model=embedding_model,
        )

    @staticmethod
    def _create_storage(config: Config) -> ContextStorageABC:
        """Factory method to create storage based on config.

        Currently only supports in-memory storage.
        Override this method in subclasses to support other backends.
        """
        return InMemoryStorage()

    # ==================== Session ====================

    def create_session(self, session_id: str, original_ask: str, high_level_plan: str):
        """Create session."""
        return self.storage.create_session(session_id, original_ask, high_level_plan)

    def get_current_session(self):
        """Get current session."""
        return self.storage.get_current_session()

    # ==================== Activity ====================

    def create_activity(
        self,
        activity_id: str,
        session_id: str,
        description: str,
        tags: List[str],
        mode: str,
        component: str,
        planner_analysis: str,
    ):
        """Create activity."""
        return self.storage.create_activity(
            activity_id, session_id, description, tags, mode, component, planner_analysis
        )

    def get_current_activity(self):
        """Get current activity."""
        return self.storage.get_current_activity()

    # ==================== Task ====================

    def create_task(self, task_id: str, activity_id: str, description: str, tags: List[str]):
        """Create task."""
        return self.storage.create_task(task_id, activity_id, description, tags)

    def complete_task(self, task_id: str, result: str):
        """Mark task complete."""
        self.storage.complete_task(task_id, result)

    def get_current_task(self):
        """Get current task."""
        return self.storage.get_current_task()

    # ==================== Record Context ====================

    def record_action(self, content: str, tags: List[str]):
        """Record action."""
        return self.storage.add_item(content, ContextCategory.ACTION, tags)

    def record_decision(self, content: str, tags: List[str]):
        """Record decision."""
        return self.storage.add_item(content, ContextCategory.DECISION, tags)

    def record_learning(self, content: str, tags: List[str]):
        """Record learning."""
        return self.storage.add_item(content, ContextCategory.LEARNING, tags)

    def record_result(self, content: str, tags: List[str]):
        """Record result."""
        return self.storage.add_item(content, ContextCategory.RESULT, tags)

    # ==================== Retrieve ====================

    def retrieve(
        self, query_tags: List[str], query_description: str, max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context."""
        max_results = max_results or self.config.max_results
        return self.retriever.retrieve(query_tags, query_description, max_results)

    # ==================== Build Prompt ====================

    def build_prompt(self, include_history: bool = True) -> str:
        """Build prompt context."""
        session = self.storage.get_current_session()
        activity = self.storage.get_current_activity()
        task = self.storage.get_current_task()

        parts = []

        if session:
            parts.append("=== SESSION ===")
            parts.append(f"Ask: {session.original_ask}")
            parts.append(f"Plan: {session.high_level_plan}")
            parts.append("")

        if activity:
            parts.append("=== ACTIVITY ===")
            parts.append(f"Description: {activity.description}")
            parts.append(f"Mode: {activity.mode}")
            parts.append(f"Component: {activity.component}")
            parts.append(f"Analysis: {activity.planner_analysis}")
            parts.append(f"Tags: {', '.join(activity.tags)}")
            parts.append("")

        if task:
            parts.append("=== TASK ===")
            parts.append(f"Description: {task.description}")
            if task.result:
                parts.append(f"Result: {task.result}")
            parts.append(f"Tags: {', '.join(task.tags)}")
            parts.append("")

        if include_history and task:
            relevant = self.retrieve(task.tags, task.description)
            if relevant:
                parts.append("=== RELEVANT HISTORY ===")
                for i, item in enumerate(relevant, 1):
                    score = item["score"]
                    parts.append(f"[{i}] (score: {score:.2f})")
                    parts.append(item["item"].content)
                parts.append("")

        return "\n".join(parts)

    def clear(self):
        """Clear all storage."""
        self.storage.clear()


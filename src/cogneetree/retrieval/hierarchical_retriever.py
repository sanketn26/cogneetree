"""Hierarchical retriever with user control over historical context."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from enum import Enum

from cogneetree.core.interfaces import ContextStorageABC, EmbeddingModelABC
from cogneetree.core.models import ContextItem


class HistoryMode(Enum):
    """Control how to include historical context."""

    CURRENT_ONLY = "current_only"
    """Only search current Session→Activity→Task hierarchy."""

    CURRENT_WEIGHTED = "current_weighted"
    """Current hierarchy prioritized; include other sessions with lower weight."""

    ALL_SESSIONS = "all_sessions"
    """Search all sessions equally; proximity only matters within current task."""

    SELECTED_SESSIONS = "selected_sessions"
    """Only search specified session IDs."""


@dataclass
class RetrievalConfig:
    """Configuration for hierarchical retrieval."""

    history_mode: HistoryMode = HistoryMode.CURRENT_WEIGHTED
    """How to include historical context."""

    history_depth_days: Optional[int] = None
    """Only include items from last N days (None = all history)."""

    include_session_ids: Optional[List[str]] = None
    """For SELECTED_SESSIONS mode: which sessions to include."""

    current_weight: float = 1.0
    """Weight multiplier for current task/activity/session items (0.0 - 2.0)."""

    sibling_weight: float = 0.7
    """Weight for sibling tasks in same activity."""

    activity_weight: float = 0.5
    """Weight for items from same activity."""

    session_weight: float = 0.3
    """Weight for items from same session."""

    history_weight: float = 0.2
    """Weight for items from other sessions (used in CURRENT_WEIGHTED mode)."""

    use_semantic: bool = True
    """Enable semantic similarity scoring."""

    return_explainability: bool = True
    """Return reasoning for each result (which session, distance, why selected)."""


class HierarchicalRetriever:
    """
    Retrieve context items considering hierarchy and history.

    User can control whether past sessions are included, making the system
    transparent and giving full agency over what context influences decisions.
    """

    def __init__(
        self,
        storage: ContextStorageABC,
        embedding_model: Optional[EmbeddingModelABC] = None,
        default_config: Optional[RetrievalConfig] = None,
    ):
        self.storage = storage
        self.embedding_model = embedding_model
        self.default_config = default_config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        current_task_id: str,
        max_results: int = 5,
        config: Optional[RetrievalConfig] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve items using hierarchical search with user-controlled history.

        Args:
            query: Search query/description
            current_task_id: Current task context
            max_results: Max items to return
            config: Retrieval configuration (uses default if None)

        Returns:
            List of results with scores and explainability info.
        """
        config = config or self.default_config

        # Get current position in hierarchy
        task = self.storage.get_task(current_task_id)
        if not task:
            return []

        activity = self.storage.get_activity(task.activity_id)
        session = self.storage.get_session(activity.session_id)

        # Gather candidates based on history mode
        candidates: List[tuple[ContextItem, str, str, float]] = []
        # (item, source_session_id, source_location, proximity_weight)

        if config.history_mode == HistoryMode.CURRENT_ONLY:
            candidates = self._gather_current_only(task, activity, session, config)

        elif config.history_mode == HistoryMode.CURRENT_WEIGHTED:
            candidates = self._gather_current_weighted(
                task, activity, session, config, current_task_id
            )

        elif config.history_mode == HistoryMode.ALL_SESSIONS:
            candidates = self._gather_all_sessions(
                task, activity, session, config, current_task_id
            )

        elif config.history_mode == HistoryMode.SELECTED_SESSIONS:
            candidates = self._gather_selected_sessions(
                task, activity, session, config, current_task_id
            )

        # Score candidates
        scored = self._score_candidates(candidates, query, config)

        # Sort by final score
        scored.sort(key=lambda x: x["final_score"], reverse=True)

        return scored[:max_results]

    def _gather_current_only(
        self, task, activity, session, config: RetrievalConfig
    ) -> List[tuple[ContextItem, str, str, float]]:
        """Only search current Session→Activity→Task hierarchy."""
        candidates = []

        # This task's items
        task_items = self.storage.get_items_by_task(task.task_id) if hasattr(
            self.storage, "get_items_by_task"
        ) else []
        for item in task_items:
            candidates.append(
                (item, session.session_id, f"Task: {task.description}", 1.0)
            )

        # Sibling tasks in same activity
        activity_tasks = self.storage.get_activity_tasks(activity.activity_id) if hasattr(
            self.storage, "get_activity_tasks"
        ) else []
        for sibling_task in activity_tasks:
            if sibling_task.task_id != task.task_id:
                sibling_items = self.storage.get_items_by_task(
                    sibling_task.task_id
                ) if hasattr(self.storage, "get_items_by_task") else []
                for item in sibling_items:
                    candidates.append(
                        (
                            item,
                            session.session_id,
                            f"Sibling Task: {sibling_task.description}",
                            config.sibling_weight,
                        )
                    )

        # Activity-level items
        activity_items = (
            self.storage.get_items_by_activity(activity.activity_id)
            if hasattr(self.storage, "get_items_by_activity")
            else []
        )
        for item in activity_items:
            candidates.append(
                (
                    item,
                    session.session_id,
                    f"Activity: {activity.description}",
                    config.activity_weight,
                )
            )

        # Session-level items
        session_items = (
            self.storage.get_items_by_session(session.session_id)
            if hasattr(self.storage, "get_items_by_session")
            else []
        )
        for item in session_items:
            candidates.append(
                (
                    item,
                    session.session_id,
                    f"Session: {session.original_ask}",
                    config.session_weight,
                )
            )

        return candidates

    def _gather_current_weighted(
        self, task, activity, session, config: RetrievalConfig, current_task_id: str
    ) -> List[tuple[ContextItem, str, str, float]]:
        """Current hierarchy prioritized; include other sessions with lower weight."""
        candidates = self._gather_current_only(task, activity, session, config)

        # Add items from other sessions with lower weight
        all_sessions = (
            self.storage.get_all_sessions()
            if hasattr(self.storage, "get_all_sessions")
            else []
        )
        for other_session in all_sessions:
            if other_session.session_id == session.session_id:
                continue

            # Filter by time if configured
            if config.history_depth_days:
                age = (datetime.now() - other_session.created_at).days
                if age > config.history_depth_days:
                    continue

            # Get items from this session
            other_items = (
                self.storage.get_items_by_session(other_session.session_id)
                if hasattr(self.storage, "get_items_by_session")
                else []
            )
            for item in other_items:
                candidates.append(
                    (
                        item,
                        other_session.session_id,
                        f"Past Session: {other_session.original_ask}",
                        config.history_weight,
                    )
                )

        return candidates

    def _gather_all_sessions(
        self, task, activity, session, config: RetrievalConfig, current_task_id: str
    ) -> List[tuple[ContextItem, str, str, float]]:
        """Search all sessions equally; proximity only matters within current task."""
        candidates = []

        # Current task items get priority
        task_items = self.storage.get_items_by_task(task.task_id) if hasattr(
            self.storage, "get_items_by_task"
        ) else []
        for item in task_items:
            candidates.append(
                (item, session.session_id, f"Task: {task.description}", 1.0)
            )

        # All items from all sessions (except current task's items)
        all_sessions = (
            self.storage.get_all_sessions()
            if hasattr(self.storage, "get_all_sessions")
            else []
        )
        for other_session in all_sessions:
            other_items = (
                self.storage.get_items_by_session(other_session.session_id)
                if hasattr(self.storage, "get_items_by_session")
                else []
            )
            for item in other_items:
                # Don't duplicate current task items
                if not hasattr(item, "task_id") or item.task_id != task.task_id:
                    candidates.append(
                        (
                            item,
                            other_session.session_id,
                            f"Session: {other_session.original_ask}",
                            config.session_weight,
                        )
                    )

        return candidates

    def _gather_selected_sessions(
        self, task, activity, session, config: RetrievalConfig, current_task_id: str
    ) -> List[tuple[ContextItem, str, str, float]]:
        """Only search specified session IDs."""
        if not config.include_session_ids:
            # Fallback to current only if no sessions specified
            return self._gather_current_only(task, activity, session, config)

        candidates = []
        selected_ids = set(config.include_session_ids)

        for session_id in selected_ids:
            other_session = self.storage.get_session(session_id)
            if not other_session:
                continue

            # Get items from this session
            session_items = (
                self.storage.get_items_by_session(session_id)
                if hasattr(self.storage, "get_items_by_session")
                else []
            )

            # Boost weight if this is the current session
            weight = (
                config.current_weight if session_id == session.session_id else config.history_weight
            )

            for item in session_items:
                candidates.append(
                    (
                        item,
                        session_id,
                        f"Session: {other_session.original_ask}",
                        weight,
                    )
                )

        return candidates

    def _score_candidates(
        self,
        candidates: List[tuple[ContextItem, str, str, float]],
        query: str,
        config: RetrievalConfig,
    ) -> List[Dict[str, Any]]:
        """Score candidates by semantic similarity × proximity weight."""
        scored = []

        for item, source_session_id, source_location, proximity_weight in candidates:
            # Semantic score
            semantic_score = 0.0
            if config.use_semantic and self.embedding_model:
                try:
                    query_emb = self.embedding_model.encode(query)
                    item_emb = self.embedding_model.encode(item.content)
                    # Cosine similarity
                    import numpy as np

                    norm1 = np.linalg.norm(query_emb)
                    norm2 = np.linalg.norm(item_emb)
                    if norm1 > 0 and norm2 > 0:
                        similarity = float(
                            np.dot(query_emb, item_emb) / (norm1 * norm2)
                        )
                        semantic_score = max(0.0, min(1.0, (similarity + 1) / 2))
                except Exception:
                    pass

            # Final score = semantic × proximity
            final_score = semantic_score * proximity_weight

            result = {
                "item": item,
                "semantic_score": semantic_score,
                "proximity_weight": proximity_weight,
                "final_score": final_score,
            }

            # Add explainability if requested
            if config.return_explainability:
                result["explanation"] = {
                    "source_session_id": source_session_id,
                    "source_location": source_location,
                    "why": f"Semantic match ({semantic_score:.2f}) × proximity ({proximity_weight:.2f})",
                }

            scored.append(result)

        return scored


# Convenience retrieval modes for common use cases
RETRIEVAL_PRESETS = {
    "fresh": RetrievalConfig(
        history_mode=HistoryMode.CURRENT_ONLY,
        return_explainability=True,
    ),
    "balanced": RetrievalConfig(
        history_mode=HistoryMode.CURRENT_WEIGHTED,
        history_depth_days=90,  # Last 90 days of history
        return_explainability=True,
    ),
    "learn_from_all": RetrievalConfig(
        history_mode=HistoryMode.ALL_SESSIONS,
        return_explainability=True,
    ),
    "debug": RetrievalConfig(
        history_mode=HistoryMode.CURRENT_WEIGHTED,
        return_explainability=True,
        history_weight=1.0,  # Equal weight to history for debugging
    ),
}

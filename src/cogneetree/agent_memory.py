"""
AgentMemory - Frictionless memory access for agents.

Provides the simplest possible interface for agents to access context:
- Automatic context awareness (knows current task)
- One-line memory recalls
- Natural language queries
- Automatic explanation building
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from cogneetree.core.interfaces import ContextStorageABC
from cogneetree.retrieval.hierarchical_retriever import (
    HierarchicalRetriever,
    RetrievalConfig,
    HistoryMode,
    RETRIEVAL_PRESETS,
)


@dataclass
class ContextResult:
    """A single context result with metadata."""

    content: str
    category: str  # "decision", "learning", "action", "result"
    source: str  # Human readable: "Project X, Task Y"
    confidence: float  # Relevance score
    explanation: Optional[Dict[str, Any]] = None


class AgentMemory:
    """
    Frictionless memory interface for agents.

    Usage in agent code:
        memory = AgentMemory(manager, current_task_id="t4")

        # One-line recall - agent gets relevant history
        context = memory.recall("JWT validation")

        # Get specific type
        decisions = memory.decisions("authentication")

        # Build context block for prompts
        prompt = memory.build_context("What should I do next?")

        # Explore related work
        related = memory.similar_tasks("microservices")
    """

    def __init__(
        self,
        storage: ContextStorageABC,
        current_task_id: str,
        retriever: Optional[HierarchicalRetriever] = None,
        default_scope: str = "balanced",
    ):
        """
        Initialize agent memory for a specific task.

        Args:
            storage: The context storage system
            current_task_id: The task agent is currently working on
            retriever: Optional custom retriever (uses default if None)
            default_scope: Default retrieval scope ("micro", "macro", "balanced")
        """
        self.storage = storage
        self.current_task_id = current_task_id
        self.retriever = retriever or HierarchicalRetriever(storage)
        self.default_scope = self._map_scope(default_scope)

        # Cache current context
        self.task = storage.get_task(current_task_id)
        self.activity = storage.get_activity(self.task.activity_id) if self.task else None
        self.session = (
            storage.get_session(self.activity.session_id) if self.activity else None
        )

    def recall(self, query: str, scope: Optional[str] = None) -> List[ContextResult]:
        """
        Agent asks: "What do I know about JWT validation?"

        Returns relevant context from history with full explanation.
        """
        config = scope and self._map_scope(scope) or self.default_scope

        results = self.retriever.retrieve(
            query=query,
            current_task_id=self.current_task_id,
            max_results=5,
            config=config,
        )

        return [self._format_result(r) for r in results]

    def decisions(self, query: str, scope: Optional[str] = None) -> List[ContextResult]:
        """Get decisions made about a topic across projects."""
        config = scope and self._map_scope(scope) or self.default_scope
        results = self.retriever.retrieve(
            query=query,
            current_task_id=self.current_task_id,
            max_results=5,
            config=config,
        )

        # Filter to decisions only
        decisions = [
            self._format_result(r)
            for r in results
            if r["item"].category.value == "decision"
        ]
        return decisions

    def learnings(self, query: str, scope: Optional[str] = None) -> List[ContextResult]:
        """Get learnings discovered about a topic."""
        config = scope and self._map_scope(scope) or self.default_scope
        results = self.retriever.retrieve(
            query=query,
            current_task_id=self.current_task_id,
            max_results=5,
            config=config,
        )

        # Filter to learnings only
        learnings = [
            self._format_result(r)
            for r in results
            if r["item"].category.value == "learning"
        ]
        return learnings

    def build_context(
        self, agent_question: str, scope: Optional[str] = None, max_items: int = 3
    ) -> str:
        """
        Build a context block for agent prompts.

        Returns formatted markdown that agent can include in its reasoning.
        """
        results = self.recall(agent_question, scope=scope)[:max_items]

        if not results:
            return ""

        lines = [
            "## Relevant Historical Context",
            "",
        ]

        for i, result in enumerate(results, 1):
            lines.append(f"### Past Decision {i}")
            lines.append(f"**From**: {result.source}")
            lines.append(f"**Confidence**: {result.confidence:.2%}")
            lines.append("")
            lines.append(f"> {result.content}")
            lines.append("")

            if result.explanation:
                lines.append(f"*Why this matters: {result.explanation['why']}*")
                lines.append("")

        return "\n".join(lines)

    def similar_tasks(self, topic: str) -> List[Dict[str, Any]]:
        """
        Find similar tasks agent has done before.

        Useful for: "Show me other auth tasks we've done"
        """
        items = self.storage.get_items_by_tags([topic.lower()])
        if not items:
            return []

        # Group by task
        tasks_map = {}
        for item in items:
            if item.parent_id:
                if item.parent_id not in tasks_map:
                    task = self.storage.get_task(item.parent_id)
                    if task:
                        tasks_map[item.parent_id] = {
                            "task_id": task.task_id,
                            "description": task.description,
                            "activity_id": task.activity_id,
                            "items": [],
                        }
                if item.parent_id in tasks_map:
                    tasks_map[item.parent_id]["items"].append(item)

        return list(tasks_map.values())

    def cite(self, result: ContextResult) -> str:
        """
        Format a citation for agent to include in reasoning.

        Usage: agent says "As we learned previously (cite: learning)"
        """
        return f"[From {result.source}: {result.category}]"

    # ==================== Private Methods ====================

    def _map_scope(self, scope: str) -> RetrievalConfig:
        """Map user-friendly scope names to configs."""
        scope_map = {
            "micro": RETRIEVAL_PRESETS["fresh"],
            "macro": RETRIEVAL_PRESETS["learn_from_all"],
            "balanced": RETRIEVAL_PRESETS["balanced"],
            "debug": RETRIEVAL_PRESETS["debug"],
        }
        return scope_map.get(scope.lower(), RETRIEVAL_PRESETS["balanced"])

    def _format_result(self, result: Dict[str, Any]) -> ContextResult:
        """Format hierarchical retriever result for agent consumption."""
        item = result["item"]
        explanation = result.get("explanation", {})

        # Build human-readable source
        source = explanation.get("source_location", "Unknown")

        return ContextResult(
            content=item.content,
            category=item.category.value,
            source=source,
            confidence=result.get("final_score", 0.0),
            explanation=explanation,
        )

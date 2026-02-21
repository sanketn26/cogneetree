"""
AgentMemory - Frictionless memory access for agents.

Provides the simplest possible interface for agents to access context:
- Automatic context awareness (knows current task)
- One-line memory recalls
- Natural language queries
- Automatic explanation building
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

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


@dataclass
class ConflictWarning:
    """Warning about two contradictory decisions returned in the same recall."""

    item_a: ContextResult
    item_b: ContextResult
    conflict_type: str  # "contradiction" | "ambiguous"
    explanation: str


@dataclass
class RecallResult:
    """Extended recall result that bundles retrieved items with any detected conflicts."""

    items: List[ContextResult] = field(default_factory=list)
    conflicts: List[ConflictWarning] = field(default_factory=list)


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
        self,
        agent_question: str,
        scope: Optional[str] = None,
        max_items: int = 3,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Build a context block for agent prompts.

        Returns formatted markdown that agent can include in its reasoning.

        Args:
            agent_question: The query to search relevant context for.
            scope: Retrieval scope ("micro", "balanced", "macro").
            max_items: Maximum number of items to include (used when max_tokens is None).
            max_tokens: Token budget. When set, items are selected greedily until the
                        budget is exhausted, regardless of max_items.
        """
        # Fetch more candidates than needed when using token budget
        fetch_limit = (max_items * 2) if max_tokens else max_items
        candidates = self.recall(agent_question, scope=scope)[:fetch_limit]

        if max_tokens is not None:
            results = []
            token_count = 0
            for item in candidates:
                item_tokens = self._estimate_tokens(item)
                if token_count + item_tokens > max_tokens:
                    break
                results.append(item)
                token_count += item_tokens
        else:
            results = candidates[:max_items]

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

    @staticmethod
    def _estimate_tokens(item: "ContextResult") -> int:
        """Estimate the token cost of a single context result.

        Uses the ~4 chars per token heuristic for English text plus a fixed
        overhead for formatting metadata (source label, confidence line, etc.).
        """
        overhead = 30  # tokens for category, source, confidence, blockquote syntax
        content_tokens = len(item.content) // 4
        return overhead + content_tokens

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

    def session_context(self) -> Dict[str, Any]:
        """Return accumulated decisions and learnings for the current session (hot path).

        This is the "always available" context that doesn't require a search.
        Decisions and learnings that propagated upward from tasks are returned
        grouped by their primary tag.

        Returns:
            Dict with "decisions" and "learnings" keys, each mapping tag → list of strings.

        Example::

            ctx = memory.session_context()
            # {"decisions": {"auth": ["Use JWT with RS256"]}, "learnings": {...}}
        """
        if not self.session:
            return {"decisions": {}, "learnings": {}}

        decisions = {
            tag: [entry.content for entry in entries]
            for tag, entries in self.session.decisions.items()
        }
        learnings = {
            tag: [entry.content for entry in entries]
            for tag, entries in self.session.learnings.items()
        }
        return {"decisions": decisions, "learnings": learnings}

    def recall_with_conflicts(
        self, query: str, scope: Optional[str] = None
    ) -> RecallResult:
        """Recall context and surface any contradictory decisions in the results.

        This is a non-breaking extension of ``recall()``. Existing code that calls
        ``recall()`` is unaffected; use this method when you want conflict awareness.

        Args:
            query: What to search for.
            scope: Retrieval scope ("micro", "balanced", "macro").

        Returns:
            RecallResult with ``items`` (same as recall()) and ``conflicts`` (list of
            ConflictWarning for any pairs of contradictory decisions).
        """
        items = self.recall(query, scope=scope)
        decisions = [r for r in items if r.category == "decision"]
        conflicts = self._detect_conflicts(decisions)
        return RecallResult(items=items, conflicts=conflicts)

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

    def _detect_conflicts(
        self, decisions: List[ContextResult]
    ) -> List[ConflictWarning]:
        """Check all pairs of decisions for contradictions.

        Uses two heuristics:
        1. **Negation pairs**: one decision contains a negation word while the other
           doesn't, and both reference the same tech keyword.
        2. **Known antonym pairs**: a curated list of common tech trade-offs where
           choosing one option contradicts the other.
        """
        conflicts: List[ConflictWarning] = []
        for i, a in enumerate(decisions):
            for b in decisions[i + 1:]:
                result = self._are_contradictory(a, b)
                if result:
                    conflict_type, explanation = result
                    conflicts.append(
                        ConflictWarning(
                            item_a=a,
                            item_b=b,
                            conflict_type=conflict_type,
                            explanation=explanation,
                        )
                    )
        return conflicts

    # Pairs of tech terms where picking one excludes the other
    _ANTONYM_PAIRS: List[Tuple[str, str]] = [
        ("hs256", "rs256"),
        ("sql", "nosql"),
        ("sync", "async"),
        ("jwt", "session"),
        ("redis", "memcached"),
        ("postgres", "mysql"),
        ("rest", "graphql"),
        ("monolith", "microservices"),
    ]

    _NEGATION_WORDS = frozenset({"not", "never", "avoid", "don't", "dont", "no", "without"})

    @staticmethod
    def _term_in_text(term: str, text: str) -> bool:
        """Return True if ``term`` appears as a whole word in ``text``.

        Pads both sides with a space to avoid substring false positives
        (e.g., "sql" must not match inside "nosql").
        """
        return f" {term} " in f" {text} "

    def _are_contradictory(
        self, a: ContextResult, b: ContextResult
    ) -> Optional[Tuple[str, str]]:
        """Return (conflict_type, explanation) if a and b contradict, else None."""
        a_lower = a.content.lower()
        b_lower = b.content.lower()

        # Check known antonym pairs using whole-word matching
        for term_x, term_y in self._ANTONYM_PAIRS:
            a_has_x = self._term_in_text(term_x, a_lower)
            a_has_y = self._term_in_text(term_y, a_lower)
            b_has_x = self._term_in_text(term_x, b_lower)
            b_has_y = self._term_in_text(term_y, b_lower)
            if (a_has_x and b_has_y and not b_has_x) or (a_has_y and b_has_x and not a_has_x):
                return (
                    "contradiction",
                    f"'{term_x}' vs '{term_y}': both options appear across decisions.",
                )

        # Check negation flip: one item negates a keyword the other affirms
        a_words = set(a_lower.split())
        b_words = set(b_lower.split())
        a_negated = bool(a_words & self._NEGATION_WORDS)
        b_negated = bool(b_words & self._NEGATION_WORDS)
        if a_negated != b_negated:
            shared_keywords = (a_words - self._NEGATION_WORDS) & (b_words - self._NEGATION_WORDS)
            meaningful = {w for w in shared_keywords if len(w) > 3}
            if meaningful:
                return (
                    "ambiguous",
                    f"One decision negates while the other affirms shared keywords: {meaningful}.",
                )

        return None

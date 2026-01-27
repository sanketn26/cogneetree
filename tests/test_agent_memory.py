"""Tests for AgentMemory - frictionless agent context access."""

import pytest
from cogneetree.storage.in_memory_storage import InMemoryStorage
from cogneetree.core.models import ContextCategory
from cogneetree.agent_memory import AgentMemory, ContextResult


@pytest.fixture
def memory_setup():
    """Setup storage with context for agent testing."""
    storage = InMemoryStorage()

    # Project 1: OAuth2 (past)
    storage.create_session("s1", "Implement OAuth2", "Auth system")
    storage.create_activity("a1", "s1", "OAuth2 flow", ["auth", "oauth2"], "learner", "core", "analysis")
    storage.create_task("t1", "a1", "Learn grant types", ["oauth2"])

    storage.add_item(
        "Use auth code grant for web apps - most secure option",
        ContextCategory.DECISION,
        ["oauth2", "decision"],
        parent_id="t1",
    )
    storage.add_item(
        "OAuth2 has 4 main grant types: auth code, implicit, client creds, resource owner",
        ContextCategory.LEARNING,
        ["oauth2", "learning"],
        parent_id="t1",
    )

    # Project 2: JWT (past)
    storage.create_session("s2", "Implement JWT", "Token system")
    storage.create_activity("a2", "s2", "JWT tokens", ["jwt", "auth"], "coder", "core", "analysis")
    storage.create_task("t2", "a2", "JWT validation", ["jwt"])

    storage.add_item(
        "Use HS256 for symmetric signing",
        ContextCategory.DECISION,
        ["jwt", "signing"],
        parent_id="t2",
    )
    storage.add_item(
        "JWT validation must happen before request processing",
        ContextCategory.LEARNING,
        ["jwt", "validation"],
        parent_id="t2",
    )
    storage.add_item(
        "Added middleware to validate token signature",
        ContextCategory.ACTION,
        ["jwt", "middleware"],
        parent_id="t2",
    )

    # Current project: Microservices JWT (now)
    storage.create_session("s3", "Microservices auth", "Multi-service JWT")
    storage.create_activity(
        "a3", "s3", "Inter-service JWT", ["jwt", "microservices"], "coder", "core", "analysis"
    )
    storage.create_task("t3", "a3", "Add JWT validation middleware", ["jwt", "middleware"])

    return storage


class TestAgentMemoryBasics:
    """Test basic memory recall operations."""

    def test_memory_initialization(self, memory_setup):
        """Agent memory initializes with current task context."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        assert memory.current_task_id == "t3"
        assert memory.task is not None
        assert memory.task.task_id == "t3"
        assert memory.activity is not None
        assert memory.session is not None

    def test_recall_returns_context_results(self, memory_setup):
        """recall() returns formatted ContextResult objects."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        results = memory.recall("JWT validation")

        assert len(results) > 0
        assert all(isinstance(r, ContextResult) for r in results)
        assert all(hasattr(r, "content") for r in results)
        assert all(hasattr(r, "category") for r in results)
        assert all(hasattr(r, "source") for r in results)
        assert all(hasattr(r, "confidence") for r in results)

    def test_recall_finds_jwt_context(self, memory_setup):
        """recall() finds relevant JWT context."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        results = memory.recall("JWT validation")

        # Should find JWT validation items
        jwt_results = [r for r in results if "jwt" in r.content.lower() or "jwt" in r.source.lower()]
        assert len(jwt_results) > 0

    def test_recall_with_scope_micro(self, memory_setup):
        """recall() with micro scope only searches current task."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        results = memory.recall("JWT", scope="micro")

        # Current task t3 has no items yet, so micro scope returns empty
        assert len(results) == 0

    def test_recall_with_scope_balanced(self, memory_setup):
        """recall() with balanced scope searches current + history."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        results = memory.recall("JWT", scope="balanced")

        # Should find JWT items from past sessions
        assert len(results) > 0

    def test_recall_with_scope_macro(self, memory_setup):
        """recall() with macro scope searches all sessions equally."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        results = memory.recall("JWT", scope="macro")

        # Should find items from all sessions
        assert len(results) > 0


class TestAgentMemoryFiltering:
    """Test filtering by category."""

    def test_decisions_filter(self, memory_setup):
        """decisions() returns only decision items."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        results = memory.decisions("JWT")

        assert len(results) > 0
        assert all(r.category == "decision" for r in results)

    def test_learnings_filter(self, memory_setup):
        """learnings() returns only learning items."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        results = memory.learnings("JWT")

        assert len(results) > 0
        assert all(r.category == "learning" for r in results)

    def test_decisions_finds_jwt_decision(self, memory_setup):
        """decisions() finds the JWT signing decision from past."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        decisions = memory.decisions("signing")

        assert any("HS256" in r.content for r in decisions)

    def test_learnings_finds_jwt_learning(self, memory_setup):
        """learnings() finds JWT validation learning from past."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        learnings = memory.learnings("validation")

        assert any("before request" in r.content for r in learnings)


class TestAgentMemoryPromptBuilding:
    """Test building context for prompts."""

    def test_build_context_returns_markdown(self, memory_setup):
        """build_context() returns formatted markdown."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        context = memory.build_context("JWT validation")

        assert isinstance(context, str)
        assert "Relevant Historical Context" in context
        assert "Past Decision" in context or len(context) == 0  # May be empty if no results

    def test_build_context_includes_source(self, memory_setup):
        """build_context() includes source information."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        context = memory.build_context("JWT")

        # Should include some source info if results found
        if context:
            assert "From:" in context or "Decision" in context

    def test_build_context_respects_max_items(self, memory_setup):
        """build_context() limits results to max_items."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        context = memory.build_context("JWT", max_items=1)

        # Count "Past Decision" headers
        decision_count = context.count("### Past Decision")
        assert decision_count <= 1


class TestAgentMemorySimilarTasks:
    """Test finding similar past work."""

    def test_similar_tasks_finds_auth_tasks(self, memory_setup):
        """similar_tasks() finds tasks with matching tags."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        similar = memory.similar_tasks("jwt")

        # Should find multiple JWT tasks from different sessions
        assert len(similar) > 0

    def test_similar_tasks_includes_metadata(self, memory_setup):
        """similar_tasks() returns task metadata."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        similar = memory.similar_tasks("jwt")

        assert all("task_id" in task for task in similar)
        assert all("description" in task for task in similar)
        assert all("items" in task for task in similar)

    def test_similar_tasks_by_auth_tag(self, memory_setup):
        """similar_tasks() finds auth-related tasks."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        similar = memory.similar_tasks("oauth2")

        # Should find tasks with oauth2 tag
        assert len(similar) >= 1
        assert similar[0]["description"] == "Learn grant types"


class TestAgentMemoryCitation:
    """Test citation formatting."""

    def test_cite_formats_result(self, memory_setup):
        """cite() returns properly formatted citation."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        results = memory.recall("JWT")
        if results:
            citation = memory.cite(results[0])

            assert isinstance(citation, str)
            assert "[From" in citation
            assert "]" in citation

    def test_cite_includes_category(self, memory_setup):
        """cite() includes item category in citation."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        results = memory.recall("JWT")
        if results:
            citation = memory.cite(results[0])

            # Citation should include category
            assert any(cat in citation for cat in ["decision", "learning", "action"])


class TestAgentMemoryIntegration:
    """Test realistic agent workflows."""

    def test_agent_workflow_recall_then_record(self, memory_setup):
        """Agent recalls context then records its work."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        # Agent recalls what to do
        context = memory.recall("JWT validation")
        assert len(context) > 0

        # Agent records a decision (simulated)
        memory.storage.add_item(
            "Followed past decision: use HS256",
            ContextCategory.DECISION,
            ["jwt", "signing"],
            parent_id="t3",
        )

        # Verify recording
        task_items = memory.storage.get_items_by_task("t3")
        assert len(task_items) == 1
        assert "HS256" in task_items[0].content

    def test_agent_explores_patterns(self, memory_setup):
        """Agent explores patterns across projects."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        # Agent explores similar work
        similar = memory.similar_tasks("jwt")
        assert len(similar) > 0

        # Agent gets macro view
        all_jwt = memory.recall("JWT", scope="macro")
        assert len(all_jwt) > 0

    def test_agent_builds_informed_prompt(self, memory_setup):
        """Agent builds a prompt informed by history."""
        memory = AgentMemory(memory_setup, current_task_id="t3")

        # Agent builds context for LLM
        context_block = memory.build_context("JWT validation middleware")

        # Simulated agent prompt
        prompt = f"""Implement JWT validation middleware.

{context_block}

Based on our history, what's the best approach?"""

        # Should have context if items were found
        if context_block:
            assert "Past Decision" in prompt or "Relevant" in prompt

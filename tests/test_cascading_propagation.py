"""Tests for Phase 0.2: Cascading Context Propagation."""

import pytest
from cogneetree.core.context_manager import ContextManager
from cogneetree.core.models import DecisionEntry
from cogneetree.config import Config
from cogneetree.agent_memory import AgentMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_manager() -> ContextManager:
    """Return a ContextManager with a session/activity/task in context."""
    manager = ContextManager(Config.default())
    # original_ask intentionally contains "jwt" so tag-based propagation reaches session level
    manager.create_session("s1", "Implement JWT authentication system", "JWT-based auth")
    manager.create_activity(
        "a1", "s1", "Implement JWT authentication", ["jwt", "auth"], "coder", "core", "analysis"
    )
    manager.create_task("t1", "a1", "Add JWT middleware", ["jwt", "auth"])
    return manager


# ---------------------------------------------------------------------------
# Decision propagation
# ---------------------------------------------------------------------------

class TestDecisionPropagation:

    def test_decision_propagates_to_activity(self):
        """record_decision() stores a DecisionEntry in the current activity."""
        manager = _setup_manager()
        manager.record_decision("Use JWT with RS256 for token signing", ["jwt", "auth"])

        activity = manager.storage.get_current_activity()
        assert activity is not None
        assert "jwt" in activity.decisions
        entries = activity.decisions["jwt"]
        assert len(entries) == 1
        assert isinstance(entries[0], DecisionEntry)
        assert "RS256" in entries[0].content

    def test_decision_propagates_to_session(self):
        """record_decision() stores a DecisionEntry in the current session."""
        manager = _setup_manager()
        manager.record_decision("Use JWT with RS256 for token signing", ["jwt", "auth"])

        session = manager.storage.get_current_session()
        assert session is not None
        assert "jwt" in session.decisions
        entries = session.decisions["jwt"]
        assert len(entries) == 1
        assert "RS256" in entries[0].content

    def test_action_does_not_propagate(self):
        """record_action() stays at task level — no propagation to activity/session."""
        manager = _setup_manager()
        manager.record_action("Read JWT RFC 7519", ["jwt"])

        activity = manager.storage.get_current_activity()
        session = manager.storage.get_current_session()
        assert activity.decisions == {}
        assert activity.learnings == {}
        assert session.decisions == {}
        assert session.learnings == {}

    def test_result_does_not_propagate(self):
        """record_result() stays at task level — no propagation."""
        manager = _setup_manager()
        manager.record_result("JWT middleware implemented", ["jwt"])

        activity = manager.storage.get_current_activity()
        assert activity.decisions == {}
        assert activity.learnings == {}

    def test_primary_tag_used_as_bucket_key(self):
        """The first tag in the list is used as the bucket key."""
        manager = _setup_manager()
        manager.record_decision("Use RS256", ["auth", "jwt", "signing"])

        activity = manager.storage.get_current_activity()
        assert "auth" in activity.decisions  # first tag

    def test_no_tags_uses_general_bucket(self):
        """When tags list is empty, promote=True is needed; 'general' bucket is used."""
        manager = _setup_manager()
        # Empty tags cannot determine relevance, so promote=True is required
        manager.record_decision("Use RS256 algorithm", [], promote=True)

        activity = manager.storage.get_current_activity()
        assert "general" in activity.decisions


# ---------------------------------------------------------------------------
# Learning propagation
# ---------------------------------------------------------------------------

class TestLearningPropagation:

    def test_learning_propagates_to_activity(self):
        """record_learning() stores a DecisionEntry in the current activity.learnings."""
        manager = _setup_manager()
        manager.record_learning("RS256 requires public key distribution", ["jwt", "auth"])

        activity = manager.storage.get_current_activity()
        assert "jwt" in activity.learnings
        assert "RS256" in activity.learnings["jwt"][0].content

    def test_learning_propagates_to_session(self):
        """record_learning() stores a DecisionEntry in the current session.learnings."""
        manager = _setup_manager()
        manager.record_learning("RS256 requires public key distribution", ["jwt", "auth"])

        session = manager.storage.get_current_session()
        assert "jwt" in session.learnings
        assert "RS256" in session.learnings["jwt"][0].content


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:

    def test_deduplication_increments_count(self):
        """A near-duplicate decision increments count instead of creating a new entry."""
        manager = _setup_manager()
        # Use the same words in a different order so Jaccard similarity > 0.8
        manager.record_decision("Use JWT RS256 signing for auth", ["jwt"])
        manager.record_decision("JWT RS256 signing use for auth", ["jwt"])  # rearranged

        activity = manager.storage.get_current_activity()
        entries = activity.decisions.get("jwt", [])
        # Should have only 1 entry with count == 2 (same words → Jaccard = 1.0)
        assert len(entries) == 1
        assert entries[0].count == 2

    def test_non_duplicate_creates_new_entry(self):
        """Two clearly different decisions both get stored."""
        manager = _setup_manager()
        manager.record_decision("Use RS256 for JWT signing", ["jwt"])
        manager.record_decision("Set token expiry to 24 hours", ["jwt"])

        activity = manager.storage.get_current_activity()
        entries = activity.decisions.get("jwt", [])
        assert len(entries) == 2

    def test_deduplication_also_applies_to_session(self):
        """Deduplication works the same at session level."""
        manager = _setup_manager()
        manager.record_decision("Use RS256 for JWT signing auth", ["jwt"])
        manager.record_decision("RS256 for JWT signing auth use", ["jwt"])  # rearranged

        session = manager.storage.get_current_session()
        entries = session.decisions.get("jwt", [])
        assert len(entries) == 1
        assert entries[0].count == 2


# ---------------------------------------------------------------------------
# Promote flag (forced propagation)
# ---------------------------------------------------------------------------

class TestPromoteFlag:

    def test_promote_forces_propagation_regardless_of_relevance(self):
        """promote=True bypasses relevance gating."""
        manager = ContextManager(Config.default())
        manager.create_session("s1", "Build ticketing system", "REST API based")
        manager.create_activity(
            "a1", "s1", "Frontend styling", ["css", "ui"], "designer", "frontend", "analysis"
        )
        manager.create_task("t1", "a1", "Style ticket list", ["css"])

        # This decision is clearly unrelated to "Frontend styling" activity,
        # but promote=True should force it to propagate.
        manager.record_decision("Use PostgreSQL for persistence", ["db"], promote=True)

        activity = manager.storage.get_current_activity()
        session = manager.storage.get_current_session()
        assert "db" in activity.decisions
        assert "db" in session.decisions

    def test_without_promote_irrelevant_stays_local(self):
        """Without promote=True, irrelevant decisions stay at task level only."""
        manager = ContextManager(Config.default())
        manager.create_session("s1", "Build ticketing system", "REST API based")
        manager.create_activity(
            "a1", "s1", "Frontend styling", ["css", "ui"], "designer", "frontend", "analysis"
        )
        manager.create_task("t1", "a1", "Style ticket list", ["css"])

        # "PostgreSQL for persistence" has almost no word overlap with "Frontend styling"
        manager.record_decision("PostgreSQL for persistence layer", ["db"])

        activity = manager.storage.get_current_activity()
        # With threshold=0.3 and very low word overlap, should NOT propagate
        assert "db" not in activity.decisions or activity.decisions.get("db", []) == []


# ---------------------------------------------------------------------------
# Word overlap helper
# ---------------------------------------------------------------------------

class TestWordOverlap:

    def test_identical_strings_overlap_is_1(self):
        overlap = ContextManager._word_overlap("use jwt auth", "use jwt auth")
        assert overlap == 1.0

    def test_completely_different_overlap_is_0(self):
        overlap = ContextManager._word_overlap("jwt token signing", "css flexbox grid")
        assert overlap == 0.0

    def test_partial_overlap(self):
        overlap = ContextManager._word_overlap("use jwt signing", "jwt authentication signing")
        # Intersection: {jwt, signing} = 2; Union = {use, jwt, signing, authentication} = 4
        assert abs(overlap - 0.5) < 1e-9

    def test_empty_string_returns_0(self):
        assert ContextManager._word_overlap("", "anything") == 0.0
        assert ContextManager._word_overlap("anything", "") == 0.0


# ---------------------------------------------------------------------------
# Session context hot path (AgentMemory)
# ---------------------------------------------------------------------------

class TestSessionContextHotPath:

    def test_session_context_returns_accumulated_decisions(self):
        """AgentMemory.session_context() returns propagated decisions."""
        manager = _setup_manager()
        manager.record_decision("Use JWT with RS256 for token signing", ["jwt", "auth"])

        memory = AgentMemory(manager.storage, current_task_id="t1")
        ctx = memory.session_context()

        assert "decisions" in ctx
        assert "jwt" in ctx["decisions"]
        assert any("RS256" in d for d in ctx["decisions"]["jwt"])

    def test_session_context_returns_accumulated_learnings(self):
        """AgentMemory.session_context() returns propagated learnings."""
        manager = _setup_manager()
        manager.record_learning("RS256 requires public key distribution", ["jwt"])

        memory = AgentMemory(manager.storage, current_task_id="t1")
        ctx = memory.session_context()

        assert "learnings" in ctx
        assert "jwt" in ctx["learnings"]

    def test_session_context_empty_when_no_propagation(self):
        """session_context() returns empty dicts when only actions/results recorded."""
        manager = _setup_manager()
        manager.record_action("Read RFC 7519", ["jwt"])

        memory = AgentMemory(manager.storage, current_task_id="t1")
        ctx = memory.session_context()

        assert ctx["decisions"] == {}
        assert ctx["learnings"] == {}

    def test_session_context_groups_by_tag(self):
        """Multiple decisions under the same tag are grouped together."""
        manager = _setup_manager()
        manager.record_decision("Use RS256 for signing", ["jwt"])
        manager.record_decision("Set JWT expiry to 24h", ["jwt"])

        memory = AgentMemory(manager.storage, current_task_id="t1")
        ctx = memory.session_context()

        # Both decisions are under the "jwt" tag
        assert len(ctx["decisions"]["jwt"]) == 2

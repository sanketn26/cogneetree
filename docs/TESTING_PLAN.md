# Cogneetree Testing Plan

## Overview

This document outlines a comprehensive testing strategy for Cogneetree, covering unit tests, integration tests, performance benchmarks, and effectiveness validation. The goal is to ensure the library reliably builds and maintains long-term memory for AI agents.

**Target Audience**: QA engineers, developers, and maintainers

---

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Structure](#test-structure)
3. [Unit Tests](#unit-tests)
4. [Integration Tests](#integration-tests)
5. [Performance Tests](#performance-tests)
6. [Effectiveness Tests](#effectiveness-tests)
7. [Storage Backend Tests](#storage-backend-tests)
8. [Retrieval Strategy Tests](#retrieval-strategy-tests)
9. [Agent Integration Tests](#agent-integration-tests)
10. [Test Execution](#test-execution)
11. [Metrics and Reporting](#metrics-and-reporting)

---

## Testing Philosophy

### Core Principles

1. **Reliability**: Tests must catch real bugs, not false positives
2. **Clarity**: Tests document expected behavior
3. **Speed**: Fast feedback for developers
4. **Coverage**: Critical paths covered; 80%+ line coverage target
5. **Real-world**: Tests mirror actual usage patterns

### Coverage Goals

| Component | Target Coverage |
|-----------|-----------------|
| Core models | 95%+ |
| Context manager | 90%+ |
| Retrieval logic | 90%+ |
| Storage backends | 85%+ |
| Utilities | 80%+ |
| **Overall** | **85%+** |

---

## Test Structure

### Directory Organization

```
tests/
├── __init__.py
├── conftest.py                          # Shared fixtures
├── unit/
│   ├── test_models.py
│   ├── test_context_manager.py
│   ├── test_hierarchical_retriever.py
│   └── test_tag_normalization.py
├── integration/
│   ├── test_context_workflow.py
│   ├── test_retrieval_workflow.py
│   ├── test_semantic_retrieval.py
│   └── test_multi_session_learning.py
├── storage/
│   ├── test_in_memory_storage.py
│   ├── test_file_storage.py
│   ├── test_sql_storage.py
│   └── test_redis_storage.py
├── performance/
│   ├── test_retrieval_performance.py
│   ├── test_storage_performance.py
│   └── test_scaling.py
├── effectiveness/
│   ├── test_semantic_accuracy.py
│   ├── test_proximity_weighting.py
│   ├── test_context_quality.py
│   └── test_agent_learning.py
└── fixtures/
    ├── sample_data.py
    ├── knowledge_bases.py
    └── test_scenarios.py
```

### Test Naming Convention

```
test_[component]_[scenario]_[expected_outcome]

Examples:
- test_hierarchical_retriever_current_weighted_prioritizes_current_task
- test_context_manager_create_session_sets_current_session
- test_retrieval_semantic_similarity_above_threshold
```

---

## Unit Tests

### 1. Data Models Tests

**File**: `tests/unit/test_models.py`

```python
"""Tests for core data models."""

import pytest
from datetime import datetime
from cogneetree.core.models import (
    ContextItem, ContextCategory, Session, Activity, Task
)


class TestContextItem:
    """Test ContextItem model."""

    def test_context_item_creation(self):
        """Test basic ContextItem creation."""
        item = ContextItem(
            content="Test learning",
            category=ContextCategory.LEARNING,
            tags=["test", "learning"]
        )
        assert item.content == "Test learning"
        assert item.category == ContextCategory.LEARNING
        assert item.tags == ["test", "learning"]
        assert item.timestamp is not None
        assert isinstance(item.timestamp, datetime)

    def test_context_item_with_metadata(self):
        """Test ContextItem with custom metadata."""
        metadata = {"reviewed": True, "priority": "high"}
        item = ContextItem(
            content="Important decision",
            category=ContextCategory.DECISION,
            metadata=metadata
        )
        assert item.metadata["reviewed"] is True
        assert item.metadata["priority"] == "high"

    def test_context_item_parent_tracking(self):
        """Test parent_id tracking for hierarchy."""
        item = ContextItem(
            content="Task learning",
            category=ContextCategory.LEARNING,
            parent_id="task_123"
        )
        assert item.parent_id == "task_123"

    def test_all_context_categories(self):
        """Test all category types."""
        categories = [
            ContextCategory.SESSION,
            ContextCategory.ACTIVITY,
            ContextCategory.TASK,
            ContextCategory.ACTION,
            ContextCategory.DECISION,
            ContextCategory.LEARNING,
            ContextCategory.RESULT
        ]
        for cat in categories:
            item = ContextItem(content="test", category=cat)
            assert item.category == cat


class TestSession:
    """Test Session model."""

    def test_session_creation(self):
        """Test basic Session creation."""
        session = Session(
            session_id="sess_001",
            original_ask="Implement auth",
            high_level_plan="JWT-based"
        )
        assert session.session_id == "sess_001"
        assert session.original_ask == "Implement auth"
        assert session.high_level_plan == "JWT-based"
        assert session.created_at is not None


class TestActivity:
    """Test Activity model."""

    def test_activity_creation(self):
        """Test basic Activity creation."""
        activity = Activity(
            activity_id="act_001",
            session_id="sess_001",
            description="Research JWT",
            tags=["jwt", "auth"],
            mode="learner",
            component="core",
            planner_analysis="Need understanding"
        )
        assert activity.activity_id == "act_001"
        assert activity.session_id == "sess_001"
        assert "jwt" in activity.tags


class TestTask:
    """Test Task model."""

    def test_task_creation(self):
        """Test basic Task creation."""
        task = Task(
            task_id="task_001",
            activity_id="act_001",
            description="Learn JWT structure",
            tags=["jwt", "structure"]
        )
        assert task.task_id == "task_001"
        assert task.activity_id == "act_001"

    def test_task_with_result(self):
        """Test Task with result."""
        task = Task(
            task_id="task_001",
            activity_id="act_001",
            description="Learn JWT",
            tags=["jwt"],
            result="Understood JWT structure"
        )
        assert task.result == "Understood JWT structure"
```

### 2. Context Manager Tests

**File**: `tests/unit/test_context_manager.py`

```python
"""Tests for ContextManager."""

import pytest
from cogneetree import ContextManager, ContextCategory


class TestContextManagerHierarchy:
    """Test context creation and hierarchy."""

    def test_create_session(self):
        """Test session creation."""
        manager = ContextManager()
        session = manager.create_session(
            "sess_001",
            "Implement auth",
            "JWT-based"
        )
        assert session.session_id == "sess_001"
        assert manager.get_current_session_id() == "sess_001"

    def test_create_activity(self):
        """Test activity creation under session."""
        manager = ContextManager()
        manager.create_session("sess_001", "Implement auth", "JWT")
        activity = manager.create_activity(
            "act_001",
            "sess_001",
            "Research JWT",
            ["jwt"],
            "learner",
            "core",
            "Need understanding"
        )
        assert activity.session_id == "sess_001"
        assert manager.get_current_activity_id() == "act_001"

    def test_create_task(self):
        """Test task creation under activity."""
        manager = ContextManager()
        manager.create_session("sess_001", "Auth", "JWT")
        manager.create_activity("act_001", "sess_001", "Research", ["jwt"], "learner", "core", "...")
        task = manager.create_task("task_001", "act_001", "Learn JWT", ["jwt"])
        assert task.activity_id == "act_001"
        assert manager.get_current_task_id() == "task_001"

    def test_context_stack_isolation(self):
        """Test that context stack maintains isolation."""
        manager = ContextManager()
        manager.create_session("sess_001", "Auth", "JWT")
        manager.create_session("sess_002", "API", "REST")

        # Current session should be sess_002
        assert manager.get_current_session_id() == "sess_002"


class TestContextManagerRecording:
    """Test knowledge recording."""

    def test_record_learning(self):
        """Test recording a learning."""
        manager = ContextManager()
        manager.create_session("sess_001", "Auth", "JWT")
        manager.create_activity("act_001", "sess_001", "Research", ["jwt"], "learner", "core", "...")
        manager.create_task("task_001", "act_001", "Learn JWT", ["jwt"])

        manager.record_learning(
            "JWT has three parts",
            tags=["jwt", "structure"]
        )

        # Verify item was recorded
        items = manager.storage.get_items_by_task("task_001")
        assert len(items) == 1
        assert items[0].content == "JWT has three parts"
        assert items[0].category == ContextCategory.LEARNING

    def test_record_decision(self):
        """Test recording a decision."""
        manager = ContextManager()
        manager.create_session("sess_001", "Auth", "JWT")
        manager.create_activity("act_001", "sess_001", "Research", ["jwt"], "learner", "core", "...")
        manager.create_task("task_001", "act_001", "Implement", ["jwt"])

        manager.record_decision(
            "Use HS256",
            tags=["jwt", "algorithm"]
        )

        items = manager.storage.get_items_by_task("task_001")
        assert items[0].category == ContextCategory.DECISION

    def test_record_action(self):
        """Test recording an action."""
        manager = ContextManager()
        manager.create_session("sess_001", "Auth", "JWT")
        manager.create_activity("act_001", "sess_001", "Research", ["jwt"], "learner", "core", "...")
        manager.create_task("task_001", "act_001", "Research", ["jwt"])

        manager.record_action(
            "Read RFC 7519",
            tags=["jwt", "research"]
        )

        items = manager.storage.get_items_by_task("task_001")
        assert items[0].category == ContextCategory.ACTION

    def test_record_result(self):
        """Test recording a result."""
        manager = ContextManager()
        manager.create_session("sess_001", "Auth", "JWT")
        manager.create_activity("act_001", "sess_001", "Research", ["jwt"], "learner", "core", "...")
        manager.create_task("task_001", "act_001", "Research", ["jwt"])

        manager.record_result(
            "Complete understanding of JWT",
            tags=["jwt", "complete"]
        )

        items = manager.storage.get_items_by_task("task_001")
        assert items[0].category == ContextCategory.RESULT

    def test_parent_id_tracking(self):
        """Test that items track parent task correctly."""
        manager = ContextManager()
        manager.create_session("sess_001", "Auth", "JWT")
        manager.create_activity("act_001", "sess_001", "Research", ["jwt"], "learner", "core", "...")
        manager.create_task("task_001", "act_001", "Learn", ["jwt"])

        manager.record_learning("JWT info", tags=["jwt"])

        items = manager.storage.get_items_by_task("task_001")
        assert items[0].parent_id == "task_001"
```

### 3. Tag Normalization Tests

**File**: `tests/unit/test_tag_normalization.py`

```python
"""Tests for tag normalization utilities."""

import pytest
from cogneetree.retrieval.tag_normalization import (
    normalize_tag, normalize_tags, get_tag_variations
)


class TestTagNormalization:
    """Test tag normalization."""

    def test_normalize_tag_lowercase(self):
        """Test tag lowercasing."""
        assert normalize_tag("JWT") == "jwt"
        assert normalize_tag("OAuth2") == "oauth2"

    def test_normalize_tag_whitespace(self):
        """Test tag whitespace handling."""
        assert normalize_tag("  jwt  ") == "jwt"
        assert normalize_tag("jwt auth") == "jwt_auth"

    def test_normalize_tag_special_chars(self):
        """Test special character handling."""
        assert normalize_tag("jwt-auth") == "jwt_auth"
        assert normalize_tag("oauth/2") == "oauth_2"

    def test_normalize_tags_batch(self):
        """Test normalizing multiple tags."""
        tags = ["JWT", "  Auth  ", "OAuth2"]
        normalized = normalize_tags(tags)
        assert "jwt" in normalized
        assert "auth" in normalized
        assert "oauth2" in normalized

    def test_tag_variations(self):
        """Test getting tag variations for search."""
        variations = get_tag_variations("jwt")
        assert "jwt" in variations
        # Should include variations like "jwt_auth", "jwt_token", etc.


class TestTagConsistency:
    """Test tag consistency across system."""

    def test_tag_case_insensitivity(self):
        """Test that tags work case-insensitively."""
        manager = ContextManager()
        manager.create_session("s1", "Test", "Test")
        manager.create_activity("a1", "s1", "Test", ["JWT"], "learner", "core", "...")
        manager.create_task("t1", "a1", "Test", ["JWT"])

        # Record with different casing
        manager.record_learning("JWT info", tags=["jwt"])
        manager.record_learning("More info", tags=["JWT"])

        # Both should be findable
        items_lower = manager.storage.get_items_by_tags("jwt")
        items_upper = manager.storage.get_items_by_tags("JWT")
        assert len(items_lower) == 2
        assert len(items_upper) == 2
```

---

## Integration Tests

### 1. Full Workflow Tests

**File**: `tests/integration/test_context_workflow.py`

```python
"""Integration tests for complete workflows."""

import pytest
from cogneetree import ContextManager, AgentMemory


class TestCompleteWorkflow:
    """Test end-to-end workflows."""

    def test_oauth_implementation_workflow(self):
        """Test complete OAuth implementation workflow."""
        manager = ContextManager()

        # Session 1: Research
        session = manager.create_session(
            "oauth_impl",
            "Implement OAuth2",
            "Research → Design → Implement"
        )
        activity = manager.create_activity(
            "oauth_research",
            "oauth_impl",
            "Research OAuth2",
            ["oauth2", "research"],
            "learner",
            "core",
            "..."
        )
        task = manager.create_task(
            "task_research",
            "oauth_research",
            "Learn OAuth2 flows",
            ["oauth2", "flows"]
        )

        # Record learnings
        manager.record_learning(
            "Authorization code flow for web apps",
            tags=["oauth2", "flow"]
        )
        manager.record_learning(
            "Use PKCE for public clients",
            tags=["oauth2", "security"]
        )

        # Create implementation task
        task2 = manager.create_task(
            "task_impl",
            "oauth_research",
            "Implement OAuth2",
            ["oauth2", "implementation"]
        )

        # Retrieve context
        memory = AgentMemory(manager.storage, current_task_id="task_impl")
        context = memory.recall("OAuth2 implementation best practices")

        # Verify we found research learnings
        assert len(context) > 0
        assert any("PKCE" in item.content for item in context)

    def test_multi_activity_workflow(self):
        """Test workflow spanning multiple activities."""
        manager = ContextManager()
        manager.create_session("s1", "Build API", "Complete API")

        # Activity 1: Authentication
        manager.create_activity(
            "act_auth",
            "s1",
            "Authentication",
            ["auth"],
            "builder",
            "core",
            "..."
        )
        manager.create_task("t_auth", "act_auth", "Implement JWT", ["jwt"])
        manager.record_learning("JWT structure", tags=["jwt"])

        # Activity 2: Authorization
        manager.create_activity(
            "act_authz",
            "s1",
            "Authorization",
            ["authz"],
            "builder",
            "core",
            "..."
        )
        manager.create_task("t_authz", "act_authz", "Implement RBAC", ["rbac"])

        # Authorization task should find JWT learning from auth activity
        memory = AgentMemory(manager.storage, current_task_id="t_authz")
        context = memory.recall("Token validation for authorization")

        # Should find JWT learning from different activity
        assert any("JWT" in item.content for item in context)
```

### 2. Retrieval Workflow Tests

**File**: `tests/integration/test_retrieval_workflow.py`

```python
"""Tests for retrieval workflows."""

import pytest
from cogneetree import ContextManager, AgentMemory, HistoryMode


class TestRetrievalScopes:
    """Test different retrieval scopes."""

    def test_current_only_scope(self):
        """Test CURRENT_ONLY scope returns only current task items."""
        manager = ContextManager()
        manager.create_session("s1", "Test", "Test")
        manager.create_activity("a1", "s1", "Activity", ["test"], "learner", "core", "...")

        # Task 1: Record learning
        manager.create_task("t1", "a1", "Task 1", ["tag1"])
        manager.record_learning("Learning from t1", tags=["tag1"])

        # Task 2: Different task
        manager.create_task("t2", "a1", "Task 2", ["tag2"])

        # Query with CURRENT_ONLY
        memory = AgentMemory(manager.storage, current_task_id="t2")
        context = memory.recall("Learning from t1", scope="micro")

        # Should not find t1's learning (different task)
        assert not any("Learning from t1" in item.content for item in context)

    def test_current_weighted_scope(self):
        """Test CURRENT_WEIGHTED includes current + history."""
        manager = ContextManager()
        manager.create_session("s1", "Test", "Test")
        manager.create_activity("a1", "s1", "Activity", ["test"], "learner", "core", "...")

        # Task 1: Record learning
        manager.create_task("t1", "a1", "Task 1", ["jwt"])
        manager.record_learning("JWT fundamentals", tags=["jwt"])

        # Task 2: Same activity
        manager.create_task("t2", "a1", "Task 2", ["jwt", "implementation"])

        # Query with CURRENT_WEIGHTED
        memory = AgentMemory(manager.storage, current_task_id="t2")
        context = memory.recall("JWT", scope="balanced")

        # Should find JWT learning from t1 (weighted)
        assert any("JWT" in item.content for item in context)

    def test_all_sessions_scope(self):
        """Test ALL_SESSIONS searches all projects equally."""
        manager = ContextManager()

        # Session 1
        manager.create_session("s1", "OAuth", "OAuth impl")
        manager.create_activity("a1", "s1", "Activity", ["oauth"], "learner", "core", "...")
        manager.create_task("t1", "a1", "Task", ["oauth"])
        manager.record_learning("OAuth2 flow", tags=["oauth"])

        # Session 2
        manager.create_session("s2", "JWT", "JWT impl")
        manager.create_activity("a2", "s2", "Activity", ["jwt"], "learner", "core", "...")
        manager.create_task("t2", "a2", "Task", ["jwt"])
        manager.record_learning("JWT structure", tags=["jwt"])

        # Query in s2 with ALL_SESSIONS
        memory = AgentMemory(manager.storage, current_task_id="t2")
        context = memory.recall("authentication", scope="macro")

        # Should find items from both sessions
        assert any("OAuth" in item.content for item in context)
        assert any("JWT" in item.content for item in context)
```

---

## Performance Tests

### 1. Retrieval Performance

**File**: `tests/performance/test_retrieval_performance.py`

```python
"""Performance tests for retrieval."""

import pytest
import time
from cogneetree import ContextManager, AgentMemory


class TestRetrievalPerformance:
    """Test retrieval performance."""

    def test_retrieval_with_1000_items(self):
        """Test retrieval performance with 1000 items."""
        manager = ContextManager()
        manager.create_session("s1", "Perf test", "Test")
        manager.create_activity("a1", "s1", "Activity", ["test"], "learner", "core", "...")
        manager.create_task("t1", "a1", "Task", ["test"])

        # Add 1000 items
        for i in range(1000):
            manager.record_learning(
                f"Learning {i}: Some learning content",
                tags=[f"tag_{i % 10}", "common"]
            )

        # Measure retrieval time
        memory = AgentMemory(manager.storage, current_task_id="t1")

        start = time.time()
        context = memory.recall("common learning", max_items=10)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 1 second)
        assert elapsed < 1.0
        assert len(context) > 0

    def test_retrieval_scales_linearly(self):
        """Test that retrieval scales linearly with items."""
        manager = ContextManager()
        manager.create_session("s1", "Scale test", "Test")
        manager.create_activity("a1", "s1", "Activity", ["test"], "learner", "core", "...")
        manager.create_task("t1", "a1", "Task", ["test"])

        times = []
        for count in [100, 500, 1000]:
            # Add items
            for i in range(count - (100 if times else 0)):
                manager.record_learning(
                    f"Learning {i}: Content",
                    tags=["test"]
                )

            # Measure retrieval
            memory = AgentMemory(manager.storage, current_task_id="t1")
            start = time.time()
            context = memory.recall("test", max_items=10)
            elapsed = time.time() - start
            times.append((count, elapsed))

        # Verify roughly linear scaling
        ratio1 = times[1][1] / times[0][1]  # Should be ~5x
        assert 4 < ratio1 < 6

    @pytest.mark.benchmark
    def test_retrieval_latency_sla(self):
        """Test retrieval meets SLA (< 100ms for 10k items)."""
        manager = ContextManager()
        manager.create_session("s1", "SLA test", "Test")
        manager.create_activity("a1", "s1", "Activity", ["test"], "learner", "core", "...")
        manager.create_task("t1", "a1", "Task", ["test"])

        # Add 10k items
        for i in range(10000):
            manager.record_learning(f"Item {i}", tags=["test"])

        # Measure retrieval
        memory = AgentMemory(manager.storage, current_task_id="t1")
        start = time.time()
        context = memory.recall("test", max_items=10)
        elapsed = time.time() - start

        # Should meet SLA
        assert elapsed < 0.1  # 100ms SLA
```

### 2. Storage Performance

**File**: `tests/performance/test_storage_performance.py`

```python
"""Performance tests for storage."""

import pytest
import time
from cogneetree import ContextManager
from cogneetree.storage.in_memory_storage import InMemoryStorage


class TestStoragePerformance:
    """Test storage performance."""

    def test_in_memory_storage_write_throughput(self):
        """Test in-memory storage write throughput."""
        manager = ContextManager(storage=InMemoryStorage())
        manager.create_session("s1", "Throughput test", "Test")
        manager.create_activity("a1", "s1", "Activity", ["test"], "learner", "core", "...")
        manager.create_task("t1", "a1", "Task", ["test"])

        start = time.time()
        for i in range(5000):
            manager.record_learning(f"Item {i}", tags=["test"])
        elapsed = time.time() - start

        # Should handle 5k writes per second
        throughput = 5000 / elapsed
        assert throughput > 1000  # At least 1000 writes/sec

    def test_in_memory_storage_read_throughput(self):
        """Test in-memory storage read throughput."""
        manager = ContextManager(storage=InMemoryStorage())
        manager.create_session("s1", "Read test", "Test")
        manager.create_activity("a1", "s1", "Activity", ["test"], "learner", "core", "...")
        manager.create_task("t1", "a1", "Task", ["test"])

        # Add items
        for i in range(1000):
            manager.record_learning(f"Item {i}", tags=["test"])

        # Measure read throughput
        start = time.time()
        for i in range(1000):
            items = manager.storage.get_items_by_task("t1")
        elapsed = time.time() - start

        throughput = 1000 / elapsed
        assert throughput > 100  # At least 100 reads/sec
```

---

## Effectiveness Tests

### 1. Semantic Accuracy Tests

**File**: `tests/effectiveness/test_semantic_accuracy.py`

```python
"""Tests for semantic retrieval accuracy."""

import pytest
from cogneetree import ContextManager, AgentMemory


class TestSemanticAccuracy:
    """Test semantic retrieval accuracy."""

    def test_retrieve_related_content(self):
        """Test that semantically related content is retrieved."""
        manager = ContextManager()
        manager.create_session("s1", "Auth", "Auth impl")
        manager.create_activity("a1", "s1", "Auth", ["auth"], "learner", "core", "...")
        manager.create_task("t1", "a1", "Task", ["auth"])

        # Record related but differently-worded items
        manager.record_learning(
            "JSON Web Tokens use three base64-encoded segments",
            tags=["jwt"]
        )
        manager.record_learning(
            "JWT structure: header.payload.signature",
            tags=["jwt"]
        )
        manager.record_learning(
            "The three parts of a JWT token are separated by periods",
            tags=["jwt"]
        )

        # Unrelated item
        manager.record_learning(
            "REST APIs use HTTP verbs like GET, POST, PUT",
            tags=["rest"]
        )

        # Query with semantic similarity
        memory = AgentMemory(manager.storage, current_task_id="t1")
        context = memory.recall("What is the structure of JWT tokens?")

        # Should find JWT-related items, not REST item
        jwt_items = [item for item in context if "JWT" in item.content or "token" in item.content.lower()]
        assert len(jwt_items) >= 2
        assert not any("REST" in item.content for item in context)

    def test_semantic_threshold(self):
        """Test semantic similarity threshold."""
        manager = ContextManager()
        manager.create_session("s1", "Test", "Test")
        manager.create_activity("a1", "s1", "Activity", ["test"], "learner", "core", "...")
        manager.create_task("t1", "a1", "Task", ["test"])

        # Highly relevant
        manager.record_learning(
            "JWT tokens should be validated on each request",
            tags=["jwt", "validation"]
        )

        # Slightly relevant
        manager.record_learning(
            "Web applications should validate user input",
            tags=["validation"]
        )

        # Unrelated
        manager.record_learning(
            "Cookies are stored in browser local storage",
            tags=["browser"]
        )

        # Query
        memory = AgentMemory(manager.storage, current_task_id="t1")
        context = memory.recall("How do we validate JWT?", max_items=3)

        # Highly relevant should rank higher
        jwt_validation = next(
            (item for item in context if "JWT" in item.content),
            None
        )
        assert jwt_validation is not None
```

### 2. Proximity Weighting Tests

**File**: `tests/effectiveness/test_proximity_weighting.py`

```python
"""Tests for proximity weighting in retrieval."""

import pytest
from cogneetree import ContextManager, AgentMemory


class TestProximityWeighting:
    """Test that proximity affects ranking."""

    def test_same_task_highest_priority(self):
        """Test that current task items rank highest."""
        manager = ContextManager()
        manager.create_session("s1", "Test", "Test")
        manager.create_activity("a1", "s1", "Activity", ["test"], "learner", "core", "...")

        # Task 1: Record learning
        manager.create_task("t1", "a1", "Task 1", ["jwt"])
        manager.record_learning("JWT structure for task 1", tags=["jwt"])

        # Task 2: Record similar learning
        manager.create_task("t2", "a1", "Task 2", ["jwt"])
        manager.record_learning("JWT structure for task 2", tags=["jwt"])

        # Query from task 2
        memory = AgentMemory(manager.storage, current_task_id="t2")
        context = memory.recall("JWT structure", scope="balanced")

        # Task 2's learning should rank higher (same task)
        task2_item = next(
            (item for item in context if "task 2" in item.content.lower()),
            None
        )
        assert task2_item is not None
        assert context.index(task2_item) == 0  # First result

    def test_sibling_task_secondary_priority(self):
        """Test that sibling task items rank after current task."""
        manager = ContextManager()
        manager.create_session("s1", "Test", "Test")
        manager.create_activity("a1", "s1", "Activity", ["test"], "learner", "core", "...")

        # Task 1: Record learning
        manager.create_task("t1", "a1", "Task 1", ["jwt"])
        manager.record_learning("JWT learning", tags=["jwt"])

        # Task 2: Same activity
        manager.create_task("t2", "a1", "Task 2", ["jwt"])

        # Task 3: Different activity
        manager.create_activity("a2", "s1", "Activity 2", ["auth"], "learner", "core", "...")
        manager.create_task("t3", "a2", "Task 3", ["jwt"])
        manager.record_learning("JWT learning from different activity", tags=["jwt"])

        # Query from task 2
        memory = AgentMemory(manager.storage, current_task_id="t2")
        context = memory.recall("JWT", scope="balanced", max_items=2)

        # Task 1 (sibling) should rank higher than Task 3 (different activity)
        if len(context) >= 2:
            assert context[0].parent_id == "t1"  # Sibling ranked first
            assert context[1].parent_id == "t3"  # Different activity ranked second

    def test_temporal_weighting(self):
        """Test that recent items rank higher."""
        from datetime import datetime, timedelta
        import time

        manager = ContextManager()
        manager.create_session("s1", "Test", "Test")
        manager.create_activity("a1", "s1", "Activity", ["test"], "learner", "core", "...")
        manager.create_task("t1", "a1", "Task", ["jwt"])

        # Record old learning
        old_learning = "JWT old learning"
        manager.record_learning(old_learning, tags=["jwt"])

        # Wait a bit
        time.sleep(0.1)

        # Record new learning
        new_learning = "JWT new learning"
        manager.record_learning(new_learning, tags=["jwt"])

        # Query
        memory = AgentMemory(manager.storage, current_task_id="t1")
        context = memory.recall("JWT learning")

        # Newer item should rank first (in same task, proximity weight is same)
        assert context[0].content == new_learning
        assert context[1].content == old_learning
```

### 3. Context Quality Tests

**File**: `tests/effectiveness/test_context_quality.py`

```python
"""Tests for context quality and usefulness."""

import pytest
from cogneetree import ContextManager, AgentMemory


class TestContextQuality:
    """Test context quality for agents."""

    def test_context_completeness(self):
        """Test that context includes all relevant information."""
        manager = ContextManager()
        manager.create_session("s1", "Implement JWT", "Build auth system")
        manager.create_activity("a1", "s1", "JWT Research", ["jwt"], "learner", "core", "...")
        manager.create_task("t1", "a1", "Learn JWT", ["jwt"])

        # Record comprehensive knowledge
        manager.record_action("Read RFC 7519", tags=["jwt", "research"])
        manager.record_learning("JWT has 3 parts", tags=["jwt", "structure"])
        manager.record_decision("Use RS256 for asymmetric signing", tags=["jwt", "security"])
        manager.record_result("Fully understand JWT", tags=["jwt"])

        # Create new task
        manager.create_task("t2", "a1", "Implement JWT", ["jwt", "implementation"])

        # Retrieve context
        memory = AgentMemory(manager.storage, current_task_id="t2")
        context = memory.recall("Implement JWT securely")

        # Should have action, learning, and decision
        categories = {item.category.value for item in context}
        assert "action" in categories
        assert "learning" in categories
        assert "decision" in categories

    def test_context_relevance(self):
        """Test that retrieved context is relevant to query."""
        manager = ContextManager()
        manager.create_session("s1", "Auth", "Auth impl")
        manager.create_activity("a1", "s1", "Auth", ["auth"], "learner", "core", "...")
        manager.create_task("t1", "a1", "Task", ["auth"])

        # Record mixed relevant and irrelevant
        manager.record_learning("Use HTTPS for auth endpoints", tags=["auth", "security"])
        manager.record_learning("JWT uses base64 encoding", tags=["jwt"])
        manager.record_learning("Always hash passwords with bcrypt", tags=["auth", "security"])
        manager.record_learning("Browser caches are configured with headers", tags=["caching"])

        # Query
        memory = AgentMemory(manager.storage, current_task_id="t1")
        context = memory.recall("secure authentication", max_items=3)

        # Most relevant should be auth+security items
        auth_items = [item for item in context if "auth" in " ".join(item.tags).lower() or "security" in " ".join(item.tags).lower()]
        assert len(auth_items) >= 2

    def test_context_deduplication(self):
        """Test that very similar items aren't duplicated."""
        manager = ContextManager()
        manager.create_session("s1", "Test", "Test")
        manager.create_activity("a1", "s1", "Activity", ["test"], "learner", "core", "...")
        manager.create_task("t1", "a1", "Task", ["jwt"])

        # Record similar items (could happen from multiple agents)
        manager.record_learning("JWT tokens consist of header.payload.signature", tags=["jwt"])
        manager.record_learning("JWT is header.payload.signature format", tags=["jwt"])
        manager.record_learning("Three part JWT: header, payload, signature", tags=["jwt"])

        # Retrieve
        memory = AgentMemory(manager.storage, current_task_id="t1")
        context = memory.recall("JWT structure", max_items=10)

        # Should have all 3 (not duplicated)
        assert len(context) == 3
```

### 4. Agent Learning Tests

**File**: `tests/effectiveness/test_agent_learning.py`

```python
"""Tests for agent learning over time."""

import pytest
from cogneetree import ContextManager, AgentMemory


class TestAgentLearning:
    """Test agent learning patterns."""

    def test_agent_builds_knowledge_over_time(self):
        """Test that agent knowledge compounds over multiple tasks."""
        manager = ContextManager()
        manager.create_session("s1", "API Project", "Build complete API")
        manager.create_activity("a1", "s1", "Authentication", ["auth"], "builder", "core", "...")

        # Task 1: Learn about JWT
        manager.create_task("t1", "a1", "Learn JWT", ["jwt"])
        manager.record_learning("JWT structure: header.payload.signature", tags=["jwt"])
        manager.record_learning("Use RS256 for asymmetric signing", tags=["jwt"])

        # Task 2: Learn about OAuth
        manager.create_task("t2", "a1", "Learn OAuth", ["oauth"])
        manager.record_learning("OAuth2 is for authorization delegation", tags=["oauth"])

        # Task 3: Implement auth (should have access to both JWT and OAuth knowledge)
        manager.create_task("t3", "a1", "Implement Auth", ["auth", "jwt", "oauth"])

        memory = AgentMemory(manager.storage, current_task_id="t3")
        context = memory.recall("implement authentication")

        # Should find both JWT and OAuth learnings
        jwt_found = any("JWT" in item.content for item in context)
        oauth_found = any("OAuth" in item.content for item in context)

        assert jwt_found and oauth_found

    def test_agent_applies_past_decisions(self):
        """Test that agent can apply past decisions."""
        manager = ContextManager()
        manager.create_session("s1", "API Project", "Build API")
        manager.create_activity("a1", "s1", "Architecture", ["architecture"], "builder", "core", "...")

        # Project 1: Made decision about error handling
        manager.create_session("s2", "Previous Project", "Build previous project")
        manager.create_activity("a2", "s2", "Error Handling", ["error"], "builder", "core", "...")
        manager.create_task("t1", "a2", "Design Error Handling", ["error"])
        manager.record_decision(
            "Use consistent error response format: {error: string, code: number, details: object}",
            tags=["error_handling", "api"]
        )

        # Project 2: New project should find this decision
        manager.create_task("t2", "a1", "Implement Error Handling", ["error_handling"])

        memory = AgentMemory(manager.storage, current_task_id="t2")
        context = memory.recall("error response format")

        # Should find previous decision
        decision_found = any("error response format" in item.content for item in context)
        assert decision_found
```

---

## Storage Backend Tests

### 1. In-Memory Storage Tests

**File**: `tests/storage/test_in_memory_storage.py`

```python
"""Tests for InMemoryStorage."""

import pytest
from cogneetree.storage.in_memory_storage import InMemoryStorage
from cogneetree.core.models import ContextItem, ContextCategory


class TestInMemoryStorage:
    """Test InMemoryStorage implementation."""

    def test_create_session(self):
        """Test session creation."""
        storage = InMemoryStorage()
        session = storage.create_session("s1", "Ask", "Plan")
        assert session.session_id == "s1"

    def test_session_isolation(self):
        """Test that sessions are isolated."""
        storage = InMemoryStorage()
        storage.create_session("s1", "Ask 1", "Plan 1")
        storage.create_session("s2", "Ask 2", "Plan 2")

        # Get items from each session (should be separate)
        items_s1 = storage.get_items_by_session("s1")
        items_s2 = storage.get_items_by_session("s2")

        # Sessions should be separate
        assert len(items_s1) == 0
        assert len(items_s2) == 0

    def test_add_and_retrieve_item(self):
        """Test adding and retrieving items."""
        storage = InMemoryStorage()
        storage.create_session("s1", "Test", "Test")

        item = ContextItem(
            content="Test item",
            category=ContextCategory.LEARNING,
            tags=["test"]
        )
        storage.add_item(item)

        # Should be able to retrieve
        items = storage.get_items_by_tags(["test"])
        assert len(items) == 1
        assert items[0].content == "Test item"

    def test_context_stack_management(self):
        """Test context stack (current session/activity/task)."""
        storage = InMemoryStorage()
        session = storage.create_session("s1", "Test", "Test")
        assert storage.get_current_session_id() == "s1"

        activity = storage.create_activity("a1", "s1", "Activity", ["tag"], "mode", "comp", "plan")
        assert storage.get_current_activity_id() == "a1"

        task = storage.create_task("t1", "a1", "Task", ["tag"])
        assert storage.get_current_task_id() == "t1"
```

### 2. SQL Storage Tests

**File**: `tests/storage/test_sql_storage.py`

```python
"""Tests for SQLStorage."""

import pytest
import tempfile
import os
from cogneetree.storage.sql_storage import SQLStorage
from cogneetree.core.models import ContextItem, ContextCategory


class TestSQLStorage:
    """Test SQLStorage with SQLite."""

    @pytest.fixture
    def storage(self):
        """Create temporary SQLite storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLStorage(f"sqlite:///{db_path}")
            yield storage

    def test_persistence(self, storage):
        """Test that data persists."""
        session = storage.create_session("s1", "Test", "Test")
        storage.create_activity("a1", "s1", "Activity", ["tag"], "mode", "comp", "plan")
        storage.create_task("t1", "a1", "Task", ["tag"])

        item = ContextItem(
            content="Persistent item",
            category=ContextCategory.LEARNING,
            tags=["persist"]
        )
        storage.add_item(item)

        # Retrieve
        items = storage.get_items_by_tags(["persist"])
        assert len(items) == 1
        assert items[0].content == "Persistent item"

    def test_transaction_safety(self, storage):
        """Test transaction safety."""
        storage.create_session("s1", "Test", "Test")
        storage.create_activity("a1", "s1", "Activity", ["tag"], "mode", "comp", "plan")
        storage.create_task("t1", "a1", "Task", ["tag"])

        # Add multiple items (should be atomic)
        for i in range(100):
            item = ContextItem(
                content=f"Item {i}",
                category=ContextCategory.LEARNING,
                tags=["batch"]
            )
            storage.add_item(item)

        items = storage.get_items_by_tags(["batch"])
        assert len(items) == 100
```

---

## Retrieval Strategy Tests

### Tag-Based Retrieval

```python
"""Tests for tag-based retrieval."""

def test_single_tag_retrieval(manager):
    """Test retrieving items by single tag."""
    manager.create_session("s1", "Test", "Test")
    manager.create_activity("a1", "s1", "Activity", ["test"], "learner", "core", "...")
    manager.create_task("t1", "a1", "Task", ["jwt"])

    manager.record_learning("JWT learning", tags=["jwt", "auth"])
    manager.record_learning("OAuth learning", tags=["oauth"])

    # Retrieve by single tag
    items = manager.storage.get_items_by_tags(["jwt"])
    assert len(items) == 1
    assert "JWT" in items[0].content

def test_multiple_tag_retrieval(manager):
    """Test retrieving items with multiple tags."""
    manager.create_session("s1", "Test", "Test")
    manager.create_activity("a1", "s1", "Activity", ["test"], "learner", "core", "...")
    manager.create_task("t1", "a1", "Task", ["jwt"])

    manager.record_learning("JWT auth", tags=["jwt", "auth"])
    manager.record_learning("OAuth auth", tags=["oauth", "auth"])

    # Both have "auth" tag
    items = manager.storage.get_items_by_tags(["auth"])
    assert len(items) == 2
```

---

## Agent Integration Tests

### LLM Integration Tests

```python
"""Tests for LLM integration."""

def test_context_formatting_for_llm():
    """Test that context is properly formatted for LLM."""
    manager = ContextManager()
    manager.create_session("s1", "Build API", "Complete")
    manager.create_activity("a1", "s1", "Auth", ["auth"], "builder", "core", "...")
    manager.create_task("t1", "a1", "Implement", ["auth"])

    manager.record_learning("JWT structure: header.payload.signature", tags=["jwt"])
    manager.record_decision("Use HS256", tags=["jwt"])

    memory = AgentMemory(manager.storage, current_task_id="t1")
    formatted = memory.build_context("How to implement JWT?")

    # Should be readable markdown
    assert isinstance(formatted, str)
    assert "JWT" in formatted
    assert "header.payload.signature" in formatted
    assert "#" in formatted or "##" in formatted  # Markdown headers
```

---

## Test Execution

### Run All Tests

```bash
# Run complete test suite
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=cogneetree --cov-report=html

# Run specific test file
pytest tests/unit/test_context_manager.py

# Run specific test class
pytest tests/unit/test_context_manager.py::TestContextManagerHierarchy

# Run specific test
pytest tests/unit/test_context_manager.py::TestContextManagerHierarchy::test_create_session

# Run with markers
pytest -m "not slow"  # Skip slow tests
pytest -m "benchmark"  # Run only benchmarks
```

### Test Configuration

**File**: `pytest.ini`

```ini
[pytest]
minversion = 7.0
addopts = -v --strict-markers
testpaths = tests
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance/benchmark tests
    effectiveness: Effectiveness tests
    storage: Storage backend tests
    slow: Slow running tests
    benchmark: Benchmark tests
```

**File**: `conftest.py`

```python
"""Pytest configuration and fixtures."""

import pytest
from cogneetree import ContextManager


@pytest.fixture
def manager():
    """Provide a fresh ContextManager for each test."""
    return ContextManager()


@pytest.fixture
def sample_session(manager):
    """Provide a sample session with activity and task."""
    session = manager.create_session("test_session", "Test ask", "Test plan")
    activity = manager.create_activity(
        "test_activity",
        "test_session",
        "Test activity",
        ["test"],
        "learner",
        "core",
        "Test analysis"
    )
    task = manager.create_task(
        "test_task",
        "test_activity",
        "Test task",
        ["test"]
    )
    return {"session": session, "activity": activity, "task": task, "manager": manager}
```

---

## Metrics and Reporting

### Coverage Report

```bash
# Generate HTML coverage report
pytest --cov=cogneetree --cov-report=html

# View report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov\index.html  # Windows
```

### Performance Report

```bash
# Run performance tests with timing
pytest tests/performance/ -v --durations=10
```

### Test Summary

```
Test Results Summary
====================
Unit Tests:         85/85 passed (100%)
Integration Tests:  32/32 passed (100%)
Storage Tests:      40/40 passed (100%)
Performance Tests:  12/12 passed (100%)
Effectiveness Tests: 28/28 passed (100%)

Code Coverage:      87%
  - cogneetree.core: 95%
  - cogneetree.storage: 85%
  - cogneetree.retrieval: 90%

Performance Metrics:
  - Retrieval (10k items): 87ms (SLA: <100ms) ✓
  - Write throughput: 2,500 items/sec
  - Read throughput: 1,200 reads/sec
```

### Continuous Integration

**File**: `.github/workflows/test.yml`

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    - name: Run tests
      run: pytest
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

---

## Effectiveness Evaluation Checklist

Use this checklist to assess whether Cogneetree effectively builds long-term memory:

### Memory Persistence ✓

- [ ] Data survives across sessions with persistent storage
- [ ] Context stack (current session/activity/task) properly maintained
- [ ] Retrieval finds items from previous sessions
- [ ] Metadata preserved with timestamps

### Knowledge Organization ✓

- [ ] Hierarchical structure naturally mirrors work (Session → Activity → Task)
- [ ] Items properly categorized (action, decision, learning, result)
- [ ] Tags enable flexible organization
- [ ] Parent-child relationships maintained correctly

### Intelligent Retrieval ✓

- [ ] Items ranked by semantic relevance
- [ ] Proximity weighting prioritizes nearby items
- [ ] Temporal weighting favors recent items
- [ ] Different scopes (micro/balanced/macro) return appropriate results

### Agent Integration ✓

- [ ] Simple recall() API for agents
- [ ] Context builds properly formatted for LLM injection
- [ ] Agents can record learnings/decisions
- [ ] Multi-session learning enables knowledge compounding

### Performance ✓

- [ ] Retrieval SLA met (<100ms for 10k items)
- [ ] Linear scaling as items grow
- [ ] Write throughput sufficient for real-time recording
- [ ] Storage backends scale appropriately

---

## Summary

This testing plan ensures Cogneetree reliably:

1. **Stores** knowledge hierarchically and persistently
2. **Retrieves** context intelligently based on relevance and proximity
3. **Scales** to handle 10k+ items without performance degradation
4. **Integrates** seamlessly with AI agents for real-time learning
5. **Compounds** expertise as agents work across multiple projects

By following this comprehensive testing strategy, we maintain confidence that Cogneetree delivers genuine long-term memory for AI applications.

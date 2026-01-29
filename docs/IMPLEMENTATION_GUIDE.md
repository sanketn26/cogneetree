# Cogneetree Implementation Guide

## Overview

This guide provides a comprehensive walkthrough for implementing Cogneetree in your AI application. It covers setup, core concepts, usage patterns, and best practices for building long-term memory systems.

**Target Audience**: Developers implementing Cogneetree in their projects

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Architecture Fundamentals](#architecture-fundamentals)
3. [Core Components](#core-components)
4. [Basic Implementation](#basic-implementation)
5. [Advanced Patterns](#advanced-patterns)
6. [Storage Configuration](#storage-configuration)
7. [Retrieval Strategies](#retrieval-strategies)
8. [Integration with AI Agents](#integration-with-ai-agents)
9. [Troubleshooting](#troubleshooting)
10. [Performance Considerations](#performance-considerations)

---

## Getting Started

### Installation

```bash
# Basic installation
pip install cogneetree

# Development installation
git clone https://github.com/cogneetree/cogneetree.git
cd cogneetree
pip install -e ".[dev]"
```

### Dependencies

- **Core**: Zero external dependencies for basic functionality
- **Optional**: For semantic retrieval, install embeddings library (e.g., `sentence-transformers`)

### Verify Installation

```python
from cogneetree import ContextManager, AgentMemory
print(f"Cogneetree installed successfully")
```

---

## Architecture Fundamentals

### The Hierarchical Model

Cogneetree organizes memory into three natural levels:

```
Session (Project/Conversation)
  ├── Original Question
  ├── High-level Plan
  └── Activities (Work Areas)
      ├── Description
      ├── Tags (e.g., "authentication", "api")
      ├── Mode (e.g., "learner", "builder")
      └── Tasks (Atomic Work Items)
          ├── Description
          ├── Tags
          ├── Result
          └── Context Items
              ├── Actions (what you did)
              ├── Decisions (why you chose it)
              ├── Learnings (what you discovered)
              └── Results (what was accomplished)
```

### Why This Structure?

1. **Natural to work**: Mirrors how projects actually unfold
2. **Efficient scoping**: Search just this task, or all projects
3. **Proximity weighting**: Nearby items matter more; distant items still surface
4. **Flexible retrieval**: Same data, multiple access patterns

### Core Principles

| Principle | Meaning | Example |
|-----------|---------|---------|
| **Hierarchical** | Work organized in parent-child relationships | Session contains Activities; Activities contain Tasks |
| **Transparent** | Understand why context was retrieved | Each result includes similarity score and proximity weight |
| **Flexible** | Multiple storage and retrieval options | Switch from in-memory to PostgreSQL without code changes |
| **Lightweight** | Minimal dependencies, maximum utility | Zero external deps for core library |
| **Agent-friendly** | Simple API designed for LLM integration | One-line context retrieval |

---

## Core Components

### 1. ContextManager

The primary interface for creating and managing context hierarchies.

**Responsibilities**:
- Creating sessions, activities, and tasks
- Recording actions, decisions, learnings, and results
- Managing the context stack (current session/activity/task)

**Key Methods**:

```python
from cogneetree import ContextManager

manager = ContextManager()

# Create hierarchy
session = manager.create_session(
    session_id="proj_auth_001",
    original_ask="Implement OAuth2 authentication",
    high_level_plan="JWT tokens + refresh mechanism"
)

activity = manager.create_activity(
    activity_id="act_jwt_001",
    session_id="proj_auth_001",
    description="Understand JWT fundamentals",
    tags=["jwt", "auth", "research"],
    mode="learner",
    component="core",
    planner_analysis="Need to understand JWT structure before implementation"
)

task = manager.create_task(
    task_id="task_jwt_structure_001",
    activity_id="act_jwt_001",
    description="Learn JWT structure and components",
    tags=["jwt", "structure"]
)

# Record learning
manager.record_learning(
    "JWT consists of three parts separated by dots: header.payload.signature",
    tags=["jwt", "structure"]
)

# Record decision
manager.record_decision(
    "Use HS256 algorithm for token signing",
    tags=["jwt", "algorithm"]
)

# Record action
manager.record_action(
    "Analyzed IETF JWT RFC specification",
    tags=["jwt", "research"]
)

# Record result
manager.record_result(
    "Understanding of JWT structure and best practices",
    tags=["jwt", "structure"]
)
```

### 2. AgentMemory

High-level interface optimized for AI agents. Wraps HierarchicalRetriever with convenient recall/build methods.

**Responsibilities**:
- Semantic and hierarchical retrieval
- Context building for LLM prompts
- Scope management (micro/balanced/macro)

**Key Methods**:

```python
from cogneetree import AgentMemory, HistoryMode

memory = AgentMemory(manager.storage, current_task_id="task_jwt_structure_001")

# Simple recall (default: CURRENT_WEIGHTED scope)
context = memory.recall("JWT validation strategies")

# Specify scope
context_focused = memory.recall(
    "JWT validation",
    scope="micro"  # Only this task
)

context_learning = memory.recall(
    "authentication patterns",
    scope="macro"  # All projects equally
)

# Build context for LLM
context_str = memory.build_context(
    query="How should we validate JWT tokens?",
    max_items=5
)
# Returns formatted markdown ready for prompt injection

# Detailed retrieval with config
from cogneetree import RetrievalConfig, HistoryMode

config = RetrievalConfig(
    history_mode=HistoryMode.CURRENT_WEIGHTED,
    max_results=10,
    time_depth_days=90,
    include_explanation=True
)
context = memory.recall_with_config("JWT implementation", config)
```

### 3. Storage Backend (ContextStorageABC)

Abstract base class for all storage implementations.

**Responsibilities**:
- Storing/retrieving sessions, activities, tasks, and items
- Managing context state (current session/activity/task)
- Providing item filtering and querying

**Built-in Implementations**:

| Storage | Use Case | Persistence |
|---------|----------|-------------|
| **InMemoryStorage** | Development, testing | Per-process only |
| **FileStorage** | Single-machine persistence | JSON files |
| **SQLStorage** | Multi-user applications | SQLite, PostgreSQL |
| **RedisStorage** | Distributed systems | Redis backend |

**Custom Implementation Example**:

```python
from cogneetree import ContextStorageABC
from cogneetree.core.models import Session, Activity, Task, ContextItem

class MongoDBStorage(ContextStorageABC):
    """MongoDB-backed context storage."""

    def __init__(self, db_url: str):
        from pymongo import MongoClient
        self.client = MongoClient(db_url)
        self.db = self.client["cogneetree"]
        self.items_collection = self.db["items"]
        self.context_collection = self.db["context"]

        # Initialize context stack
        self.current_session_id = None
        self.current_activity_id = None
        self.current_task_id = None

    def create_session(self, session_id: str, original_ask: str, high_level_plan: str) -> Session:
        """Create a new session."""
        session = Session(session_id, original_ask, high_level_plan)
        self.context_collection.insert_one({
            "_id": session_id,
            "original_ask": original_ask,
            "high_level_plan": high_level_plan,
            "type": "session"
        })
        self.current_session_id = session_id
        return session

    def create_activity(self, activity_id: str, session_id: str, description: str,
                       tags: list, mode: str, component: str, planner_analysis: str) -> Activity:
        """Create a new activity."""
        activity = Activity(activity_id, session_id, description, tags, mode, component, planner_analysis)
        self.context_collection.insert_one({
            "_id": activity_id,
            "session_id": session_id,
            "description": description,
            "tags": tags,
            "mode": mode,
            "component": component,
            "planner_analysis": planner_analysis,
            "type": "activity"
        })
        self.current_activity_id = activity_id
        return activity

    def create_task(self, task_id: str, activity_id: str, description: str, tags: list) -> Task:
        """Create a new task."""
        task = Task(task_id, activity_id, description, tags)
        self.context_collection.insert_one({
            "_id": task_id,
            "activity_id": activity_id,
            "description": description,
            "tags": tags,
            "type": "task"
        })
        self.current_task_id = task_id
        return task

    def add_item(self, item: ContextItem) -> None:
        """Add a context item."""
        self.items_collection.insert_one({
            "content": item.content,
            "category": item.category.value,
            "tags": item.tags,
            "timestamp": item.timestamp,
            "parent_id": item.parent_id,
            "metadata": item.metadata
        })

    # ... implement remaining abstract methods
```

### 4. HierarchicalRetriever

Implements the core retrieval logic with four history modes.

**Features**:
- Semantic similarity scoring
- Proximity weighting (based on hierarchy position)
- Temporal weighting (recent items rank higher)
- Explainability (see why results were ranked)

**History Modes**:

```python
from cogneetree import HistoryMode, RetrievalConfig, HierarchicalRetriever

config = RetrievalConfig(
    history_mode=HistoryMode.CURRENT_ONLY,  # Only this task
    # OR
    # history_mode=HistoryMode.CURRENT_WEIGHTED,  # Task + history (default)
    # history_mode=HistoryMode.ALL_SESSIONS,  # All projects equally
    # history_mode=HistoryMode.SELECTED_SESSIONS,  # Specific projects
)

retriever = HierarchicalRetriever(storage, config)
results = retriever.retrieve("JWT implementation", current_task_id="task_123")

for result in results:
    print(f"Content: {result.content}")
    print(f"Similarity: {result.semantic_similarity:.2f}")
    print(f"Proximity Weight: {result.proximity_weight:.2f}")
    print(f"Final Score: {result.final_score:.2f}")
    print(f"Explanation: {result.explanation}")
```

---

## Basic Implementation

### Step 1: Initialize Context

```python
from cogneetree import ContextManager

# Create manager with default in-memory storage
manager = ContextManager()

# Create a session (project/conversation)
session = manager.create_session(
    session_id="oauth_impl_2024",
    original_ask="Implement OAuth2 authentication flow",
    high_level_plan="Research → Design → Implement → Test"
)
```

### Step 2: Create Activity and Task

```python
# Create an activity (work area)
activity = manager.create_activity(
    activity_id="oauth_research",
    session_id="oauth_impl_2024",
    description="Research OAuth2 flows and best practices",
    tags=["oauth2", "authentication", "research"],
    mode="learner",
    component="core",
    planner_analysis="Need to understand OAuth2 flows before implementation"
)

# Create a task (atomic work item)
task = manager.create_task(
    task_id="oauth_flow_analysis",
    activity_id="oauth_research",
    description="Analyze authorization code flow",
    tags=["oauth2", "authorization_code_flow"]
)
```

### Step 3: Record Knowledge

```python
# Record learnings
manager.record_learning(
    "OAuth2 authorization code flow best for web apps",
    tags=["oauth2", "flow_selection"]
)

manager.record_learning(
    "Always use PKCE (Proof Key for Code Exchange) for public clients",
    tags=["oauth2", "security", "pkce"]
)

# Record decisions
manager.record_decision(
    "Use Auth0 as OAuth provider for simplicity",
    tags=["oauth2", "provider"]
)

manager.record_decision(
    "Store tokens in httpOnly cookies for security",
    tags=["oauth2", "token_storage", "security"]
)

# Record actions
manager.record_action(
    "Read RFC 6749 OAuth2 specification",
    tags=["oauth2", "research"]
)

# Record results
manager.record_result(
    "Clear understanding of OAuth2 flows, security considerations, and implementation best practices",
    tags=["oauth2", "research"]
)
```

### Step 4: Retrieve and Use Context

```python
from cogneetree import AgentMemory

# Create new task requiring previous knowledge
new_task = manager.create_task(
    task_id="oauth_implementation",
    activity_id="oauth_research",
    description="Implement OAuth2 flow in application",
    tags=["oauth2", "implementation"]
)

# Get memory interface
memory = AgentMemory(manager.storage, current_task_id="oauth_implementation")

# Retrieve relevant context
context = memory.recall("OAuth2 security best practices")

print("Retrieved Context:")
for item in context:
    print(f"- {item.category.value}: {item.content}")

# Build formatted context for LLM
llm_context = memory.build_context(
    query="How should we implement OAuth2 securely?",
    max_items=5
)
print(llm_context)
```

---

## Advanced Patterns

### Pattern 1: Multi-Session Learning

Enable agents to learn across multiple projects:

```python
from cogneetree import ContextManager, AgentMemory, HistoryMode

# Session 1: OAuth2 project
manager = ContextManager()
manager.create_session("proj_oauth", "Implement OAuth2", "Use Auth0")
# ... record learnings ...

# Session 2: JWT project (separate)
manager.create_session("proj_jwt", "Implement JWT tokens", "Use RS256")
# ... record learnings ...

# Later: Query across both sessions
memory = AgentMemory(manager.storage, current_task_id="new_task")

# Learn from both OAuth2 and JWT work
context = memory.recall(
    "Token authentication approaches",
    scope="macro"  # Search all projects equally
)
```

### Pattern 2: Decision Tracking

Build an audit trail of architectural decisions:

```python
manager.record_decision(
    "Use PostgreSQL instead of MongoDB for ACID transactions",
    tags=["database", "architecture", "decision"],
    metadata={
        "decision_date": "2024-01-15",
        "trade_offs": "ACID compliance vs. flexibility",
        "reviewed_by": "senior_architect"
    }
)

# Later, retrieve decision history
memory = AgentMemory(manager.storage)
decisions = memory.recall("database technology decision")
```

### Pattern 3: Learning Chains

Connect related learnings for progressive understanding:

```python
# First learning
manager.record_learning(
    "JWT structure: three base64-encoded parts",
    tags=["jwt", "structure"]
)

# Related learning
manager.record_learning(
    "Header contains algorithm; decoded as: {'alg': 'HS256', 'typ': 'JWT'}",
    tags=["jwt", "header"]
)

# Connected learning
manager.record_learning(
    "Always validate algorithm matches expected value to prevent signature bypass",
    tags=["jwt", "security", "validation"]
)

# Retrieve complete chain
context = memory.recall("JWT security validation", scope="balanced")
# Returns all related learnings in proximity order
```

### Pattern 4: Task-Specific Context

Keep focused context when working on specific tasks:

```python
# Create a specific task
manager.create_task(
    task_id="bug_fix_001",
    activity_id="debugging",
    description="Fix JWT validation vulnerability",
    tags=["jwt", "bug", "security"]
)

memory = AgentMemory(manager.storage, current_task_id="bug_fix_001")

# Retrieve ONLY context from this task (micro scope)
focused_context = memory.recall(
    "What JWT validation issues were identified?",
    scope="micro"  # Only this task
)

# Retrieve context from this task + recent related history (balanced scope)
balanced_context = memory.recall(
    "JWT validation issues and fixes",
    scope="balanced"  # Default: task + history
)
```

---

## Storage Configuration

### In-Memory Storage (Default)

Fastest, zero configuration, no persistence:

```python
from cogneetree import ContextManager

manager = ContextManager()  # Uses InMemoryStorage by default
# All data lost when process ends
```

### File Storage

Persistent JSON files:

```python
from cogneetree import ContextManager
from cogneetree.storage.file_storage import FileStorage

storage = FileStorage(base_path="./cogneetree_data")
manager = ContextManager(storage=storage)
# Data persists in JSON files
```

### SQL Storage

Multi-user, queryable, scalable:

```python
from cogneetree import ContextManager
from cogneetree.storage.sql_storage import SQLStorage

# SQLite (development)
storage = SQLStorage(db_url="sqlite:///cogneetree.db")

# PostgreSQL (production)
storage = SQLStorage(db_url="postgresql://user:pass@localhost/cogneetree")

manager = ContextManager(storage=storage)
# Data persists in database
```

### Redis Storage

Distributed, high-performance caching:

```python
from cogneetree import ContextManager
from cogneetree.storage.redis_storage import RedisStorage

storage = RedisStorage(
    host="localhost",
    port=6379,
    db=0,
    prefix="cogneetree:"
)

manager = ContextManager(storage=storage)
# Data in Redis with configurable TTL
```

---

## Retrieval Strategies

### Simple Tag-Based Retrieval

```python
from cogneetree import AgentMemory

memory = AgentMemory(manager.storage)

# Items tagged with "jwt" OR "oauth2"
context = memory.recall_by_tags(tags=["jwt", "oauth2"])
```

### Semantic Retrieval with Embeddings

```python
from cogneetree import AgentMemory, RetrievalConfig

config = RetrievalConfig(
    semantic_threshold=0.7,  # Require 70% similarity
    use_semantic_search=True
)

memory = AgentMemory(manager.storage, config=config)

# Uses embeddings for similarity matching
context = memory.recall(
    "How do we validate authentication tokens?",
    config=config
)
```

### Combined Tag + Semantic Retrieval

```python
from cogneetree import AgentMemory

memory = AgentMemory(manager.storage)

# Retrieve items matching both tags AND semantic similarity
context = memory.recall_hybrid(
    query="JWT implementation",
    tags=["jwt", "security"],
    semantic_weight=0.7,
    tag_weight=0.3
)
```

---

## Integration with AI Agents

### Pattern: Claude/OpenAI Integration

```python
from cogneetree import AgentMemory
import anthropic

# Initialize Cogneetree
memory = AgentMemory(manager.storage, current_task_id="current_task")

# Get relevant context
context = memory.recall("Implement feature X")
context_str = memory.build_context("How should we implement feature X?")

# Call Claude with context
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2000,
    messages=[
        {
            "role": "user",
            "content": f"""You are an expert software engineer. Consider the following context from past work:

{context_str}

Now, help me with: Implement feature X following established patterns."""
        }
    ]
)

# Record Claude's response and learnings
manager.record_action(f"Asked Claude about feature X implementation")
manager.record_result(response.content[0].text)
```

### Pattern: Agent-Driven Knowledge Recording

```python
from cogneetree import ContextManager, AgentMemory

class SmartAgent:
    def __init__(self, task_id: str):
        self.manager = ContextManager()
        self.memory = AgentMemory(self.manager.storage, current_task_id=task_id)

    def work(self, task_description: str):
        # Get relevant context
        context = self.memory.recall(task_description)

        # Do work...
        result = self.execute_task(task_description, context)

        # Record learning
        self.manager.record_action(f"Executed: {task_description}")
        self.manager.record_result(result)

        # Record any learnings discovered
        if self.discovered_insights:
            self.manager.record_learning(
                self.discovered_insights,
                tags=["auto_discovered"]
            )

        return result
```

---

## Troubleshooting

### Issue: Empty Context Results

**Cause**: No matching items or incorrect task_id

**Solution**:
```python
# Verify task exists
task = manager.get_task_by_id("task_id")
assert task is not None, "Task does not exist"

# Try broader search
context = memory.recall("search query", scope="macro")

# Check tags
items = memory.recall_by_tags(tags=["debug_tag"])
```

### Issue: Semantic Search Not Working

**Cause**: Embeddings provider not configured

**Solution**:
```python
from cogneetree.retrieval.semantic_retrieval import SemanticRetrieval
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
semantic_retrieval = SemanticRetrieval(model)

# Now semantic search works
context = memory.recall("query", use_semantic=True)
```

### Issue: Storage Persistence Not Working

**Cause**: Using InMemoryStorage instead of persistent backend

**Solution**:
```python
from cogneetree import ContextManager
from cogneetree.storage.sql_storage import SQLStorage

# Switch to persistent storage
storage = SQLStorage("postgresql://user:pass@host/db")
manager = ContextManager(storage=storage)
```

---

## Performance Considerations

### Optimization Tips

1. **Use Appropriate Scope**
   ```python
   # Don't use macro scope for every query
   context = memory.recall("query", scope="micro")  # Faster
   ```

2. **Limit Max Results**
   ```python
   context = memory.recall("query", max_items=5)  # Not 100
   ```

3. **Filter by Time**
   ```python
   config = RetrievalConfig(time_depth_days=30)
   context = memory.recall_with_config("query", config)
   ```

4. **Cache Embeddings**
   ```python
   # Embeddings computed once per retrieval call, reused across items
   ```

5. **Use Redis for Distributed Systems**
   ```python
   # RedisStorage provides caching and distribution
   storage = RedisStorage(host="localhost", ttl_seconds=3600)
   ```

### Benchmarking

```python
import time
from cogneetree import AgentMemory

memory = AgentMemory(manager.storage)

# Measure retrieval time
start = time.time()
context = memory.recall("test query", max_items=10)
elapsed = time.time() - start

print(f"Retrieval took {elapsed:.3f} seconds")
print(f"Retrieved {len(context)} items")
```

---

## Summary

Cogneetree enables building genuine long-term memory systems for AI agents:

1. **Structure** knowledge in natural hierarchies (Session → Activity → Task)
2. **Record** actions, decisions, learnings, and results
3. **Retrieve** context at appropriate scopes (micro, balanced, macro)
4. **Integrate** with LLMs for intelligent decision-making

By following these patterns, agents can compound expertise across projects and make increasingly informed decisions over time.

---

## Roadmap Features: Detailed Implementation

This section provides detailed implementation guides for planned features that will differentiate Cogneetree from competitors.

### Table of Contents - Roadmap

- [Phase 1: Critical Fixes](#phase-1-critical-fixes)
  - [1.1 Complete Storage Backends](#11-complete-storage-backends)
  - [1.2 Persistent Embedding Cache](#12-persistent-embedding-cache)
  - [1.3 Thread-Safe Context Management](#13-thread-safe-context-management)
- [Phase 2: Table Stakes Features](#phase-2-table-stakes-features)
  - [2.1 Batch Operations](#21-batch-operations)
  - [2.2 Update/Delete Operations](#22-updatedelete-operations)
  - [2.3 Analytics & Observability](#23-analytics--observability)
- [Phase 3: Differentiation Features](#phase-3-differentiation-features)
  - [3.1 Automatic Knowledge Extraction](#31-automatic-knowledge-extraction)
  - [3.2 Causal Chains & Reasoning Traces](#32-causal-chains--reasoning-traces)
  - [3.3 Memory Consolidation & Forgetting](#33-memory-consolidation--forgetting)
  - [3.4 Time-Travel Queries](#34-time-travel-queries)
- [Phase 4: Enterprise Features](#phase-4-enterprise-features)
  - [4.1 Multi-Agent Shared Memory](#41-multi-agent-shared-memory)
  - [4.2 REST API Wrapper](#42-rest-api-wrapper)
  - [4.3 Structured Output Integration](#43-structured-output-integration)

---

## Phase 1: Critical Fixes

### 1.1 Complete Storage Backends

**Problem**: SQL and Redis storage implementations are missing required methods, causing `AttributeError` when used with `HierarchicalRetriever`.

**File**: `src/cogneetree/storage/sql_storage.py`

```python
"""Complete SQL Storage Implementation."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from cogneetree.core.interfaces import ContextStorageABC
from cogneetree.core.models import (
    Session, Activity, Task, ContextItem, ContextCategory
)


class SQLStorage(ContextStorageABC):
    """SQL-backed storage with complete implementation."""

    def __init__(self, db_url: str):
        """
        Initialize SQL storage.

        Args:
            db_url: Database connection string
                    - SQLite: "sqlite:///cogneetree.db"
                    - PostgreSQL: "postgresql://user:pass@host/db"
        """
        self.db_url = db_url
        self._init_db()

        # Context stack
        self.current_session_id: Optional[str] = None
        self.current_activity_id: Optional[str] = None
        self.current_task_id: Optional[str] = None

    def _init_db(self):
        """Initialize database schema."""
        import sqlite3

        if self.db_url.startswith("sqlite"):
            db_path = self.db_url.replace("sqlite:///", "")
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        else:
            # PostgreSQL support
            import psycopg2
            from psycopg2.extras import RealDictCursor
            self.conn = psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)

        self._create_tables()

    def _create_tables(self):
        """Create required tables."""
        cursor = self.conn.cursor()

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                original_ask TEXT NOT NULL,
                high_level_plan TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Activities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS activities (
                activity_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                description TEXT NOT NULL,
                tags TEXT NOT NULL,  -- JSON array
                mode TEXT NOT NULL,
                component TEXT NOT NULL,
                planner_analysis TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        # Tasks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                activity_id TEXT NOT NULL,
                description TEXT NOT NULL,
                tags TEXT NOT NULL,  -- JSON array
                result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (activity_id) REFERENCES activities(activity_id)
            )
        """)

        # Context items table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS items (
                item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                tags TEXT NOT NULL,  -- JSON array
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                parent_id TEXT,
                metadata TEXT,  -- JSON object
                embedding BLOB,
                FOREIGN KEY (parent_id) REFERENCES tasks(task_id)
            )
        """)

        # Indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_items_parent ON items(parent_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_items_category ON items(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_activity ON tasks(activity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_activities_session ON activities(session_id)")

        self.conn.commit()

    # ===== SESSION METHODS =====

    def create_session(self, session_id: str, original_ask: str,
                       high_level_plan: str) -> Session:
        """Create a new session."""
        import json
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (session_id, original_ask, high_level_plan) VALUES (?, ?, ?)",
            (session_id, original_ask, high_level_plan)
        )
        self.conn.commit()
        self.current_session_id = session_id
        return Session(session_id, original_ask, high_level_plan)

    def get_session_by_id(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if row:
            return Session(
                session_id=row["session_id"],
                original_ask=row["original_ask"],
                high_level_plan=row["high_level_plan"],
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now()
            )
        return None

    def get_all_sessions(self) -> List[Session]:
        """Get all sessions ordered by creation time."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sessions ORDER BY created_at DESC")
        sessions = []
        for row in cursor.fetchall():
            sessions.append(Session(
                session_id=row["session_id"],
                original_ask=row["original_ask"],
                high_level_plan=row["high_level_plan"],
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now()
            ))
        return sessions

    def get_current_session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self.current_session_id

    # ===== ACTIVITY METHODS =====

    def create_activity(self, activity_id: str, session_id: str, description: str,
                        tags: List[str], mode: str, component: str,
                        planner_analysis: str) -> Activity:
        """Create a new activity."""
        import json
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO activities
               (activity_id, session_id, description, tags, mode, component, planner_analysis)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (activity_id, session_id, description, json.dumps(tags), mode, component, planner_analysis)
        )
        self.conn.commit()
        self.current_activity_id = activity_id
        return Activity(activity_id, session_id, description, tags, mode, component, planner_analysis)

    def get_activity_by_id(self, activity_id: str) -> Optional[Activity]:
        """Get activity by ID."""
        import json
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM activities WHERE activity_id = ?", (activity_id,))
        row = cursor.fetchone()
        if row:
            return Activity(
                activity_id=row["activity_id"],
                session_id=row["session_id"],
                description=row["description"],
                tags=json.loads(row["tags"]),
                mode=row["mode"],
                component=row["component"],
                planner_analysis=row["planner_analysis"]
            )
        return None

    def get_session_activities(self, session_id: str) -> List[Activity]:
        """Get all activities for a session."""
        import json
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM activities WHERE session_id = ? ORDER BY created_at",
            (session_id,)
        )
        activities = []
        for row in cursor.fetchall():
            activities.append(Activity(
                activity_id=row["activity_id"],
                session_id=row["session_id"],
                description=row["description"],
                tags=json.loads(row["tags"]),
                mode=row["mode"],
                component=row["component"],
                planner_analysis=row["planner_analysis"]
            ))
        return activities

    def get_current_activity_id(self) -> Optional[str]:
        """Get current activity ID."""
        return self.current_activity_id

    # ===== TASK METHODS =====

    def create_task(self, task_id: str, activity_id: str, description: str,
                    tags: List[str]) -> Task:
        """Create a new task."""
        import json
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO tasks (task_id, activity_id, description, tags) VALUES (?, ?, ?, ?)",
            (task_id, activity_id, description, json.dumps(tags))
        )
        self.conn.commit()
        self.current_task_id = task_id
        return Task(task_id, activity_id, description, tags)

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        import json
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        if row:
            return Task(
                task_id=row["task_id"],
                activity_id=row["activity_id"],
                description=row["description"],
                tags=json.loads(row["tags"]),
                result=row["result"]
            )
        return None

    def get_activity_tasks(self, activity_id: str) -> List[Task]:
        """Get all tasks for an activity."""
        import json
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM tasks WHERE activity_id = ? ORDER BY created_at",
            (activity_id,)
        )
        tasks = []
        for row in cursor.fetchall():
            tasks.append(Task(
                task_id=row["task_id"],
                activity_id=row["activity_id"],
                description=row["description"],
                tags=json.loads(row["tags"]),
                result=row["result"]
            ))
        return tasks

    def get_current_task_id(self) -> Optional[str]:
        """Get current task ID."""
        return self.current_task_id

    # ===== ITEM METHODS (CRITICAL - THESE WERE MISSING) =====

    def add_item(self, item: ContextItem) -> None:
        """Add a context item."""
        import json
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO items (content, category, tags, timestamp, parent_id, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                item.content,
                item.category.value,
                json.dumps(item.tags),
                item.timestamp.isoformat(),
                item.parent_id,
                json.dumps(item.metadata)
            )
        )
        self.conn.commit()

    def get_items_by_task(self, task_id: str) -> List[ContextItem]:
        """Get all items for a specific task."""
        return self._query_items("parent_id = ?", (task_id,))

    def get_items_by_activity(self, activity_id: str) -> List[ContextItem]:
        """Get all items for tasks within an activity."""
        import json
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT i.* FROM items i
               JOIN tasks t ON i.parent_id = t.task_id
               WHERE t.activity_id = ?
               ORDER BY i.timestamp DESC""",
            (activity_id,)
        )
        return self._rows_to_items(cursor.fetchall())

    def get_items_by_session(self, session_id: str) -> List[ContextItem]:
        """Get all items for a session (across all activities/tasks)."""
        import json
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT i.* FROM items i
               JOIN tasks t ON i.parent_id = t.task_id
               JOIN activities a ON t.activity_id = a.activity_id
               WHERE a.session_id = ?
               ORDER BY i.timestamp DESC""",
            (session_id,)
        )
        return self._rows_to_items(cursor.fetchall())

    def get_items_by_category(self, category: ContextCategory) -> List[ContextItem]:
        """Get all items of a specific category."""
        return self._query_items("category = ?", (category.value,))

    def get_items_by_tags(self, tags: List[str]) -> List[ContextItem]:
        """Get items matching any of the provided tags."""
        import json
        cursor = self.conn.cursor()
        # SQLite JSON support - check if any tag matches
        placeholders = " OR ".join(["tags LIKE ?" for _ in tags])
        params = [f'%"{tag}"%' for tag in tags]
        cursor.execute(f"SELECT * FROM items WHERE {placeholders} ORDER BY timestamp DESC", params)
        return self._rows_to_items(cursor.fetchall())

    def get_all_items(self) -> List[ContextItem]:
        """Get all items."""
        return self._query_items("1=1", ())

    def _query_items(self, where_clause: str, params: tuple) -> List[ContextItem]:
        """Execute item query with WHERE clause."""
        cursor = self.conn.cursor()
        cursor.execute(
            f"SELECT * FROM items WHERE {where_clause} ORDER BY timestamp DESC",
            params
        )
        return self._rows_to_items(cursor.fetchall())

    def _rows_to_items(self, rows) -> List[ContextItem]:
        """Convert database rows to ContextItem objects."""
        import json
        items = []
        for row in rows:
            items.append(ContextItem(
                content=row["content"],
                category=ContextCategory(row["category"]),
                tags=json.loads(row["tags"]),
                timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else datetime.now(),
                parent_id=row["parent_id"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {}
            ))
        return items

    # ===== STATISTICS (NEW) =====

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) as count FROM sessions")
        sessions_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM activities")
        activities_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM tasks")
        tasks_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM items")
        items_count = cursor.fetchone()["count"]

        cursor.execute(
            "SELECT category, COUNT(*) as count FROM items GROUP BY category"
        )
        by_category = {row["category"]: row["count"] for row in cursor.fetchall()}

        return {
            "sessions": sessions_count,
            "activities": activities_count,
            "tasks": tasks_count,
            "items": items_count,
            "items_by_category": by_category
        }

    def close(self):
        """Close database connection."""
        self.conn.close()
```

---

### 1.2 Persistent Embedding Cache

**Problem**: Embeddings are recomputed on every query, causing massive latency at scale.

**File**: `src/cogneetree/retrieval/embedding_cache.py`

```python
"""Persistent embedding cache for efficient semantic retrieval."""

import hashlib
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CachedEmbedding:
    """Cached embedding with metadata."""
    content_hash: str
    embedding: np.ndarray
    model_id: str
    created_at: datetime


class EmbeddingCacheABC(ABC):
    """Abstract base class for embedding caches."""

    @abstractmethod
    def get(self, content: str, model_id: str) -> Optional[np.ndarray]:
        """Get cached embedding for content."""
        pass

    @abstractmethod
    def set(self, content: str, embedding: np.ndarray, model_id: str) -> None:
        """Cache embedding for content."""
        pass

    @abstractmethod
    def get_batch(self, contents: List[str], model_id: str) -> Dict[str, Optional[np.ndarray]]:
        """Get cached embeddings for multiple contents."""
        pass

    @abstractmethod
    def set_batch(self, embeddings: Dict[str, np.ndarray], model_id: str) -> None:
        """Cache multiple embeddings."""
        pass

    @abstractmethod
    def invalidate(self, model_id: Optional[str] = None) -> int:
        """Invalidate cache entries. Returns count of invalidated entries."""
        pass

    @staticmethod
    def content_hash(content: str) -> str:
        """Generate hash for content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class SQLiteEmbeddingCache(EmbeddingCacheABC):
    """SQLite-backed embedding cache for persistence."""

    def __init__(self, db_path: str = "embeddings_cache.db"):
        """
        Initialize SQLite embedding cache.

        Args:
            db_path: Path to SQLite database file
        """
        import sqlite3
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                content_hash TEXT NOT NULL,
                model_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (content_hash, model_id)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_id)"
        )
        self.conn.commit()

    def get(self, content: str, model_id: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        content_hash = self.content_hash(content)
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT embedding FROM embeddings WHERE content_hash = ? AND model_id = ?",
            (content_hash, model_id)
        )
        row = cursor.fetchone()
        if row:
            return pickle.loads(row["embedding"])
        return None

    def set(self, content: str, embedding: np.ndarray, model_id: str) -> None:
        """Cache embedding."""
        content_hash = self.content_hash(content)
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT OR REPLACE INTO embeddings (content_hash, model_id, embedding)
               VALUES (?, ?, ?)""",
            (content_hash, model_id, pickle.dumps(embedding))
        )
        self.conn.commit()

    def get_batch(self, contents: List[str], model_id: str) -> Dict[str, Optional[np.ndarray]]:
        """Get cached embeddings for multiple contents."""
        results = {}
        hashes = {self.content_hash(c): c for c in contents}

        cursor = self.conn.cursor()
        placeholders = ",".join(["?" for _ in hashes])
        cursor.execute(
            f"""SELECT content_hash, embedding FROM embeddings
                WHERE content_hash IN ({placeholders}) AND model_id = ?""",
            (*hashes.keys(), model_id)
        )

        found = {row["content_hash"]: pickle.loads(row["embedding"])
                 for row in cursor.fetchall()}

        for hash_val, content in hashes.items():
            results[content] = found.get(hash_val)

        return results

    def set_batch(self, embeddings: Dict[str, np.ndarray], model_id: str) -> None:
        """Cache multiple embeddings."""
        cursor = self.conn.cursor()
        for content, embedding in embeddings.items():
            content_hash = self.content_hash(content)
            cursor.execute(
                """INSERT OR REPLACE INTO embeddings (content_hash, model_id, embedding)
                   VALUES (?, ?, ?)""",
                (content_hash, model_id, pickle.dumps(embedding))
            )
        self.conn.commit()

    def invalidate(self, model_id: Optional[str] = None) -> int:
        """Invalidate cache entries."""
        cursor = self.conn.cursor()
        if model_id:
            cursor.execute("DELETE FROM embeddings WHERE model_id = ?", (model_id,))
        else:
            cursor.execute("DELETE FROM embeddings")
        count = cursor.rowcount
        self.conn.commit()
        return count

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM embeddings")
        total = cursor.fetchone()["count"]

        cursor.execute(
            "SELECT model_id, COUNT(*) as count FROM embeddings GROUP BY model_id"
        )
        by_model = {row["model_id"]: row["count"] for row in cursor.fetchall()}

        return {"total_entries": total, "by_model": by_model}

    def close(self):
        """Close database connection."""
        self.conn.close()


class RedisEmbeddingCache(EmbeddingCacheABC):
    """Redis-backed embedding cache for distributed systems."""

    def __init__(self, host: str = "localhost", port: int = 6379,
                 db: int = 0, prefix: str = "emb:", ttl_seconds: int = 86400 * 30):
        """
        Initialize Redis embedding cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            prefix: Key prefix for cache entries
            ttl_seconds: Time-to-live for cache entries (default: 30 days)
        """
        import redis
        self.client = redis.Redis(host=host, port=port, db=db)
        self.prefix = prefix
        self.ttl = ttl_seconds

    def _key(self, content_hash: str, model_id: str) -> str:
        """Generate Redis key."""
        return f"{self.prefix}{model_id}:{content_hash}"

    def get(self, content: str, model_id: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        key = self._key(self.content_hash(content), model_id)
        data = self.client.get(key)
        if data:
            return pickle.loads(data)
        return None

    def set(self, content: str, embedding: np.ndarray, model_id: str) -> None:
        """Cache embedding."""
        key = self._key(self.content_hash(content), model_id)
        self.client.setex(key, self.ttl, pickle.dumps(embedding))

    def get_batch(self, contents: List[str], model_id: str) -> Dict[str, Optional[np.ndarray]]:
        """Get cached embeddings for multiple contents."""
        results = {}
        keys = [self._key(self.content_hash(c), model_id) for c in contents]
        values = self.client.mget(keys)

        for content, value in zip(contents, values):
            results[content] = pickle.loads(value) if value else None

        return results

    def set_batch(self, embeddings: Dict[str, np.ndarray], model_id: str) -> None:
        """Cache multiple embeddings."""
        pipe = self.client.pipeline()
        for content, embedding in embeddings.items():
            key = self._key(self.content_hash(content), model_id)
            pipe.setex(key, self.ttl, pickle.dumps(embedding))
        pipe.execute()

    def invalidate(self, model_id: Optional[str] = None) -> int:
        """Invalidate cache entries."""
        pattern = f"{self.prefix}{model_id}:*" if model_id else f"{self.prefix}*"
        keys = list(self.client.scan_iter(match=pattern))
        if keys:
            return self.client.delete(*keys)
        return 0


class CachedEmbeddingModel:
    """Wrapper that adds caching to any embedding model."""

    def __init__(self, model: Any, cache: EmbeddingCacheABC, model_id: str):
        """
        Initialize cached embedding model.

        Args:
            model: Underlying embedding model (must have encode() method)
            cache: Embedding cache instance
            model_id: Unique identifier for this model (for cache invalidation)
        """
        self.model = model
        self.cache = cache
        self.model_id = model_id
        self._hits = 0
        self._misses = 0

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts with caching.

        Args:
            texts: List of texts to encode

        Returns:
            numpy array of embeddings
        """
        # Check cache for all texts
        cached = self.cache.get_batch(texts, self.model_id)

        # Identify cache misses
        to_encode = [t for t, emb in cached.items() if emb is None]

        # Track stats
        self._hits += len(texts) - len(to_encode)
        self._misses += len(to_encode)

        # Encode missing texts
        if to_encode:
            new_embeddings = self.model.encode(to_encode)
            # Cache new embeddings
            new_cache = {text: emb for text, emb in zip(to_encode, new_embeddings)}
            self.cache.set_batch(new_cache, self.model_id)

            # Update cached dict
            for text, emb in new_cache.items():
                cached[text] = emb

        # Return in original order
        return np.array([cached[t] for t in texts])

    @property
    def cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        """Get caching statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.cache_hit_rate,
            "model_id": self.model_id
        }
```

**Usage Example**:

```python
from cogneetree.retrieval.embedding_cache import (
    SQLiteEmbeddingCache, CachedEmbeddingModel
)
from sentence_transformers import SentenceTransformer

# Create cache
cache = SQLiteEmbeddingCache("./embeddings.db")

# Wrap embedding model with cache
base_model = SentenceTransformer("all-MiniLM-L6-v2")
cached_model = CachedEmbeddingModel(
    model=base_model,
    cache=cache,
    model_id="all-MiniLM-L6-v2"
)

# First call: computes and caches
embeddings = cached_model.encode(["JWT authentication", "OAuth2 flow"])

# Second call: retrieves from cache (instant)
embeddings = cached_model.encode(["JWT authentication", "OAuth2 flow"])

# Check stats
print(cached_model.stats())
# {"hits": 2, "misses": 2, "hit_rate": 0.5, "model_id": "all-MiniLM-L6-v2"}
```

---

### 1.3 Thread-Safe Context Management

**Problem**: Context stack uses instance variables that cause race conditions in multi-threaded environments.

**File**: `src/cogneetree/core/thread_safe_context.py`

```python
"""Thread-safe context management."""

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Dict, Any
from cogneetree.core.interfaces import ContextStorageABC
from cogneetree.core.models import Session, Activity, Task


@dataclass
class ContextState:
    """Thread-local context state."""
    session_id: Optional[str] = None
    activity_id: Optional[str] = None
    task_id: Optional[str] = None


class ThreadLocalContextStack:
    """Thread-local context stack for safe concurrent access."""

    def __init__(self):
        self._local = threading.local()

    def _get_state(self) -> ContextState:
        """Get or create thread-local state."""
        if not hasattr(self._local, 'state'):
            self._local.state = ContextState()
        return self._local.state

    @property
    def current_session_id(self) -> Optional[str]:
        return self._get_state().session_id

    @current_session_id.setter
    def current_session_id(self, value: Optional[str]):
        self._get_state().session_id = value

    @property
    def current_activity_id(self) -> Optional[str]:
        return self._get_state().activity_id

    @current_activity_id.setter
    def current_activity_id(self, value: Optional[str]):
        self._get_state().activity_id = value

    @property
    def current_task_id(self) -> Optional[str]:
        return self._get_state().task_id

    @current_task_id.setter
    def current_task_id(self, value: Optional[str]):
        self._get_state().task_id = value

    def clear(self):
        """Clear thread-local state."""
        self._local.state = ContextState()


class ThreadSafeContextManager:
    """Thread-safe context manager for concurrent agent access."""

    def __init__(self, storage: ContextStorageABC):
        """
        Initialize thread-safe context manager.

        Args:
            storage: Storage backend (should also be thread-safe)
        """
        self.storage = storage
        self._context_stack = ThreadLocalContextStack()
        self._lock = threading.RLock()  # Reentrant lock for nested operations

    def create_session(self, session_id: str, original_ask: str,
                       high_level_plan: str) -> Session:
        """Create session (thread-safe)."""
        with self._lock:
            session = self.storage.create_session(session_id, original_ask, high_level_plan)
            self._context_stack.current_session_id = session_id
            return session

    def create_activity(self, activity_id: str, session_id: str, description: str,
                        tags: list, mode: str, component: str,
                        planner_analysis: str) -> Activity:
        """Create activity (thread-safe)."""
        with self._lock:
            activity = self.storage.create_activity(
                activity_id, session_id, description, tags, mode, component, planner_analysis
            )
            self._context_stack.current_activity_id = activity_id
            return activity

    def create_task(self, task_id: str, activity_id: str, description: str,
                    tags: list) -> Task:
        """Create task (thread-safe)."""
        with self._lock:
            task = self.storage.create_task(task_id, activity_id, description, tags)
            self._context_stack.current_task_id = task_id
            return task

    def get_current_session_id(self) -> Optional[str]:
        """Get current session ID for this thread."""
        return self._context_stack.current_session_id

    def get_current_activity_id(self) -> Optional[str]:
        """Get current activity ID for this thread."""
        return self._context_stack.current_activity_id

    def get_current_task_id(self) -> Optional[str]:
        """Get current task ID for this thread."""
        return self._context_stack.current_task_id

    @contextmanager
    def session_context(self, session_id: str, original_ask: str,
                        high_level_plan: str):
        """Context manager for session scope."""
        session = self.create_session(session_id, original_ask, high_level_plan)
        try:
            yield session
        finally:
            self._context_stack.current_session_id = None

    @contextmanager
    def activity_context(self, activity_id: str, session_id: str, description: str,
                         tags: list, mode: str, component: str, planner_analysis: str):
        """Context manager for activity scope."""
        activity = self.create_activity(
            activity_id, session_id, description, tags, mode, component, planner_analysis
        )
        try:
            yield activity
        finally:
            self._context_stack.current_activity_id = None

    @contextmanager
    def task_context(self, task_id: str, activity_id: str, description: str, tags: list):
        """Context manager for task scope."""
        task = self.create_task(task_id, activity_id, description, tags)
        try:
            yield task
        finally:
            self._context_stack.current_task_id = None


class ThreadSafeStorage(ContextStorageABC):
    """Thread-safe wrapper for any storage backend."""

    def __init__(self, storage: ContextStorageABC):
        """
        Wrap storage with thread safety.

        Args:
            storage: Underlying storage backend
        """
        self._storage = storage
        self._lock = threading.RLock()
        self._context_stack = ThreadLocalContextStack()

    def create_session(self, session_id: str, original_ask: str,
                       high_level_plan: str) -> Session:
        with self._lock:
            session = self._storage.create_session(session_id, original_ask, high_level_plan)
            self._context_stack.current_session_id = session_id
            return session

    def create_activity(self, activity_id: str, session_id: str, description: str,
                        tags: list, mode: str, component: str,
                        planner_analysis: str) -> Activity:
        with self._lock:
            activity = self._storage.create_activity(
                activity_id, session_id, description, tags, mode, component, planner_analysis
            )
            self._context_stack.current_activity_id = activity_id
            return activity

    def create_task(self, task_id: str, activity_id: str, description: str,
                    tags: list) -> Task:
        with self._lock:
            task = self._storage.create_task(task_id, activity_id, description, tags)
            self._context_stack.current_task_id = task_id
            return task

    def add_item(self, item) -> None:
        with self._lock:
            self._storage.add_item(item)

    def get_items_by_task(self, task_id: str):
        with self._lock:
            return self._storage.get_items_by_task(task_id)

    def get_items_by_activity(self, activity_id: str):
        with self._lock:
            return self._storage.get_items_by_activity(activity_id)

    def get_items_by_session(self, session_id: str):
        with self._lock:
            return self._storage.get_items_by_session(session_id)

    def get_all_sessions(self):
        with self._lock:
            return self._storage.get_all_sessions()

    # Thread-local context accessors
    def get_current_session_id(self) -> Optional[str]:
        return self._context_stack.current_session_id

    def get_current_activity_id(self) -> Optional[str]:
        return self._context_stack.current_activity_id

    def get_current_task_id(self) -> Optional[str]:
        return self._context_stack.current_task_id

    # Delegate remaining methods with locking
    def __getattr__(self, name):
        attr = getattr(self._storage, name)
        if callable(attr):
            def locked_method(*args, **kwargs):
                with self._lock:
                    return attr(*args, **kwargs)
            return locked_method
        return attr
```

**Usage Example**:

```python
from cogneetree import ContextManager
from cogneetree.storage.in_memory_storage import InMemoryStorage
from cogneetree.core.thread_safe_context import ThreadSafeStorage
import threading

# Wrap storage for thread safety
base_storage = InMemoryStorage()
safe_storage = ThreadSafeStorage(base_storage)
manager = ContextManager(storage=safe_storage)

def agent_work(agent_id: str):
    """Work performed by each agent thread."""
    # Each thread has isolated context
    manager.create_session(f"session_{agent_id}", f"Agent {agent_id} work", "Plan")
    manager.create_activity(f"act_{agent_id}", f"session_{agent_id}", "Activity", ["tag"], "builder", "core", "...")
    manager.create_task(f"task_{agent_id}", f"act_{agent_id}", "Task", ["tag"])

    # Record learnings - thread-safe
    manager.record_learning(f"Learning from agent {agent_id}", tags=["agent", agent_id])

# Run multiple agents concurrently
threads = []
for i in range(5):
    t = threading.Thread(target=agent_work, args=(f"agent_{i}",))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

# All data safely recorded
print(safe_storage.get_stats())
```

---

## Phase 2: Table Stakes Features

### 2.1 Batch Operations

**Problem**: No way to efficiently add multiple items or import/export data.

**File**: `src/cogneetree/core/batch_operations.py`

```python
"""Batch operations for efficient bulk data handling."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from cogneetree.core.interfaces import ContextStorageABC
from cogneetree.core.models import (
    ContextItem, ContextCategory, Session, Activity, Task
)


class BatchOperations:
    """Batch operations for efficient bulk data handling."""

    def __init__(self, storage: ContextStorageABC):
        """
        Initialize batch operations.

        Args:
            storage: Storage backend
        """
        self.storage = storage

    # ===== BATCH RECORDING =====

    def record_items(self, items: List[Tuple[str, str, List[str]]],
                     parent_id: Optional[str] = None) -> List[ContextItem]:
        """
        Record multiple items efficiently.

        Args:
            items: List of (category, content, tags) tuples
                   category: "action", "decision", "learning", "result"
            parent_id: Parent task ID (defaults to current task)

        Returns:
            List of created ContextItem objects

        Example:
            batch.record_items([
                ("learning", "JWT has 3 parts", ["jwt"]),
                ("decision", "Use HS256", ["jwt", "security"]),
                ("action", "Read RFC 7519", ["jwt", "research"]),
            ])
        """
        if parent_id is None:
            parent_id = self.storage.get_current_task_id()

        category_map = {
            "action": ContextCategory.ACTION,
            "decision": ContextCategory.DECISION,
            "learning": ContextCategory.LEARNING,
            "result": ContextCategory.RESULT,
        }

        created_items = []
        for category_str, content, tags in items:
            category = category_map.get(category_str.lower())
            if not category:
                raise ValueError(f"Invalid category: {category_str}")

            item = ContextItem(
                content=content,
                category=category,
                tags=tags,
                parent_id=parent_id
            )
            self.storage.add_item(item)
            created_items.append(item)

        return created_items

    def record_learnings(self, learnings: List[Tuple[str, List[str]]],
                         parent_id: Optional[str] = None) -> List[ContextItem]:
        """
        Record multiple learnings.

        Args:
            learnings: List of (content, tags) tuples

        Example:
            batch.record_learnings([
                ("JWT structure: header.payload.signature", ["jwt"]),
                ("Use RS256 for asymmetric signing", ["jwt", "security"]),
            ])
        """
        items = [("learning", content, tags) for content, tags in learnings]
        return self.record_items(items, parent_id)

    def record_decisions(self, decisions: List[Tuple[str, List[str]]],
                         parent_id: Optional[str] = None) -> List[ContextItem]:
        """Record multiple decisions."""
        items = [("decision", content, tags) for content, tags in decisions]
        return self.record_items(items, parent_id)

    # ===== EXPORT OPERATIONS =====

    def export_session(self, session_id: str, format: str = "json") -> Union[str, Dict]:
        """
        Export a complete session with all activities, tasks, and items.

        Args:
            session_id: Session to export
            format: "json" or "dict"

        Returns:
            JSON string or dictionary
        """
        session = self.storage.get_session_by_id(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        export_data = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "session": {
                "session_id": session.session_id,
                "original_ask": session.original_ask,
                "high_level_plan": session.high_level_plan,
                "created_at": session.created_at.isoformat() if session.created_at else None
            },
            "activities": [],
            "tasks": [],
            "items": []
        }

        # Export activities
        activities = self.storage.get_session_activities(session_id)
        for activity in activities:
            export_data["activities"].append({
                "activity_id": activity.activity_id,
                "session_id": activity.session_id,
                "description": activity.description,
                "tags": activity.tags,
                "mode": activity.mode,
                "component": activity.component,
                "planner_analysis": activity.planner_analysis
            })

            # Export tasks for this activity
            tasks = self.storage.get_activity_tasks(activity.activity_id)
            for task in tasks:
                export_data["tasks"].append({
                    "task_id": task.task_id,
                    "activity_id": task.activity_id,
                    "description": task.description,
                    "tags": task.tags,
                    "result": task.result
                })

                # Export items for this task
                items = self.storage.get_items_by_task(task.task_id)
                for item in items:
                    export_data["items"].append({
                        "content": item.content,
                        "category": item.category.value,
                        "tags": item.tags,
                        "timestamp": item.timestamp.isoformat() if item.timestamp else None,
                        "parent_id": item.parent_id,
                        "metadata": item.metadata
                    })

        if format == "json":
            return json.dumps(export_data, indent=2)
        return export_data

    def export_to_file(self, session_id: str, file_path: str) -> None:
        """Export session to file."""
        data = self.export_session(session_id, format="json")
        Path(file_path).write_text(data)

    # ===== IMPORT OPERATIONS =====

    def import_session(self, data: Union[str, Dict],
                       session_id_override: Optional[str] = None) -> Session:
        """
        Import a complete session from exported data.

        Args:
            data: JSON string or dictionary from export_session()
            session_id_override: Optional new session ID (avoids conflicts)

        Returns:
            Created Session object
        """
        if isinstance(data, str):
            data = json.loads(data)

        # Create session
        session_data = data["session"]
        session_id = session_id_override or session_data["session_id"]
        session = self.storage.create_session(
            session_id,
            session_data["original_ask"],
            session_data["high_level_plan"]
        )

        # Map old IDs to new IDs
        activity_id_map = {}
        task_id_map = {}

        # Import activities
        for act_data in data["activities"]:
            old_id = act_data["activity_id"]
            new_id = f"{session_id}_{old_id}" if session_id_override else old_id

            activity = self.storage.create_activity(
                new_id,
                session_id,
                act_data["description"],
                act_data["tags"],
                act_data["mode"],
                act_data["component"],
                act_data["planner_analysis"]
            )
            activity_id_map[old_id] = new_id

        # Import tasks
        for task_data in data["tasks"]:
            old_id = task_data["task_id"]
            new_activity_id = activity_id_map.get(
                task_data["activity_id"],
                task_data["activity_id"]
            )
            new_id = f"{session_id}_{old_id}" if session_id_override else old_id

            task = self.storage.create_task(
                new_id,
                new_activity_id,
                task_data["description"],
                task_data["tags"]
            )
            task_id_map[old_id] = new_id

        # Import items
        for item_data in data["items"]:
            new_parent_id = task_id_map.get(
                item_data["parent_id"],
                item_data["parent_id"]
            )

            item = ContextItem(
                content=item_data["content"],
                category=ContextCategory(item_data["category"]),
                tags=item_data["tags"],
                timestamp=datetime.fromisoformat(item_data["timestamp"]) if item_data["timestamp"] else datetime.now(),
                parent_id=new_parent_id,
                metadata=item_data.get("metadata", {})
            )
            self.storage.add_item(item)

        return session

    def import_from_file(self, file_path: str,
                         session_id_override: Optional[str] = None) -> Session:
        """Import session from file."""
        data = Path(file_path).read_text()
        return self.import_session(data, session_id_override)

    # ===== MERGE OPERATIONS =====

    def merge_sessions(self, source_session_id: str, target_session_id: str,
                       delete_source: bool = False) -> int:
        """
        Merge one session into another.

        Args:
            source_session_id: Session to merge from
            target_session_id: Session to merge into
            delete_source: Whether to delete source session after merge

        Returns:
            Number of items merged
        """
        # Export source
        source_data = self.export_session(source_session_id, format="dict")

        # Count items
        item_count = len(source_data["items"])

        # Import into target (with ID remapping)
        # This creates new activities/tasks under target session
        for activity in source_data["activities"]:
            activity["session_id"] = target_session_id

        # Re-import (simplified - in production, handle ID conflicts better)
        source_data["session"]["session_id"] = target_session_id
        self.import_session(source_data, session_id_override=f"{target_session_id}_merged")

        return item_count
```

**Usage Example**:

```python
from cogneetree import ContextManager
from cogneetree.core.batch_operations import BatchOperations

manager = ContextManager()
batch = BatchOperations(manager.storage)

# Setup context
manager.create_session("s1", "JWT Implementation", "Plan")
manager.create_activity("a1", "s1", "Research", ["jwt"], "learner", "core", "...")
manager.create_task("t1", "a1", "Learn JWT", ["jwt"])

# Batch record multiple learnings
batch.record_learnings([
    ("JWT has three parts: header.payload.signature", ["jwt", "structure"]),
    ("Header contains algorithm and token type", ["jwt", "header"]),
    ("Payload contains claims (sub, exp, iat)", ["jwt", "payload"]),
    ("Signature verifies token integrity", ["jwt", "signature"]),
])

# Batch record mixed items
batch.record_items([
    ("decision", "Use HS256 for symmetric signing", ["jwt", "security"]),
    ("action", "Read RFC 7519 specification", ["jwt", "research"]),
    ("result", "Complete understanding of JWT structure", ["jwt"]),
])

# Export session
json_data = batch.export_session("s1")
batch.export_to_file("s1", "./session_backup.json")

# Import session (with new ID to avoid conflicts)
batch.import_from_file("./session_backup.json", session_id_override="s1_restored")
```

---

### 2.2 Update/Delete Operations

**Problem**: Items are immutable once created - no way to edit or delete.

**File**: `src/cogneetree/core/mutable_operations.py`

```python
"""Mutable operations for updating and deleting items."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable
from cogneetree.core.interfaces import ContextStorageABC
from cogneetree.core.models import ContextItem, ContextCategory


@dataclass
class ItemUpdate:
    """Represents an update to an item."""
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class MutableStorage:
    """Wrapper that adds update/delete operations to storage."""

    def __init__(self, storage: ContextStorageABC):
        """
        Initialize mutable storage wrapper.

        Args:
            storage: Underlying storage backend
        """
        self.storage = storage
        self._item_index: Dict[str, ContextItem] = {}  # item_id -> item
        self._deleted_ids: set = set()

    def _generate_item_id(self, item: ContextItem) -> str:
        """Generate unique ID for item."""
        import hashlib
        content = f"{item.content}:{item.timestamp.isoformat()}:{item.parent_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def add_item(self, item: ContextItem) -> str:
        """
        Add item and return its ID.

        Args:
            item: Item to add

        Returns:
            Generated item ID
        """
        item_id = self._generate_item_id(item)
        item.metadata["_item_id"] = item_id
        self.storage.add_item(item)
        self._item_index[item_id] = item
        return item_id

    def get_item(self, item_id: str) -> Optional[ContextItem]:
        """
        Get item by ID.

        Args:
            item_id: Item ID

        Returns:
            ContextItem or None if not found/deleted
        """
        if item_id in self._deleted_ids:
            return None
        return self._item_index.get(item_id)

    def update_item(self, item_id: str, update: ItemUpdate) -> Optional[ContextItem]:
        """
        Update an existing item.

        Args:
            item_id: ID of item to update
            update: ItemUpdate with fields to change

        Returns:
            Updated ContextItem or None if not found

        Example:
            storage.update_item("abc123", ItemUpdate(
                content="Updated content",
                tags=["new", "tags"]
            ))
        """
        item = self.get_item(item_id)
        if not item:
            return None

        # Apply updates
        if update.content is not None:
            item.content = update.content
        if update.tags is not None:
            item.tags = update.tags
        if update.metadata is not None:
            item.metadata.update(update.metadata)

        # Track modification
        item.metadata["_modified_at"] = datetime.now().isoformat()
        item.metadata["_version"] = item.metadata.get("_version", 0) + 1

        return item

    def delete_item(self, item_id: str) -> bool:
        """
        Soft-delete an item.

        Args:
            item_id: ID of item to delete

        Returns:
            True if deleted, False if not found
        """
        if item_id not in self._item_index:
            return False

        self._deleted_ids.add(item_id)
        item = self._item_index[item_id]
        item.metadata["_deleted_at"] = datetime.now().isoformat()
        return True

    def restore_item(self, item_id: str) -> bool:
        """
        Restore a soft-deleted item.

        Args:
            item_id: ID of item to restore

        Returns:
            True if restored, False if not found
        """
        if item_id not in self._deleted_ids:
            return False

        self._deleted_ids.discard(item_id)
        item = self._item_index[item_id]
        item.metadata.pop("_deleted_at", None)
        return True

    def hard_delete(self, item_id: str) -> bool:
        """
        Permanently delete an item.

        Args:
            item_id: ID of item to permanently delete

        Returns:
            True if deleted, False if not found
        """
        if item_id not in self._item_index:
            return False

        self._deleted_ids.discard(item_id)
        del self._item_index[item_id]
        return True

    def get_items_by_task(self, task_id: str) -> List[ContextItem]:
        """Get items for task, excluding deleted items."""
        all_items = self.storage.get_items_by_task(task_id)
        return [
            item for item in all_items
            if item.metadata.get("_item_id") not in self._deleted_ids
        ]

    def find_and_update(self, predicate: Callable[[ContextItem], bool],
                        update: ItemUpdate) -> int:
        """
        Find items matching predicate and update them.

        Args:
            predicate: Function that returns True for items to update
            update: ItemUpdate to apply

        Returns:
            Number of items updated
        """
        count = 0
        for item_id, item in self._item_index.items():
            if item_id not in self._deleted_ids and predicate(item):
                self.update_item(item_id, update)
                count += 1
        return count

    def find_and_delete(self, predicate: Callable[[ContextItem], bool]) -> int:
        """
        Find items matching predicate and delete them.

        Args:
            predicate: Function that returns True for items to delete

        Returns:
            Number of items deleted
        """
        count = 0
        for item_id, item in list(self._item_index.items()):
            if item_id not in self._deleted_ids and predicate(item):
                self.delete_item(item_id)
                count += 1
        return count

    def cleanup_deleted(self, older_than_days: int = 30) -> int:
        """
        Permanently remove items deleted more than N days ago.

        Args:
            older_than_days: Only cleanup items deleted this many days ago

        Returns:
            Number of items permanently removed
        """
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=older_than_days)
        count = 0

        for item_id in list(self._deleted_ids):
            item = self._item_index.get(item_id)
            if item:
                deleted_at = item.metadata.get("_deleted_at")
                if deleted_at:
                    deleted_time = datetime.fromisoformat(deleted_at)
                    if deleted_time < cutoff:
                        self.hard_delete(item_id)
                        count += 1

        return count

    # Delegate to underlying storage
    def __getattr__(self, name):
        return getattr(self.storage, name)
```

**Usage Example**:

```python
from cogneetree import ContextManager
from cogneetree.core.mutable_operations import MutableStorage, ItemUpdate

# Wrap storage with mutable operations
manager = ContextManager()
mutable = MutableStorage(manager.storage)

# Setup
manager.create_session("s1", "Test", "Plan")
manager.create_activity("a1", "s1", "Activity", ["test"], "learner", "core", "...")
manager.create_task("t1", "a1", "Task", ["test"])

# Add item and get ID
from cogneetree.core.models import ContextItem, ContextCategory
item = ContextItem(
    content="Initial learning",
    category=ContextCategory.LEARNING,
    tags=["initial"],
    parent_id="t1"
)
item_id = mutable.add_item(item)
print(f"Created item: {item_id}")

# Update item
mutable.update_item(item_id, ItemUpdate(
    content="Updated learning with more detail",
    tags=["initial", "updated"]
))

# Soft delete
mutable.delete_item(item_id)

# Restore
mutable.restore_item(item_id)

# Bulk update: add "reviewed" tag to all learnings
mutable.find_and_update(
    predicate=lambda item: item.category == ContextCategory.LEARNING,
    update=ItemUpdate(tags=["reviewed"])
)

# Bulk delete: remove all items older than 90 days
from datetime import datetime, timedelta
cutoff = datetime.now() - timedelta(days=90)
mutable.find_and_delete(
    predicate=lambda item: item.timestamp < cutoff
)
```

---

### 2.3 Analytics & Observability

**Problem**: No visibility into memory usage, retrieval patterns, or agent learning.

**File**: `src/cogneetree/analytics/analytics.py`

```python
"""Analytics and observability for Cogneetree."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from cogneetree.core.interfaces import ContextStorageABC
from cogneetree.core.models import ContextCategory


@dataclass
class QueryLogEntry:
    """Log entry for a retrieval query."""
    timestamp: datetime
    query: str
    scope: str
    results_count: int
    latency_ms: float
    top_result_score: Optional[float] = None
    tags_searched: List[str] = field(default_factory=list)


@dataclass
class AnalyticsSnapshot:
    """Point-in-time analytics snapshot."""
    timestamp: datetime
    total_items: int
    items_by_category: Dict[str, int]
    items_by_session: Dict[str, int]
    top_tags: List[tuple]
    avg_items_per_task: float
    learning_velocity: float  # items per day


class MemoryAnalytics:
    """Analytics engine for Cogneetree memory."""

    def __init__(self, storage: ContextStorageABC):
        """
        Initialize analytics.

        Args:
            storage: Storage backend to analyze
        """
        self.storage = storage
        self._query_log: List[QueryLogEntry] = []
        self._enabled = True

    # ===== QUERY LOGGING =====

    def log_query(self, query: str, scope: str, results_count: int,
                  latency_ms: float, top_score: Optional[float] = None,
                  tags: Optional[List[str]] = None) -> None:
        """Log a retrieval query."""
        if not self._enabled:
            return

        entry = QueryLogEntry(
            timestamp=datetime.now(),
            query=query,
            scope=scope,
            results_count=results_count,
            latency_ms=latency_ms,
            top_result_score=top_score,
            tags_searched=tags or []
        )
        self._query_log.append(entry)

    def enable_logging(self) -> None:
        """Enable query logging."""
        self._enabled = True

    def disable_logging(self) -> None:
        """Disable query logging."""
        self._enabled = False

    def query_history(self, limit: int = 100,
                      since: Optional[datetime] = None) -> List[QueryLogEntry]:
        """
        Get recent query history.

        Args:
            limit: Maximum entries to return
            since: Only return entries after this time

        Returns:
            List of QueryLogEntry objects
        """
        entries = self._query_log
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        return entries[-limit:]

    # ===== STORAGE STATISTICS =====

    def stats(self) -> Dict[str, Any]:
        """
        Get comprehensive storage statistics.

        Returns:
            Dictionary with storage stats
        """
        # Get all items
        all_items = self.storage.get_all_items() if hasattr(self.storage, 'get_all_items') else []

        # Count by category
        by_category = defaultdict(int)
        for item in all_items:
            by_category[item.category.value] += 1

        # Count by session
        by_session = defaultdict(int)
        sessions = self.storage.get_all_sessions() if hasattr(self.storage, 'get_all_sessions') else []
        for session in sessions:
            items = self.storage.get_items_by_session(session.session_id) if hasattr(self.storage, 'get_items_by_session') else []
            by_session[session.session_id] = len(items)

        # Tag frequency
        tag_counts = defaultdict(int)
        for item in all_items:
            for tag in item.tags:
                tag_counts[tag] += 1
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]

        # Storage backend stats
        storage_stats = {}
        if hasattr(self.storage, 'get_stats'):
            storage_stats = self.storage.get_stats()

        return {
            "total_items": len(all_items),
            "items_by_category": dict(by_category),
            "items_by_session": dict(by_session),
            "top_tags": top_tags,
            "sessions_count": len(sessions),
            "storage_backend": storage_stats
        }

    # ===== RETRIEVAL STATISTICS =====

    def retrieval_stats(self, window_hours: int = 24) -> Dict[str, Any]:
        """
        Get retrieval performance statistics.

        Args:
            window_hours: Time window to analyze

        Returns:
            Dictionary with retrieval stats
        """
        since = datetime.now() - timedelta(hours=window_hours)
        recent_queries = [q for q in self._query_log if q.timestamp >= since]

        if not recent_queries:
            return {
                "queries_count": 0,
                "avg_latency_ms": 0,
                "avg_results_count": 0,
                "cache_hit_rate": 0,
                "scope_distribution": {}
            }

        latencies = [q.latency_ms for q in recent_queries]
        results_counts = [q.results_count for q in recent_queries]

        # Scope distribution
        scope_counts = defaultdict(int)
        for q in recent_queries:
            scope_counts[q.scope] += 1

        return {
            "queries_count": len(recent_queries),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 20 else max(latencies),
            "avg_results_count": sum(results_counts) / len(results_counts),
            "zero_result_queries": sum(1 for q in recent_queries if q.results_count == 0),
            "scope_distribution": dict(scope_counts)
        }

    # ===== LEARNING VELOCITY =====

    def learning_velocity(self, window_days: int = 7) -> Dict[str, Any]:
        """
        Calculate learning velocity (items recorded per day).

        Args:
            window_days: Time window to analyze

        Returns:
            Dictionary with learning velocity metrics
        """
        since = datetime.now() - timedelta(days=window_days)
        all_items = self.storage.get_all_items() if hasattr(self.storage, 'get_all_items') else []

        # Filter to recent items
        recent_items = [
            item for item in all_items
            if item.timestamp and item.timestamp >= since
        ]

        # Group by day
        by_day = defaultdict(int)
        for item in recent_items:
            day = item.timestamp.date().isoformat()
            by_day[day] += 1

        # Group by category
        by_category = defaultdict(int)
        for item in recent_items:
            by_category[item.category.value] += 1

        total_days = max(window_days, 1)
        total_items = len(recent_items)

        return {
            "window_days": window_days,
            "total_items": total_items,
            "items_per_day": total_items / total_days,
            "by_day": dict(by_day),
            "by_category": dict(by_category),
            "trend": self._calculate_trend(by_day, window_days)
        }

    def _calculate_trend(self, by_day: Dict[str, int], window_days: int) -> str:
        """Calculate trend direction."""
        if len(by_day) < 3:
            return "insufficient_data"

        days = sorted(by_day.keys())
        first_half = sum(by_day.get(d, 0) for d in days[:len(days)//2])
        second_half = sum(by_day.get(d, 0) for d in days[len(days)//2:])

        if second_half > first_half * 1.2:
            return "increasing"
        elif second_half < first_half * 0.8:
            return "decreasing"
        return "stable"

    # ===== KNOWLEDGE GAPS =====

    def knowledge_gaps(self) -> Dict[str, Any]:
        """
        Identify potential knowledge gaps.

        Returns:
            Dictionary with knowledge gap analysis
        """
        all_items = self.storage.get_all_items() if hasattr(self.storage, 'get_all_items') else []

        # Tags with few items
        tag_counts = defaultdict(int)
        for item in all_items:
            for tag in item.tags:
                tag_counts[tag] += 1

        sparse_tags = [(tag, count) for tag, count in tag_counts.items() if count < 3]

        # Categories with few items
        category_counts = defaultdict(int)
        for item in all_items:
            category_counts[item.category.value] += 1

        # Tasks with no items
        sessions = self.storage.get_all_sessions() if hasattr(self.storage, 'get_all_sessions') else []
        empty_tasks = []
        for session in sessions:
            activities = self.storage.get_session_activities(session.session_id) if hasattr(self.storage, 'get_session_activities') else []
            for activity in activities:
                tasks = self.storage.get_activity_tasks(activity.activity_id) if hasattr(self.storage, 'get_activity_tasks') else []
                for task in tasks:
                    items = self.storage.get_items_by_task(task.task_id)
                    if not items:
                        empty_tasks.append({
                            "task_id": task.task_id,
                            "description": task.description
                        })

        return {
            "sparse_tags": sparse_tags,
            "category_distribution": dict(category_counts),
            "empty_tasks": empty_tasks[:10],  # Limit to 10
            "total_empty_tasks": len(empty_tasks)
        }

    # ===== SNAPSHOT =====

    def snapshot(self) -> AnalyticsSnapshot:
        """
        Create a point-in-time analytics snapshot.

        Returns:
            AnalyticsSnapshot object
        """
        stats = self.stats()
        velocity = self.learning_velocity(window_days=7)

        # Calculate avg items per task
        sessions = self.storage.get_all_sessions() if hasattr(self.storage, 'get_all_sessions') else []
        task_count = 0
        for session in sessions:
            activities = self.storage.get_session_activities(session.session_id) if hasattr(self.storage, 'get_session_activities') else []
            for activity in activities:
                tasks = self.storage.get_activity_tasks(activity.activity_id) if hasattr(self.storage, 'get_activity_tasks') else []
                task_count += len(tasks)

        avg_items_per_task = stats["total_items"] / max(task_count, 1)

        return AnalyticsSnapshot(
            timestamp=datetime.now(),
            total_items=stats["total_items"],
            items_by_category=stats["items_by_category"],
            items_by_session=stats["items_by_session"],
            top_tags=stats["top_tags"],
            avg_items_per_task=avg_items_per_task,
            learning_velocity=velocity["items_per_day"]
        )
```

**Usage Example**:

```python
from cogneetree import ContextManager, AgentMemory
from cogneetree.analytics.analytics import MemoryAnalytics
import time

manager = ContextManager()
analytics = MemoryAnalytics(manager.storage)

# Setup and record items
manager.create_session("s1", "JWT Project", "Plan")
manager.create_activity("a1", "s1", "Research", ["jwt"], "learner", "core", "...")
manager.create_task("t1", "a1", "Learn JWT", ["jwt"])
manager.record_learning("JWT has 3 parts", tags=["jwt"])
manager.record_decision("Use HS256", tags=["jwt", "security"])

# Log retrieval queries (normally done automatically by memory)
start = time.time()
memory = AgentMemory(manager.storage, current_task_id="t1")
results = memory.recall("JWT structure")
latency = (time.time() - start) * 1000

analytics.log_query(
    query="JWT structure",
    scope="balanced",
    results_count=len(results),
    latency_ms=latency,
    top_score=results[0].metadata.get("score") if results else None
)

# Get comprehensive stats
print("=== Storage Stats ===")
print(analytics.stats())

print("\n=== Retrieval Stats ===")
print(analytics.retrieval_stats(window_hours=24))

print("\n=== Learning Velocity ===")
print(analytics.learning_velocity(window_days=7))

print("\n=== Knowledge Gaps ===")
print(analytics.knowledge_gaps())

print("\n=== Snapshot ===")
snapshot = analytics.snapshot()
print(f"Total items: {snapshot.total_items}")
print(f"Learning velocity: {snapshot.learning_velocity:.2f} items/day")
```

---

## Phase 3: Differentiation Features

### 3.1 Automatic Knowledge Extraction

**Problem**: Agents must manually call `record_learning()`, `record_decision()` - high friction.

**Solution**: Auto-extract structured knowledge from LLM conversations.

**File**: `src/cogneetree/extraction/knowledge_extractor.py`

```python
"""Automatic knowledge extraction from conversations."""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from cogneetree.core.models import ContextItem, ContextCategory


class ExtractionConfidence(Enum):
    """Confidence level of extraction."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ExtractedKnowledge:
    """Extracted knowledge item with confidence."""
    category: ContextCategory
    content: str
    tags: List[str]
    confidence: ExtractionConfidence
    source_text: str
    reasoning: str


class KnowledgeExtractor:
    """Extract structured knowledge from conversations."""

    # Signal phrases for each category
    DECISION_SIGNALS = [
        r"(?:I |we |let's |should |will |decided to |choosing |choose |chose |opting for |opted for |going with |went with )",
        r"(?:use |using |implement |implementing |adopt |adopting )",
        r"(?:instead of |rather than |over |prefer |better to )",
    ]

    LEARNING_SIGNALS = [
        r"(?:I |we )(?:learned |discovered |found out |realized |understood |noticed )",
        r"(?:turns out |apparently |it seems |TIL |today I learned )",
        r"(?:the |a )(?:key |important |interesting |notable )(?:thing |insight |point |observation )",
        r"(?:consists of |is made of |has \d+ parts |structure is )",
    ]

    ACTION_SIGNALS = [
        r"(?:I |we )(?:read |wrote |created |built |implemented |fixed |updated |modified |refactored )",
        r"(?:analyzed |reviewed |tested |debugged |deployed |configured |installed )",
    ]

    RESULT_SIGNALS = [
        r"(?:now |finally |successfully |completed |finished |done |working |works )",
        r"(?:achieved |accomplished |delivered |shipped |launched )",
        r"(?:the result |outcome |conclusion )(?:is |was )",
    ]

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize knowledge extractor.

        Args:
            llm_client: Optional LLM client for enhanced extraction
        """
        self.llm_client = llm_client
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for faster matching."""
        self.decision_pattern = re.compile(
            "|".join(self.DECISION_SIGNALS), re.IGNORECASE
        )
        self.learning_pattern = re.compile(
            "|".join(self.LEARNING_SIGNALS), re.IGNORECASE
        )
        self.action_pattern = re.compile(
            "|".join(self.ACTION_SIGNALS), re.IGNORECASE
        )
        self.result_pattern = re.compile(
            "|".join(self.RESULT_SIGNALS), re.IGNORECASE
        )

    def extract_from_text(self, text: str,
                          context_tags: Optional[List[str]] = None) -> List[ExtractedKnowledge]:
        """
        Extract knowledge from text using pattern matching.

        Args:
            text: Text to extract from
            context_tags: Optional tags to add to all extractions

        Returns:
            List of ExtractedKnowledge items
        """
        extractions = []
        sentences = self._split_sentences(text)
        context_tags = context_tags or []

        for sentence in sentences:
            # Try each category
            if self.decision_pattern.search(sentence):
                extractions.append(self._create_extraction(
                    sentence, ContextCategory.DECISION, context_tags, "Pattern: decision signal"
                ))
            elif self.learning_pattern.search(sentence):
                extractions.append(self._create_extraction(
                    sentence, ContextCategory.LEARNING, context_tags, "Pattern: learning signal"
                ))
            elif self.action_pattern.search(sentence):
                extractions.append(self._create_extraction(
                    sentence, ContextCategory.ACTION, context_tags, "Pattern: action signal"
                ))
            elif self.result_pattern.search(sentence):
                extractions.append(self._create_extraction(
                    sentence, ContextCategory.RESULT, context_tags, "Pattern: result signal"
                ))

        return extractions

    def extract_from_conversation(
        self,
        messages: List[Dict[str, str]],
        context_tags: Optional[List[str]] = None
    ) -> List[ExtractedKnowledge]:
        """
        Extract knowledge from a conversation.

        Args:
            messages: List of {"role": str, "content": str} messages
            context_tags: Optional tags to add to all extractions

        Returns:
            List of ExtractedKnowledge items
        """
        extractions = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Focus on assistant responses (where decisions/learnings are explained)
            if role == "assistant":
                extractions.extend(self.extract_from_text(content, context_tags))

        # Deduplicate similar extractions
        return self._deduplicate(extractions)

    def extract_with_llm(
        self,
        text: str,
        context_tags: Optional[List[str]] = None
    ) -> List[ExtractedKnowledge]:
        """
        Extract knowledge using LLM for higher accuracy.

        Args:
            text: Text to extract from
            context_tags: Optional tags to add

        Returns:
            List of ExtractedKnowledge items
        """
        if not self.llm_client:
            return self.extract_from_text(text, context_tags)

        prompt = f"""Analyze the following text and extract structured knowledge items.

For each item, identify:
1. Category: DECISION (choices made), LEARNING (facts discovered), ACTION (work done), RESULT (outcomes achieved)
2. Content: A clear, standalone statement
3. Tags: Relevant keywords (lowercase, underscore-separated)
4. Confidence: HIGH (explicit statement), MEDIUM (implied), LOW (inferred)

Text to analyze:
{text}

Return as JSON array:
[{{"category": "DECISION", "content": "...", "tags": ["tag1", "tag2"], "confidence": "HIGH"}}]

Extract ONLY clear, actionable knowledge. Skip conversational filler."""

        # Call LLM (implementation depends on client)
        try:
            response = self._call_llm(prompt)
            return self._parse_llm_response(response, text, context_tags)
        except Exception as e:
            # Fallback to pattern matching
            return self.extract_from_text(text, context_tags)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (could use nltk for better results)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _create_extraction(
        self,
        sentence: str,
        category: ContextCategory,
        context_tags: List[str],
        reasoning: str
    ) -> ExtractedKnowledge:
        """Create an extraction from a sentence."""
        # Extract tags from sentence
        tags = self._extract_tags(sentence) + context_tags

        # Determine confidence based on signal strength
        confidence = ExtractionConfidence.MEDIUM

        return ExtractedKnowledge(
            category=category,
            content=sentence.strip(),
            tags=list(set(tags)),  # Dedupe
            confidence=confidence,
            source_text=sentence,
            reasoning=reasoning
        )

    def _extract_tags(self, text: str) -> List[str]:
        """Extract potential tags from text."""
        # Common technical terms to tag
        tech_terms = [
            "jwt", "oauth", "api", "rest", "graphql", "database", "sql",
            "redis", "cache", "auth", "security", "encryption", "hash",
            "token", "session", "cookie", "http", "https", "ssl", "tls"
        ]

        text_lower = text.lower()
        found_tags = [term for term in tech_terms if term in text_lower]

        return found_tags

    def _deduplicate(self, extractions: List[ExtractedKnowledge]) -> List[ExtractedKnowledge]:
        """Remove near-duplicate extractions."""
        seen_content = set()
        unique = []

        for ext in extractions:
            # Simple dedup by normalized content
            normalized = ext.content.lower().strip()
            if normalized not in seen_content:
                seen_content.add(normalized)
                unique.append(ext)

        return unique

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt."""
        # Implementation depends on client type
        if hasattr(self.llm_client, 'messages'):
            # Anthropic client
            response = self.llm_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        elif hasattr(self.llm_client, 'chat'):
            # OpenAI client
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        else:
            raise ValueError("Unsupported LLM client")

    def _parse_llm_response(
        self,
        response: str,
        source_text: str,
        context_tags: Optional[List[str]]
    ) -> List[ExtractedKnowledge]:
        """Parse LLM response into extractions."""
        import json

        try:
            # Find JSON array in response
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if not match:
                return []

            data = json.loads(match.group())
            extractions = []

            category_map = {
                "DECISION": ContextCategory.DECISION,
                "LEARNING": ContextCategory.LEARNING,
                "ACTION": ContextCategory.ACTION,
                "RESULT": ContextCategory.RESULT,
            }

            confidence_map = {
                "HIGH": ExtractionConfidence.HIGH,
                "MEDIUM": ExtractionConfidence.MEDIUM,
                "LOW": ExtractionConfidence.LOW,
            }

            for item in data:
                category = category_map.get(item.get("category", "").upper())
                if not category:
                    continue

                tags = item.get("tags", []) + (context_tags or [])
                confidence = confidence_map.get(
                    item.get("confidence", "MEDIUM").upper(),
                    ExtractionConfidence.MEDIUM
                )

                extractions.append(ExtractedKnowledge(
                    category=category,
                    content=item.get("content", ""),
                    tags=list(set(tags)),
                    confidence=confidence,
                    source_text=source_text,
                    reasoning="LLM extraction"
                ))

            return extractions

        except (json.JSONDecodeError, KeyError):
            return []


class ConversationObserver:
    """Observes conversations and auto-records knowledge."""

    def __init__(self, manager, extractor: Optional[KnowledgeExtractor] = None,
                 auto_record: bool = True, min_confidence: ExtractionConfidence = ExtractionConfidence.MEDIUM):
        """
        Initialize conversation observer.

        Args:
            manager: ContextManager instance
            extractor: KnowledgeExtractor instance
            auto_record: Whether to automatically record extractions
            min_confidence: Minimum confidence to auto-record
        """
        self.manager = manager
        self.extractor = extractor or KnowledgeExtractor()
        self.auto_record = auto_record
        self.min_confidence = min_confidence
        self._pending: List[ExtractedKnowledge] = []

    def observe(self, messages: List[Dict[str, str]],
                context_tags: Optional[List[str]] = None) -> List[ExtractedKnowledge]:
        """
        Observe a conversation and extract knowledge.

        Args:
            messages: Conversation messages
            context_tags: Tags to add to all extractions

        Returns:
            List of extracted knowledge items
        """
        extractions = self.extractor.extract_from_conversation(messages, context_tags)

        if self.auto_record:
            self._record_extractions(extractions)
        else:
            self._pending.extend(extractions)

        return extractions

    def observe_message(self, role: str, content: str,
                        context_tags: Optional[List[str]] = None) -> List[ExtractedKnowledge]:
        """
        Observe a single message.

        Args:
            role: Message role (user/assistant)
            content: Message content
            context_tags: Tags to add

        Returns:
            List of extracted knowledge items
        """
        return self.observe([{"role": role, "content": content}], context_tags)

    def _record_extractions(self, extractions: List[ExtractedKnowledge]) -> None:
        """Record extractions that meet confidence threshold."""
        confidence_order = [
            ExtractionConfidence.LOW,
            ExtractionConfidence.MEDIUM,
            ExtractionConfidence.HIGH
        ]

        min_idx = confidence_order.index(self.min_confidence)

        for ext in extractions:
            ext_idx = confidence_order.index(ext.confidence)
            if ext_idx >= min_idx:
                self._record_single(ext)

    def _record_single(self, extraction: ExtractedKnowledge) -> None:
        """Record a single extraction."""
        # Add extraction metadata
        metadata = {
            "auto_extracted": True,
            "confidence": extraction.confidence.value,
            "source_text": extraction.source_text[:200],  # Truncate
            "reasoning": extraction.reasoning
        }

        # Record based on category
        if extraction.category == ContextCategory.DECISION:
            self.manager.record_decision(extraction.content, tags=extraction.tags)
        elif extraction.category == ContextCategory.LEARNING:
            self.manager.record_learning(extraction.content, tags=extraction.tags)
        elif extraction.category == ContextCategory.ACTION:
            self.manager.record_action(extraction.content, tags=extraction.tags)
        elif extraction.category == ContextCategory.RESULT:
            self.manager.record_result(extraction.content, tags=extraction.tags)

    def get_pending(self) -> List[ExtractedKnowledge]:
        """Get pending extractions (when auto_record=False)."""
        return self._pending

    def approve_pending(self, indices: Optional[List[int]] = None) -> int:
        """
        Approve and record pending extractions.

        Args:
            indices: Specific indices to approve (None = all)

        Returns:
            Number of items recorded
        """
        to_record = self._pending if indices is None else [self._pending[i] for i in indices]

        for ext in to_record:
            self._record_single(ext)

        count = len(to_record)
        self._pending = [] if indices is None else [
            ext for i, ext in enumerate(self._pending) if i not in indices
        ]

        return count

    def clear_pending(self) -> None:
        """Clear pending extractions."""
        self._pending = []
```

**Usage Example**:

```python
from cogneetree import ContextManager
from cogneetree.extraction.knowledge_extractor import (
    KnowledgeExtractor, ConversationObserver
)

manager = ContextManager()
manager.create_session("s1", "JWT Implementation", "Plan")
manager.create_activity("a1", "s1", "Research", ["jwt"], "learner", "core", "...")
manager.create_task("t1", "a1", "Learn JWT", ["jwt"])

# Create observer
observer = ConversationObserver(manager, auto_record=True)

# Observe a conversation
conversation = [
    {"role": "user", "content": "How should we implement JWT authentication?"},
    {"role": "assistant", "content": """For JWT authentication, I recommend using RS256 algorithm
    instead of HS256 for better security in distributed systems.

    I learned that JWT tokens consist of three base64-encoded parts: header, payload, and signature.

    We should store tokens in httpOnly cookies to prevent XSS attacks.

    I've analyzed the RFC 7519 specification and the implementation is straightforward."""}
]

# Auto-extracts and records:
# - Decision: "using RS256 algorithm instead of HS256"
# - Learning: "JWT tokens consist of three base64-encoded parts"
# - Decision: "store tokens in httpOnly cookies"
# - Action: "analyzed the RFC 7519 specification"
extractions = observer.observe(conversation, context_tags=["jwt"])

print(f"Extracted {len(extractions)} knowledge items:")
for ext in extractions:
    print(f"  [{ext.category.value}] {ext.content[:50]}... (confidence: {ext.confidence.value})")

# Manual approval mode
observer_manual = ConversationObserver(manager, auto_record=False)
observer_manual.observe(conversation)

# Review pending
for i, ext in enumerate(observer_manual.get_pending()):
    print(f"{i}: [{ext.category.value}] {ext.content[:50]}...")

# Approve specific items
observer_manual.approve_pending(indices=[0, 2])  # Approve first and third
```

---

### 3.2 Causal Chains & Reasoning Traces

**Problem**: Items are isolated; no explicit relationships between decisions.

**Solution**: Track cause-effect relationships for transparent reasoning.

**File**: `src/cogneetree/reasoning/causal_chains.py`

```python
"""Causal chains and reasoning traces for decision tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from cogneetree.core.models import ContextItem, ContextCategory


class RelationType(Enum):
    """Types of causal relationships."""
    BECAUSE = "because"          # A because B (A is caused by B)
    LEADS_TO = "leads_to"        # A leads to B (A causes B)
    ENABLES = "enables"          # A enables B (A is prerequisite for B)
    CONTRADICTS = "contradicts"  # A contradicts B
    SUPERSEDES = "supersedes"    # A supersedes B (A replaces B)
    SUPPORTS = "supports"        # A supports B (A provides evidence for B)


@dataclass
class CausalLink:
    """A causal relationship between two items."""
    source_id: str
    target_id: str
    relation: RelationType
    reasoning: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningTrace:
    """A complete reasoning trace from root cause to final decision."""
    decision_id: str
    decision_content: str
    chain: List[CausalLink]
    root_causes: List[str]
    depth: int


class CausalChainManager:
    """Manages causal relationships between context items."""

    def __init__(self, storage):
        """
        Initialize causal chain manager.

        Args:
            storage: Storage backend
        """
        self.storage = storage
        self._links: Dict[str, CausalLink] = {}  # link_id -> CausalLink
        self._forward_index: Dict[str, Set[str]] = {}  # source_id -> set of link_ids
        self._backward_index: Dict[str, Set[str]] = {}  # target_id -> set of link_ids
        self._item_index: Dict[str, ContextItem] = {}  # item_id -> item

    def _generate_link_id(self, link: CausalLink) -> str:
        """Generate unique ID for link."""
        import hashlib
        content = f"{link.source_id}:{link.target_id}:{link.relation.value}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _ensure_item_indexed(self, item: ContextItem) -> str:
        """Ensure item is indexed and return its ID."""
        item_id = item.metadata.get("_item_id")
        if not item_id:
            import hashlib
            content = f"{item.content}:{item.timestamp.isoformat()}"
            item_id = hashlib.sha256(content.encode()).hexdigest()[:12]
            item.metadata["_item_id"] = item_id
        self._item_index[item_id] = item
        return item_id

    def link(self, source: ContextItem, target: ContextItem,
             relation: RelationType, reasoning: str = "") -> CausalLink:
        """
        Create a causal link between two items.

        Args:
            source: Source item (the cause or prerequisite)
            target: Target item (the effect or result)
            relation: Type of relationship
            reasoning: Explanation of the relationship

        Returns:
            Created CausalLink
        """
        source_id = self._ensure_item_indexed(source)
        target_id = self._ensure_item_indexed(target)

        link = CausalLink(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            reasoning=reasoning
        )

        link_id = self._generate_link_id(link)

        # Store link
        self._links[link_id] = link

        # Update indexes
        if source_id not in self._forward_index:
            self._forward_index[source_id] = set()
        self._forward_index[source_id].add(link_id)

        if target_id not in self._backward_index:
            self._backward_index[target_id] = set()
        self._backward_index[target_id].add(link_id)

        return link

    def link_decision(self, decision: ContextItem, because: ContextItem,
                      reasoning: str = "") -> CausalLink:
        """
        Link a decision to its cause.

        Args:
            decision: The decision item
            because: The item that caused this decision
            reasoning: Explanation

        Returns:
            Created CausalLink

        Example:
            causal.link_decision(
                decision=jwt_decision,
                because=security_requirement,
                reasoning="RS256 needed for distributed key management"
            )
        """
        return self.link(because, decision, RelationType.BECAUSE, reasoning)

    def link_consequence(self, decision: ContextItem, leads_to: str,
                         tags: Optional[List[str]] = None) -> ContextItem:
        """
        Record a consequence of a decision.

        Args:
            decision: The decision that leads to this consequence
            leads_to: Description of the consequence
            tags: Tags for the consequence item

        Returns:
            Created consequence item
        """
        consequence = ContextItem(
            content=leads_to,
            category=ContextCategory.LEARNING,  # Consequences are learnings
            tags=tags or [],
            parent_id=decision.parent_id
        )

        self.link(decision, consequence, RelationType.LEADS_TO,
                  f"Consequence of: {decision.content[:50]}")

        return consequence

    def get_causes(self, item: ContextItem, depth: int = 1) -> List[ContextItem]:
        """
        Get items that caused this item.

        Args:
            item: Item to find causes for
            depth: How many levels back to trace

        Returns:
            List of cause items
        """
        item_id = self._ensure_item_indexed(item)
        causes = []
        visited = set()

        def trace_back(current_id: str, current_depth: int):
            if current_depth > depth or current_id in visited:
                return
            visited.add(current_id)

            link_ids = self._backward_index.get(current_id, set())
            for link_id in link_ids:
                link = self._links.get(link_id)
                if link and link.relation in [RelationType.BECAUSE, RelationType.ENABLES]:
                    source_item = self._item_index.get(link.source_id)
                    if source_item:
                        causes.append(source_item)
                        trace_back(link.source_id, current_depth + 1)

        trace_back(item_id, 1)
        return causes

    def get_effects(self, item: ContextItem, depth: int = 1) -> List[ContextItem]:
        """
        Get items caused by this item.

        Args:
            item: Item to find effects for
            depth: How many levels forward to trace

        Returns:
            List of effect items
        """
        item_id = self._ensure_item_indexed(item)
        effects = []
        visited = set()

        def trace_forward(current_id: str, current_depth: int):
            if current_depth > depth or current_id in visited:
                return
            visited.add(current_id)

            link_ids = self._forward_index.get(current_id, set())
            for link_id in link_ids:
                link = self._links.get(link_id)
                if link and link.relation in [RelationType.LEADS_TO, RelationType.ENABLES]:
                    target_item = self._item_index.get(link.target_id)
                    if target_item:
                        effects.append(target_item)
                        trace_forward(link.target_id, current_depth + 1)

        trace_forward(item_id, 1)
        return effects

    def get_reasoning_trace(self, decision: ContextItem,
                            max_depth: int = 10) -> ReasoningTrace:
        """
        Get complete reasoning trace for a decision.

        Args:
            decision: Decision to trace
            max_depth: Maximum trace depth

        Returns:
            ReasoningTrace with full causal chain
        """
        decision_id = self._ensure_item_indexed(decision)
        chain = []
        root_causes = []
        visited = set()

        def trace(current_id: str, depth: int):
            if depth > max_depth or current_id in visited:
                return
            visited.add(current_id)

            link_ids = self._backward_index.get(current_id, set())

            if not link_ids:
                # This is a root cause
                root_causes.append(current_id)
                return

            for link_id in link_ids:
                link = self._links.get(link_id)
                if link:
                    chain.append(link)
                    trace(link.source_id, depth + 1)

        trace(decision_id, 0)

        return ReasoningTrace(
            decision_id=decision_id,
            decision_content=decision.content,
            chain=chain,
            root_causes=root_causes,
            depth=len(set(link.source_id for link in chain))
        )

    def explain_decision(self, decision: ContextItem) -> str:
        """
        Generate human-readable explanation of a decision.

        Args:
            decision: Decision to explain

        Returns:
            Formatted explanation string
        """
        trace = self.get_reasoning_trace(decision)

        lines = [
            f"## Decision: {decision.content}",
            "",
            "### Reasoning Chain:",
        ]

        # Group by depth
        for link in reversed(trace.chain):
            source_item = self._item_index.get(link.source_id)
            target_item = self._item_index.get(link.target_id)

            if source_item and target_item:
                lines.append(f"- {source_item.content[:80]}...")
                lines.append(f"  → {link.relation.value}: {target_item.content[:80]}...")
                if link.reasoning:
                    lines.append(f"    Reasoning: {link.reasoning}")
                lines.append("")

        # Root causes
        if trace.root_causes:
            lines.append("### Root Causes:")
            for cause_id in trace.root_causes:
                cause_item = self._item_index.get(cause_id)
                if cause_item:
                    lines.append(f"- {cause_item.content}")

        return "\n".join(lines)

    def find_contradictions(self) -> List[tuple]:
        """
        Find potentially contradicting decisions.

        Returns:
            List of (item1, item2, reasoning) tuples
        """
        contradictions = []

        for link_id, link in self._links.items():
            if link.relation == RelationType.CONTRADICTS:
                source = self._item_index.get(link.source_id)
                target = self._item_index.get(link.target_id)
                if source and target:
                    contradictions.append((source, target, link.reasoning))

        return contradictions

    def supersede_decision(self, old_decision: ContextItem,
                           new_decision: ContextItem,
                           reasoning: str = "") -> CausalLink:
        """
        Mark a decision as superseded by a new one.

        Args:
            old_decision: The old decision being replaced
            new_decision: The new decision
            reasoning: Why the change was made

        Returns:
            Created CausalLink
        """
        return self.link(new_decision, old_decision, RelationType.SUPERSEDES, reasoning)

    def get_active_decisions(self, tags: Optional[List[str]] = None) -> List[ContextItem]:
        """
        Get decisions that haven't been superseded.

        Args:
            tags: Filter by tags

        Returns:
            List of active decision items
        """
        # Find all superseded items
        superseded_ids = set()
        for link in self._links.values():
            if link.relation == RelationType.SUPERSEDES:
                superseded_ids.add(link.target_id)

        # Filter decisions
        active = []
        for item_id, item in self._item_index.items():
            if item.category == ContextCategory.DECISION:
                if item_id not in superseded_ids:
                    if tags is None or any(t in item.tags for t in tags):
                        active.append(item)

        return active
```

**Usage Example**:

```python
from cogneetree import ContextManager
from cogneetree.core.models import ContextItem, ContextCategory
from cogneetree.reasoning.causal_chains import CausalChainManager, RelationType

manager = ContextManager()
causal = CausalChainManager(manager.storage)

# Create items
requirement = ContextItem(
    content="System must support multiple authentication servers",
    category=ContextCategory.LEARNING,
    tags=["requirements", "architecture"]
)

jwt_decision = ContextItem(
    content="Use RS256 algorithm for JWT signing",
    category=ContextCategory.DECISION,
    tags=["jwt", "security"]
)

key_management = ContextItem(
    content="Implement public key distribution via JWKS endpoint",
    category=ContextCategory.DECISION,
    tags=["jwt", "keys"]
)

# Link decisions to causes
causal.link_decision(
    decision=jwt_decision,
    because=requirement,
    reasoning="RS256 allows asymmetric keys, enabling distributed verification"
)

causal.link(
    jwt_decision, key_management,
    RelationType.LEADS_TO,
    "Asymmetric signing requires key distribution mechanism"
)

# Get reasoning trace
trace = causal.get_reasoning_trace(key_management)
print(f"Decision depth: {trace.depth}")
print(f"Root causes: {len(trace.root_causes)}")

# Generate explanation
explanation = causal.explain_decision(key_management)
print(explanation)
# Output:
# ## Decision: Implement public key distribution via JWKS endpoint
#
# ### Reasoning Chain:
# - System must support multiple authentication servers...
#   → because: Use RS256 algorithm for JWT signing...
#     Reasoning: RS256 allows asymmetric keys, enabling distributed verification
# - Use RS256 algorithm for JWT signing...
#   → leads_to: Implement public key distribution via JWKS endpoint...
#     Reasoning: Asymmetric signing requires key distribution mechanism
#
# ### Root Causes:
# - System must support multiple authentication servers

# Later: supersede a decision
new_jwt_decision = ContextItem(
    content="Switch to ES256 for better performance",
    category=ContextCategory.DECISION,
    tags=["jwt", "security", "performance"]
)

causal.supersede_decision(
    old_decision=jwt_decision,
    new_decision=new_jwt_decision,
    reasoning="ES256 has smaller signatures and faster verification"
)

# Get active decisions (excludes superseded)
active = causal.get_active_decisions(tags=["jwt"])
print(f"Active JWT decisions: {len(active)}")  # Returns new_jwt_decision only
```

---

### 3.3 Memory Consolidation & Forgetting

**Problem**: Items accumulate forever with no cleanup, causing context pollution.

**Solution**: Smart consolidation that mimics human memory with forgetting curves.

**File**: `src/cogneetree/memory/consolidation.py`

```python
"""Memory consolidation and forgetting mechanisms."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import math
from cogneetree.core.models import ContextItem, ContextCategory


@dataclass
class ConsolidatedItem:
    """A consolidated memory item."""
    content: str
    category: ContextCategory
    tags: List[str]
    confidence: float  # 0.0 to 1.0 - strengthened by repetition
    access_count: int
    last_accessed: datetime
    source_items: List[str]  # IDs of items that were consolidated
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryStrength:
    """Memory strength with decay calculation."""
    initial_strength: float
    decay_rate: float  # Higher = faster forgetting
    last_reinforced: datetime
    reinforcement_count: int

    def current_strength(self) -> float:
        """Calculate current strength with exponential decay."""
        elapsed = (datetime.now() - self.last_reinforced).total_seconds()
        elapsed_days = elapsed / 86400

        # Ebbinghaus forgetting curve: R = e^(-t/S)
        # Where S is stability (inversely related to decay_rate)
        stability = (1 / self.decay_rate) * (1 + 0.1 * self.reinforcement_count)
        retention = math.exp(-elapsed_days / stability)

        return self.initial_strength * retention


class MemoryConsolidator:
    """Consolidates similar memories and applies forgetting curves."""

    def __init__(self, storage, similarity_threshold: float = 0.85,
                 base_decay_rate: float = 0.1):
        """
        Initialize memory consolidator.

        Args:
            storage: Storage backend
            similarity_threshold: Minimum similarity to consolidate (0.0-1.0)
            base_decay_rate: Base forgetting rate (higher = faster forgetting)
        """
        self.storage = storage
        self.similarity_threshold = similarity_threshold
        self.base_decay_rate = base_decay_rate
        self._consolidated: Dict[str, ConsolidatedItem] = {}
        self._strengths: Dict[str, MemoryStrength] = {}
        self._access_log: List[Tuple[str, datetime]] = []

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity (simple Jaccard for demo)."""
        # In production, use embeddings for semantic similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = words1 & words2
        union = words1 | words2

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _generate_id(self, content: str) -> str:
        """Generate ID for content."""
        import hashlib
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def consolidate(self, items: Optional[List[ContextItem]] = None) -> Dict[str, Any]:
        """
        Consolidate similar items into stronger memories.

        Args:
            items: Items to consolidate (default: all items from storage)

        Returns:
            Dictionary with consolidation statistics
        """
        if items is None:
            items = self.storage.get_all_items() if hasattr(self.storage, 'get_all_items') else []

        # Group by category and tags
        groups = defaultdict(list)
        for item in items:
            key = (item.category.value, tuple(sorted(item.tags)))
            groups[key].append(item)

        consolidated_count = 0
        merged_count = 0

        for (category, tags), group_items in groups.items():
            if len(group_items) < 2:
                continue

            # Find similar items within group
            clusters = self._cluster_similar(group_items)

            for cluster in clusters:
                if len(cluster) > 1:
                    # Merge into single consolidated item
                    merged = self._merge_cluster(cluster)
                    self._consolidated[merged.content[:50]] = merged
                    consolidated_count += 1
                    merged_count += len(cluster)

        return {
            "consolidated_items": consolidated_count,
            "merged_items": merged_count,
            "compression_ratio": merged_count / max(len(items), 1)
        }

    def _cluster_similar(self, items: List[ContextItem]) -> List[List[ContextItem]]:
        """Cluster similar items together."""
        clusters = []
        used = set()

        for i, item1 in enumerate(items):
            if i in used:
                continue

            cluster = [item1]
            used.add(i)

            for j, item2 in enumerate(items[i+1:], start=i+1):
                if j in used:
                    continue

                similarity = self._compute_similarity(item1.content, item2.content)
                if similarity >= self.similarity_threshold:
                    cluster.append(item2)
                    used.add(j)

            clusters.append(cluster)

        return clusters

    def _merge_cluster(self, cluster: List[ContextItem]) -> ConsolidatedItem:
        """Merge a cluster of similar items."""
        # Use most recent item's content as base
        cluster_sorted = sorted(cluster, key=lambda x: x.timestamp, reverse=True)
        base_item = cluster_sorted[0]

        # Combine tags
        all_tags = set()
        for item in cluster:
            all_tags.update(item.tags)

        # Generate source IDs
        source_ids = [
            item.metadata.get("_item_id", self._generate_id(item.content))
            for item in cluster
        ]

        return ConsolidatedItem(
            content=base_item.content,
            category=base_item.category,
            tags=list(all_tags),
            confidence=min(1.0, 0.5 + 0.1 * len(cluster)),  # Increases with repetition
            access_count=sum(item.metadata.get("_access_count", 1) for item in cluster),
            last_accessed=max(item.timestamp for item in cluster),
            source_items=source_ids
        )

    def reinforce(self, item_id: str) -> None:
        """
        Reinforce a memory (called when accessed/retrieved).

        Args:
            item_id: ID of item being accessed
        """
        self._access_log.append((item_id, datetime.now()))

        if item_id not in self._strengths:
            self._strengths[item_id] = MemoryStrength(
                initial_strength=1.0,
                decay_rate=self.base_decay_rate,
                last_reinforced=datetime.now(),
                reinforcement_count=1
            )
        else:
            strength = self._strengths[item_id]
            strength.last_reinforced = datetime.now()
            strength.reinforcement_count += 1
            # Reduce decay rate with reinforcement (memory becomes more stable)
            strength.decay_rate *= 0.9

    def get_strength(self, item_id: str) -> float:
        """
        Get current memory strength for an item.

        Args:
            item_id: Item ID

        Returns:
            Current strength (0.0-1.0)
        """
        if item_id not in self._strengths:
            return 1.0  # New items have full strength

        return self._strengths[item_id].current_strength()

    def apply_forgetting(self, min_strength: float = 0.3) -> List[str]:
        """
        Apply forgetting curve and identify weak memories.

        Args:
            min_strength: Minimum strength to retain

        Returns:
            List of item IDs below threshold (candidates for removal)
        """
        weak_items = []

        for item_id, strength in self._strengths.items():
            current = strength.current_strength()
            if current < min_strength:
                weak_items.append(item_id)

        return weak_items

    def cleanup(self, keep_decisions: bool = True,
                max_age_days: int = 180,
                min_strength: float = 0.2) -> Dict[str, Any]:
        """
        Clean up old and weak memories.

        Args:
            keep_decisions: Never remove decisions (important for auditing)
            max_age_days: Maximum age for items
            min_strength: Minimum strength to retain

        Returns:
            Cleanup statistics
        """
        all_items = self.storage.get_all_items() if hasattr(self.storage, 'get_all_items') else []

        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        removed = []
        retained = []

        for item in all_items:
            item_id = item.metadata.get("_item_id", self._generate_id(item.content))

            # Never remove decisions if flag set
            if keep_decisions and item.category == ContextCategory.DECISION:
                retained.append(item_id)
                continue

            # Check age
            if item.timestamp and item.timestamp < cutoff_date:
                strength = self.get_strength(item_id)
                if strength < min_strength:
                    removed.append(item_id)
                    continue

            retained.append(item_id)

        return {
            "removed_count": len(removed),
            "retained_count": len(retained),
            "removed_ids": removed
        }

    def summarize_memories(self, tags: Optional[List[str]] = None,
                           max_items: int = 10) -> str:
        """
        Generate a summary of strongest memories.

        Args:
            tags: Filter by tags
            max_items: Maximum items to include

        Returns:
            Formatted summary string
        """
        all_items = self.storage.get_all_items() if hasattr(self.storage, 'get_all_items') else []

        # Filter by tags
        if tags:
            all_items = [item for item in all_items if any(t in item.tags for t in tags)]

        # Score by strength and recency
        scored_items = []
        for item in all_items:
            item_id = item.metadata.get("_item_id", self._generate_id(item.content))
            strength = self.get_strength(item_id)
            recency = 1.0 / (1 + (datetime.now() - item.timestamp).days) if item.timestamp else 0.5
            score = strength * 0.7 + recency * 0.3
            scored_items.append((item, score))

        # Sort by score
        scored_items.sort(key=lambda x: x[1], reverse=True)

        # Format summary
        lines = ["## Memory Summary", ""]
        for item, score in scored_items[:max_items]:
            lines.append(f"- **[{item.category.value}]** {item.content[:80]}...")
            lines.append(f"  Tags: {', '.join(item.tags)} | Strength: {score:.2f}")
            lines.append("")

        return "\n".join(lines)


class RetentionPolicy:
    """Configurable retention policies for automatic cleanup."""

    def __init__(self, consolidator: MemoryConsolidator):
        """
        Initialize retention policy.

        Args:
            consolidator: MemoryConsolidator instance
        """
        self.consolidator = consolidator
        self.rules: List[Dict[str, Any]] = []

    def add_rule(self, name: str, category: Optional[ContextCategory] = None,
                 tags: Optional[List[str]] = None,
                 max_age_days: int = 90,
                 min_strength: float = 0.3,
                 keep_count: Optional[int] = None):
        """
        Add a retention rule.

        Args:
            name: Rule name for logging
            category: Apply to specific category
            tags: Apply to items with these tags
            max_age_days: Maximum age before considering removal
            min_strength: Minimum strength to retain
            keep_count: Always keep at least this many items
        """
        self.rules.append({
            "name": name,
            "category": category,
            "tags": tags,
            "max_age_days": max_age_days,
            "min_strength": min_strength,
            "keep_count": keep_count
        })

    def apply(self) -> Dict[str, Any]:
        """
        Apply all retention rules.

        Returns:
            Statistics about cleanup
        """
        results = {"rules_applied": [], "total_removed": 0}

        for rule in self.rules:
            # Get matching items
            all_items = self.consolidator.storage.get_all_items() if hasattr(self.consolidator.storage, 'get_all_items') else []

            matching = all_items
            if rule["category"]:
                matching = [i for i in matching if i.category == rule["category"]]
            if rule["tags"]:
                matching = [i for i in matching if any(t in i.tags for t in rule["tags"])]

            # Apply age and strength filters
            cutoff = datetime.now() - timedelta(days=rule["max_age_days"])
            candidates = []
            for item in matching:
                if item.timestamp and item.timestamp < cutoff:
                    item_id = item.metadata.get("_item_id", "")
                    strength = self.consolidator.get_strength(item_id)
                    if strength < rule["min_strength"]:
                        candidates.append((item, strength))

            # Sort by strength (remove weakest first)
            candidates.sort(key=lambda x: x[1])

            # Keep minimum count if specified
            if rule["keep_count"]:
                candidates = candidates[:-rule["keep_count"]] if len(candidates) > rule["keep_count"] else []

            removed_count = len(candidates)
            results["rules_applied"].append({
                "rule": rule["name"],
                "removed": removed_count
            })
            results["total_removed"] += removed_count

        return results
```

**Usage Example**:

```python
from cogneetree import ContextManager
from cogneetree.memory.consolidation import MemoryConsolidator, RetentionPolicy

manager = ContextManager()
manager.create_session("s1", "Test", "Plan")
manager.create_activity("a1", "s1", "Activity", ["jwt"], "learner", "core", "...")
manager.create_task("t1", "a1", "Task", ["jwt"])

# Record similar learnings over time
manager.record_learning("JWT has three parts: header, payload, signature", tags=["jwt"])
manager.record_learning("JWT structure is header.payload.signature", tags=["jwt"])
manager.record_learning("A JWT token consists of 3 base64 parts", tags=["jwt"])

consolidator = MemoryConsolidator(manager.storage, similarity_threshold=0.7)

# Consolidate similar memories
stats = consolidator.consolidate()
print(f"Consolidated {stats['consolidated_items']} groups from {stats['merged_items']} items")
print(f"Compression ratio: {stats['compression_ratio']:.2%}")

# Simulate access patterns (reinforce some memories)
consolidator.reinforce("jwt_learning_1")
consolidator.reinforce("jwt_learning_1")  # Accessed twice

# Check memory strength after time passes
strength = consolidator.get_strength("jwt_learning_1")
print(f"Memory strength: {strength:.2f}")

# Find weak memories
weak = consolidator.apply_forgetting(min_strength=0.3)
print(f"Weak memories to consider removing: {len(weak)}")

# Setup retention policies
policy = RetentionPolicy(consolidator)

# Keep decisions forever, but clean up old learnings
policy.add_rule(
    name="cleanup_old_learnings",
    category=ContextCategory.LEARNING,
    max_age_days=90,
    min_strength=0.3,
    keep_count=100  # Always keep at least 100
)

policy.add_rule(
    name="cleanup_old_actions",
    category=ContextCategory.ACTION,
    max_age_days=30,
    min_strength=0.2
)

# Apply policies
cleanup_result = policy.apply()
print(f"Removed {cleanup_result['total_removed']} items")

# Generate summary of strongest memories
summary = consolidator.summarize_memories(tags=["jwt"], max_items=5)
print(summary)
```

---

### 3.4 Time-Travel Queries

**Problem**: Only current state is queryable - can't see what agent knew at a point in time.

**Solution**: Query memory as it was at any point in time.

**File**: `src/cogneetree/temporal/time_travel.py`

```python
"""Time-travel queries for historical memory state."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Iterator
from cogneetree.core.models import ContextItem, ContextCategory, Session


@dataclass
class MemoryEvent:
    """A single event in memory history."""
    timestamp: datetime
    event_type: str  # "created", "updated", "deleted", "accessed"
    item_id: str
    content: str
    category: ContextCategory
    tags: List[str]
    metadata: Dict[str, Any]


@dataclass
class MemoryDiff:
    """Difference between two memory states."""
    from_time: datetime
    to_time: datetime
    added: List[ContextItem]
    removed: List[ContextItem]
    modified: List[tuple]  # (old_item, new_item)


class TemporalMemory:
    """Memory with time-travel capabilities."""

    def __init__(self, storage):
        """
        Initialize temporal memory.

        Args:
            storage: Storage backend
        """
        self.storage = storage
        self._event_log: List[MemoryEvent] = []
        self._snapshots: Dict[str, List[ContextItem]] = {}  # timestamp_key -> items

    def _log_event(self, event_type: str, item: ContextItem) -> None:
        """Log a memory event."""
        item_id = item.metadata.get("_item_id", "")
        if not item_id:
            import hashlib
            item_id = hashlib.sha256(item.content.encode()).hexdigest()[:12]

        event = MemoryEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            item_id=item_id,
            content=item.content,
            category=item.category,
            tags=item.tags,
            metadata=item.metadata.copy()
        )
        self._event_log.append(event)

    def record_item(self, item: ContextItem) -> None:
        """Record item with event logging."""
        self._log_event("created", item)
        self.storage.add_item(item)

    def create_snapshot(self, name: Optional[str] = None) -> str:
        """
        Create a named snapshot of current memory state.

        Args:
            name: Optional snapshot name

        Returns:
            Snapshot key
        """
        key = name or datetime.now().isoformat()
        all_items = self.storage.get_all_items() if hasattr(self.storage, 'get_all_items') else []
        self._snapshots[key] = list(all_items)
        return key

    def at_time(self, target_time: datetime) -> 'HistoricalView':
        """
        Get memory state as it was at a specific time.

        Args:
            target_time: Point in time to query

        Returns:
            HistoricalView of memory at that time
        """
        return HistoricalView(self, target_time)

    def at_snapshot(self, snapshot_key: str) -> 'HistoricalView':
        """
        Get memory state from a named snapshot.

        Args:
            snapshot_key: Snapshot key/name

        Returns:
            HistoricalView from snapshot
        """
        if snapshot_key not in self._snapshots:
            raise ValueError(f"Snapshot not found: {snapshot_key}")

        items = self._snapshots[snapshot_key]
        # Find the timestamp of the snapshot
        for event in reversed(self._event_log):
            if event.timestamp.isoformat().startswith(snapshot_key[:19]):
                return HistoricalView(self, event.timestamp, items)

        return HistoricalView(self, datetime.now(), items)

    def diff(self, from_time: datetime, to_time: datetime) -> MemoryDiff:
        """
        Get difference between two points in time.

        Args:
            from_time: Start time
            to_time: End time

        Returns:
            MemoryDiff showing changes
        """
        from_view = self.at_time(from_time)
        to_view = self.at_time(to_time)

        from_items = {self._item_key(i): i for i in from_view.get_all_items()}
        to_items = {self._item_key(i): i for i in to_view.get_all_items()}

        from_keys = set(from_items.keys())
        to_keys = set(to_items.keys())

        added = [to_items[k] for k in to_keys - from_keys]
        removed = [from_items[k] for k in from_keys - to_keys]

        # Find modified (same key but different content)
        modified = []
        for key in from_keys & to_keys:
            if from_items[key].content != to_items[key].content:
                modified.append((from_items[key], to_items[key]))

        return MemoryDiff(
            from_time=from_time,
            to_time=to_time,
            added=added,
            removed=removed,
            modified=modified
        )

    def _item_key(self, item: ContextItem) -> str:
        """Generate stable key for item comparison."""
        return item.metadata.get("_item_id", item.content[:50])

    def replay(self, session_id: Optional[str] = None,
               from_time: Optional[datetime] = None,
               to_time: Optional[datetime] = None) -> Iterator[MemoryEvent]:
        """
        Replay memory events.

        Args:
            session_id: Filter to specific session
            from_time: Start time (default: beginning)
            to_time: End time (default: now)

        Yields:
            MemoryEvent objects in chronological order
        """
        for event in self._event_log:
            # Apply filters
            if from_time and event.timestamp < from_time:
                continue
            if to_time and event.timestamp > to_time:
                continue
            if session_id and event.metadata.get("session_id") != session_id:
                continue

            yield event

    def timeline(self, days: int = 30,
                 group_by: str = "day") -> Dict[str, Dict[str, int]]:
        """
        Get timeline of memory activity.

        Args:
            days: Number of days to include
            group_by: "day", "hour", or "category"

        Returns:
            Dictionary of activity counts
        """
        cutoff = datetime.now() - timedelta(days=days)
        timeline = {}

        for event in self._event_log:
            if event.timestamp < cutoff:
                continue

            if group_by == "day":
                key = event.timestamp.date().isoformat()
            elif group_by == "hour":
                key = event.timestamp.strftime("%Y-%m-%d %H:00")
            else:
                key = event.category.value

            if key not in timeline:
                timeline[key] = {"created": 0, "accessed": 0, "total": 0}

            timeline[key][event.event_type] = timeline[key].get(event.event_type, 0) + 1
            timeline[key]["total"] += 1

        return timeline

    def what_changed_since(self, reference: datetime) -> Dict[str, Any]:
        """
        Summarize what changed since a reference time.

        Args:
            reference: Reference time to compare against

        Returns:
            Summary of changes
        """
        diff = self.diff(reference, datetime.now())

        # Categorize changes
        changes_by_category = {}
        for item in diff.added:
            cat = item.category.value
            if cat not in changes_by_category:
                changes_by_category[cat] = {"added": [], "removed": []}
            changes_by_category[cat]["added"].append(item.content[:50])

        for item in diff.removed:
            cat = item.category.value
            if cat not in changes_by_category:
                changes_by_category[cat] = {"added": [], "removed": []}
            changes_by_category[cat]["removed"].append(item.content[:50])

        return {
            "reference_time": reference.isoformat(),
            "current_time": datetime.now().isoformat(),
            "items_added": len(diff.added),
            "items_removed": len(diff.removed),
            "items_modified": len(diff.modified),
            "by_category": changes_by_category
        }


class HistoricalView:
    """A read-only view of memory at a point in time."""

    def __init__(self, temporal: TemporalMemory, target_time: datetime,
                 cached_items: Optional[List[ContextItem]] = None):
        """
        Initialize historical view.

        Args:
            temporal: Parent TemporalMemory
            target_time: Point in time for this view
            cached_items: Pre-cached items (for snapshots)
        """
        self.temporal = temporal
        self.target_time = target_time
        self._cached_items = cached_items

    def get_all_items(self) -> List[ContextItem]:
        """Get all items that existed at target_time."""
        if self._cached_items is not None:
            return self._cached_items

        # Reconstruct state from event log
        items_at_time = {}

        for event in self.temporal._event_log:
            if event.timestamp > self.target_time:
                break

            if event.event_type == "created":
                items_at_time[event.item_id] = ContextItem(
                    content=event.content,
                    category=event.category,
                    tags=event.tags,
                    timestamp=event.timestamp,
                    metadata=event.metadata
                )
            elif event.event_type == "deleted":
                items_at_time.pop(event.item_id, None)

        return list(items_at_time.values())

    def recall(self, query: str, max_items: int = 10) -> List[ContextItem]:
        """
        Recall items from historical state.

        Args:
            query: Search query
            max_items: Maximum items to return

        Returns:
            List of matching items from historical state
        """
        all_items = self.get_all_items()

        # Simple keyword matching (in production, use embeddings)
        query_words = set(query.lower().split())
        scored = []

        for item in all_items:
            content_words = set(item.content.lower().split())
            tag_words = set(t.lower() for t in item.tags)
            all_words = content_words | tag_words

            overlap = len(query_words & all_words)
            if overlap > 0:
                scored.append((item, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scored[:max_items]]

    def decisions(self) -> List[ContextItem]:
        """Get decisions that existed at target_time."""
        return [i for i in self.get_all_items() if i.category == ContextCategory.DECISION]

    def learnings(self) -> List[ContextItem]:
        """Get learnings that existed at target_time."""
        return [i for i in self.get_all_items() if i.category == ContextCategory.LEARNING]

    def stats(self) -> Dict[str, int]:
        """Get statistics for historical state."""
        items = self.get_all_items()
        by_category = {}
        for item in items:
            cat = item.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

        return {
            "total_items": len(items),
            "by_category": by_category,
            "as_of": self.target_time.isoformat()
        }
```

**Usage Example**:

```python
from datetime import datetime, timedelta
from cogneetree import ContextManager
from cogneetree.temporal.time_travel import TemporalMemory

manager = ContextManager()
temporal = TemporalMemory(manager.storage)

# Setup context
manager.create_session("s1", "JWT Project", "Plan")
manager.create_activity("a1", "s1", "Research", ["jwt"], "learner", "core", "...")
manager.create_task("t1", "a1", "Learn JWT", ["jwt"])

# Record items over time (simulating a project timeline)
from cogneetree.core.models import ContextItem, ContextCategory

# Day 1: Initial learning
item1 = ContextItem(
    content="JWT uses HS256 algorithm",
    category=ContextCategory.LEARNING,
    tags=["jwt"],
    metadata={"session_id": "s1"}
)
temporal.record_item(item1)

# Create a snapshot at this point
snapshot_key = temporal.create_snapshot("day1_snapshot")

# Day 2: Decision made
item2 = ContextItem(
    content="Switch to RS256 for distributed systems",
    category=ContextCategory.DECISION,
    tags=["jwt", "security"],
    metadata={"session_id": "s1"}
)
temporal.record_item(item2)

# Day 3: More learnings
item3 = ContextItem(
    content="RS256 requires public key distribution",
    category=ContextCategory.LEARNING,
    tags=["jwt", "keys"],
    metadata={"session_id": "s1"}
)
temporal.record_item(item3)

# Time-travel: What did the agent know on Day 1?
day1_view = temporal.at_snapshot("day1_snapshot")
print(f"Items on Day 1: {len(day1_view.get_all_items())}")
print(f"Decisions on Day 1: {len(day1_view.decisions())}")

# Query historical state
historical_results = day1_view.recall("JWT algorithm")
print(f"Historical recall: {len(historical_results)} results")

# Get diff between two times
yesterday = datetime.now() - timedelta(days=1)
diff = temporal.diff(yesterday, datetime.now())
print(f"Since yesterday: +{len(diff.added)} added, -{len(diff.removed)} removed")

# Replay events
print("\nEvent replay:")
for event in temporal.replay(session_id="s1"):
    print(f"  {event.timestamp}: [{event.event_type}] {event.content[:40]}...")

# Get timeline
timeline = temporal.timeline(days=7, group_by="day")
print(f"\nTimeline: {timeline}")

# What changed since a reference point?
changes = temporal.what_changed_since(yesterday)
print(f"\nChanges summary: {changes['items_added']} added, {changes['items_removed']} removed")
```

---

## Phase 4: Enterprise Features

### 4.1 Multi-Agent Shared Memory

**Problem**: Single-agent focus - no collaboration primitives for multi-agent systems.

**Solution**: Shared memory spaces with access control and conflict resolution.

**File**: `src/cogneetree/multiagent/shared_memory.py`

```python
"""Multi-agent shared memory with access control."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from threading import RLock
from cogneetree.core.interfaces import ContextStorageABC
from cogneetree.core.models import ContextItem, ContextCategory


class Permission(Enum):
    """Access permissions."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"  # Can modify access control


@dataclass
class AgentIdentity:
    """Agent identity with metadata."""
    agent_id: str
    name: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessControl:
    """Access control entry."""
    agent_id: str
    permissions: Set[Permission]
    granted_by: str
    granted_at: datetime = field(default_factory=datetime.now)


class SharedMemory:
    """Shared memory space for multiple agents."""

    def __init__(self, storage: ContextStorageABC, workspace_id: str,
                 owner_agent_id: str):
        """
        Initialize shared memory workspace.

        Args:
            storage: Underlying storage backend
            workspace_id: Unique workspace identifier
            owner_agent_id: Agent that owns this workspace
        """
        self.storage = storage
        self.workspace_id = workspace_id
        self.owner_agent_id = owner_agent_id

        self._agents: Dict[str, AgentIdentity] = {}
        self._access_control: Dict[str, AccessControl] = {}
        self._item_ownership: Dict[str, str] = {}  # item_id -> agent_id
        self._lock = RLock()

        # Register owner with admin permissions
        self.register_agent(owner_agent_id, f"Owner-{owner_agent_id}")
        self.grant_access(owner_agent_id, {Permission.READ, Permission.WRITE, Permission.ADMIN},
                          granted_by="system")

    def register_agent(self, agent_id: str, name: str,
                       metadata: Optional[Dict[str, Any]] = None) -> AgentIdentity:
        """
        Register a new agent in the workspace.

        Args:
            agent_id: Unique agent identifier
            name: Human-readable agent name
            metadata: Optional agent metadata

        Returns:
            AgentIdentity object
        """
        with self._lock:
            agent = AgentIdentity(
                agent_id=agent_id,
                name=name,
                metadata=metadata or {}
            )
            self._agents[agent_id] = agent
            return agent

    def grant_access(self, agent_id: str, permissions: Set[Permission],
                     granted_by: str) -> None:
        """
        Grant access permissions to an agent.

        Args:
            agent_id: Agent to grant access to
            permissions: Set of permissions to grant
            granted_by: Agent granting the access
        """
        with self._lock:
            # Verify granter has admin permission
            if granted_by != "system":
                granter_perms = self._access_control.get(granted_by)
                if not granter_perms or Permission.ADMIN not in granter_perms.permissions:
                    raise PermissionError(f"Agent {granted_by} lacks admin permission")

            self._access_control[agent_id] = AccessControl(
                agent_id=agent_id,
                permissions=permissions,
                granted_by=granted_by
            )

    def revoke_access(self, agent_id: str, revoker_id: str) -> None:
        """Revoke all access for an agent."""
        with self._lock:
            revoker_perms = self._access_control.get(revoker_id)
            if not revoker_perms or Permission.ADMIN not in revoker_perms.permissions:
                raise PermissionError(f"Agent {revoker_id} lacks admin permission")

            self._access_control.pop(agent_id, None)

    def _check_permission(self, agent_id: str, required: Permission) -> None:
        """Check if agent has required permission."""
        access = self._access_control.get(agent_id)
        if not access or required not in access.permissions:
            raise PermissionError(
                f"Agent {agent_id} lacks {required.value} permission"
            )

    def as_agent(self, agent_id: str) -> 'AgentView':
        """
        Get a view of shared memory for a specific agent.

        Args:
            agent_id: Agent requesting the view

        Returns:
            AgentView with permission-checked methods
        """
        return AgentView(self, agent_id)

    def add_item(self, item: ContextItem, agent_id: str) -> str:
        """
        Add item to shared memory.

        Args:
            item: Item to add
            agent_id: Agent adding the item

        Returns:
            Item ID
        """
        with self._lock:
            self._check_permission(agent_id, Permission.WRITE)

            # Generate ID and track ownership
            import hashlib
            item_id = hashlib.sha256(
                f"{item.content}:{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]

            item.metadata["_item_id"] = item_id
            item.metadata["_workspace_id"] = self.workspace_id
            item.metadata["_created_by"] = agent_id
            item.metadata["_created_at"] = datetime.now().isoformat()

            self._item_ownership[item_id] = agent_id
            self.storage.add_item(item)

            return item_id

    def get_items(self, agent_id: str,
                  category: Optional[ContextCategory] = None,
                  tags: Optional[List[str]] = None,
                  created_by: Optional[str] = None) -> List[ContextItem]:
        """
        Get items from shared memory.

        Args:
            agent_id: Agent requesting items
            category: Filter by category
            tags: Filter by tags
            created_by: Filter by creating agent

        Returns:
            List of matching items
        """
        with self._lock:
            self._check_permission(agent_id, Permission.READ)

            # Get all items in workspace
            all_items = self.storage.get_all_items() if hasattr(self.storage, 'get_all_items') else []

            # Filter to workspace
            workspace_items = [
                i for i in all_items
                if i.metadata.get("_workspace_id") == self.workspace_id
            ]

            # Apply filters
            if category:
                workspace_items = [i for i in workspace_items if i.category == category]
            if tags:
                workspace_items = [i for i in workspace_items if any(t in i.tags for t in tags)]
            if created_by:
                workspace_items = [i for i in workspace_items if i.metadata.get("_created_by") == created_by]

            return workspace_items

    def merge_duplicates(self, strategy: str = "keep_most_recent",
                         admin_agent_id: str = None) -> Dict[str, Any]:
        """
        Merge duplicate items from different agents.

        Args:
            strategy: "keep_most_recent", "keep_highest_confidence", "merge_all"
            admin_agent_id: Agent with admin permission to perform merge

        Returns:
            Merge statistics
        """
        if admin_agent_id:
            self._check_permission(admin_agent_id, Permission.ADMIN)

        with self._lock:
            all_items = self.get_items(self.owner_agent_id)

            # Group similar items
            groups = self._find_similar_items(all_items)
            merged_count = 0

            for group in groups:
                if len(group) <= 1:
                    continue

                if strategy == "keep_most_recent":
                    # Keep newest, mark others as superseded
                    sorted_group = sorted(
                        group,
                        key=lambda x: x.metadata.get("_created_at", ""),
                        reverse=True
                    )
                    keeper = sorted_group[0]
                    for item in sorted_group[1:]:
                        item.metadata["_superseded_by"] = keeper.metadata.get("_item_id")
                        merged_count += 1

            return {"merged_count": merged_count, "strategy": strategy}

    def _find_similar_items(self, items: List[ContextItem],
                            threshold: float = 0.8) -> List[List[ContextItem]]:
        """Find groups of similar items."""
        groups = []
        used = set()

        for i, item1 in enumerate(items):
            if i in used:
                continue

            group = [item1]
            used.add(i)

            for j, item2 in enumerate(items[i+1:], start=i+1):
                if j in used:
                    continue

                # Simple similarity check (use embeddings in production)
                words1 = set(item1.content.lower().split())
                words2 = set(item2.content.lower().split())
                similarity = len(words1 & words2) / max(len(words1 | words2), 1)

                if similarity >= threshold:
                    group.append(item2)
                    used.add(j)

            groups.append(group)

        return groups

    def get_agent_activity(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get activity statistics for agents."""
        all_items = self.get_items(self.owner_agent_id)

        activity = {}
        for item in all_items:
            creator = item.metadata.get("_created_by", "unknown")
            if agent_id and creator != agent_id:
                continue

            if creator not in activity:
                activity[creator] = {
                    "items_created": 0,
                    "by_category": {},
                    "last_activity": None
                }

            activity[creator]["items_created"] += 1
            cat = item.category.value
            activity[creator]["by_category"][cat] = activity[creator]["by_category"].get(cat, 0) + 1

            created_at = item.metadata.get("_created_at")
            if created_at:
                if not activity[creator]["last_activity"] or created_at > activity[creator]["last_activity"]:
                    activity[creator]["last_activity"] = created_at

        return activity


class AgentView:
    """Permission-scoped view of shared memory for a specific agent."""

    def __init__(self, shared: SharedMemory, agent_id: str):
        """
        Initialize agent view.

        Args:
            shared: Parent SharedMemory
            agent_id: Agent this view is for
        """
        self.shared = shared
        self.agent_id = agent_id

    def record_learning(self, content: str, tags: List[str]) -> str:
        """Record a learning."""
        item = ContextItem(
            content=content,
            category=ContextCategory.LEARNING,
            tags=tags
        )
        return self.shared.add_item(item, self.agent_id)

    def record_decision(self, content: str, tags: List[str]) -> str:
        """Record a decision."""
        item = ContextItem(
            content=content,
            category=ContextCategory.DECISION,
            tags=tags
        )
        return self.shared.add_item(item, self.agent_id)

    def record_action(self, content: str, tags: List[str]) -> str:
        """Record an action."""
        item = ContextItem(
            content=content,
            category=ContextCategory.ACTION,
            tags=tags
        )
        return self.shared.add_item(item, self.agent_id)

    def recall(self, query: Optional[str] = None,
               tags: Optional[List[str]] = None,
               category: Optional[ContextCategory] = None,
               include_others: bool = True) -> List[ContextItem]:
        """
        Recall items from shared memory.

        Args:
            query: Search query
            tags: Filter by tags
            category: Filter by category
            include_others: Include items from other agents

        Returns:
            List of matching items
        """
        created_by = None if include_others else self.agent_id
        items = self.shared.get_items(
            self.agent_id,
            category=category,
            tags=tags,
            created_by=created_by
        )

        if query:
            query_words = set(query.lower().split())
            items = [
                i for i in items
                if query_words & set(i.content.lower().split())
            ]

        return items

    def my_items(self) -> List[ContextItem]:
        """Get items created by this agent."""
        return self.shared.get_items(self.agent_id, created_by=self.agent_id)

    def my_activity(self) -> Dict[str, Any]:
        """Get activity stats for this agent."""
        return self.shared.get_agent_activity(self.agent_id).get(self.agent_id, {})
```

**Usage Example**:

```python
from cogneetree.storage.in_memory_storage import InMemoryStorage
from cogneetree.multiagent.shared_memory import SharedMemory, Permission

# Create shared workspace
storage = InMemoryStorage()
workspace = SharedMemory(
    storage=storage,
    workspace_id="auth-team",
    owner_agent_id="lead_agent"
)

# Register additional agents
workspace.register_agent("research_agent", "Research Agent")
workspace.register_agent("coding_agent", "Coding Agent")

# Grant permissions
workspace.grant_access(
    "research_agent",
    {Permission.READ, Permission.WRITE},
    granted_by="lead_agent"
)
workspace.grant_access(
    "coding_agent",
    {Permission.READ, Permission.WRITE},
    granted_by="lead_agent"
)

# Research agent records findings
research = workspace.as_agent("research_agent")
research.record_learning(
    "JWT RS256 requires public key infrastructure",
    tags=["jwt", "security"]
)
research.record_learning(
    "Auth0 supports RS256 out of the box",
    tags=["jwt", "auth0"]
)

# Coding agent can see research findings
coding = workspace.as_agent("coding_agent")
jwt_knowledge = coding.recall(tags=["jwt"])
print(f"Coding agent found {len(jwt_knowledge)} JWT items from research")

# Coding agent records decisions based on research
coding.record_decision(
    "Use Auth0's RS256 implementation for JWT signing",
    tags=["jwt", "auth0", "decision"]
)

# Lead agent reviews all activity
activity = workspace.get_agent_activity()
print("Team activity:")
for agent_id, stats in activity.items():
    print(f"  {agent_id}: {stats['items_created']} items")

# Merge duplicates from different agents
merge_result = workspace.merge_duplicates(
    strategy="keep_most_recent",
    admin_agent_id="lead_agent"
)
print(f"Merged {merge_result['merged_count']} duplicate items")
```

---

### 4.2 REST API Wrapper

**Problem**: Cogneetree is Python-only - no way to use from other languages or services.

**Solution**: REST API wrapper for language-agnostic access.

**File**: `src/cogneetree/api/rest_api.py`

```python
"""REST API wrapper for Cogneetree."""

from dataclasses import asdict
from datetime import datetime
from typing import Dict, Any, Optional, List
from cogneetree import ContextManager, AgentMemory
from cogneetree.core.models import ContextCategory


class CogneetreeAPI:
    """REST API interface for Cogneetree (use with Flask/FastAPI)."""

    def __init__(self, storage=None):
        """
        Initialize API.

        Args:
            storage: Optional storage backend (defaults to InMemory)
        """
        self.manager = ContextManager(storage=storage)
        self._memories: Dict[str, AgentMemory] = {}

    def _serialize_item(self, item) -> Dict[str, Any]:
        """Serialize a ContextItem to dict."""
        return {
            "content": item.content,
            "category": item.category.value,
            "tags": item.tags,
            "timestamp": item.timestamp.isoformat() if item.timestamp else None,
            "parent_id": item.parent_id,
            "metadata": item.metadata
        }

    def _serialize_session(self, session) -> Dict[str, Any]:
        """Serialize a Session to dict."""
        return {
            "session_id": session.session_id,
            "original_ask": session.original_ask,
            "high_level_plan": session.high_level_plan,
            "created_at": session.created_at.isoformat() if session.created_at else None
        }

    # ===== SESSION ENDPOINTS =====

    def create_session(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST /sessions

        Body: {
            "session_id": "string",
            "original_ask": "string",
            "high_level_plan": "string"
        }
        """
        session = self.manager.create_session(
            data["session_id"],
            data["original_ask"],
            data["high_level_plan"]
        )
        return {"status": "created", "session": self._serialize_session(session)}

    def get_sessions(self) -> Dict[str, Any]:
        """GET /sessions"""
        sessions = self.manager.storage.get_all_sessions()
        return {"sessions": [self._serialize_session(s) for s in sessions]}

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """GET /sessions/{session_id}"""
        session = self.manager.storage.get_session_by_id(session_id)
        if not session:
            return {"error": "Session not found", "status": 404}
        return {"session": self._serialize_session(session)}

    # ===== ACTIVITY ENDPOINTS =====

    def create_activity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST /activities

        Body: {
            "activity_id": "string",
            "session_id": "string",
            "description": "string",
            "tags": ["string"],
            "mode": "string",
            "component": "string",
            "planner_analysis": "string"
        }
        """
        activity = self.manager.create_activity(
            data["activity_id"],
            data["session_id"],
            data["description"],
            data.get("tags", []),
            data.get("mode", "builder"),
            data.get("component", "core"),
            data.get("planner_analysis", "")
        )
        return {"status": "created", "activity_id": activity.activity_id}

    # ===== TASK ENDPOINTS =====

    def create_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST /tasks

        Body: {
            "task_id": "string",
            "activity_id": "string",
            "description": "string",
            "tags": ["string"]
        }
        """
        task = self.manager.create_task(
            data["task_id"],
            data["activity_id"],
            data["description"],
            data.get("tags", [])
        )
        return {"status": "created", "task_id": task.task_id}

    # ===== ITEM ENDPOINTS =====

    def record_item(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST /items

        Body: {
            "category": "learning|decision|action|result",
            "content": "string",
            "tags": ["string"]
        }
        """
        category = data["category"].lower()
        content = data["content"]
        tags = data.get("tags", [])

        if category == "learning":
            self.manager.record_learning(content, tags=tags)
        elif category == "decision":
            self.manager.record_decision(content, tags=tags)
        elif category == "action":
            self.manager.record_action(content, tags=tags)
        elif category == "result":
            self.manager.record_result(content, tags=tags)
        else:
            return {"error": f"Invalid category: {category}", "status": 400}

        return {"status": "created", "category": category}

    def get_items(self, task_id: Optional[str] = None,
                  category: Optional[str] = None,
                  tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        GET /items?task_id=X&category=Y&tags=a,b,c
        """
        if task_id:
            items = self.manager.storage.get_items_by_task(task_id)
        elif tags:
            items = self.manager.storage.get_items_by_tags(tags)
        else:
            items = self.manager.storage.get_all_items() if hasattr(self.manager.storage, 'get_all_items') else []

        if category:
            cat_enum = ContextCategory(category)
            items = [i for i in items if i.category == cat_enum]

        return {"items": [self._serialize_item(i) for i in items]}

    # ===== RECALL ENDPOINTS =====

    def recall(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST /recall

        Body: {
            "query": "string",
            "task_id": "string" (optional),
            "scope": "micro|balanced|macro" (optional),
            "max_items": int (optional)
        }
        """
        task_id = data.get("task_id", self.manager.get_current_task_id())

        if task_id not in self._memories:
            self._memories[task_id] = AgentMemory(
                self.manager.storage,
                current_task_id=task_id
            )

        memory = self._memories[task_id]

        results = memory.recall(
            data["query"],
            scope=data.get("scope", "balanced"),
            max_items=data.get("max_items", 10)
        )

        return {
            "query": data["query"],
            "results": [self._serialize_item(r) for r in results],
            "count": len(results)
        }

    def build_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST /context

        Body: {
            "query": "string",
            "task_id": "string" (optional),
            "max_items": int (optional)
        }
        """
        task_id = data.get("task_id", self.manager.get_current_task_id())

        if task_id not in self._memories:
            self._memories[task_id] = AgentMemory(
                self.manager.storage,
                current_task_id=task_id
            )

        memory = self._memories[task_id]
        context = memory.build_context(
            data["query"],
            max_items=data.get("max_items", 5)
        )

        return {"context": context, "query": data["query"]}

    # ===== STATS ENDPOINTS =====

    def stats(self) -> Dict[str, Any]:
        """GET /stats"""
        if hasattr(self.manager.storage, 'get_stats'):
            return self.manager.storage.get_stats()
        return {"error": "Stats not available for this storage backend"}


# ===== FLASK INTEGRATION =====

def create_flask_app(storage=None):
    """
    Create Flask app with Cogneetree API.

    Usage:
        from cogneetree.api.rest_api import create_flask_app
        app = create_flask_app()
        app.run(port=8080)
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        raise ImportError("Flask required: pip install flask")

    app = Flask(__name__)
    api = CogneetreeAPI(storage=storage)

    @app.route("/sessions", methods=["GET", "POST"])
    def sessions():
        if request.method == "POST":
            return jsonify(api.create_session(request.json))
        return jsonify(api.get_sessions())

    @app.route("/sessions/<session_id>", methods=["GET"])
    def get_session(session_id):
        return jsonify(api.get_session(session_id))

    @app.route("/activities", methods=["POST"])
    def create_activity():
        return jsonify(api.create_activity(request.json))

    @app.route("/tasks", methods=["POST"])
    def create_task():
        return jsonify(api.create_task(request.json))

    @app.route("/items", methods=["GET", "POST"])
    def items():
        if request.method == "POST":
            return jsonify(api.record_item(request.json))
        return jsonify(api.get_items(
            task_id=request.args.get("task_id"),
            category=request.args.get("category"),
            tags=request.args.get("tags", "").split(",") if request.args.get("tags") else None
        ))

    @app.route("/recall", methods=["POST"])
    def recall():
        return jsonify(api.recall(request.json))

    @app.route("/context", methods=["POST"])
    def build_context():
        return jsonify(api.build_context(request.json))

    @app.route("/stats", methods=["GET"])
    def stats():
        return jsonify(api.stats())

    return app


# ===== FASTAPI INTEGRATION =====

def create_fastapi_app(storage=None):
    """
    Create FastAPI app with Cogneetree API.

    Usage:
        from cogneetree.api.rest_api import create_fastapi_app
        app = create_fastapi_app()
        # Run with: uvicorn module:app --port 8080
    """
    try:
        from fastapi import FastAPI, Query
        from pydantic import BaseModel
        from typing import List, Optional
    except ImportError:
        raise ImportError("FastAPI required: pip install fastapi uvicorn")

    app = FastAPI(title="Cogneetree API", version="1.0.0")
    api = CogneetreeAPI(storage=storage)

    class SessionCreate(BaseModel):
        session_id: str
        original_ask: str
        high_level_plan: str

    class ItemCreate(BaseModel):
        category: str
        content: str
        tags: List[str] = []

    class RecallRequest(BaseModel):
        query: str
        task_id: Optional[str] = None
        scope: str = "balanced"
        max_items: int = 10

    @app.post("/sessions")
    def create_session(data: SessionCreate):
        return api.create_session(data.dict())

    @app.get("/sessions")
    def get_sessions():
        return api.get_sessions()

    @app.post("/items")
    def record_item(data: ItemCreate):
        return api.record_item(data.dict())

    @app.post("/recall")
    def recall(data: RecallRequest):
        return api.recall(data.dict())

    @app.get("/stats")
    def stats():
        return api.stats()

    return app
```

**Usage Example**:

```bash
# Start the API server
python -c "from cogneetree.api.rest_api import create_flask_app; create_flask_app().run(port=8080)"
```

```python
# Client usage (any language with HTTP)
import requests

BASE = "http://localhost:8080"

# Create session
requests.post(f"{BASE}/sessions", json={
    "session_id": "s1",
    "original_ask": "Implement JWT",
    "high_level_plan": "Research and implement"
})

# Create activity and task
requests.post(f"{BASE}/activities", json={
    "activity_id": "a1",
    "session_id": "s1",
    "description": "Research JWT",
    "tags": ["jwt", "research"]
})

requests.post(f"{BASE}/tasks", json={
    "task_id": "t1",
    "activity_id": "a1",
    "description": "Learn JWT structure",
    "tags": ["jwt"]
})

# Record items
requests.post(f"{BASE}/items", json={
    "category": "learning",
    "content": "JWT has three parts",
    "tags": ["jwt", "structure"]
})

# Recall
response = requests.post(f"{BASE}/recall", json={
    "query": "JWT structure",
    "scope": "balanced"
})
print(response.json())
```

---

### 4.3 Structured Output Integration

**Problem**: Returns raw text - no native support for modern LLM structured outputs.

**Solution**: Native Pydantic integration for type-safe knowledge retrieval.

**File**: `src/cogneetree/structured/structured_output.py`

```python
"""Structured output integration for type-safe retrieval."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Type, TypeVar, Generic
from cogneetree.core.models import ContextItem, ContextCategory

T = TypeVar('T')


@dataclass
class StructuredResult(Generic[T]):
    """Result with structured data."""
    data: T
    source_item: ContextItem
    confidence: float
    extraction_method: str


class StructuredMemory:
    """Memory with structured output support."""

    def __init__(self, memory, llm_client=None):
        """
        Initialize structured memory.

        Args:
            memory: AgentMemory instance
            llm_client: Optional LLM client for extraction
        """
        self.memory = memory
        self.llm_client = llm_client

    def recall_structured(
        self,
        query: str,
        schema: Type[T],
        max_items: int = 5,
        min_confidence: float = 0.5
    ) -> List[StructuredResult[T]]:
        """
        Recall and parse items into structured format.

        Args:
            query: Search query
            schema: Pydantic model or dataclass to parse into
            max_items: Maximum items to return
            min_confidence: Minimum confidence to include

        Returns:
            List of StructuredResult with parsed data
        """
        # Get raw items
        raw_items = self.memory.recall(query, max_items=max_items * 2)

        results = []
        for item in raw_items:
            try:
                parsed, confidence = self._parse_item(item, schema)
                if confidence >= min_confidence:
                    results.append(StructuredResult(
                        data=parsed,
                        source_item=item,
                        confidence=confidence,
                        extraction_method="pattern" if not self.llm_client else "llm"
                    ))
            except Exception:
                continue

        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:max_items]

    def _parse_item(self, item: ContextItem, schema: Type[T]) -> tuple:
        """Parse item into schema."""
        if self.llm_client:
            return self._parse_with_llm(item, schema)
        return self._parse_with_patterns(item, schema)

    def _parse_with_patterns(self, item: ContextItem, schema: Type[T]) -> tuple:
        """Parse using pattern matching."""
        # Get schema fields
        if hasattr(schema, '__dataclass_fields__'):
            fields = schema.__dataclass_fields__
        elif hasattr(schema, 'model_fields'):  # Pydantic v2
            fields = schema.model_fields
        elif hasattr(schema, '__fields__'):  # Pydantic v1
            fields = schema.__fields__
        else:
            raise ValueError(f"Unsupported schema type: {type(schema)}")

        # Extract values from content
        extracted = {}
        content_lower = item.content.lower()

        for field_name, field_info in fields.items():
            # Try to find field value in content
            value = self._extract_field(content_lower, field_name, field_info)
            if value is not None:
                extracted[field_name] = value

        # Calculate confidence based on fields extracted
        confidence = len(extracted) / len(fields)

        # Create instance
        try:
            instance = schema(**extracted)
            return instance, confidence
        except Exception:
            # Fill missing required fields with defaults
            return None, 0.0

    def _extract_field(self, content: str, field_name: str,
                       field_info: Any) -> Optional[Any]:
        """Extract a field value from content."""
        import re

        # Common patterns for field extraction
        patterns = [
            rf"{field_name}[:\s]+([^\.,]+)",  # "field: value"
            rf"{field_name}\s+is\s+([^\.,]+)",  # "field is value"
            rf"([^\.,]+)\s+{field_name}",  # "value field"
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                return self._convert_value(value, field_info)

        return None

    def _convert_value(self, value: str, field_info: Any) -> Any:
        """Convert string value to appropriate type."""
        # Get type annotation
        if hasattr(field_info, 'annotation'):
            type_hint = field_info.annotation
        elif hasattr(field_info, 'outer_type_'):
            type_hint = field_info.outer_type_
        else:
            return value

        # Convert based on type
        if type_hint == int:
            return int(value) if value.isdigit() else None
        elif type_hint == float:
            try:
                return float(value)
            except ValueError:
                return None
        elif type_hint == bool:
            return value.lower() in ('true', 'yes', '1')
        else:
            return value

    def _parse_with_llm(self, item: ContextItem, schema: Type[T]) -> tuple:
        """Parse using LLM extraction."""
        # Generate schema description
        if hasattr(schema, 'model_json_schema'):  # Pydantic v2
            schema_desc = schema.model_json_schema()
        elif hasattr(schema, 'schema'):  # Pydantic v1
            schema_desc = schema.schema()
        else:
            schema_desc = str(schema)

        prompt = f"""Extract structured data from the following text.

Schema: {schema_desc}

Text: {item.content}

Return a JSON object matching the schema. If a field cannot be determined, use null."""

        try:
            response = self._call_llm(prompt)
            import json
            data = json.loads(response)
            instance = schema(**data)
            return instance, 0.9  # High confidence for LLM extraction
        except Exception:
            return self._parse_with_patterns(item, schema)

    def _call_llm(self, prompt: str) -> str:
        """Call LLM for extraction."""
        if hasattr(self.llm_client, 'messages'):  # Anthropic
            response = self.llm_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        elif hasattr(self.llm_client, 'chat'):  # OpenAI
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        raise ValueError("Unsupported LLM client")

    def decisions_as(self, schema: Type[T],
                     tags: Optional[List[str]] = None) -> List[StructuredResult[T]]:
        """Get decisions parsed into structured format."""
        query = " ".join(tags) if tags else "decision"
        return self.recall_structured(query, schema)

    def learnings_as(self, schema: Type[T],
                     tags: Optional[List[str]] = None) -> List[StructuredResult[T]]:
        """Get learnings parsed into structured format."""
        query = " ".join(tags) if tags else "learning"
        return self.recall_structured(query, schema)
```

**Usage Example**:

```python
from dataclasses import dataclass
from typing import Optional, Literal
from cogneetree import ContextManager, AgentMemory
from cogneetree.structured.structured_output import StructuredMemory

# Define structured schema
@dataclass
class JWTDecision:
    algorithm: str
    reason: str
    security_level: Literal["high", "medium", "low"] = "medium"


@dataclass
class SecurityLearning:
    vulnerability: str
    mitigation: str
    severity: Optional[str] = None


# Setup
manager = ContextManager()
manager.create_session("s1", "Security Review", "Plan")
manager.create_activity("a1", "s1", "JWT Security", ["jwt", "security"], "builder", "core", "...")
manager.create_task("t1", "a1", "Review JWT", ["jwt"])

# Record items with structured content
manager.record_decision(
    "Algorithm: RS256. Reason: Supports key rotation. Security level: high",
    tags=["jwt", "algorithm"]
)
manager.record_learning(
    "Vulnerability: Algorithm confusion attack. Mitigation: Always validate alg header. Severity: critical",
    tags=["jwt", "security"]
)

# Create structured memory
memory = AgentMemory(manager.storage, current_task_id="t1")
structured = StructuredMemory(memory)

# Retrieve as structured data
jwt_decisions = structured.decisions_as(JWTDecision, tags=["jwt"])
for result in jwt_decisions:
    print(f"Decision: {result.data.algorithm}")
    print(f"  Reason: {result.data.reason}")
    print(f"  Security: {result.data.security_level}")
    print(f"  Confidence: {result.confidence:.2f}")

security_learnings = structured.learnings_as(SecurityLearning, tags=["security"])
for result in security_learnings:
    print(f"Vulnerability: {result.data.vulnerability}")
    print(f"  Mitigation: {result.data.mitigation}")
    print(f"  Severity: {result.data.severity}")
```

---

## Summary: Implementation Roadmap

| Phase | Feature | Priority | Effort |
|-------|---------|----------|--------|
| **1.1** | Complete Storage Backends | Critical | 2 days |
| **1.2** | Persistent Embedding Cache | Critical | 2 days |
| **1.3** | Thread-Safe Context | Critical | 1 day |
| **2.1** | Batch Operations | High | 1 day |
| **2.2** | Update/Delete Operations | High | 1 day |
| **2.3** | Analytics & Observability | High | 2 days |
| **3.1** | Auto Knowledge Extraction | High | 3 days |
| **3.2** | Causal Chains | Medium | 3 days |
| **3.3** | Memory Consolidation | Medium | 2 days |
| **3.4** | Time-Travel Queries | Medium | 2 days |
| **4.1** | Multi-Agent Shared Memory | Medium | 3 days |
| **4.2** | REST API Wrapper | Medium | 1 day |
| **4.3** | Structured Output Integration | Low | 2 days |

**Total Estimated Effort**: ~25 days

These implementations will transform Cogneetree from a solid foundation into a differentiated, production-ready memory system for AI agents.

---

## See Also

- [TESTING_PLAN.md](TESTING_PLAN.md) - Comprehensive testing strategy
- [CLAUDE.md](../CLAUDE.md) - Architecture and design philosophy
- [AGENT_MEMORY.md](../AGENT_MEMORY.md) - Agent API reference
- [README.md](../README.md) - Quick start guide

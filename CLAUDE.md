# CLAUDE.md

Guidance for working with Cogneetree: hierarchical agentic memory with micro/macro search.

---

## Philosophy

**Cogneetree enables AI agents to build true long-term memory across projects, not just remember the current session.**

### Core Insight: Knowledge Grows in Branches

Every conversation, project, or task is a **branch** in a growing knowledge tree:

```
Knowledge Tree (your agentic memory)
├── Project 1: "Implement OAuth2" (Session 1)
│   ├── Learn grant types → Actions, Decisions, Learnings
│   ├── Understand JWT → Actions, Decisions, Learnings
│   └── Result: Working auth system
│
├── Project 2: "Build microservices" (Session 2)
│   ├── Service authentication → Actions, Decisions, Learnings
│   └── Result: Microservices communicating
│
└── Project 3: "New project" (Session 3) ← You are here
    ├── Setup authentication → micro-search finds relevant past learnings
    └── (future work)
```

**At each branch (task), you record:**
- **Actions**: What you did
- **Decisions**: Why you chose that approach
- **Learnings**: What you discovered
- **Results**: What was accomplished

**This becomes your knowledge bank.** When the agent faces a similar problem later, it can ask:
- **Micro Search** ("Fresh"): "Just show me what I learned in THIS task"
- **Macro Search** ("Learn from All"): "Show me EVERYTHING related to this pattern across all my projects"
- **Balanced Search** (Default): "Prioritize THIS project but remind me what I learned before"

---

## Design Structure: Hierarchical Retrieval

### Three-Level Hierarchy + Micro/Macro Search

```
Session (Project scope)
  "Implement OAuth2 authentication"

  ├── Activity (Work area)
  │   "Understand OAuth2 fundamentals"
  │
  │   ├── Task (Atomic work)
  │   │   "Learn grant types"
  │   │   │
  │   │   ├── Action: "Read RFC 6749"
  │   │   ├── Decision: "Use auth code flow for web apps"
  │   │   ├── Learning: "OAuth2 has 4 grant types"
  │   │   └── Result: "Understand complete"
  │   │
  │   └── Task (Atomic work)
  │       "Understand JWT"
  │       │
  │       ├── Action: "Decode JWT on jwt.io"
  │       ├── Decision: "Use HS256 for symmetric signing"
  │       ├── Learning: "JWT: header.payload.signature"
  │       └── Result: "JWT structure clear"
  │
  └── Activity (Work area)
      "Implement in API"
      └── Task
          "Add JWT validation middleware"
          └── (Records actions/decisions/learnings)
```

**The hierarchy IS your memory structure.** No separate concept graph—work is organized hierarchically.

### Retrieval Modes: Choosing Your Search Scope

**Micro Search (Current Only)**
```python
config = RETRIEVAL_PRESETS["fresh"]
results = retriever.retrieve("JWT validation", task_id="t4", config=config)
# Returns: Items from THIS task only
# Use: Focused work, avoid distractions
```

**Macro Search (All Sessions)**
```python
config = RETRIEVAL_PRESETS["learn_from_all"]
results = retriever.retrieve("JWT validation", task_id="t4", config=config)
# Returns: Items from ALL projects, weighted equally
# Use: Pattern finding, learning how you've solved this before
```

**Balanced Search (Current Weighted) — Default**
```python
config = RETRIEVAL_PRESETS["balanced"]
results = retriever.retrieve("JWT validation", task_id="t4", config=config)
# Returns: Current task/activity prioritized + relevant history
# Use: Normal development—stay focused but learn from past
```

**Selective Search (Specific Projects)**
```python
config = RetrievalConfig(
    history_mode=HistoryMode.SELECTED_SESSIONS,
    include_session_ids=["s1", "s3"]  # Only these projects
)
# Use: Comparing specific projects or focused debugging
```

### Scoring Formula

```
final_score = semantic_similarity × proximity_weight × temporal_weight

Where:
- semantic_similarity: How relevant is this to my query? (0.0-1.0)
- proximity_weight: How close in hierarchy?
  * Current task: 1.0
  * Sibling task (same activity): 0.7
  * Same activity: 0.5
  * Same session: 0.3
  * Other sessions: 0.2 (configurable)
- temporal_weight: How recent? (favor newer learning)
```

**Result:** Items close to current work rank highest, but highly relevant items from months ago still surface.

---

## Knowledge Bank: Recorded at Each Branch

### Task Level (Most Granular)

```python
with task.task("Implement JWT validation") as ctx:
    ctx.record_action("Implemented middleware using express-jwt")
    ctx.record_decision("Use RS256 instead of HS256 for key rotation")
    ctx.record_learning("RS256 requires public key distribution")
    ctx.set_result("Validation working with refresh token flow")
```

These become **searchable knowledge** via hierarchical retrieval.

### Activity Level (Aggregated)

All task-level items accessible at activity scope. Activity-level decisions can override tasks.

### Session Level (Project Level)

Highest-level outcomes and project-wide decisions.

### Query at Any Level

```python
# Task-level
results = retriever.retrieve("JWT validation", task_id="t4")

# Activity-level
results = retriever.retrieve("authentication", activity_id="a1")

# Session-level (entire project)
results = retriever.retrieve("oauth", session_id="s1")
```

---

## Development Best Practices

### 1. SOLID Principles

**Single Responsibility**
- Each class has ONE reason to change
- `ContextManager`: Orchestration
- `Retriever`: Retrieval logic
- `ContextStorage`: Storage contract

**Open/Closed**
- New storage? Subclass `ContextStorageABC`
- New embeddings? Subclass `EmbeddingModelABC`
- Existing code unchanged

**Liskov Substitution**
- All storage implementations interchangeable
- All embedding models interchangeable

**Interface Segregation**
- `ContextStorageABC`: Only what storage needs
- `EmbeddingModelABC`: Only encoding/similarity

**Dependency Inversion**
- Depend on abstractions, not concrete classes
- Easy to swap implementations

### 2. DRY (Don't Repeat Yourself)

**Embedding Cache**
```python
# Reuse embeddings within single retrieval
embedding_cache["query"] = compute_once(query)
embedding_cache[item_id] = compute_once(item)
```

**Storage Factory Pattern**
```python
# Single place to create storage
storage = ContextManager._create_storage(config)
# Override once, all code uses it
```

**Proximity Configuration**
```python
# Define weights once in RetrievalConfig
config.current_weight = 1.0
config.sibling_weight = 0.7
# No magic numbers scattered
```

### 3. Code Clarity (1-Minute Readability)

**Explicit Names**
- `HierarchicalRetriever` not `Retriever`
- `HistoryMode` enum not string magic
- `RetrievalConfig` self-documents

**Docstrings Explain Why**
```python
"""
Current hierarchy prioritized; include other sessions with lower weight.

This ensures agents focus on current work while learning from past.
"""
```

**Tests Document Usage**
```python
def test_micro_search(self):
    """MICRO: Only search current task."""
    config = RetrievalConfig(history_mode=HistoryMode.CURRENT_ONLY)
    # Shows exactly how to use
```

### 4. Testing Strategy

**Test Each Mode**
- `test_current_only_mode()` - Focused work
- `test_current_weighted_mode()` - Default balanced
- `test_all_sessions_mode()` - Learning across projects
- `test_selected_sessions_mode()` - Comparing specific projects

**Test User Control**
- `test_weight_configuration()` - Proximity weights
- `test_time_filtering()` - Age-based filtering
- `test_explainability_on_off()` - Reasoning output

**Test Hierarchy**
- `test_same_task_items_highest_priority()` - Proximity matters
- `test_sibling_tasks_boost()` - Hierarchy respected

### 5. Configuration Philosophy

**Presets for Common Patterns**
```python
RETRIEVAL_PRESETS = {
    "fresh":         # Current task only
    "balanced":      # Weighted history (default)
    "learn_from_all": # All equal
    "debug":         # Everything with reasoning
}
```

**User Agency**
- Choose `history_mode`
- Filter by time: `history_depth_days`
- Select specific sessions
- Control output: `return_explainability=True`

**Transparency**
```python
result = {
    "item": item,
    "semantic_score": 0.85,      # How relevant?
    "proximity_weight": 0.7,     # How close?
    "final_score": 0.595,        # Combined
    "explanation": {              # Why?
        "source_session_id": "s1",
        "source_location": "Task: Learn grant types",
        "why": "Semantic match (0.85) × proximity (0.7)"
    }
}
```

---

## Development Commands

### Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests with coverage report
pytest

# Run specific test file
pytest tests/test_agentic_context_config.py

# Run single test function
pytest tests/test_agentic_context_config.py::test_config_defaults

# Run tests with verbose output
pytest -v

# Run tests and generate HTML coverage report
pytest --cov-report=html
```

### Code Quality
```bash
# Format code
black src tests

# Sort imports
isort src tests

# Run linter
flake8 src tests

# Type checking
mypy src
```

### Building & Publishing
```bash
# Build distribution
python -m build

# Test upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*
```

## Project Structure

### Core Architecture: Three-Level Hierarchy

The library implements a three-level hierarchical context structure:

```
Session (top-level context)
  ├── original_ask: the main question/request
  ├── high_level_plan: initial strategic plan
  └─── Activity (mid-level work unit)
       ├── activity_id, description
       ├── tags: categorization tags
       ├── mode, component: execution context
       ├── planner_analysis: planning info
       └─── Task (atomic work item)
            ├── task_id, description
            ├── tags: inherits from activity
            ├── result: outcome
            └─── Context Items (recorded during task execution)
                 ├── Actions: what was done
                 ├── Decisions: why choices were made
                 ├── Learnings: what was discovered
                 └─── Results: outcomes and summaries
```

Each context level is tracked in `ContextStorage` with parent-child relationships, allowing retrieval of context at any granularity.

### Module Organization

**[src/cogneetree/workflow.py](src/cogneetree/workflow.py)** - Entry point with context managers
- `ContextWorkflow`: Main entry class that manages sessions
- `SessionContext`, `ActivityContext`, `TaskContext`: Nested context managers
- Provides clean `with` statement API for hierarchical context tracking

**[src/cogneetree/core/](src/cogneetree/core/)** - Core context management
- `context_manager.py`: `ContextManager` orchestrates storage and retrieval, delegates to storage and retriever
- `context_storage.py`: `ContextStorage` flat in-memory storage with parent tracking, tracks current context via `current_session_id`, `current_activity_id`, `current_task_id`
- Data models: `Session`, `Activity`, `Task`, `ContextItem` (dataclasses)

**[src/cogneetree/retrieval/](src/cogneetree/retrieval/)** - Context retrieval
- `retrieval_strategies.py`: `Retriever` implements hybrid retrieval (tag-based + optional semantic)
- `semantic_retrieval.py`: Optional semantic embeddings using transformers (only loaded if `config.use_semantic=True`)
- `tag_normalization.py`: Tag matching and normalization utilities

**[src/cogneetree/config.py](src/cogneetree/config.py)** - Configuration
- `Config` dataclass with retrieval settings
- `use_semantic`: Enable embedding-based retrieval
- `max_results`, `min_score`: Retrieval constraints
- `embedding_model`: Which transformer model to use for embeddings

### Data Flow

1. **Recording Context**: Task context manager methods call `manager.record_action/decision/learning()` → stores in `ContextStorage.items` with tags
2. **Retrieval**: `manager.retrieve(query_tags, query_description)` → `Retriever` filters by tags, optionally scores by semantic similarity
3. **Prompt Building**: `manager.build_prompt()` → collects Session/Activity/Task info + relevant context items → formatted string for LLM
4. **Storage**: All items stored in-memory by default; designed to support pluggable storage backends (SQLite, Redis, etc.)

### Configuration Patterns

The library uses optional semantic retrieval:
- Default: tag-based retrieval only (no ML dependencies)
- Optional: `Config.semantic()` enables embedding-based ranking
- Embeddings only computed if explicitly enabled to avoid unnecessary overhead

## Key Implementation Notes

- **Minimal Dependencies**: No required external dependencies (semantic retrieval is optional)
- **In-Memory Storage**: Current implementation uses flat storage with parent tracking; designed for extension with persistent backends
- **Context Stack**: Uses Python's context managers and instance variables to maintain current session/activity/task
- **Tag-Based Org**: Primary organization is via tags; tags flow from Activity to Task to all recorded items
- **Timestamp Tracking**: All context items and entities include creation timestamps for temporal awareness

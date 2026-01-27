# CLAUDE.md - Development Guide

Guidance for developers extending Cogneetree and understanding its architecture.

**For AI agents using Cogneetree, see [AGENT_MEMORY.md](AGENT_MEMORY.md).**

---

## Philosophy

**Cogneetree enables AI agents to build true long-term memory across projects.**

### Core Insight: Knowledge Grows in Branches

Every project, conversation, or task is a **branch** in a growing knowledge tree:

```
Knowledge Tree
├── Project 1: "Implement OAuth2" (Session 1)
│   ├── Learn grant types → Actions, Decisions, Learnings
│   ├── Understand JWT → Actions, Decisions, Learnings
│   └── Result: Working auth system
├── Project 2: "Build microservices" (Session 2)
│   └── Service authentication → Actions, Decisions, Learnings
└── Project 3: "Current work" (Session 3) ← You are here
    └── Agent searches past branches, finds relevant knowledge
```

At each branch (task), agents record **actions, decisions, learnings, results**. This becomes searchable knowledge via hierarchical retrieval.

When agents face similar problems later, they can search with different scopes:
- **Micro**: "Show me THIS task only"
- **Balanced**: "Prioritize current task + relevant history" (default)
- **Macro**: "Show me ALL projects equally"

The hierarchy **IS the memory structure**—no separate concept graph. Work organized naturally.

---

## Architecture: Hierarchical Retrieval

### Three-Level Hierarchy

```
Session (Project)
  ├── original_ask, high_level_plan
  └── Activity (Work Area)
      ├── description, tags, mode, component
      └── Task (Atomic Work)
          ├── description, tags, result
          └── Context Items (Actions, Decisions, Learnings, Results)
```

### Retrieval Modes

Agents control what history influences decisions:

| Mode | Search Scope | Use Case |
|------|--------------|----------|
| **CURRENT_ONLY** | This task only | Focused work, avoid noise |
| **CURRENT_WEIGHTED** | Task prioritized + history | Normal work (default) |
| **ALL_SESSIONS** | All projects equally | Learning patterns |
| **SELECTED_SESSIONS** | Specific projects only | Comparing projects |

Exposed to agents as simple scopes: `micro`, `balanced`, `macro`.

### Scoring Formula

```
final_score = semantic_similarity × proximity_weight × temporal_weight

proximity_weight:
  • Current task: 1.0
  • Sibling task (same activity): 0.7
  • Same activity: 0.5
  • Same session: 0.3
  • Other sessions: 0.2 (configurable)

temporal_weight: Recent items rank higher
```

Result: Items close to current work rank highest, but relevant items from months ago still surface.

---

## Design Decisions

### Why Hierarchical, Not Flat?

- **Natural to work**: Sessions → Activities → Tasks mirrors real work structure
- **Smart scoping**: Can search "just this task" or "all projects" with same API
- **Proximity weighting**: Closer items matter more, but history still surfaces
- **Storage friendly**: Parent tracking works with any backend (SQL, Redis, etc.)

### Why Micro/Macro Terminology?

- **Clear intentions**: "Micro" = focused, "Macro" = broad learning
- **Agent-friendly**: Simple language, not config details
- **Transparent defaults**: "Balanced" is optimal for 95% of cases

### Why Scoring × Proximity × Semantic?

Agents need three orthogonal signals:
1. **Semantic relevance**: Is this conceptually related?
2. **Proximity**: How close in the hierarchy?
3. **Temporal**: How recent?

Multiplying them prevents any signal from dominating.

### Embedding Cache

DRY principle: Compute embeddings once per retrieval call.

```python
embedding_cache: Dict[str, Any] = {}
# Reuse same embeddings across all items
```

---

## Knowledge Bank: What Gets Recorded

### Task Level (Most Granular)

```python
manager.record_action("Implemented JWT middleware")
manager.record_decision("Use RS256 for key rotation")
manager.record_learning("RS256 requires public key distribution")
manager.record_result("Validation working with refresh tokens")
```

These become searchable via hierarchical retrieval.

### Activity & Session Levels

Activity-level items accessed when querying by activity. Session-level items for project-wide decisions.

---

## Development Best Practices

### 1. SOLID Principles

**Single Responsibility**
- `ContextManager`: Orchestration
- `HierarchicalRetriever`: Retrieval logic
- `ContextStorageABC`: Storage contract

**Open/Closed**
- New storage? Subclass `ContextStorageABC`
- Existing code unchanged

**Dependency Inversion**
- Depend on `ContextStorageABC`, not concrete classes
- Easy to swap implementations

### 2. DRY (Don't Repeat Yourself)

**Use Factory Pattern**
```python
# Single place to create storage
self.storage = self._create_storage(config)
# Override once, all code uses it
```

**Cache Expensive Computations**
```python
embedding_cache: Dict[str, Any] = {}
# Reuse embeddings within retrieval call
```

**Configuration Over Magic Numbers**
```python
# Define weights once
config.current_weight = 1.0
config.sibling_weight = 0.7
# No magic numbers scattered throughout
```

### 3. Code Clarity (1-Minute Readability)

**Explicit Names**
- `HierarchicalRetriever` vs generic `Retriever`
- `HistoryMode` enum vs string magic
- `RetrievalConfig` self-documents fields

**Tests Document Usage**
```python
def test_current_weighted_mode():
    """CURRENT_WEIGHTED: Prioritize current, include history."""
    config = RetrievalConfig(history_mode=HistoryMode.CURRENT_WEIGHTED)
    # Shows exactly how to use
```

### 4. Testing Strategy

**Test Each Mode**
- `test_current_only_mode()` - Focused work
- `test_current_weighted_mode()` - Balanced search
- `test_all_sessions_mode()` - Pattern learning
- `test_selected_sessions_mode()` - Comparing projects

**Test User Control**
- Weight configuration
- Time filtering
- Explainability toggle

**Test Hierarchy**
- Same-task priority
- Sibling task boost

### 5. Configuration Philosophy

**Presets for Common Patterns**
```python
RETRIEVAL_PRESETS = {
    "fresh":         # Current task only
    "balanced":      # Weighted history (default)
    "learn_from_all": # All sessions equal
    "debug":         # Everything + reasoning
}
```

**User Agency**
- Choose history mode
- Filter by time depth
- Select specific sessions
- Toggle explainability

**Transparency**
All results include:
- Semantic similarity score
- Proximity weight
- Final score
- Explanation of why selected

---

## Development Commands

### Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_hierarchical_retriever.py

# Run single test
pytest tests/test_hierarchical_retriever.py::test_current_weighted_mode

# Verbose output
pytest -v

# With coverage
pytest --cov-report=html
```

### Code Quality

```bash
# Format code
black src tests

# Sort imports
isort src tests

# Lint
flake8 src tests

# Type checking
mypy src
```

### Publishing

```bash
# Build
python -m build

# Test upload
python -m twine upload --repository testpypi dist/*

# Publish
python -m twine upload dist/*
```

---

## Project Structure

### Core Modules

**[src/cogneetree/](src/cogneetree/)**

- `__init__.py` - Public API exports
- `config.py` - Configuration dataclass
- `core/` - Context management
  - `interfaces.py` - Abstract base classes (storage, embeddings)
  - `models.py` - Data models (Session, Activity, Task, ContextItem)
  - `context_manager.py` - Simple interface to storage + retrieval
- `storage/` - Storage implementations
  - `in_memory_storage.py` - Default in-memory backend
- `retrieval/` - Retrieval system
  - `hierarchical_retriever.py` - Hierarchical retrieval (4 modes + scoring)
  - `retrieval_strategies.py` - Tag-based + semantic hybrid retrieval
  - `tag_normalization.py` - Tag utilities
- `agent_memory.py` - Agent-friendly interface (wraps HierarchicalRetriever)
- `workflow.py` - Context manager API

### Data Flow

1. **Recording**: `manager.record_action()` → `storage.add_item()` with tags + parent_id
2. **Retrieval**: `memory.recall()` → `HierarchicalRetriever` gathers candidates by mode → scores by semantic + proximity → returns sorted results
3. **Prompt Building**: `memory.build_context()` → formats results as markdown → ready for LLM
4. **Storage**: Flat item list + parent tracking + context stack (current session/activity/task)

---

## Storage Backend Implementation

To implement a custom storage backend:

```python
from cogneetree.core.interfaces import ContextStorageABC

class CustomStorage(ContextStorageABC):
    def __init__(self):
        # Track current context
        self.current_session_id = None
        self.current_activity_id = None
        self.current_task_id = None

    def create_session(self, session_id, original_ask, high_level_plan):
        # Store and set as current
        self.current_session_id = session_id
        return Session(session_id, original_ask, high_level_plan)

    # Implement all abstract methods
    # get_items_by_task, get_items_by_activity, get_items_by_session, etc.
```

Required methods:
- Context creation: `create_session`, `create_activity`, `create_task`
- Context access: `get_current_*`, `get_*_by_id`
- Item operations: `add_item`, `get_items_by_*`
- Context stack management (current_session_id, etc.)

---

## Key Implementation Notes

- **Zero Required Dependencies**: Core library has no external dependencies; semantic retrieval optional
- **Embedding Cache**: Computed once per retrieval call, reused across items
- **Parent Tracking**: Items linked to parent task via `parent_id` field
- **Timestamp Tracking**: All entities include `created_at` for temporal filtering
- **Context Stack**: Using instance variables for current session/activity/task (not thread-safe by design—use separate managers for threads)

---

## See Also

- [README.md](README.md) - Overview and installation
- [AGENT_MEMORY.md](AGENT_MEMORY.md) - Agent API reference and integration examples
- [examples/](examples/) - Real-world usage patterns

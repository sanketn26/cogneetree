# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

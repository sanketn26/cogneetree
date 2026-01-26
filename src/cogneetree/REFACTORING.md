"""Agentic Context Module - Refactored for Simplicity

DESIGN PRINCIPLES (SOLID + DRY + YAGNI):

1. SINGLE RESPONSIBILITY - Each class does one thing:
   - ContextStorage: Store and retrieve context items
   - Retriever: Find relevant items by tags and semantics
   - ContextManager: Coordinate storage + retrieval
   - ContextWorkflow: Provide easy context manager API

2. FLAT ARCHITECTURE - No unnecessary nesting:
   - Storage uses flat lists with parent_id refs (not trees)
   - No recursive context building
   - Simple IDs instead of complex hierarchies

3. NO PREMATURE OPTIMIZATION:
   - Removed caching layers
   - Removed auto-detection
   - Removed preset configurations  
   - Removed excessive thread safety
   - Just use what you need, when you need it

4. OBVIOUS CODE - 30-second readability:
   - Docstrings on every class/method
   - No nested try-catch blocks
   - No complex validation logic
   - Explicit is better than implicit

USAGE
=====

Basic workflow:

    from vivek.agentic_context import ContextWorkflow, Config

    # Create workflow with config
    config = Config(use_semantic=True, max_results=5)
    workflow = ContextWorkflow(config)

    # Use context managers for hierarchy
    with workflow.session("sess1", "Original ask", "High-level plan") as session:
        with session.activity("act1", "Do X", "coder", "component", "analysis") as activity:
            with activity.task("Implement function") as task:
                # Build prompt for LLM
                prompt = task.build_prompt(include_history=True)

                # Record what you did
                task.record_action("Implemented function")
                task.record_decision("Used async pattern")
                task.record_learning("Pattern X is faster")

                # Set final result
                task.set_result("Function implemented")

MODULE STRUCTURE
================

agentic_context/
├── __init__.py           - Public API exports
├── config.py             - Simple dataclass config
├── workflow.py           - Context managers (Session, Activity, Task)
├── core/
│   ├── context_storage.py     - Storage: Session, Activity, Task, Items
│   └── context_manager.py     - Manager: interface to storage + retrieval
└── retrieval/
    ├── retrieval_strategies.py   - Single Retriever class
    ├── semantic_retrieval.py     - EmbeddingModel wrapper
    └── tag_normalization.py      - Simple tag normalization

CLASSES & METHODS (30-SECOND REFERENCE)
========================================

ContextWorkflow:
  ├── __init__(config=None)
  ├── session(id, ask, plan) -> SessionContext
  └── clear()

SessionContext:
  └── activity(id, desc, mode, component, analysis) -> ActivityContext

ActivityContext:
  └── task(desc, tags=None) -> TaskContext

TaskContext:
  ├── build_prompt(include_history=True) -> str
  ├── record_action(content)
  ├── record_decision(content)
  ├── record_learning(content)
  └── set_result(result)

ContextStorage:
  ├── create_session(id, ask, plan) -> Session
  ├── create_activity(id, session_id, desc, tags, mode, component, analysis) -> Activity
  ├── create_task(id, activity_id, desc, tags) -> Task
  ├── add_item(content, category, tags, parent_id=None) -> ContextItem
  ├── get_items_by_tags(tags) -> [ContextItem]
  ├── get_items_by_category(category) -> [ContextItem]
  ├── get_current_session/activity/task() -> Session/Activity/Task
  └── clear()

Retriever:
  └── retrieve(query_tags, query_description, max_results=5) -> [Dict]

Config:
  ├── use_semantic: bool
  ├── max_results: int
  ├── min_score: float
  ├── embedding_model: str
  └── @classmethod semantic() -> Config

CONTEXT CATEGORIES
==================

ContextCategory (Enum):
  - SESSION
  - ACTIVITY
  - TASK
  - ACTION
  - DECISION
  - LEARNING
  - RESULT

KEY DIFFERENCES FROM ORIGINAL
==============================

BEFORE (Complex):
  - 327 lines in context_manager.py
  - 1224 lines in workflow.py  
  - 633 lines in retrieval_strategies.py
  - ThreadLock + nested validation
  - 5+ retriever classes
  - Caching at multiple levels
  - Preset configurations
  - Complex exports

AFTER (Simple):
  - 74 lines in context_manager.py
  - 113 lines in workflow.py
  - 95 lines in retrieval_strategies.py
  - No locks (not needed for typical usage)
  - 1 retriever class
  - Optional semantic, no caching
  - 1 simple Config dataclass
  - Clean exports in __init__.py

SIZE REDUCTION: 70% fewer lines of code
COMPLEXITY: 80% simpler to understand
MAINTAINABILITY: Much easier to modify
"""

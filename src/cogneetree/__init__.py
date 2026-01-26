"""
cogneetree - Hierarchical context memory for AI agents

A lightweight, flexible library for managing hierarchical context in AI applications.
Supports Session → Activity → Task hierarchies with tag-based and semantic retrieval.
"""

__version__ = "0.1.0"

from cogneetree.workflow import ContextWorkflow
from cogneetree.config import Config
from cogneetree.core.context_manager import ContextManager
from cogneetree.core.context_storage import (
    ContextStorage,
    ContextItem,
    ContextCategory,
    Session,
    Activity,
    Task,
)

__all__ = [
    "ContextWorkflow",
    "Config",
    "ContextManager",
    "ContextStorage",
    "ContextItem",
    "ContextCategory",
    "Session",
    "Activity",
    "Task",
]

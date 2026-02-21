"""
cogneetree - Hierarchical context memory for AI agents

A lightweight, flexible library for managing hierarchical context in AI applications.
Supports Session → Activity → Task hierarchies with tag-based and semantic retrieval.
"""

__version__ = "0.1.0"

from cogneetree.workflow import ContextWorkflow
from cogneetree.config import Config
from cogneetree.core.context_manager import ContextManager
from cogneetree.core.interfaces import ContextStorageABC
from cogneetree.storage.in_memory_storage import InMemoryStorage
from cogneetree.core.models import (
    ContextItem,
    ContextCategory,
    DecisionEntry,
    ImportanceTier,
    Session,
    Activity,
    Task,
)
from cogneetree.retrieval.hierarchical_retriever import (
    HierarchicalRetriever,
    RetrievalConfig,
    HistoryMode,
    RETRIEVAL_PRESETS,
)
from cogneetree.agent_memory import AgentMemory, ContextResult, ConflictWarning, RecallResult

__all__ = [
    # Context management
    "ContextWorkflow",
    "Config",
    "ContextManager",
    # Storage
    "ContextStorageABC",
    "InMemoryStorage",
    # Data models
    "ContextItem",
    "ContextCategory",
    "DecisionEntry",
    "ImportanceTier",
    "Session",
    "Activity",
    "Task",
    # Retrieval (hierarchical)
    "HierarchicalRetriever",
    "RetrievalConfig",
    "HistoryMode",
    "RETRIEVAL_PRESETS",
    # Agent interface (recommended for agents)
    "AgentMemory",
    "ContextResult",
    "ConflictWarning",
    "RecallResult",
]

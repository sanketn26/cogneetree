"""Core data models for Cogneetree."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


@dataclass
class DecisionEntry:
    """A decision or learning that has propagated to Activity or Session level.

    The ``count`` field acts as an implicit importance signal: decisions
    mentioned multiple times across tasks are more significant.
    """

    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    count: int = 1


class ContextCategory(Enum):
    """Context item categories."""

    SESSION = "session"
    ACTIVITY = "activity"
    TASK = "task"
    ACTION = "action"
    DECISION = "decision"
    LEARNING = "learning"
    RESULT = "result"


class ImportanceTier(Enum):
    """Importance tier for a context item.

    Used as a score multiplier during retrieval so that critical knowledge
    surfaces before minor notes even when semantic similarity is similar.

    Multipliers applied in HierarchicalRetriever:
        CRITICAL → 2.0   (architectural decisions, blockers)
        MAJOR    → 1.5   (significant choices)
        MINOR    → 1.0   (default, routine notes)
        NOISE    → 0.1   (low-signal, can be suppressed)
    """

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    NOISE = "noise"


@dataclass
class ContextItem:
    """Single context item with metadata."""

    content: str
    category: ContextCategory
    tags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[Any] = None
    tier: "ImportanceTier" = field(default_factory=lambda: ImportanceTier.MINOR)


@dataclass
class Session:
    """Session context."""

    session_id: str
    original_ask: str
    high_level_plan: str
    created_at: datetime = field(default_factory=datetime.now)

    # Accumulated decisions/learnings propagated up from tasks via activities.
    # Keyed by primary tag for easy grouping (e.g. {"auth": [DecisionEntry, ...]}).
    decisions: Dict[str, List["DecisionEntry"]] = field(default_factory=dict)
    learnings: Dict[str, List["DecisionEntry"]] = field(default_factory=dict)


@dataclass
class Activity:
    """Activity context."""

    activity_id: str
    session_id: str
    description: str
    tags: List[str]
    mode: str
    component: str
    planner_analysis: str

    # Accumulated decisions/learnings propagated up from tasks.
    # Keyed by primary tag for easy grouping.
    decisions: Dict[str, List["DecisionEntry"]] = field(default_factory=dict)
    learnings: Dict[str, List["DecisionEntry"]] = field(default_factory=dict)


@dataclass
class Task:
    """Task context."""

    task_id: str
    activity_id: str
    description: str
    tags: List[str]
    result: Optional[str] = None

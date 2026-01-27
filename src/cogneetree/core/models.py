"""Core data models for Cogneetree."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class ContextCategory(Enum):
    """Context item categories."""

    SESSION = "session"
    ACTIVITY = "activity"
    TASK = "task"
    ACTION = "action"
    DECISION = "decision"
    LEARNING = "learning"
    RESULT = "result"


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


@dataclass
class Session:
    """Session context."""

    session_id: str
    original_ask: str
    high_level_plan: str
    created_at: datetime = field(default_factory=datetime.now)


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


@dataclass
class Task:
    """Task context."""

    task_id: str
    activity_id: str
    description: str
    tags: List[str]
    result: Optional[str] = None

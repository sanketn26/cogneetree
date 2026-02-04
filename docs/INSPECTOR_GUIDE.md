# Cogneetree Inspector Implementation Guide

A lightweight, optional web UI for visualizing and manipulating Cogneetree memory state.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Core Components](#core-components)
6. [Backend API](#backend-api)
7. [Frontend UI](#frontend-ui)
8. [WebSocket Events](#websocket-events)
9. [State Operations](#state-operations)
10. [Implementation Details](#implementation-details)
11. [Security Considerations](#security-considerations)
12. [Extending the Inspector](#extending-the-inspector)

---

## Overview

### Purpose

The Inspector provides:

- **Visibility**: See accumulated decisions/learnings at each hierarchy level
- **Understanding**: View propagation paths and why items were filtered
- **Control**: Add, edit, remove, or promote items between levels
- **Debugging**: Trace context flow through the system

### When to Use

| Scenario | Use Inspector? |
|----------|---------------|
| Development/debugging | Yes - understand what's propagating |
| Correcting wrong context | Yes - remove or edit items |
| Manual promotion | Yes - force items that were filtered |
| Production monitoring | Optional - read-only mode recommended |
| Automated pipelines | No - use API directly |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (UI)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Tree View  │  │  Edit Modal │  │  Propagation Diagram    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Inspector Server                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  REST API   │  │  WebSocket  │  │  State Manager          │  │
│  │  /api/*     │  │  /ws        │  │  (reads/writes memory)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AgentMemory / Storage                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Session   │  │  Activity   │  │  Task / Permanent       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Optional**: Core library works without inspector
2. **Lightweight**: Minimal dependencies (FastAPI + basic HTML/JS)
3. **Non-invasive**: Read/write through existing AgentMemory API
4. **Real-time**: WebSocket for live updates
5. **Embeddable**: Can run in-process or standalone

---

## Installation

### As Optional Dependency

```bash
# Core only (no inspector)
pip install cogneetree

# With inspector
pip install cogneetree[inspector]
```

### Dependencies Added

```toml
# pyproject.toml
[project.optional-dependencies]
inspector = [
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "websockets>=11.0",
    "jinja2>=3.0.0",
]
```

### Verify Installation

```python
from cogneetree.inspector import StateInspector

# Raises ImportError with helpful message if dependencies missing
inspector = StateInspector(memory)
```

---

## Quick Start

### In-Process (Development)

```python
from cogneetree import AgentMemory
from cogneetree.inspector import serve_inspector

memory = AgentMemory(...)

# Non-blocking: starts server in background thread
serve_inspector(memory, port=8080)
# Browser opens to http://localhost:8080

# Continue using memory normally
memory.record_decision("Use JWT", tags=["auth"])
# UI updates in real-time via WebSocket
```

### Standalone (Production/Debugging)

```bash
# Inspect file-based storage
cogneetree inspect --storage ./cogneetree_data --port 8080

# Inspect SQLite storage
cogneetree inspect --storage sqlite:///memory.db --port 8080

# Connect to running agent via socket
cogneetree inspect --socket /tmp/cogneetree.sock --port 8080

# Read-only mode (no modifications allowed)
cogneetree inspect --storage ./data --readonly --port 8080
```

### Programmatic Control

```python
from cogneetree.inspector import StateInspector, InspectorServer

memory = AgentMemory(...)
inspector = StateInspector(memory)

# Get server instance for custom configuration
server = InspectorServer(
    inspector=inspector,
    host="0.0.0.0",
    port=8080,
    readonly=False,
    auth_token="secret123",  # Optional authentication
)

# Start with custom settings
server.start(open_browser=True)

# Later: stop server
server.stop()
```

---

## Core Components

### StateInspector

The core class that provides state inspection and manipulation.

```python
from cogneetree.inspector import StateInspector

class StateInspector:
    """Inspects and manipulates AgentMemory state."""

    def __init__(
        self,
        memory: AgentMemory,
        embedding_model: Optional[EmbeddingModelABC] = None,
    ):
        """
        Args:
            memory: AgentMemory instance to inspect
            embedding_model: For computing similarity scores (optional)
        """
        self.memory = memory
        self.embedding_model = embedding_model

    # ==================== View Operations ====================

    def snapshot(self) -> StateSnapshot:
        """Get complete state across all levels."""
        pass

    def view_task(self, task_id: Optional[str] = None) -> TaskState:
        """View task-level state."""
        pass

    def view_activity(self, activity_id: Optional[str] = None) -> ActivityState:
        """View activity-level state."""
        pass

    def view_session(self, session_id: Optional[str] = None) -> SessionState:
        """View session-level state."""
        pass

    def view_permanent(self) -> PermanentState:
        """View permanent memory state."""
        pass

    # ==================== Understand Operations ====================

    def explain(self, content: str) -> PropagationExplanation:
        """Explain why content was filtered or propagated."""
        pass

    def trace_task(self, task_id: str) -> List[PropagationTrace]:
        """Trace all decisions/learnings from a task."""
        pass

    def provenance(self, level: str) -> ProvenanceMap:
        """Show where each item at a level came from."""
        pass

    # ==================== Modify Operations ====================

    def add(
        self,
        content: str,
        category: str,  # "decision" or "learning"
        level: str,     # "task", "activity", "session", "permanent"
        tag: str,
        task_id: Optional[str] = None,
        activity_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Add item at specified level."""
        pass

    def edit(
        self,
        level: str,
        category: str,
        tag: str,
        index: int,
        new_content: str,
        session_id: Optional[str] = None,
        activity_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> None:
        """Edit item at specified level."""
        pass

    def remove(
        self,
        level: str,
        category: str,
        tag: str,
        index: int,
        session_id: Optional[str] = None,
        activity_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> None:
        """Remove item from specified level."""
        pass

    def remove_everywhere(self, content: str) -> RemovalReport:
        """Remove item from all levels where it exists."""
        pass

    def promote(
        self,
        content: str,
        from_level: str,
        to_level: str,
        tag: str,
    ) -> None:
        """Promote item to higher level (bypass gating)."""
        pass

    def clear_level(
        self,
        level: str,
        session_id: Optional[str] = None,
        activity_id: Optional[str] = None,
    ) -> None:
        """Clear all decisions/learnings at specified level."""
        pass

    # ==================== Export/Import ====================

    def export_state(self) -> Dict[str, Any]:
        """Export complete state as JSON-serializable dict."""
        pass

    def import_state(self, state: Dict[str, Any], merge: bool = False) -> None:
        """Import state from dict. If merge=False, replaces existing."""
        pass
```

### Data Models

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

class Level(Enum):
    TASK = "task"
    ACTIVITY = "activity"
    SESSION = "session"
    PERMANENT = "permanent"

class Category(Enum):
    DECISION = "decision"
    LEARNING = "learning"

@dataclass
class StateItem:
    """An item at any level."""
    content: str
    category: Category
    tag: str
    timestamp: datetime
    count: int = 1  # For deduplication tracking

    # Propagation metadata
    propagated_to: List[Level] = field(default_factory=list)
    filtered_at: Optional[Level] = None
    filter_reason: Optional[str] = None
    similarity_scores: Dict[Level, float] = field(default_factory=dict)

@dataclass
class TaskState:
    """State at task level."""
    task_id: str
    description: str
    tags: List[str]
    decisions: List[StateItem]
    learnings: List[StateItem]
    actions: List[str]
    results: List[str]

@dataclass
class ActivityState:
    """State at activity level."""
    activity_id: str
    session_id: str
    description: str
    tags: List[str]
    decisions: Dict[str, List[StateItem]]  # grouped by tag
    learnings: Dict[str, List[StateItem]]
    tasks: List[TaskState]

@dataclass
class SessionState:
    """State at session level."""
    session_id: str
    original_ask: str
    high_level_plan: str
    decisions: Dict[str, List[StateItem]]
    learnings: Dict[str, List[StateItem]]
    activities: List[ActivityState]

@dataclass
class PermanentState:
    """State at permanent memory level."""
    decisions: Dict[str, List[StateItem]]
    learnings: Dict[str, List[StateItem]]

@dataclass
class StateSnapshot:
    """Complete state across all levels."""
    permanent: PermanentState
    sessions: List[SessionState]
    current_session_id: Optional[str]
    current_activity_id: Optional[str]
    current_task_id: Optional[str]

@dataclass
class PropagationStep:
    """One step in propagation path."""
    level: Level
    passed: bool
    similarity: Optional[float]
    threshold: Optional[float]
    reason: str

@dataclass
class PropagationExplanation:
    """Full explanation of item propagation."""
    content: str
    category: Category
    origin_task_id: str
    steps: List[PropagationStep]

@dataclass
class PropagationTrace:
    """Trace of an item from a task."""
    content: str
    category: Category
    path: List[PropagationStep]

@dataclass
class ProvenanceEntry:
    """Where an item came from."""
    content: str
    origin_task_id: str
    origin_activity_id: str
    timestamp: datetime

@dataclass
class ProvenanceMap:
    """Provenance for all items at a level."""
    level: Level
    decisions: Dict[str, List[ProvenanceEntry]]  # by tag
    learnings: Dict[str, List[ProvenanceEntry]]

@dataclass
class RemovalReport:
    """Report of what was removed."""
    content: str
    removed_from: List[Level]
    not_found_at: List[Level]
```

---

## Backend API

### REST Endpoints

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer
from pydantic import BaseModel

app = FastAPI(title="Cogneetree Inspector")

# ==================== Read Operations ====================

@app.get("/api/state")
async def get_full_state() -> StateSnapshot:
    """Get complete state snapshot."""
    return inspector.snapshot()

@app.get("/api/session/{session_id}")
async def get_session_state(session_id: str) -> SessionState:
    """Get session state."""
    return inspector.view_session(session_id)

@app.get("/api/activity/{activity_id}")
async def get_activity_state(activity_id: str) -> ActivityState:
    """Get activity state."""
    return inspector.view_activity(activity_id)

@app.get("/api/task/{task_id}")
async def get_task_state(task_id: str) -> TaskState:
    """Get task state."""
    return inspector.view_task(task_id)

@app.get("/api/permanent")
async def get_permanent_state() -> PermanentState:
    """Get permanent memory state."""
    return inspector.view_permanent()

@app.get("/api/explain")
async def explain_propagation(content: str) -> PropagationExplanation:
    """Explain why content was filtered or propagated."""
    return inspector.explain(content)

@app.get("/api/trace/{task_id}")
async def trace_task(task_id: str) -> List[PropagationTrace]:
    """Trace all items from a task."""
    return inspector.trace_task(task_id)

@app.get("/api/provenance/{level}")
async def get_provenance(level: str) -> ProvenanceMap:
    """Get provenance map for a level."""
    return inspector.provenance(level)

# ==================== Write Operations ====================

class AddItemRequest(BaseModel):
    content: str
    category: str  # "decision" or "learning"
    level: str     # "task", "activity", "session", "permanent"
    tag: str
    task_id: Optional[str] = None
    activity_id: Optional[str] = None
    session_id: Optional[str] = None

@app.post("/api/items")
async def add_item(request: AddItemRequest) -> dict:
    """Add item at specified level."""
    inspector.add(
        content=request.content,
        category=request.category,
        level=request.level,
        tag=request.tag,
        task_id=request.task_id,
        activity_id=request.activity_id,
        session_id=request.session_id,
    )
    return {"status": "added"}

class EditItemRequest(BaseModel):
    level: str
    category: str
    tag: str
    index: int
    new_content: str
    session_id: Optional[str] = None
    activity_id: Optional[str] = None
    task_id: Optional[str] = None

@app.put("/api/items")
async def edit_item(request: EditItemRequest) -> dict:
    """Edit item at specified level."""
    inspector.edit(
        level=request.level,
        category=request.category,
        tag=request.tag,
        index=request.index,
        new_content=request.new_content,
        session_id=request.session_id,
        activity_id=request.activity_id,
        task_id=request.task_id,
    )
    return {"status": "edited"}

class RemoveItemRequest(BaseModel):
    level: str
    category: str
    tag: str
    index: int
    session_id: Optional[str] = None
    activity_id: Optional[str] = None
    task_id: Optional[str] = None

@app.delete("/api/items")
async def remove_item(request: RemoveItemRequest) -> dict:
    """Remove item from specified level."""
    inspector.remove(
        level=request.level,
        category=request.category,
        tag=request.tag,
        index=request.index,
        session_id=request.session_id,
        activity_id=request.activity_id,
        task_id=request.task_id,
    )
    return {"status": "removed"}

@app.delete("/api/items/everywhere")
async def remove_everywhere(content: str) -> RemovalReport:
    """Remove item from all levels."""
    return inspector.remove_everywhere(content)

class PromoteRequest(BaseModel):
    content: str
    from_level: str
    to_level: str
    tag: str

@app.post("/api/promote")
async def promote_item(request: PromoteRequest) -> dict:
    """Promote item to higher level."""
    inspector.promote(
        content=request.content,
        from_level=request.from_level,
        to_level=request.to_level,
        tag=request.tag,
    )
    return {"status": "promoted"}

@app.delete("/api/level/{level}")
async def clear_level(
    level: str,
    session_id: Optional[str] = None,
    activity_id: Optional[str] = None,
) -> dict:
    """Clear all items at specified level."""
    inspector.clear_level(level, session_id, activity_id)
    return {"status": "cleared"}

# ==================== Export/Import ====================

@app.get("/api/export")
async def export_state() -> dict:
    """Export complete state as JSON."""
    return inspector.export_state()

class ImportRequest(BaseModel):
    state: dict
    merge: bool = False

@app.post("/api/import")
async def import_state(request: ImportRequest) -> dict:
    """Import state from JSON."""
    inspector.import_state(request.state, request.merge)
    return {"status": "imported"}
```

### Error Handling

```python
from fastapi import HTTPException

class InspectorError(Exception):
    """Base exception for inspector errors."""
    pass

class ItemNotFoundError(InspectorError):
    """Item not found at specified location."""
    pass

class ReadOnlyError(InspectorError):
    """Modification attempted in read-only mode."""
    pass

class InvalidLevelError(InspectorError):
    """Invalid level specified."""
    pass

@app.exception_handler(InspectorError)
async def inspector_error_handler(request, exc: InspectorError):
    status_codes = {
        ItemNotFoundError: 404,
        ReadOnlyError: 403,
        InvalidLevelError: 400,
    }
    return JSONResponse(
        status_code=status_codes.get(type(exc), 500),
        content={"error": str(exc)},
    )
```

---

## Frontend UI

### Technology Stack

- **HTML/CSS**: Vanilla, minimal dependencies
- **JavaScript**: Vanilla JS + htmx for reactivity
- **Icons**: Simple Unicode or inline SVG
- **No build step**: Served directly by FastAPI

### Page Structure

```html
<!DOCTYPE html>
<html>
<head>
    <title>Cogneetree Inspector</title>
    <script src="https://unpkg.com/htmx.org@1.9.0"></script>
    <style>
        /* Minimal CSS for tree view and modals */
    </style>
</head>
<body>
    <header>
        <h1>🌳 Cogneetree Inspector</h1>
        <button hx-get="/api/state" hx-target="#state-tree">⟳ Refresh</button>
        <span id="connection-status">● Connected</span>
    </header>

    <main id="state-tree">
        <!-- Populated by htmx -->
    </main>

    <div id="modal-container">
        <!-- Edit/Explain modals rendered here -->
    </div>

    <script>
        // WebSocket connection for live updates
        // Modal handlers
        // Tree expand/collapse
    </script>
</body>
</html>
```

### Component Templates

#### Tree Node (Jinja2)

```html
<!-- templates/tree_node.html -->
{% macro render_level(level, name, id, decisions, learnings, children=None, collapsed=False) %}
<div class="level level-{{ level }}" data-id="{{ id }}">
    <div class="level-header" onclick="toggleCollapse(this)">
        <span class="collapse-icon">{{ '▶' if collapsed else '▼' }}</span>
        <span class="level-name">{{ name }}</span>
        <span class="level-id">{{ id }}</span>
    </div>

    <div class="level-content" {% if collapsed %}style="display:none"{% endif %}>
        <div class="columns">
            <div class="column decisions">
                <h4>Decisions</h4>
                {% for tag, items in decisions.items() %}
                <div class="tag-group">
                    <span class="tag">📁 {{ tag }}</span>
                    {% for item in items %}
                    <div class="item" data-index="{{ loop.index0 }}">
                        <span class="content">{{ item.content }}</span>
                        <span class="propagation-indicator">
                            {% if item.propagated_to %}✓↑{% endif %}
                            {% if item.filtered_at %}⊘{% endif %}
                        </span>
                        <button onclick="editItem('{{ level }}', 'decision', '{{ tag }}', {{ loop.index0 }})">✏️</button>
                        <button onclick="removeItem('{{ level }}', 'decision', '{{ tag }}', {{ loop.index0 }})">🗑️</button>
                    </div>
                    {% endfor %}
                </div>
                {% endfor %}
                <button onclick="addItem('{{ level }}', 'decision')">+ Add Decision</button>
            </div>

            <div class="column learnings">
                <h4>Learnings</h4>
                {% for tag, items in learnings.items() %}
                <div class="tag-group">
                    <span class="tag">📁 {{ tag }}</span>
                    {% for item in items %}
                    <div class="item">
                        <span class="content">{{ item.content }}</span>
                        <button onclick="removeItem('{{ level }}', 'learning', '{{ tag }}', {{ loop.index0 }})">🗑️</button>
                    </div>
                    {% endfor %}
                </div>
                {% endfor %}
                <button onclick="addItem('{{ level }}', 'learning')">+ Add Learning</button>
            </div>
        </div>

        {% if children %}
        <div class="children">
            {% for child in children %}
                {{ render_level(child.level, child.name, child.id, child.decisions, child.learnings, child.children) }}
            {% endfor %}
        </div>
        {% endif %}
    </div>
</div>
{% endmacro %}
```

#### Edit Modal

```html
<!-- templates/edit_modal.html -->
<div class="modal" id="edit-modal">
    <div class="modal-content">
        <h3>Edit {{ category | title }}</h3>
        <form hx-put="/api/items" hx-target="#state-tree" hx-swap="innerHTML">
            <input type="hidden" name="level" value="{{ level }}">
            <input type="hidden" name="category" value="{{ category }}">
            <input type="hidden" name="tag" value="{{ tag }}">
            <input type="hidden" name="index" value="{{ index }}">

            <label>Content:</label>
            <textarea name="new_content" rows="3">{{ content }}</textarea>

            <div class="modal-actions">
                <button type="submit">Save</button>
                <button type="button" onclick="closeModal()">Cancel</button>
            </div>
        </form>
    </div>
</div>
```

#### Propagation Diagram

```html
<!-- templates/propagation_diagram.html -->
<div class="propagation-diagram">
    <h3>Propagation: "{{ content }}"</h3>

    <div class="path">
        {% for step in steps %}
        <div class="step {{ 'passed' if step.passed else 'blocked' }}">
            <span class="level-name">{{ step.level.value | title }}</span>
            {% if step.similarity is not none %}
            <span class="similarity">
                similarity = {{ "%.2f" | format(step.similarity) }}
                (threshold: {{ "%.2f" | format(step.threshold) }})
            </span>
            {% endif %}
            <span class="status">{{ '✓' if step.passed else '✗' }}</span>
            <span class="reason">{{ step.reason }}</span>
        </div>
        {% if not loop.last %}
        <div class="arrow">▼</div>
        {% endif %}
        {% endfor %}
    </div>

    {% if not steps[-1].passed %}
    <div class="actions">
        <button onclick="promote('{{ content }}', '{{ steps[-1].level.value }}', 'activity')">
            Force Promote to Activity
        </button>
        <button onclick="promote('{{ content }}', '{{ steps[-1].level.value }}', 'session')">
            Force Promote to Session
        </button>
    </div>
    {% endif %}

    <button onclick="closeModal()">Close</button>
</div>
```

### CSS Styles

```css
/* inspector.css */

:root {
    --bg-primary: #1a1a2e;
    --bg-secondary: #16213e;
    --bg-tertiary: #0f3460;
    --text-primary: #e8e8e8;
    --text-secondary: #a0a0a0;
    --accent: #e94560;
    --success: #4ade80;
    --warning: #fbbf24;
    --border: #333;
}

body {
    font-family: 'SF Mono', 'Consolas', monospace;
    background: var(--bg-primary);
    color: var(--text-primary);
    margin: 0;
    padding: 20px;
}

header {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
}

/* Tree Structure */
.level {
    border: 1px solid var(--border);
    border-radius: 4px;
    margin: 10px 0;
    background: var(--bg-secondary);
}

.level-permanent { border-left: 3px solid #a855f7; }
.level-session { border-left: 3px solid #3b82f6; }
.level-activity { border-left: 3px solid #22c55e; }
.level-task { border-left: 3px solid #eab308; }

.level-header {
    padding: 10px 15px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 10px;
    background: var(--bg-tertiary);
}

.level-header:hover {
    background: #1a4a7a;
}

.level-content {
    padding: 15px;
}

.children {
    margin-left: 20px;
    border-left: 2px dashed var(--border);
    padding-left: 15px;
}

/* Columns for decisions/learnings */
.columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}

.column h4 {
    margin: 0 0 10px 0;
    color: var(--text-secondary);
}

/* Tag groups */
.tag-group {
    margin-bottom: 15px;
}

.tag {
    display: inline-block;
    padding: 2px 8px;
    background: var(--bg-tertiary);
    border-radius: 3px;
    font-size: 12px;
    margin-bottom: 5px;
}

/* Items */
.item {
    display: flex;
    align-items: center;
    padding: 5px 10px;
    margin: 3px 0;
    background: var(--bg-primary);
    border-radius: 3px;
}

.item .content {
    flex: 1;
}

.item button {
    background: none;
    border: none;
    cursor: pointer;
    opacity: 0.5;
    padding: 2px 5px;
}

.item:hover button {
    opacity: 1;
}

/* Propagation indicators */
.propagation-indicator {
    margin-left: 10px;
    font-size: 12px;
}

.propagation-indicator .passed { color: var(--success); }
.propagation-indicator .blocked { color: var(--warning); }

/* Modal */
.modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
}

.modal-content {
    background: var(--bg-secondary);
    padding: 20px;
    border-radius: 8px;
    min-width: 400px;
    max-width: 600px;
}

/* Propagation diagram */
.propagation-diagram .path {
    padding: 20px;
}

.propagation-diagram .step {
    padding: 10px 15px;
    margin: 5px 0;
    border-radius: 4px;
    display: flex;
    align-items: center;
    gap: 15px;
}

.propagation-diagram .step.passed {
    background: rgba(74, 222, 128, 0.1);
    border: 1px solid var(--success);
}

.propagation-diagram .step.blocked {
    background: rgba(251, 191, 36, 0.1);
    border: 1px solid var(--warning);
}

.propagation-diagram .arrow {
    text-align: center;
    color: var(--text-secondary);
}

/* Buttons */
button {
    background: var(--accent);
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
}

button:hover {
    opacity: 0.9;
}

/* Connection status */
#connection-status {
    margin-left: auto;
}

#connection-status.connected { color: var(--success); }
#connection-status.disconnected { color: var(--accent); }
```

### JavaScript

```javascript
// inspector.js

// ==================== WebSocket Connection ====================

let ws = null;
let reconnectAttempts = 0;

function connectWebSocket() {
    ws = new WebSocket(`ws://${window.location.host}/ws`);

    ws.onopen = () => {
        document.getElementById('connection-status').textContent = '● Connected';
        document.getElementById('connection-status').className = 'connected';
        reconnectAttempts = 0;
    };

    ws.onclose = () => {
        document.getElementById('connection-status').textContent = '● Disconnected';
        document.getElementById('connection-status').className = 'disconnected';

        // Exponential backoff reconnect
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
        reconnectAttempts++;
        setTimeout(connectWebSocket, delay);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleStateUpdate(data);
    };
}

function handleStateUpdate(data) {
    switch (data.type) {
        case 'full_state':
            renderFullState(data.state);
            break;
        case 'item_added':
            addItemToUI(data.level, data.category, data.tag, data.item);
            break;
        case 'item_removed':
            removeItemFromUI(data.level, data.category, data.tag, data.index);
            break;
        case 'item_edited':
            editItemInUI(data.level, data.category, data.tag, data.index, data.new_content);
            break;
    }
}

// ==================== Tree Operations ====================

function toggleCollapse(header) {
    const content = header.nextElementSibling;
    const icon = header.querySelector('.collapse-icon');

    if (content.style.display === 'none') {
        content.style.display = 'block';
        icon.textContent = '▼';
    } else {
        content.style.display = 'none';
        icon.textContent = '▶';
    }
}

// ==================== CRUD Operations ====================

async function addItem(level, category) {
    const content = prompt(`Enter ${category}:`);
    if (!content) return;

    const tag = prompt('Enter tag:');
    if (!tag) return;

    await fetch('/api/items', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, category, level, tag })
    });
}

async function editItem(level, category, tag, index) {
    // Fetch current content
    const response = await fetch(`/api/state`);
    const state = await response.json();

    // Find the item
    const currentContent = findItem(state, level, category, tag, index);

    const newContent = prompt('Edit content:', currentContent);
    if (!newContent || newContent === currentContent) return;

    await fetch('/api/items', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ level, category, tag, index, new_content: newContent })
    });
}

async function removeItem(level, category, tag, index) {
    if (!confirm('Remove this item?')) return;

    await fetch('/api/items', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ level, category, tag, index })
    });
}

async function promote(content, fromLevel, toLevel) {
    const tag = prompt('Enter tag for promoted item:');
    if (!tag) return;

    await fetch('/api/promote', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, from_level: fromLevel, to_level: toLevel, tag })
    });
}

// ==================== Explain/Trace ====================

async function explainItem(content) {
    const response = await fetch(`/api/explain?content=${encodeURIComponent(content)}`);
    const explanation = await response.json();
    showPropagationDiagram(explanation);
}

function showPropagationDiagram(explanation) {
    const modal = document.getElementById('modal-container');
    modal.innerHTML = renderPropagationDiagram(explanation);
    modal.style.display = 'block';
}

// ==================== Modal ====================

function closeModal() {
    document.getElementById('modal-container').style.display = 'none';
}

// ==================== Initialize ====================

document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();

    // Initial state load
    htmx.ajax('GET', '/api/state', { target: '#state-tree' });
});
```

---

## WebSocket Events

### Server → Client Events

```python
from fastapi import WebSocket
from typing import Set
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        # Send initial state
        state = inspector.snapshot()
        await websocket.send_json({
            "type": "full_state",
            "state": state.dict()
        })

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                self.active_connections.discard(connection)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, handle client messages if needed
            data = await websocket.receive_text()
            # Handle client requests via WebSocket if needed
    except:
        manager.disconnect(websocket)
```

### Event Types

```python
# Item added
{
    "type": "item_added",
    "level": "session",
    "category": "decision",
    "tag": "auth",
    "item": {
        "content": "Use JWT with RS256",
        "timestamp": "2024-01-15T10:30:00Z",
        "count": 1
    }
}

# Item removed
{
    "type": "item_removed",
    "level": "session",
    "category": "decision",
    "tag": "auth",
    "index": 0
}

# Item edited
{
    "type": "item_edited",
    "level": "session",
    "category": "decision",
    "tag": "auth",
    "index": 0,
    "new_content": "Use JWT with RS256 (with HMAC fallback)"
}

# Full state refresh
{
    "type": "full_state",
    "state": { ... }
}
```

### Hooking into AgentMemory

```python
class InspectorHooks:
    """Hooks to broadcast state changes to inspector."""

    def __init__(self, manager: ConnectionManager):
        self.manager = manager

    async def on_decision_recorded(
        self,
        content: str,
        tags: List[str],
        propagated_to: List[Level],
    ):
        for level in propagated_to:
            await self.manager.broadcast({
                "type": "item_added",
                "level": level.value,
                "category": "decision",
                "tag": tags[0] if tags else "general",
                "item": {
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                    "count": 1
                }
            })

    async def on_learning_recorded(self, content: str, tags: List[str], propagated_to: List[Level]):
        # Similar to on_decision_recorded
        pass

# Integration with AgentMemory
memory = AgentMemory(...)
hooks = InspectorHooks(manager)
memory.add_hook("decision_recorded", hooks.on_decision_recorded)
memory.add_hook("learning_recorded", hooks.on_learning_recorded)
```

---

## State Operations

### Implementation Examples

#### snapshot()

```python
def snapshot(self) -> StateSnapshot:
    """Get complete state across all levels."""
    storage = self.memory.storage

    # Get all sessions
    sessions = []
    for session in storage.get_all_sessions():
        session_state = self._build_session_state(session)
        sessions.append(session_state)

    # Get permanent memory
    permanent = self._build_permanent_state()

    return StateSnapshot(
        permanent=permanent,
        sessions=sessions,
        current_session_id=storage.current_session_id,
        current_activity_id=storage.current_activity_id,
        current_task_id=storage.current_task_id,
    )

def _build_session_state(self, session: Session) -> SessionState:
    """Build session state with nested activities and tasks."""
    activities = []
    for activity in self._get_session_activities(session.session_id):
        activity_state = self._build_activity_state(activity)
        activities.append(activity_state)

    return SessionState(
        session_id=session.session_id,
        original_ask=session.original_ask,
        high_level_plan=session.high_level_plan,
        decisions=session.decisions,
        learnings=session.learnings,
        activities=activities,
    )
```

#### explain()

```python
def explain(self, content: str) -> PropagationExplanation:
    """Explain propagation path for a decision/learning."""

    # Find origin
    origin_task = self._find_origin_task(content)
    if not origin_task:
        raise ItemNotFoundError(f"Content not found: {content}")

    activity = self.memory.storage.get_activity(origin_task.activity_id)
    session = self.memory.storage.get_session(activity.session_id)

    steps = []

    # Step 1: Task (always stored)
    steps.append(PropagationStep(
        level=Level.TASK,
        passed=True,
        similarity=None,
        threshold=None,
        reason="Always stored at task level"
    ))

    # Step 2: Activity
    if self.embedding_model:
        activity_sim = self._compute_similarity(content, activity.description)
        activity_threshold = self.memory.activity_threshold
        activity_passed = activity_sim >= activity_threshold
    else:
        activity_sim = None
        activity_threshold = None
        activity_passed = content in [d.content for d in activity.decisions.get(origin_task.tags[0], [])]

    steps.append(PropagationStep(
        level=Level.ACTIVITY,
        passed=activity_passed,
        similarity=activity_sim,
        threshold=activity_threshold,
        reason="Passed similarity threshold" if activity_passed else f"Filtered: {activity_sim:.2f} < {activity_threshold}"
    ))

    if not activity_passed:
        return PropagationExplanation(
            content=content,
            category=Category.DECISION,  # or detect
            origin_task_id=origin_task.task_id,
            steps=steps
        )

    # Step 3: Session
    session_sim = self._compute_similarity(content, session.original_ask) if self.embedding_model else None
    session_threshold = self.memory.session_threshold
    session_passed = session_sim >= session_threshold if session_sim else True

    steps.append(PropagationStep(
        level=Level.SESSION,
        passed=session_passed,
        similarity=session_sim,
        threshold=session_threshold,
        reason="Passed similarity threshold" if session_passed else f"Filtered: {session_sim:.2f} < {session_threshold}"
    ))

    if not session_passed:
        return PropagationExplanation(
            content=content,
            category=Category.DECISION,
            origin_task_id=origin_task.task_id,
            steps=steps
        )

    # Step 4: Permanent
    is_novel = self._is_novel_pattern(content)
    steps.append(PropagationStep(
        level=Level.PERMANENT,
        passed=is_novel,
        similarity=None,
        threshold=None,
        reason="Novel pattern" if is_novel else "Duplicate of existing pattern"
    ))

    return PropagationExplanation(
        content=content,
        category=Category.DECISION,
        origin_task_id=origin_task.task_id,
        steps=steps
    )
```

#### promote()

```python
def promote(
    self,
    content: str,
    from_level: str,
    to_level: str,
    tag: str,
) -> None:
    """Promote item to higher level, bypassing semantic gating."""

    from_level_enum = Level(from_level)
    to_level_enum = Level(to_level)

    # Validate promotion direction
    level_order = [Level.TASK, Level.ACTIVITY, Level.SESSION, Level.PERMANENT]
    if level_order.index(to_level_enum) <= level_order.index(from_level_enum):
        raise InvalidLevelError("Can only promote to higher levels")

    # Find the item at from_level
    item = self._find_item_at_level(content, from_level_enum)
    if not item:
        raise ItemNotFoundError(f"Content not found at {from_level}: {content}")

    # Add to target level
    if to_level_enum == Level.ACTIVITY:
        activity = self.memory.storage.get_current_activity()
        if tag not in activity.decisions:
            activity.decisions[tag] = []
        activity.decisions[tag].append(DecisionEntry(
            content=content,
            timestamp=datetime.now(),
            count=1
        ))

    elif to_level_enum == Level.SESSION:
        session = self.memory.storage.get_current_session()
        if tag not in session.decisions:
            session.decisions[tag] = []
        session.decisions[tag].append(DecisionEntry(
            content=content,
            timestamp=datetime.now(),
            count=1
        ))

    elif to_level_enum == Level.PERMANENT:
        permanent = self.memory.permanent
        if tag not in permanent.decisions:
            permanent.decisions[tag] = []
        permanent.decisions[tag].append(DecisionEntry(
            content=content,
            timestamp=datetime.now(),
            count=1
        ))
```

---

## Security Considerations

### Authentication

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer(auto_error=False)

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> bool:
    if not inspector_config.auth_token:
        return True  # No auth required

    if not credentials:
        raise HTTPException(status_code=401, detail="Missing token")

    if credentials.credentials != inspector_config.auth_token:
        raise HTTPException(status_code=403, detail="Invalid token")

    return True

# Apply to all write endpoints
@app.post("/api/items", dependencies=[Depends(verify_token)])
async def add_item(request: AddItemRequest):
    ...
```

### Read-Only Mode

```python
class InspectorConfig:
    readonly: bool = False
    auth_token: Optional[str] = None

def require_write_access():
    if inspector_config.readonly:
        raise ReadOnlyError("Inspector is in read-only mode")

@app.post("/api/items")
async def add_item(request: AddItemRequest):
    require_write_access()
    ...
```

### CORS (for separate frontend)

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Dev frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Extending the Inspector

### Custom Visualizations

```python
from cogneetree.inspector import StateInspector, InspectorServer

class CustomInspector(StateInspector):
    """Extended inspector with custom features."""

    def get_decision_timeline(self) -> List[dict]:
        """Get all decisions in chronological order."""
        all_decisions = []
        for session in self.memory.storage.get_all_sessions():
            for tag, decisions in session.decisions.items():
                for d in decisions:
                    all_decisions.append({
                        "content": d.content,
                        "tag": tag,
                        "session": session.session_id,
                        "timestamp": d.timestamp
                    })
        return sorted(all_decisions, key=lambda x: x["timestamp"])

    def get_tag_statistics(self) -> Dict[str, int]:
        """Count decisions by tag across all levels."""
        stats = {}
        # ... implementation
        return stats

# Custom API endpoints
@app.get("/api/custom/timeline")
async def get_timeline():
    return inspector.get_decision_timeline()
```

### Plugin System

```python
class InspectorPlugin:
    """Base class for inspector plugins."""

    def register_routes(self, app: FastAPI) -> None:
        """Register custom API routes."""
        pass

    def register_templates(self) -> Dict[str, str]:
        """Return custom Jinja2 templates."""
        return {}

    def register_static(self) -> Dict[str, bytes]:
        """Return custom static files (CSS, JS)."""
        return {}

# Usage
server = InspectorServer(inspector)
server.register_plugin(MyCustomPlugin())
server.start()
```

---

## Summary

The Inspector provides:

1. **StateInspector** - Core class for state inspection and manipulation
2. **REST API** - Full CRUD operations on decisions/learnings
3. **WebSocket** - Real-time updates as memory changes
4. **Web UI** - Visual tree view with edit/delete/promote actions
5. **Propagation Diagrams** - Understand why items were filtered
6. **Export/Import** - Backup and restore state

### Quick Reference

```python
# Start inspector
from cogneetree import AgentMemory
from cogneetree.inspector import serve_inspector

memory = AgentMemory(...)
serve_inspector(memory, port=8080)

# CLI
cogneetree inspect --storage ./data --port 8080

# API
GET  /api/state              # Full snapshot
GET  /api/explain?content=X  # Propagation explanation
POST /api/items              # Add item
PUT  /api/items              # Edit item
DELETE /api/items            # Remove item
POST /api/promote            # Promote to higher level
```

---

## See Also

- [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - Core library implementation
- [AGENT_MEMORY.md](AGENT_MEMORY.md) - Agent API reference
- [CLAUDE.md](../CLAUDE.md) - Development guide

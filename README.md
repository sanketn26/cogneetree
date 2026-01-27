# 🌳 cogneetree

**Hierarchical context memory for AI agents**

A lightweight, flexible Python library for managing hierarchical context in AI applications. Build persistent memory across projects with Session → Activity → Task hierarchies, giving agents access to past decisions, learnings, and patterns.

## ✨ Key Features

- **Hierarchical Memory** - Session → Activity → Task structure mirrors how work actually happens
- **Multi-Scope Retrieval** - Micro (focused), Balanced (practical), Macro (learning) searches
- **Transparent Reasoning** - See why each piece of context was selected
- **Flexible Storage** - In-memory (default) plus SQLite, PostgreSQL, Redis, MongoDB
- **Tag-Based Organization** - Flexible categorization that grows with your work
- **Temporal Awareness** - Automatic timestamp tracking favors recent learnings
- **LLM-Ready** - Built-in prompt building with integrated context injection
- **Minimal Dependencies** - Core library has zero external dependencies

## 🚀 Quick Start

### Installation

```bash
pip install cogneetree
```

### Basic Usage

```python
from cogneetree import ContextManager, AgentMemory

# Initialize context storage
manager = ContextManager()

# Create a session (project)
manager.create_session("proj_1", "Build API auth", "JWT-based authentication")
manager.create_activity("a1", "proj_1", "Understand JWT", ["jwt", "auth"], "learner", "core", "research")
manager.create_task("t1", "a1", "Learn JWT structure", ["jwt"])

# Record what you learn and decide
manager.record_learning("JWT has three parts: header.payload.signature", ["jwt", "structure"])
manager.record_decision("Use HS256 for symmetric signing", ["jwt", "algorithm"])

# Later, get memory for your current task
manager.create_task("t2", "a1", "Implement JWT validation", ["jwt", "validation"])
memory = AgentMemory(manager.storage, current_task_id="t2")

# Query past knowledge - one line!
context = memory.recall("JWT validation")
for item in context:
    print(f"{item.category}: {item.content}")
```

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **[AGENT_MEMORY.md](AGENT_MEMORY.md)** | Guide for AI agents - API reference, scopes, integration examples |
| **[CLAUDE.md](CLAUDE.md)** | Development philosophy - architecture, design decisions, best practices |
| **[examples/](examples/)** | Real-world usage examples and integration patterns |

## 🏗️ How It Works

### Three-Level Hierarchy

Work is organized into three natural levels:

```
Session (Project)
  ├── Activity (Work Area)
  │   └── Task (Atomic Work Item)
  │       ├── Actions (what you did)
  │       ├── Decisions (why you chose that)
  │       ├── Learnings (what you discovered)
  │       └── Results (what was accomplished)
```

### Retrieval Scopes

When an agent needs context, it chooses a scope:

- **Micro**: This task only (focused work)
- **Balanced**: Current task + recent history (90 days) - **default**
- **Macro**: All projects equally (pattern learning)

### Why This Matters

Instead of agents starting fresh each project, they build persistent memory:

```python
# Project 1: Agent learns
manager.record_decision("Use HS256 for JWT", ["jwt"])

# Project 2: Agent finds this automatically
context = memory.recall("JWT signing")  # Finds HS256 decision!
```

Over time, agents develop genuine expertise that compounds across projects.

## 🎯 Use Cases

- **AI Coding Assistants** - Remember architecture decisions, patterns, and lessons
- **DevOps Automation** - Track incident responses and proven solutions
- **Customer Support** - Maintain context across conversations and cases
- **Research Tools** - Connect findings and build on past discoveries
- **Content Creation** - Learn from patterns in successful content

## 🔧 Storage Options

By default, Cogneetree uses in-memory storage. For persistence:

```python
from cogneetree import ContextManager
from cogneetree.storage.in_memory_storage import InMemoryStorage

# In-memory (default, per-session)
manager = ContextManager()

# For persistent storage, implement ContextStorageABC or extend InMemoryStorage
```

See [CLAUDE.md](CLAUDE.md#storage-backend-implementation) for custom storage implementation.

## 📦 What's Included

- **ContextManager** - Simple interface for creating and managing context
- **AgentMemory** - Frictionless memory access for AI agents (recommended)
- **HierarchicalRetriever** - Advanced retrieval with 4 history modes
- **InMemoryStorage** - Fast, default storage backend
- **RetrievalConfig** - Fine-grained control over search behavior

## 🤝 Contributing

Contributions welcome! Please see our development guide in [CLAUDE.md](CLAUDE.md#development-commands).

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

**Start with [AGENT_MEMORY.md](AGENT_MEMORY.md) if you're building agents. Start with [CLAUDE.md](CLAUDE.md) if you're extending the library.**

# 🌳 cogneetree

**Hierarchical context memory for AI agents**

A lightweight, flexible Python library for managing hierarchical context in AI applications. Build cognitive memory trees with Session → Activity → Task hierarchies, tag-based organization, and optional semantic retrieval.

## ✨ Features

- **Hierarchical Context** - Natural Session → Activity → Task structure
- **Flexible Storage** - In-memory (default), SQLite, PostgreSQL, Redis, MongoDB
- **Tag-Based Organization** - Organize context with flexible tagging
- **Semantic Retrieval** - Optional embedding-based context search
- **Temporal Tracking** - Built-in timestamp tracking
- **LLM-Ready** - Built-in prompt building with context injection
- **Simple & Clean** - 650 lines, 70% reduction from original design

## 🚀 Quick Start

```python
from cogneetree import ContextWorkflow, Config

# Create workflow
workflow = ContextWorkflow(config=Config.default())

# Build hierarchical context
with workflow.session("proj_1", "Build REST API", "Design → Code → Test") as session:
    with session.activity("auth", "Add authentication", "coder", "api", "JWT-based auth", tags=["auth", "security"]) as activity:
        with activity.task("implement_jwt", "JWT validation", tags=["jwt"]) as task:
            # Record context
            task.record_decision("Use RS256 for signing")
            task.record_learning("Short expiry + refresh token pattern")
            task.record_action("Created JWT middleware")
            
            # Build context-aware prompt
            prompt = task.build_prompt(include_history=True)
            
            # Mark complete
            task.set_result("JWT middleware implemented")
```

## 📦 Installation

```bash
# Basic installation
pip install cogneetree

# With semantic retrieval support
pip install cogneetree[semantic]

# With development dependencies
pip install cogneetree[dev]
```

## 🏗️ Architecture

```
Session (Top-level context)
  └── Activity (Mid-level work unit)
      └── Task (Atomic work item)
          ├── Actions (What was done)
          ├── Decisions (Why choices were made)
          ├── Learnings (What was discovered)
          └── Results (Outcomes)
```

## 🔧 Storage Backends

```python
from cogneetree import ContextWorkflow
from cogneetree.storage import SQLiteStorage, RedisStorage

# In-memory (default)
workflow = ContextWorkflow()

# SQLite persistence
workflow = ContextWorkflow(storage=SQLiteStorage(".cogneetree/memory.db"))

# Redis distributed
workflow = ContextWorkflow(storage=RedisStorage("redis://localhost:6379"))
```

## 📖 Documentation

- **[Quick Start Guide](docs/quickstart.md)** - Get started in 5 minutes
- **[API Reference](docs/api.md)** - Complete API documentation
- **[Storage Backends](docs/storage.md)** - Available storage options
- **[Examples](examples/)** - Real-world usage examples

## 🎯 Use Cases

- **AI Coding Assistants** - Track code generation context and decisions
- **Customer Support Bots** - Maintain conversation history and resolutions
- **Research Tools** - Organize findings and connect related work
- **DevOps Automation** - Track incident response and solutions
- **Content Creation** - Remember successful patterns and learnings

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Credits

Originally developed as part of the [Vivek](https://github.com/yourusername/vivek) AI coding assistant project, extracted for standalone use.

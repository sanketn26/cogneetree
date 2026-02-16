# Documentation Guide

Quick reference for finding what you need.

## I'm a User/Building Agents

**Start here:** [README.md](README.md)

Then go to: **[AGENT_MEMORY.md](AGENT_MEMORY.md)**

You'll learn:
- What AgentMemory is and why you need it
- How to initialize and use `AgentMemory` for your agents
- The three scopes: micro (focused), balanced (practical), macro (learning)
- How to integrate with Claude API or other LLM frameworks
- Best practices for recording and retrieving context

## I'm a Developer/Contributing to Cogneetree

**Start here:** [README.md](README.md) for overview

Then go to: **[CLAUDE.md](CLAUDE.md)**

You'll learn:
- Philosophy: how knowledge grows in branches
- Architecture: hierarchical retrieval with 4 modes and scoring
- Design decisions and rationale
- Development best practices (SOLID, DRY, clarity)
- How to extend with custom storage backends
- Development commands and testing strategy

## I Need Specific Information

| Question | Document | Section |
|----------|----------|---------|
| "What is Cogneetree?" | README | Overview |
| "How do I install it?" | README | Installation |
| "How do I use AgentMemory?" | AGENT_MEMORY | Core API |
| "What scopes are available?" | AGENT_MEMORY | Scopes Explained |
| "How do I integrate with Claude?" | AGENT_MEMORY | Integration |
| "How does the hierarchy work?" | CLAUDE | Architecture |
| "What's the scoring formula?" | CLAUDE | Scoring Formula |
| "How do I add custom storage?" | CLAUDE | Storage Backend |
| "How do I run tests?" | CLAUDE | Development Commands |
| "Why design it this way?" | CLAUDE | Design Decisions |

## Document Map

```
README.md (Entry Point)
├─ User/Agent Developer
│  └─ AGENT_MEMORY.md (API Reference)
│     └─ Integration Examples
├─ Library Developer
│  └─ CLAUDE.md (Architecture)
│     ├─ Philosophy
│     ├─ Design Decisions
│     ├─ Best Practices
│     └─ Development Guide
└─ Contributes Back?
   └─ CLAUDE.md (for extending)
```

## Quick Examples

### Quick Start (Agent Developer)

```python
from cogneetree import ContextManager, AgentMemory

manager = ContextManager()
manager.create_session("proj", "Build auth", "JWT-based")
manager.create_activity("a1", "proj", "Token validation", ["jwt"], "coder", "core", "design")
manager.create_task("t1", "a1", "Add middleware", ["jwt"])

memory = AgentMemory(manager.storage, current_task_id="t1")
context = memory.recall("JWT validation")  # One-line access to history!
```

See [AGENT_MEMORY.md](AGENT_MEMORY.md) for more.

### Quick Start (Library Developer)

Understanding the hierarchy:

```
Session (Project scope)
  ├── Activity (Work area)
  │   └── Task (Atomic work)
  │       ├── Action: "What I did"
  │       ├── Decision: "Why I chose that"
  │       ├── Learning: "What I discovered"
  │       └── Result: "What was accomplished"
```

Agents search with scope control:
- **Micro**: This task only
- **Balanced**: Current + recent history (default)
- **Macro**: All projects equally

See [CLAUDE.md](CLAUDE.md) for deep dive.

## File Structure

```
cogneetree/
├── README.md              ← Start here (everyone)
├── DOCUMENTATION.md       ← You are here
├── AGENT_MEMORY.md        ← Agent developers → comprehensive API guide
├── CLAUDE.md              ← Library developers → architecture guide
├── src/cogneetree/        ← Source code
│   ├── agent_memory.py    ← Frictionless memory interface
│   ├── core/              ← Storage, models, interfaces
│   ├── retrieval/         ← Hierarchical retrieval system
│   └── config.py          ← Configuration
├── tests/                 ← Test suite
│   ├── test_agent_memory.py        ← AgentMemory tests (21 tests)
│   ├── test_hierarchical_retriever.py ← Retriever tests (10 tests)
│   └── test_agentic_context_manager.py ← Manager tests (21 tests)
└── examples/              ← Real-world usage examples
    └── agent_usage.py     ← Complete agent workflow
```

## Documentation Philosophy

Each document serves its audience:

- **README**: "What is this? Does it solve my problem? How do I get started?"
- **AGENT_MEMORY**: "How do I use this for my agents? Show me examples. What are best practices?"
- **CLAUDE**: "How does it work? How do I extend it? What were the design choices?"

No redundancy. Each doc references the others. Users find what they need quickly.

## Implementation Status

Not everything documented is fully implemented yet. This transparency helps you know what to rely on today vs what's coming.

### Fully Implemented

| Feature | Status | Notes |
|---------|--------|-------|
| Three-level hierarchy (Session/Activity/Task) | Shipped | Core architecture, fully tested |
| Four retrieval modes (micro/balanced/macro/selected) | Shipped | All 4 modes working |
| Scoring: semantic x proximity x temporal | Shipped | Configurable weights |
| AgentMemory interface (recall, decisions, learnings, build_context) | Shipped | 21 tests passing |
| InMemoryStorage | Shipped | Default backend, complete |
| JsonFileStorage | Shipped | Extends InMemoryStorage with JSON persistence |
| Tag normalization | Shipped | With synonym mapping |
| ContextWorkflow (context managers) | Shipped | 22 tests passing |
| Retrieval presets (fresh/balanced/learn_from_all/debug) | Shipped | Ready to use |

### Partially Implemented

| Feature | Status | Notes |
|---------|--------|-------|
| SQLiteStorage | Tables created, methods partial | Not production-ready |
| RedisStorage | Basic structure, incomplete | Not production-ready |
| Semantic retrieval | Working but requires optional deps | `sentence-transformers` or `openai` needed |

### Documented but Not Yet Implemented

| Feature | Documented In | Notes |
|---------|---------------|-------|
| Cascading context propagation (decisions bubble up) | IMPLEMENTATION_GUIDE | Designed, not in code yet |
| RecallResult with SessionContext separation | IMPLEMENTATION_GUIDE | recall() returns flat list today |
| Permanent Memory (cross-session layer) | IMPLEMENTATION_GUIDE | No persistent cross-session layer |
| Semantic gating for propagation | IMPLEMENTATION_GUIDE | Depends on cascading propagation |
| DecisionEntry deduplication by word overlap | IMPLEMENTATION_GUIDE | Designed, not coded |
| Inspector web UI | INSPECTOR_GUIDE | Full design spec, not implemented |

### Known Gaps (Not Yet Documented Elsewhere)

These are architectural gaps that affect the "optimal token usage" goal:

1. **No token budget management** - `build_context(max_items=N)` limits by count, not tokens. No way to say "give me context that fits in 2000 tokens."
2. **No memory consolidation/forgetting** - Memory grows unbounded. Old, low-value items never expire or get summarized.
3. **No conflict detection** - Contradictory decisions across sessions aren't flagged (e.g., "Use HS256" vs "Use RS256").
4. **No importance tiers** - All items treated equally. No way to mark critical vs minor knowledge.
5. **No async API** - Blocking calls only. Multi-agent concurrent systems need async support.
6. **No MCP server integration** - Not yet exposed as Model Context Protocol tools.

See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md#roadmap-features-detailed-implementation) for the detailed roadmap.

---

## Need Help?

- Check the relevant document above
- Look at examples in [examples/](examples/)
- Read the tests in [tests/](tests/) - they document usage
- Check the code comments - they explain the "why"

## See Also

- [GitHub Issues](https://github.com/yourusername/cogneetree/issues) - Report bugs
- [Contributing Guide](CONTRIBUTING.md) - Want to contribute?

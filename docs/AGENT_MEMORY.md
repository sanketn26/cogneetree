# AgentMemory - Frictionless Context Access for AI Agents

## The Problem

AI agents working on multiple projects typically:
- Forget previous decisions and repeat mistakes
- Start each project from scratch despite accumulated knowledge
- Can't learn patterns from their own work history
- Have no transparent access to their reasoning

## The Solution

`AgentMemory` gives agents **one-line access** to relevant historical context. No configuration. No thinking about retrieval modes. Just:

```python
from cogneetree import AgentMemory

memory = AgentMemory(storage, current_task_id="t4")
context = memory.recall("What should I do next?")
```

That's it. The agent now has access to:
- Past decisions and their reasoning
- Learnings discovered across all projects
- Similar tasks and their outcomes
- Relevant patterns from all work

## Quick Start

```python
from cogneetree import ContextManager, AgentMemory

# Setup context
manager = ContextManager()
manager.create_session("project_auth", "Build authentication", "JWT-based auth")
manager.create_activity("a1", "project_auth", "Token validation", ["jwt"], "coder", "core", "analysis")
manager.create_task("t1", "a1", "Add JWT middleware", ["jwt"])

# Get memory interface for current task
memory = AgentMemory(manager.storage, current_task_id="t1")

# One-line recall - agent gets relevant history with explanations
context = memory.recall("JWT validation")

for item in context:
    print(f"Decision: {item.content}")
    print(f"From: {item.source}")
    print(f"Confidence: {item.confidence:.1%}\n")
```

## Core API

### `recall(query, scope="balanced")`

Get relevant context. Returns list of `ContextResult` objects with content, category, source, and confidence.

```python
# Balanced (default): current task + recent history (90 days)
context = memory.recall("JWT validation")

# Focused: just this task (avoid distractions)
context = memory.recall("JWT", scope="micro")

# Learn: all projects equally (pattern discovery)
context = memory.recall("JWT", scope="macro")
```

**Returns**: List of `ContextResult` objects sorted by relevance.

### `decisions(query, scope="balanced")`

Get only decisions (why choices were made).

```python
decisions = memory.decisions("JWT algorithm")
for d in decisions:
    print(f"Decision: {d.content}")
    print(f"From: {d.source}")
```

### `learnings(query, scope="balanced")`

Get only learnings (what was discovered).

```python
learnings = memory.learnings("token validation")
for l in learnings:
    print(f"Learned: {l.content}")
```

### `build_context(question, scope="balanced", max_items=3)`

Build a markdown block ready for LLM prompts.

```python
context_block = memory.build_context(
    "How should I validate JWT tokens?",
    max_items=5
)

# Use in agent prompt
prompt = f"""
Implement JWT validation.

{context_block}

Based on our past experience, what approach should we use?
"""
```

### `similar_tasks(topic)`

Find similar past work to learn from.

```python
similar = memory.similar_tasks("authentication")
for task in similar:
    print(f"Task: {task['description']}")
    print(f"Items recorded: {len(task['items'])}")
```

### `cite(result)`

Format a proper citation for agent output.

```python
results = memory.recall("JWT")
if results:
    citation = memory.cite(results[0])
    print(f"As we learned {citation}")  # Output: As we learned [From Project X, decision]
```

## Scopes Explained

### Micro (Fresh)

**Use when:** Agent needs focused, current-task context only

```python
results = memory.recall("JWT", scope="micro")
# Returns: Items from THIS task only
```

**Best for:** Focused work on specific problem, avoiding context noise

### Balanced (Default)

**Use when:** Normal agent work - stay focused on current task with historical insights

```python
results = memory.recall("JWT")  # or scope="balanced"
# Returns: Current task prioritized + last 90 days of history
```

**Best for:** Regular development work (most common case)

### Macro (Learn from All)

**Use when:** Agent is discovering patterns across all projects

```python
results = memory.recall("JWT", scope="macro")
# Returns: All sessions weighted equally
```

**Best for:** Pattern discovery, "What have I learned about X across all projects?"

For detailed scope behavior and scoring formula, see [CLAUDE.md](CLAUDE.md#retrieval-modes-choosing-your-search-scope).

## Understanding Results

Each `ContextResult` has:

```python
result.content       # The actual context (e.g., "Use HS256 for signing")
result.category      # "decision", "learning", "action", or "result"
result.source        # Human readable (e.g., "Project OAuth2, Task: Learn JWT")
result.confidence    # 0.0-1.0 relevance score
result.explanation   # Why this result was selected
```

## Integration with LLM Agents

### Using with Claude

```python
import anthropic
from cogneetree import ContextManager, AgentMemory

client = anthropic.Anthropic()
manager = ContextManager()
memory = AgentMemory(manager.storage, "current_task_id")

# Build context from history
context = memory.build_context("What should I implement?", max_items=3)

# Create prompt with history
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": f"""
You are implementing authentication middleware.

{context}

Based on our past experience, what's the best approach?
"""
        }
    ]
)

# Record what agent decided
response = message.content[0].text
manager.record_decision(response, ["jwt", "middleware"])
```

### Using with Anthropic SDK as Tool

```python
from anthropic import Anthropic

# Create AgentMemory as a tool
def recall_context(query: str, scope: str = "balanced"):
    return memory.recall(query, scope=scope)

# Agent can call it as needed during reasoning
```

### Using with LangChain

```python
from langchain.tools import tool
from cogneetree import AgentMemory

@tool
def recall_memory(query: str, scope: str = "balanced"):
    """Recall relevant context from past projects."""
    return memory.recall(query, scope=scope)

# Add to agent's toolkit
toolkit.append(recall_memory)
```

## Real Agent Workflow Example

```python
from cogneetree import ContextManager, AgentMemory

# Setup
manager = ContextManager()
manager.create_session("microservices", "Build distributed auth", "JWT across services")
manager.create_activity("a1", "microservices", "Inter-service auth", ["jwt"], "coder", "core", "design")
manager.create_task("t1", "a1", "Add JWT validation", ["jwt"])

memory = AgentMemory(manager.storage, "t1")

# ==================== Agent's Reasoning ====================
print("=== What do I know about JWT? ===")
context = memory.recall("JWT validation")
for item in context:
    print(f"  {item.category}: {item.content}")

# ==================== Agent Builds Informed Prompt ====================
print("\n=== Preparing prompt with history ===")
prompt = f"""
Implement inter-service JWT validation for microservices.

{memory.build_context("JWT validation in microservices", max_items=3)}

Recommend an approach based on our experience.
"""

# ==================== Agent Records Decisions ====================
print("\n=== Recording decisions ===")
manager.record_decision(
    "Use RS256 for asymmetric signing (public key distribution)",
    ["jwt", "signing"]
)
manager.record_action(
    "Added validation middleware before service dispatch",
    ["jwt", "middleware"]
)
manager.record_learning(
    "RS256 requires managing public key distribution",
    ["jwt", "key-management"]
)

print("✅ Decisions recorded for future agents")
```

## Performance Notes

- **Micro scope**: O(1) - just current task items
- **Balanced scope**: O(n) where n = items in last 90 days
- **Macro scope**: O(N) where N = all items in storage

**Recommendation:** Use balanced scope (default) for most work. Use micro for focused tasks. Use macro when explicitly learning patterns.

## Best Practices for Agents

1. **Start with balanced scope** - It's optimized for most work
2. **Use micro when focused** - On a specific, narrow problem
3. **Use macro for learning** - When discovering patterns
4. **Record consistently** - Future agents benefit from clear, complete records
5. **Tag thoughtfully** - Better tags = better recall
6. **Always cite sources** - Helps with transparency and debugging
7. **Review results before acting** - Check confidence scores

## Confidence Scores

Each result includes a confidence score (0.0 to 1.0):

```
confidence = semantic_similarity × proximity_weight × temporal_weight
```

- **semantic_similarity (0-1)**: How relevant is the content to your query?
- **proximity_weight**: How close in the hierarchy?
  - Current task: 1.0
  - Sibling task (same activity): 0.7
  - Same activity: 0.5
  - Same session: 0.3
  - Other sessions: 0.2
- **temporal_weight**: How recent? (newer items weighted higher)

Higher scores = more relevant results. Filter by confidence if needed:

```python
results = memory.recall("JWT")
relevant = [r for r in results if r.confidence > 0.7]
```

## Testing

```bash
pytest tests/test_agent_memory.py -v
```

All 21 tests validate:
- ✅ Recall across different scopes
- ✅ Filtering by category
- ✅ Prompt building
- ✅ Citation formatting
- ✅ Real agent workflows

## See Also

- [README.md](README.md) - Overview and features
- [DOCUMENTATION.md](DOCUMENTATION.md) - Navigation guide for all docs
- [CLAUDE.md](CLAUDE.md) - Architecture, philosophy, and development guide
- [examples/](examples/) - Real-world integration examples

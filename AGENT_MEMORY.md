# AgentMemory - Frictionless Context Access for AI Agents

## The Problem

When an agent works on multiple projects, it forgets previous decisions and repeats mistakes. Each project starts from scratch despite accumulated domain knowledge.

## The Solution

`AgentMemory` provides a **one-line interface** for agents to access relevant historical context during their work. No complex retrieval configuration. No thinking about implementation details.

## Quick Start

```python
from cogneetree import ContextManager, AgentMemory

# Initialize context
manager = ContextManager()
manager.create_session("project_auth", "Build authentication", "JWT-based auth")
manager.create_activity("a1", "project_auth", "Token validation", ["jwt"], "coder", "core", "analysis")
manager.create_task("t1", "a1", "Add JWT middleware", ["jwt"])

# Get memory for current task
memory = AgentMemory(manager.storage, current_task_id="t1")

# One-line recall - agent gets relevant history
context = memory.recall("JWT validation")

for item in context:
    print(f"Decision: {item.content}")
    print(f"From: {item.source}")
    print(f"Confidence: {item.confidence:.1%}")
```

## Core Interface

### `recall(query, scope="balanced")`
Get relevant context across projects with automatic ranking.

```python
# Default balanced scope (current + recent history)
results = memory.recall("JWT validation")

# Focused on just current task
results = memory.recall("JWT", scope="micro")

# Learn patterns across ALL projects
results = memory.recall("JWT", scope="macro")

# Equal weight all sessions (for pattern discovery)
results = memory.recall("JWT", scope="learn_from_all")
```

### `decisions(query)`
Get only decision items (what choices were made and why).

```python
decisions = memory.decisions("authentication")
for d in decisions:
    print(f"Decision: {d.content}")
    print(f"From: {d.source}")
```

### `learnings(query)`
Get only learning items (what was discovered).

```python
learnings = memory.learnings("JWT validation")
for l in learnings:
    print(f"Learned: {l.content}")
```

### `build_context(question, scope="balanced", max_items=3)`
Build a markdown block to include in agent prompts.

```python
context_block = memory.build_context(
    "How should I validate JWT tokens?",
    max_items=3
)

# Send to LLM
prompt = f"""
You are implementing JWT middleware.

{context_block}

What's the best approach?
"""
```

### `similar_tasks(topic)`
Find similar past work to learn from.

```python
similar = memory.similar_tasks("auth")
for task in similar:
    print(f"Task: {task['description']}")
    print(f"  Items: {len(task['items'])}")
```

### `cite(result)`
Format a proper citation for agent output.

```python
results = memory.recall("JWT")
if results:
    citation = memory.cite(results[0])
    agent_output = f"As we learned previously {citation}"
```

## Scopes Explained

### Micro (Fresh)
**Use when**: Agent needs focused, current-task context only
```python
results = memory.recall("JWT", scope="micro")
# Returns: Items from THIS task only
```

### Balanced (Default)
**Use when**: Normal work - apply current context with historical insights
```python
results = memory.recall("JWT")  # or scope="balanced"
# Returns: Current session prioritized + recent history (90 days)
# Best for: Regular development work
```

### Macro (Learn from All)
**Use when**: Agent is learning patterns across all projects
```python
results = memory.recall("JWT", scope="macro")
# Returns: All sessions weighted equally
# Best for: Pattern discovery, cross-project learning
```

## Real Agent Workflow

```python
from cogneetree import ContextManager, AgentMemory

# Setup
manager = ContextManager()
manager.create_session("microservices", "Build auth", "JWT across services")
manager.create_activity("a1", "microservices", "Token validation", ["jwt"], "coder", "core", "analysis")
manager.create_task("t1", "a1", "Inter-service JWT", ["jwt"])

memory = AgentMemory(manager.storage, "t1")

# Agent's reasoning:
print("=== What do I know about JWT? ===")
context = memory.recall("JWT")
for item in context:
    print(f"  {item.category}: {item.content}")

# Agent builds informed prompt
print("\n=== Building prompt with history ===")
prompt = f"""
Implement inter-service JWT validation.

{memory.build_context("JWT validation in microservices")}

Based on our experience, recommend an approach.
"""

# Agent records its work
print("\n=== Recording decisions ===")
manager.record_decision(
    "Use HS256 for symmetric validation (matches past decision)",
    ["jwt", "signing"]
)
manager.record_action(
    "Added validation middleware before service dispatch",
    ["jwt", "middleware"]
)

print("\n✅ Decisions recorded for future reference")
```

## Integration with LLM Agents

### Using with Claude API
```python
import anthropic
from cogneetree import ContextManager, AgentMemory

client = anthropic.Anthropic()
manager = ContextManager()
memory = AgentMemory(manager.storage, "current_task_id")

# Build context
context = memory.build_context("What should I implement?")

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

What's the best approach?
"""
        }
    ]
)

# Agent's response incorporates history
response = message.content[0].text

# Record what agent decided
manager.record_decision(response, ["jwt", "decision"])
```

### Using with LangChain
```python
from langchain.llms import ChatOpenAI
from cogneetree import AgentMemory

llm = ChatOpenAI()
memory = AgentMemory(storage, "task_id")

# Memory as a tool
context_tool = {
    "type": "function",
    "function": {
        "name": "recall_context",
        "description": "Recall relevant historical context",
        "execute": lambda query: memory.recall(query)
    }
}

# Agent can call memory as needed
```

## Scopes and Scoring

Every result includes a confidence score (0.0 to 1.0):

```
confidence = semantic_similarity × proximity_weight × temporal_weight
```

Where:
- **semantic_similarity**: How relevant is the content (0-1)
- **proximity_weight**: How close in the hierarchy (1.0 = current task, 0.2 = other sessions)
- **temporal_weight**: How recent (decay over time)

## Performance Notes

- **Micro scope**: O(1) - just current task
- **Balanced scope**: O(n) where n = items in last 90 days
- **Macro scope**: O(N) where N = all items (most complete)

For 10K+ items, use micro or balanced scope by default.

## Understanding Results

Each result has:
```python
result.content       # The actual context (decision/learning/action)
result.category      # "decision", "learning", "action", "result"
result.source        # Human readable: "Project X, Task Y"
result.confidence    # 0.0-1.0 relevance score
result.explanation   # Why it was selected (if returned)
```

## Best Practices

1. **Start with balanced scope** - it's optimized for most agent work
2. **Use micro for focused work** - when you need zero distraction
3. **Use macro for learning** - when discovering patterns
4. **Always cite sources** - helps with transparency
5. **Record decisions** - future agents benefit from your reasoning
6. **Tag consistently** - better recall with clear tags

## Testing

```bash
pytest tests/test_agent_memory.py -v
```

All 21 tests validate:
- Recall across different scopes
- Filtering by category (decisions vs learnings)
- Prompt building
- Citation formatting
- Real agent workflows

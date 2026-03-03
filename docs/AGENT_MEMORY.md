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

## Current Limitations

Be aware of these limitations when integrating:

1. **No token budget control** - `build_context(max_items=N)` limits by item count, not token count. For token-optimal prompts, you'll need to estimate tokens yourself:

   ```python
   # Workaround: estimate tokens and trim
   context = memory.recall("query", max_items=10)
   selected = []
   token_count = 0
   for item in context:
       # Rough estimate: 1 token ≈ 4 chars
       item_tokens = len(item.content) // 4
       if token_count + item_tokens > 2000:
           break
       selected.append(item)
       token_count += item_tokens
   ```

2. **No cross-session persistence by default** - InMemoryStorage loses data when the process ends. Use `JsonFileStorage` for persistence:

   ```python
   from cogneetree.storage.file_storage import JsonFileStorage
   storage = JsonFileStorage("./memory.json")
   manager = ContextManager(storage=storage)
   ```

3. **Memory grows unbounded** - No automatic expiration, summarization, or forgetting. For long-running agents, periodically review stored items.

4. **No conflict detection** - If different sessions record contradictory decisions, the system doesn't flag this. Agents should review confidence scores and sources carefully.

5. **Semantic retrieval requires optional dependencies** - Without `sentence-transformers` or `openai`, retrieval is tag-based only. Install for best results:

   ```bash
   pip install sentence-transformers  # Local embeddings
   # OR
   pip install openai  # OpenAI embeddings (requires API key)
   ```

6. **Not thread-safe** - Use separate `ContextManager` instances per thread. Shared storage across threads is not supported.

## Performance Notes

- **Micro scope**: O(1) - just current task items
- **Balanced scope**: O(n) where n = items in last 90 days
- **Macro scope**: O(N) where N = all items in storage

**Recommendation:** Use balanced scope (default) for most work. Use micro for focused tasks. Use macro when explicitly learning patterns.

## Token Optimization Strategies

Since the goal is to be as effective as LLMs while being optimal with token usage, follow these strategies:

### 1. Use the Right Scope

Scope directly controls how many items are searched and returned. Narrower scope = fewer tokens.

```python
# Tight focus (fewest tokens) - use when you know context is local
context = memory.recall("bug in auth", scope="micro")

# Normal work (moderate tokens) - default, good tradeoff
context = memory.recall("auth patterns", scope="balanced")

# Pattern discovery (most tokens) - use sparingly
context = memory.recall("auth patterns", scope="macro")
```

### 2. Limit Items Aggressively

```python
# Don't request more than you need
context = memory.build_context("question", max_items=3)  # Not 10
```

### 3. Filter by Category

If you only need decisions, don't pull learnings and actions too:

```python
decisions = memory.decisions("JWT signing")  # Only decisions
learnings = memory.learnings("JWT")          # Only learnings
```

### 4. Use Confidence Thresholds

Low-confidence results add tokens without adding value:

```python
results = memory.recall("JWT")
relevant = [r for r in results if r.confidence > 0.6]
```

### 5. Record Concisely

What you record today becomes tokens consumed tomorrow. Be concise:

```python
# Good: concise, searchable
manager.record_decision("Use RS256 for JWT signing - supports key rotation")

# Bad: verbose, wastes future tokens
manager.record_decision("After careful consideration of all the available signing algorithms including HS256, RS256, ES256, and PS256, we have decided to use RS256 because it supports key rotation which is important for our microservices architecture where we need to distribute public keys...")
```

### 6. Tag Precisely

Better tags = better retrieval = fewer irrelevant results = fewer wasted tokens:

```python
# Good: specific tags for precise retrieval
manager.record_learning("RS256 requires public key distribution", tags=["jwt", "key-management", "ops"])

# Bad: generic tags pull in noise
manager.record_learning("RS256 requires public key distribution", tags=["auth"])
```

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

---

## Permanent Memory — Implementation Plan

> **Status: Not yet started.**
>
> | Step | File | Status |
> |------|------|--------|
> | 1 | `src/cogneetree/core/models.py` | ⬜ Not started |
> | 2a | `src/cogneetree/storage/in_memory_storage.py` | ⬜ Not started |
> | 2b | `src/cogneetree/storage/sql_storage.py` | ⬜ Not started |
> | 2c | `src/cogneetree/storage/redis_storage.py` | ⬜ Not started |
> | 2d | `src/cogneetree/storage/async_postgres_storage.py` | ⬜ Not started |
> | 3 | `src/cogneetree/retrieval/hierarchical_retriever.py` | ⬜ Not started |
> | 4 | `src/cogneetree/core/context_manager.py` | ⬜ Not started |
> | 5 | `src/cogneetree/agent_memory.py` | ⬜ Not started |
> | 6 | `tests/test_permanent_memory.py` + contract test | ⬜ Not started |
>
> **→ Start at Step 1.**

### Problem

All existing memory is **session-scoped**. When a new session starts, past decisions and learnings are only accessible via an explicit `recall()` query. There is no concept of knowledge that *always* surfaces — architectural rules, project-wide constraints, or hard-won learnings that every future agent should be aware of without needing to ask.

Current limitation #2 in this file says:
> *"No cross-session persistence by default — InMemoryStorage loses data when the process ends."*

That addresses storage durability (solved by JsonFileStorage/Redis/SQL). Permanent memory is a separate, orthogonal problem: knowledge that transcends individual sessions and always appears in context, regardless of what the agent is querying.

### Design: The Knowledge Bank

The Knowledge Bank is a virtual permanent pool stored as normal `ContextItem` objects, but with a sentinel `parent_id = "__knowledge_bank__"` instead of a real task ID.

**Why this sentinel approach?**
- Zero schema changes — all existing storage backends already support arbitrary `parent_id` strings
- `get_items_by_task("__knowledge_bank__")` retrieves all permanent items using existing infrastructure
- `clear()` is modified to skip items with this parent_id, giving them lifetime beyond sessions

**Storage layout:**
```
Regular session items        parent_id = "t1", "t2", ...  ← cleared with clear()
Knowledge Bank items         parent_id = "__knowledge_bank__"  ← never cleared
```

### New Public API

```python
# 1. Directly write permanent knowledge (anytime, from any session)
manager.record_permanent(
    "Use RS256 for JWT signing — supports key rotation across services",
    tags=["jwt", "signing"],
    tier=ImportanceTier.CRITICAL,            # default
)

# 2. Finalize a session: promotes CRITICAL items + session decisions into the KB
promoted_count = manager.finalize_session("session_auth")

# 3. Query KB items directly
kb_items = memory.knowledge_bank()           # returns all permanent items

# 4. build_context() and build_prompt() automatically prepend a permanent section
context_block = memory.build_context("How should I handle tokens?")
# Output:
# === PERMANENT KNOWLEDGE ===
# [DECISION] Use RS256 for JWT signing — supports key rotation
#
# === RELEVANT HISTORY ===
# ...scored results...
```

### How `finalize_session()` Works

When called with a `session_id`, it promotes two classes of knowledge into the KB:

1. **Session-level accumulated decisions and learnings** (the `session.decisions` and `session.learnings` dicts that were propagated up from tasks). These become `ContextItem` objects with `tier=CRITICAL` (decisions) or `tier=MAJOR` (learnings).

2. **Any task-level `ContextItem` whose `tier == ImportanceTier.CRITICAL`** across all activities/tasks in the session.

Duplicates are avoided the same way as existing propagation: word-overlap Jaccard > 0.8 increments count rather than adding a new entry.

```python
promoted = manager.finalize_session("session_auth")
print(f"Promoted {promoted} items to permanent knowledge bank")
```

### How the Retriever Changes

The `HierarchicalRetriever` gains a `_gather_knowledge_bank()` method, called at the top of `retrieve()` before mode-specific gathering:

```python
# retrieve() — always include KB items regardless of history mode
kb_candidates = self._gather_knowledge_bank()
candidates.extend(kb_candidates)
# ... mode-specific gathering follows ...
```

`_gather_knowledge_bank()` returns all KB items with `proximity_weight = 1.0` (same as current-task items). The `CRITICAL` tier multiplier (2.0) then makes them naturally outrank non-CRITICAL candidates in scoring. They compete on semantic relevance like everything else — a permanent JWT decision won't surface when asking about CSS.

Additionally, `build_context()` / `build_prompt()` prepend a **dedicated permanent section** that always shows KB items, bypassing score filtering. This ensures architectural rules always appear in prompts even if they're not the closest semantic match.

### Implementation Steps (in order)

#### Step 1 — `src/cogneetree/core/models.py` ⬜
Add a module-level constant:
```python
KNOWLEDGE_BANK_ID: str = "__knowledge_bank__"
```

#### Step 2 — Storage backends: protect KB items from `clear()` ⬜

**`src/cogneetree/storage/in_memory_storage.py`** ⬜
In `clear()`, preserve items where `parent_id == KNOWLEDGE_BANK_ID`:
```python
def clear(self) -> None:
    self.sessions.clear()
    self.activities.clear()
    self.tasks.clear()
    self.items = [i for i in self.items if i.parent_id == KNOWLEDGE_BANK_ID]
    self.current_session_id = None
    self.current_activity_id = None
    self.current_task_id = None
```

**`src/cogneetree/storage/sql_storage.py`** (both `SQLiteStorage` and `PostgresStorage`) ⬜
Change the DELETE in `clear()`:
```python
# SQLite
self.conn.execute(
    "DELETE FROM context_items WHERE parent_id != ?", (KNOWLEDGE_BANK_ID,)
)

# Postgres (psycopg2)
c.execute("DELETE FROM context_items WHERE parent_id != %s", (KNOWLEDGE_BANK_ID,))
```

**`src/cogneetree/storage/redis_storage.py`** ⬜
In `clear()`, before deleting all keys, collect KB item keys first and restore them:
```python
def clear(self) -> None:
    # Preserve KB items
    kb_keys = [k for k in self.client.keys(self._key("item", "*"))
               if self.client.hget(k, "parent_id") == KNOWLEDGE_BANK_ID]
    # Delete everything
    keys = self.client.keys(f"{self.prefix}*")
    if keys:
        self.client.delete(*keys)
    # Restore KB items (re-write each hash)
    # ... re-store each kb_key hash back using hset ...
```
> Note: Redis `clear()` is tricky since we delete all keys by prefix. A cleaner approach is to check parent_id during deletion, or move KB items to a separate key prefix like `{prefix}kb:item:{id}`.

**`src/cogneetree/storage/async_postgres_storage.py`** ⬜
```python
async def clear(self) -> None:
    async with self._pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM context_items WHERE parent_id != $1", KNOWLEDGE_BANK_ID
        )
        await conn.execute("TRUNCATE sessions, activities, tasks")
    self.current_session_id = None
    self.current_activity_id = None
    self.current_task_id = None
```

#### Step 3 — `src/cogneetree/retrieval/hierarchical_retriever.py` ⬜

Add `_gather_knowledge_bank()`:
```python
def _gather_knowledge_bank(self) -> List[tuple]:
    """Always-included candidates from the permanent knowledge bank."""
    from cogneetree.core.models import KNOWLEDGE_BANK_ID
    kb_items = self.storage.get_items_by_task(KNOWLEDGE_BANK_ID)
    return [(item, KNOWLEDGE_BANK_ID, "Permanent Knowledge Bank", 1.0)
            for item in kb_items]
```

In `retrieve()`, inject before deduplication:
```python
# Gather mode-specific candidates
candidates = self._gather_by_mode(...)

# Always include KB candidates regardless of mode
candidates.extend(self._gather_knowledge_bank())

deduped_candidates = self._dedupe_candidates(candidates)
```

#### Step 4 — `src/cogneetree/core/context_manager.py` ⬜

Add two public methods:

```python
def record_permanent(
    self,
    content: str,
    tags: List[str],
    category: ContextCategory = ContextCategory.DECISION,
    tier: ImportanceTier = ImportanceTier.CRITICAL,
) -> ContextItem:
    """Write directly to the permanent knowledge bank. Survives session boundaries."""
    from cogneetree.core.models import KNOWLEDGE_BANK_ID
    return self.storage.add_item(
        content, category, tags, parent_id=KNOWLEDGE_BANK_ID, tier=tier
    )

def finalize_session(self, session_id: str) -> int:
    """Promote CRITICAL items and session-level decisions/learnings to the knowledge bank.

    Call this when a session is complete to preserve its key knowledge for all
    future sessions. Returns the count of items promoted.
    """
    from cogneetree.core.models import KNOWLEDGE_BANK_ID
    session = self.storage.get_session(session_id)
    if not session:
        return 0

    promoted = 0

    # Promote session-level accumulated decisions
    for tag, entries in session.decisions.items():
        for entry in entries:
            self.storage.add_item(
                entry.content, ContextCategory.DECISION, [tag],
                parent_id=KNOWLEDGE_BANK_ID, tier=ImportanceTier.CRITICAL,
            )
            promoted += 1

    # Promote session-level accumulated learnings
    for tag, entries in session.learnings.items():
        for entry in entries:
            self.storage.add_item(
                entry.content, ContextCategory.LEARNING, [tag],
                parent_id=KNOWLEDGE_BANK_ID, tier=ImportanceTier.MAJOR,
            )
            promoted += 1

    # Promote CRITICAL-tier task items from all tasks in the session
    for item in self.storage.get_items_by_session(session_id):
        if item.tier == ImportanceTier.CRITICAL:
            self.storage.add_item(
                item.content, item.category, item.tags,
                parent_id=KNOWLEDGE_BANK_ID, tier=ImportanceTier.CRITICAL,
            )
            promoted += 1

    return promoted
```

Update `build_prompt()` to prepend a permanent section:
```python
def build_prompt(self, include_history: bool = True) -> str:
    from cogneetree.core.models import KNOWLEDGE_BANK_ID
    kb_items = self.storage.get_items_by_task(KNOWLEDGE_BANK_ID)

    parts = []

    if kb_items:
        parts.append("=== PERMANENT KNOWLEDGE ===")
        for item in kb_items:
            parts.append(f"[{item.category.value.upper()}] {item.content}")
        parts.append("")

    # ... existing session / activity / task / history sections unchanged ...
```

#### Step 5 — `src/cogneetree/agent_memory.py` ⬜

Add a `knowledge_bank()` method to `AgentMemory`:
```python
def knowledge_bank(self) -> List[ContextItem]:
    """Return all items in the permanent knowledge bank."""
    from cogneetree.core.models import KNOWLEDGE_BANK_ID
    return self.storage.get_items_by_task(KNOWLEDGE_BANK_ID)
```

Update `build_context()` to prepend permanent items before scored results.

#### Step 6 — Tests: `tests/test_permanent_memory.py` ⬜

Write tests covering:
- `record_permanent()` → item has `parent_id == KNOWLEDGE_BANK_ID`
- `clear()` → KB items survive, session items do not
- `finalize_session()` → CRITICAL task items and session decisions appear in KB
- `finalize_session()` → duplicate promotion is idempotent (word-overlap check)
- Retriever → KB items always appear as candidates regardless of history mode
- `build_prompt()` → permanent section always present when KB is non-empty

Contract test addition in `tests/test_storage_contract.py`:
```python
async def test_clear_preserves_knowledge_bank(self, any_storage):
    from cogneetree.core.models import KNOWLEDGE_BANK_ID
    await any_storage.add_item("Permanent rule", ContextCategory.DECISION,
                               ["arch"], parent_id=KNOWLEDGE_BANK_ID)
    await any_storage.create_session("s1", "Ask", "Plan")
    await any_storage.add_item("Session item", ContextCategory.ACTION, ["tag"])

    await any_storage.clear()

    kb_items = await any_storage.get_items_by_task(KNOWLEDGE_BANK_ID)
    assert len(kb_items) == 1
    assert kb_items[0].content == "Permanent rule"
```

### Design Decisions & Trade-offs

| Decision | Rationale |
|----------|-----------|
| `parent_id = "__knowledge_bank__"` as sentinel | No schema changes; works with all backends via existing `get_items_by_task` |
| `proximity_weight = 1.0` for KB items | Lets tier multiplier (2.0 for CRITICAL) drive priority; KB items don't unconditionally dominate |
| Always-show section in `build_prompt()` | Permanent rules should appear even when not semantically closest match |
| `finalize_session()` is explicit, not automatic | Agent controls when a session is "done"; avoids premature promotion |
| Only `CRITICAL` task items promoted by `finalize_session()` | `MAJOR`/`MINOR` items stay in history, retrievable via `ALL_SESSIONS` — KB is reserved for truly permanent knowledge |
| Redis `clear()` complexity | Consider giving KB items a separate key prefix (`{prefix}kb:item:{id}`) to simplify the preserve-on-clear logic |

---

## See Also

- [README.md](README.md) - Overview and features
- [DOCUMENTATION.md](DOCUMENTATION.md) - Navigation guide for all docs
- [CLAUDE.md](CLAUDE.md) - Architecture, philosophy, and development guide
- [examples/](examples/) - Real-world integration examples

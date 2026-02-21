# Cogneetree Strategy: Becoming the Go-To Agent Memory Library

## Positioning

> **Cogneetree**: Hierarchical agent memory that knows what to remember, what to forget, and fits in your token budget. Self-hosted. Zero vendor lock-in. Works with any LLM.

---

## Competitive Landscape (as of Feb 2026)

| Library | Focus | Strength | Weakness | Cogneetree Advantage |
|---------|-------|----------|----------|---------------------|
| **Mem0** | Personalization, SaaS | Production-ready, graph memory | Vendor lock-in, self-hosting is painful | Fully self-hosted, no external deps |
| **Zep** | Temporal knowledge graphs | Sub-100ms retrieval, PII redaction | Enterprise-heavy, complex setup | Lightweight, zero-config start |
| **Letta** | Self-editing agent memory | Transparent, agents manage own memory | Requires its own runtime | Works with any framework |
| **LangMem** | LangChain ecosystem | Developer-friendly, 3 memory types | Tied to LangGraph, no standalone use | Framework-agnostic |

### What Nobody Solves Yet

1. **What to forget** - Every competitor stores everything and hopes retrieval handles it
2. **Token-aware retrieval** - No library lets you say "give me the best context that fits in 1500 tokens"
3. **Hierarchical proximity** - Flat vector stores can't encode "this decision is from the same work area"

### Industry Trends (2026)

- **Dual-layer architecture** is the production standard: hot path (always in-context) + cold path (retrieved on demand)
- **Selective retrieval** can cut token usage by 90% while maintaining accuracy (LoCoMo benchmark research)
- **MCP (Model Context Protocol)** is becoming the standard for agent-tool communication
- **Self-hosted** matters more as enterprises push back on sending data to external APIs

---

## The 7 Strategic Priorities

### Priority 1: Token Budget as a First-Class Concept

**Why**: Measurable differentiator. No competitor does this.

**What**:
- `build_context(max_tokens=N)` with accurate token estimation
- Integrate `tiktoken` for precise counts (not `len//4` approximation)
- Every API response should report token cost
- Publish benchmarks: "X% fewer tokens than Mem0/Zep for equivalent recall accuracy"

**Target metric**: Demonstrate 80-90% token reduction vs naive retrieval with equivalent decision quality.

**TODOs**:
- [ ] Add `max_tokens` parameter to `build_context()`
- [ ] Integrate `tiktoken` as optional dependency for accurate token counting
- [ ] Fallback to char-based estimation when tiktoken unavailable
- [ ] Add `token_cost` field to `ContextResult`
- [ ] Add `total_tokens` to recall/build_context response metadata
- [ ] Write benchmark comparing token usage: Cogneetree vs raw vector search vs Mem0
- [ ] Document token optimization strategies with measured results

---

### Priority 2: MCP Server (Zero-Friction Adoption)

**Why**: Any MCP-compatible agent gets memory for free. This is table stakes for 2026.

**What**:
- `cogneetree serve --storage ./memory.json --mcp` starts an MCP server
- Three tools: `recall`, `record`, `context`
- Works with Claude, GPT-based agents, any MCP client

**TODOs**:
- [ ] Create `src/cogneetree/mcp/` module
- [ ] Define MCP tool schemas for `cogneetree_recall`, `cogneetree_record`, `cogneetree_context`
- [ ] Implement MCP server using `mcp` Python SDK
- [ ] Add CLI command: `cogneetree serve --mcp --storage <path> --port <port>`
- [ ] Add `cogneetree[mcp]` optional dependency group in pyproject.toml
- [ ] Test with Claude Desktop / Claude Code as MCP client
- [ ] Write quickstart guide: "Add memory to your agent in 2 minutes"

---

### Priority 3: Hot/Cold Path Architecture

**Why**: Matches 2026 production pattern. Cogneetree's hierarchy naturally supports this.

**What**:
- **Hot path**: Session-level propagated decisions, always injected into prompt (~500-1000 tokens)
- **Cold path**: Cross-session retrieval, on-demand, token-budgeted
- Explicit API separation so agents know what's "always available" vs "searched"

**TODOs**:
- [ ] Implement `memory.session_context()` → returns accumulated decisions/learnings (hot path)
- [ ] Implement `RecallResult` with separate `session` (hot) and `items` (cold) fields
- [ ] Implement cascading propagation: `record_decision()` bubbles up to Activity → Session
- [ ] Implement semantic gating: only relevant decisions propagate
- [ ] Implement deduplication at each level (word overlap threshold)
- [ ] Add `DecisionEntry` model with `content`, `timestamp`, `count` fields
- [ ] Add `decisions` and `learnings` dict fields to `Activity` and `Session` models
- [ ] Update `build_context()` to include hot path first, then fill remaining budget with cold path
- [ ] Test: Activity 2 sees Activity 1's decisions without explicit retrieval
- [ ] Test: Irrelevant decisions (formatting, etc.) stay at task level

---

### Priority 4: Memory Consolidation & Forgetting

**Why**: Biggest unsolved problem in the space. Unique positioning opportunity.

**What**:
- **Deduplication**: Similar items merged, frequency tracked as importance signal
- **Summarization**: Old task-level items condensed into activity-level summaries
- **Decay**: Low-confidence, unreferenced items lose score weight over time
- **Promotion**: Frequently-referenced items promoted to permanent memory

**Pipeline**:
```
Raw items (100s)
  → Deduplicate (word overlap > 80%)
  → Summarize old tasks (> 30 days, group by tag)
  → Decay unreferenced items (score *= 0.95 per week)
  → Promote high-frequency items to permanent
  → Result: 10-20 high-value items, not 100s of noise
```

**TODOs**:
- [ ] Implement `PermanentMemory` storage layer (cross-session, never expires)
- [ ] Implement deduplication in `record_decision()`/`record_learning()` using word overlap
- [ ] Add `reference_count` tracking: increment when an item is returned by `recall()`
- [ ] Implement `memory.consolidate()` method that summarizes old items
- [ ] Add configurable decay: `config.decay_rate = 0.95` (per week)
- [ ] Add `memory.forget(item_id)` for manual removal
- [ ] Add `memory.promote(item_id)` for manual promotion to permanent
- [ ] Implement automatic promotion: items referenced > N times in different sessions → permanent
- [ ] Write consolidation tests: before/after token usage comparison
- [ ] Add `cogneetree consolidate --storage <path> --older-than 30d` CLI command

---

### Priority 5: Benchmarks & Proof

**Why**: Nobody adopts without evidence. Benchmarks make you the standard others measure against.

**What**:
- Adapt LoCoMo for agent work memory (not just conversational)
- Create "Agent Decision Consistency" benchmark
- Publish reproducible token efficiency numbers

**TODOs**:
- [ ] Create `benchmarks/` directory with reproducible test harness
- [ ] Design "Agent Decision Consistency" benchmark:
  - Record decisions in Project 1-3
  - Query in Project 4: does agent find correct prior decisions?
  - Measure: recall accuracy, token cost, latency
- [ ] Adapt LoCoMo benchmark for hierarchical agent work patterns
- [ ] Benchmark against baseline approaches:
  - Raw vector search (no hierarchy)
  - Full context dump (no retrieval)
  - Mem0 (if self-hostable)
- [ ] Measure and publish: tokens used, retrieval accuracy, latency
- [ ] Create benchmark results table in README
- [ ] Make benchmarks runnable: `python -m benchmarks.run`

---

### Priority 6: Framework Adapters

**Why**: Meet developers where they are. Reduce integration friction to near-zero.

**What**:
- LangChain Tool wrapper
- CrewAI memory adapter
- OpenAI function calling schema
- Claude tool_use schema
- AutoGen adapter

**TODOs**:
- [ ] Create `src/cogneetree/adapters/` module
- [ ] Implement `cogneetree.adapters.langchain`:
  - `CogneetreeRecallTool(BaseTool)` - LangChain tool wrapping `recall()`
  - `CogneetreeRecordTool(BaseTool)` - LangChain tool wrapping `record_*()`
- [ ] Implement `cogneetree.adapters.openai`:
  - Export function calling JSON schemas for recall/record
- [ ] Implement `cogneetree.adapters.claude`:
  - Export tool_use JSON schemas
- [ ] Implement `cogneetree.adapters.crewai`:
  - Memory adapter compatible with CrewAI's memory interface
- [ ] Add `cogneetree[langchain]`, `cogneetree[crewai]` optional dependency groups
- [ ] Write integration examples for each framework
- [ ] Test each adapter with a simple agent workflow

---

### Priority 7: The Killer Demo

**Why**: The "aha moment" that drives adoption. Shows the agent visibly getting smarter.

**What**:
- An agent works across 5 projects
- By project 5, it avoids past mistakes, applies learned patterns, cites its own history
- Token usage declines as memory consolidates
- Side-by-side comparison: agent with Cogneetree vs agent without

**TODOs**:
- [ ] Design 5-project demo scenario (e.g., building progressively complex APIs)
- [ ] Project 1: Agent makes initial decisions, records learnings
- [ ] Project 2: Agent finds Project 1 knowledge, avoids repeating research
- [ ] Project 3: Agent applies cross-project patterns (macro scope)
- [ ] Project 4: Memory consolidation kicks in, token usage drops
- [ ] Project 5: Agent cites specific past decisions, handles edge cases from prior learnings
- [ ] Add token usage tracking per project (show declining curve)
- [ ] Record screencast / create animated terminal output
- [ ] Create `examples/killer_demo.py` that runs the full scenario
- [ ] Add comparison mode: same projects without Cogneetree (full-context baseline)
- [ ] Publish demo in README with results table

---

## Implementation Order

| Phase | Items | Effort | Impact |
|-------|-------|--------|--------|
| **Phase 0** | Token budget (#1) + Hot/cold path (#3) | 2 weeks | Core differentiator |
| **Phase 1** | MCP server (#2) + Framework adapters (#6) | 1 week | Adoption channels |
| **Phase 2** | Memory consolidation (#4) | 2 weeks | Unique positioning |
| **Phase 3** | Benchmarks (#5) + Killer demo (#7) | 1 week | Proof & marketing |

**Total: ~6 weeks to differentiated, benchmarked, easy-to-adopt library.**

---

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Token efficiency | 80-90% reduction vs naive retrieval | Benchmark suite |
| Retrieval accuracy | Competitive with Mem0/Zep on adapted LoCoMo | Benchmark suite |
| Integration time | < 5 minutes to first working recall | User testing |
| Adoption signal | GitHub stars, PyPI downloads | Analytics |
| Decision consistency | Agent makes consistent decisions across projects | Decision consistency benchmark |

---

## Anti-Goals

Things Cogneetree should **NOT** try to be:

- **Not a chatbot memory** - Mem0/Zep own that. Focus on agent work memory.
- **Not a SaaS product** - Stay self-hosted, open source. That's the moat against Mem0.
- **Not a full agent framework** - Don't become Letta. Stay a library that plugs into any framework.
- **Not a vector database** - Use existing ones (or no DB at all). The value is the hierarchy and intelligence on top.

---

## References

- [Survey of AI Agent Memory Frameworks (Graphlit)](https://www.graphlit.com/blog/survey-of-ai-agent-memory-frameworks)
- [Mem0: Graph Memory for AI Agents](https://mem0.ai/blog/graph-memory-solutions-ai-agents)
- [Agent Memory: Letta vs Mem0 vs Zep vs Cognee](https://forum.letta.com/t/agent-memory-letta-vs-mem0-vs-zep-vs-cognee/88)
- [Benchmarking AI Agent Memory (Letta)](https://www.letta.com/blog/benchmarking-ai-agent-memory)
- [LoCoMo: Long-Term Conversational Memory Benchmark](https://snap-research.github.io/locomo/)
- [AI Memory Systems Benchmark 2025](https://guptadeepak.com/the-ai-memory-wars-why-one-system-crushed-the-competition-and-its-not-openai/)
- [LangChain Memory vs Mem0 vs Zep 2026](https://www.index.dev/skill-vs-skill/ai-mem0-vs-zep-vs-langchain-memory)
- [From Beta to Battle-Tested: Letta vs Mem0 vs Zep](https://medium.com/asymptotic-spaghetti-integration/from-beta-to-battle-tested-picking-between-letta-mem0-zep-for-ai-memory-6850ca8703d1)

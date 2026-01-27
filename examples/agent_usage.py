"""
Example: How an agent uses AgentMemory for frictionless context access.

This shows the actual agent code patterns - how agents interact with history
during their work without worrying about implementation details.
"""

from cogneetree.core.context_manager import ContextManager
from cogneetree.agent_memory import AgentMemory


def example_auth_agent():
    """
    Example: Agent working on JWT validation middleware.

    Shows three levels of memory interaction:
    1. One-line recalls during reasoning
    2. Building context blocks for prompts
    3. Exploring similar past work
    """

    # ==================== Setup ====================
    manager = ContextManager()

    # Create context for our work
    manager.create_session("auth_project_2024", "Add JWT validation", "Secure API endpoints")
    manager.create_activity(
        "a1",
        "auth_project_2024",
        "Implement middleware",
        ["jwt", "validation"],
        "coder",
        "security",
        "Add token validation before request processing",
    )
    manager.create_task("middleware_task", "a1", "Add JWT validation middleware", ["jwt"])

    # Get memory interface for this task
    memory = AgentMemory(manager.storage, current_task_id="middleware_task")

    # ==================== Agent's Reasoning - Micro Recall ====================
    print("=== Agent asks: 'How should I validate JWT?' ===\n")

    # Simple one-line: "What do I know about JWT validation?"
    context = memory.recall("JWT validation", scope="balanced")

    for item in context:
        print(f"📚 {item.source}")
        print(f"   Category: {item.category.upper()}")
        print(f"   Relevance: {item.confidence:.1%}")
        print(f"   > {item.content[:80]}...")
        print()

    # ==================== Building Prompts with Context ====================
    print("\n=== Agent building prompt for LLM ===\n")

    # Agent builds a context block to send to LLM
    prompt = f"""You are implementing JWT validation middleware.

{memory.build_context('JWT validation and security', max_items=3)}

Based on this history, what's the best approach for validation?
"""

    print(prompt)

    # ==================== Decisions vs Learnings ====================
    print("\n=== Agent asks: 'What decisions did we make about auth?' ===\n")

    decisions = memory.decisions("authentication")
    for decision in decisions:
        print(f"✅ Decision from {decision.source}")
        print(f"   {decision.content}")
        print()

    learnings = memory.learnings("token validation")
    print(f"\nFound {len(learnings)} past learnings about token validation")
    for learning in learnings[:2]:
        print(f"  • {learning.content[:60]}...")

    # ==================== Exploring Related Work ====================
    print("\n=== Agent explores: 'Show me other auth tasks' ===\n")

    similar = memory.similar_tasks("auth")
    print(f"Found {len(similar)} similar tasks about 'auth':")
    for task in similar:
        print(f"  - {task['description']}")
        print(f"    Items recorded: {len(task['items'])}")

    # ==================== Recording Agent's Work ====================
    print("\n=== Agent records its work ===\n")

    # As agent works, it records decisions
    manager.record_decision(
        "Use HS256 for symmetric signing (matches past decision)",
        ["jwt", "signing"],
    )
    print("✅ Recorded decision: Use HS256 for symmetric signing")

    manager.record_action("Added middleware before request.ProcessRequest()", ["jwt", "middleware"])
    print("✅ Recorded action: Added middleware")

    manager.record_learning("Validation must happen BEFORE request processing", ["jwt", "best-practice"])
    print("✅ Recorded learning: Validation timing critical")

    # ==================== Macro Memory - Learning Pattern ====================
    print("\n\n=== Agent switches to MACRO memory: 'What patterns across ALL projects?' ===\n")

    # Agent can switch scopes for learning
    all_jwt_decisions = memory.decisions("JWT", scope="macro")
    print(f"JWT decisions across ALL projects: {len(all_jwt_decisions)}")
    for decision in all_jwt_decisions[:3]:
        print(f"  • {decision.source}: {decision.content[:70]}...")

    # ==================== Citation ====================
    print("\n=== Agent cites past work ===\n")

    top_result = memory.recall("JWT validation")[0]
    citation = memory.cite(top_result)
    print(f"Agent writes: 'Following best practices {citation}'")


def example_agent_with_explicit_scopes():
    """
    Example: Agent explicitly choosing micro/macro/balanced search.

    Shows when agents would use different scopes:
    - Micro: "I need focused current task context"
    - Balanced: "Normal work - use current + recent history"
    - Macro: "I'm learning patterns across all projects"
    """

    manager = ContextManager()
    manager.create_session("microservices", "Build auth system", "Multiple services")
    manager.create_activity(
        "a1",
        "microservices",
        "Token validation",
        ["jwt"],
        "coder",
        "core",
        "Validate tokens across services",
    )
    manager.create_task("t1", "a1", "Inter-service JWT validation", ["jwt", "services"])

    memory = AgentMemory(manager.storage, current_task_id="t1")

    print("=== Different memory scopes ===\n")

    print("MICRO - Current task only (focused):")
    micro = memory.recall("JWT", scope="micro")
    print(f"  Results: {len(micro)} items from THIS task\n")

    print("BALANCED - Current + recent history (practical):")
    balanced = memory.recall("JWT", scope="balanced")
    print(f"  Results: {len(balanced)} items (current + last 90 days)\n")

    print("MACRO - All historical patterns (learning):")
    macro = memory.recall("JWT", scope="macro")
    print(f"  Results: {len(macro)} items from ALL projects\n")


if __name__ == "__main__":
    print("=" * 80)
    print("AGENT MEMORY USAGE EXAMPLE")
    print("=" * 80)
    print()

    example_auth_agent()

    print("\n" + "=" * 80)
    print("SCOPE EXAMPLES")
    print("=" * 80)
    print()

    example_agent_with_explicit_scopes()

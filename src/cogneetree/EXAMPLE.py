"""Simple example of using the refactored agentic_context module."""

from cogneetree import ContextWorkflow, Config


def example_basic():
    """Basic workflow example - no semantic retrieval."""
    config = Config.default()
    workflow = ContextWorkflow(config)

    with workflow.session("user_session", "Build a REST API", "1. Design 2. Code 3. Test") as session:
        with session.activity(
            "activity1",
            "Build user endpoint",
            "coder",
            "api_users",
            "Planner says: Use standard patterns",
            tags=["api", "users", "crud"]
        ) as activity:
            with activity.task("Create POST endpoint", tags=["endpoint", "create"]) as task:
                # Build context-aware prompt
                prompt = task.build_prompt(include_history=False)
                print("=== PROMPT FOR LLM ===")
                print(prompt)
                print()

                # Record actions taken
                task.record_action("Defined User model")
                task.record_action("Created validation schema")
                task.record_decision("Used Pydantic for validation (standard choice)")
                task.record_learning("Validation prevents invalid data early")

                # Mark task as done
                task.set_result("POST /users endpoint created with validation")

    workflow.clear()
    print("✓ Example complete")


def example_with_history():
    """Example with semantic retrieval - finds relevant past decisions."""
    config = Config(use_semantic=True, max_results=3)
    workflow = ContextWorkflow(config)

    # Create a session with prior context
    with workflow.session("session2", "Add auth to API", "Implement JWT") as session:
        with session.activity("activity1", "Create auth middleware", "coder", "auth", "Analysis here", tags=["auth", "middleware"]) as activity:
            with activity.task("Implement JWT validation", tags=["jwt", "auth"]) as task:
                # First record some historical context
                manager = task.manager
                manager.record_decision("Use JWT for stateless auth", ["auth", "jwt"])
                manager.record_learning("Short expiry (15m) + refresh token pattern", ["jwt", "refresh"])

                # Now build prompt - will include relevant history
                prompt = task.build_prompt(include_history=True)
                print("=== PROMPT WITH HISTORY ===")
                print(prompt)
                print()

                task.set_result("JWT middleware implemented")

    workflow.clear()
    print("✓ History example complete")


if __name__ == "__main__":
    print("Running basic example...\n")
    example_basic()

    print("\n" + "=" * 60 + "\n")

    print("Running history example...\n")
    example_with_history()

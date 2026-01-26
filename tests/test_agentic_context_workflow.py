"""Tests for refactored agentic_context.workflow module."""

import pytest
from cogneetree.workflow import (
    ContextWorkflow,
    SessionContext,
    ActivityContext,
    TaskContext,
)


class TestContextWorkflow:
    """Test ContextWorkflow class."""

    def test_workflow_creation(self):
        """Test creating a ContextWorkflow."""
        workflow = ContextWorkflow()
        assert workflow is not None
        assert workflow.manager is not None

    def test_workflow_session_creation(self):
        """Test creating a session in workflow."""
        workflow = ContextWorkflow()
        
        with workflow.session("s1", "Do something", "Plan: 1,2") as session_ctx:
            assert isinstance(session_ctx, SessionContext)

    def test_workflow_session_and_activity(self):
        """Test session and activity creation."""
        workflow = ContextWorkflow()
        
        with workflow.session("s1", "Do something", "Plan") as session_ctx:
            with session_ctx.activity("a1", "Implement", "coder", "comp", "analysis") as activity_ctx:
                assert isinstance(activity_ctx, ActivityContext)

    def test_workflow_session_activity_task(self):
        """Test full hierarchy."""
        workflow = ContextWorkflow()
        
        with workflow.session("s1", "Do something", "Plan") as session_ctx:
            with session_ctx.activity("a1", "Implement", "coder", "comp", "analysis") as activity_ctx:
                with activity_ctx.task("Write code", ["tag1"]) as task_ctx:
                    assert isinstance(task_ctx, TaskContext)

    def test_session_context_activity_creation(self):
        """Test creating activity from session context."""
        workflow = ContextWorkflow()
        
        with workflow.session("s1", "Task", "Plan") as session_ctx:
            with session_ctx.activity("a1", "Impl", "coder", "comp", "analysis") as ctx:
                assert ctx is not None

    def test_activity_context_task_creation(self):
        """Test creating task from activity context."""
        workflow = ContextWorkflow()
        
        with workflow.session("s1", "Task", "Plan") as session_ctx:
            with session_ctx.activity("a1", "Impl", "coder", "comp", "analysis") as activity_ctx:
                with activity_ctx.task("Subtask", ["tag"]) as task_ctx:
                    assert task_ctx is not None

    def test_task_context_build_prompt(self):
        """Test building prompt in task context."""
        workflow = ContextWorkflow()
        
        with workflow.session("s1", "Task", "Plan") as session_ctx:
            with session_ctx.activity("a1", "Impl", "coder", "comp", "analysis") as activity_ctx:
                with activity_ctx.task("Subtask", ["tag"]) as task_ctx:
                    prompt = task_ctx.build_prompt(include_history=True)
                    assert isinstance(prompt, str)

    def test_task_context_record_action(self):
        """Test recording action in task context."""
        workflow = ContextWorkflow()
        
        with workflow.session("s1", "Task", "Plan") as session_ctx:
            with session_ctx.activity("a1", "Impl", "coder", "comp", "analysis") as activity_ctx:
                with activity_ctx.task("Subtask", ["tag"]) as task_ctx:
                    task_ctx.record_action("Performed action")
                    # Should not raise error

    def test_task_context_record_decision(self):
        """Test recording decision in task context."""
        workflow = ContextWorkflow()
        
        with workflow.session("s1", "Task", "Plan") as session_ctx:
            with session_ctx.activity("a1", "Impl", "coder", "comp", "analysis") as activity_ctx:
                with activity_ctx.task("Subtask", ["tag"]) as task_ctx:
                    task_ctx.record_decision("Made decision")
                    # Should not raise error

    def test_task_context_record_learning(self):
        """Test recording learning in task context."""
        workflow = ContextWorkflow()
        
        with workflow.session("s1", "Task", "Plan") as session_ctx:
            with session_ctx.activity("a1", "Impl", "coder", "comp", "analysis") as activity_ctx:
                with activity_ctx.task("Subtask", ["tag"]) as task_ctx:
                    task_ctx.record_learning("Learned something")
                    # Should not raise error

    def test_task_context_set_result(self):
        """Test setting task result."""
        workflow = ContextWorkflow()
        
        with workflow.session("s1", "Task", "Plan") as session_ctx:
            with session_ctx.activity("a1", "Impl", "coder", "comp", "analysis") as activity_ctx:
                with activity_ctx.task("Subtask", ["tag"]) as task_ctx:
                    task_ctx.set_result("Task result here")
                    # Should not raise error

    def test_multiple_activities_in_session(self):
        """Test multiple activities in one session."""
        workflow = ContextWorkflow()
        
        with workflow.session("s1", "Task", "Plan") as session_ctx:
            with session_ctx.activity("a1", "Design", "architect", "comp", "analysis"):
                pass
            
            with session_ctx.activity("a2", "Impl", "coder", "comp", "analysis"):
                pass

    def test_multiple_tasks_in_activity(self):
        """Test multiple tasks in one activity."""
        workflow = ContextWorkflow()
        
        with workflow.session("s1", "Task", "Plan") as session_ctx:
            with session_ctx.activity("a1", "Impl", "coder", "comp", "analysis") as activity_ctx:
                with activity_ctx.task("Task 1", ["tag"]):
                    pass
                
                with activity_ctx.task("Task 2", ["tag"]):
                    pass

    def test_workflow_clear(self):
        """Test clearing workflow context."""
        workflow = ContextWorkflow()
        
        with workflow.session("s1", "Task", "Plan") as session_ctx:
            with session_ctx.activity("a1", "Impl", "coder", "comp", "analysis"):
                pass
        
        # Clear should not raise error
        workflow.clear()

    def test_activity_requires_active_session(self):
        """Test that activity requires active session."""
        workflow = ContextWorkflow()
        
        # Creating activity without session should raise
        session_ctx = SessionContext(workflow.manager)
        with pytest.raises(ValueError):
            with session_ctx.activity("a1", "Impl", "coder", "comp", "analysis"):
                pass

    def test_task_requires_active_activity(self):
        """Test that task requires active activity."""
        workflow = ContextWorkflow()
        
        with workflow.session("s1", "Task", "Plan") as session_ctx:
            activity_ctx = ActivityContext(workflow.manager)
            with pytest.raises(ValueError):
                with activity_ctx.task("Task", ["tag"]):
                    pass

    def test_workflow_complex_scenario(self):
        """Test complex multi-level workflow."""
        workflow = ContextWorkflow()
        
        with workflow.session("build_feature", "Build user auth", "1. Design 2. Code 3. Test") as session_ctx:
            # Design phase
            with session_ctx.activity("design", "Design auth system", "architect", "planning", "analysis") as design_ctx:
                with design_ctx.task("Create ER diagram", ["diagram", "database"]) as task_ctx:
                    task_ctx.record_decision("Using JWT tokens")
                    task_ctx.set_result("ER diagram in docs/")
            
            # Implementation phase
            with session_ctx.activity("coding", "Implement auth", "coder", "implementation", "coding") as code_ctx:
                with code_ctx.task("Implement login", ["auth", "api"]) as task_ctx:
                    task_ctx.record_action("Wrote login endpoint")
                    task_ctx.set_result("Endpoint implemented")
                
                with code_ctx.task("Implement signup", ["auth", "api"]) as task_ctx:
                    task_ctx.record_action("Wrote signup endpoint")
                    task_ctx.set_result("Endpoint implemented")


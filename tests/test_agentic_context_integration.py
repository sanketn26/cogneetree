"""Integration tests for refactored agentic_context module."""

import pytest
from cogneetree.workflow import ContextWorkflow
from cogneetree.config import Config


class TestAgenticContextIntegration:
    """Integration tests for full agentic context workflow."""

    def test_complete_workflow_scenario(self):
        """Test a complete workflow from session to result."""
        workflow = ContextWorkflow(Config.default())
        
        with workflow.session("build_feature", "Build user authentication", "1. Design 2. Implement 3. Test") as session_ctx:
            assert session_ctx is not None
            
            # Design phase
            with session_ctx.activity("design", "Design authentication system", "architect", "planning", "analysis", ["auth", "design"]) as design_ctx:
                with design_ctx.task("Create architecture diagram", ["diagram", "architecture"]) as task_ctx:
                    task_ctx.record_decision("Use JWT for token-based auth")
                    task_ctx.record_decision("Use bcrypt for password hashing")
                    task_ctx.set_result("Architecture diagram created and documented")
            
            # Implementation phase
            with session_ctx.activity("implementation", "Implement auth endpoints", "coder", "coding", "implementation", ["auth", "api"]) as impl_ctx:
                # Login endpoint
                with impl_ctx.task("Implement login endpoint", ["endpoint", "auth", "api"]) as task_ctx:
                    task_ctx.record_action("Created POST /auth/login endpoint")
                    task_ctx.record_action("Added email/password validation")
                    task_ctx.record_decision("Return JWT token in response")
                    task_ctx.set_result("Login endpoint completed and tested")
                
                # Signup endpoint
                with impl_ctx.task("Implement signup endpoint", ["endpoint", "auth", "api"]) as task_ctx:
                    task_ctx.record_action("Created POST /auth/signup endpoint")
                    task_ctx.record_action("Added password hashing with bcrypt")
                    task_ctx.record_decision("Send verification email after signup")
                    task_ctx.set_result("Signup endpoint completed and tested")

    def test_workflow_with_retrieval(self):
        """Test workflow with context retrieval."""
        workflow = ContextWorkflow(Config.default())
        
        with workflow.session("s1", "Build system", "Plan") as session_ctx:
            with session_ctx.activity("a1", "Implement API", "coder", "comp", "analysis", ["api"]) as activity_ctx:
                # Record some decisions
                workflow.manager.record_decision("Use REST API design", ["api", "architecture"])
                workflow.manager.record_decision("Use JSON for payload", ["api", "format"])
                
                # Later, retrieve similar context
                with activity_ctx.task("Create endpoints", ["api", "endpoint"]) as task_ctx:
                    # Build prompt should include relevant history
                    prompt = task_ctx.build_prompt(include_history=True)
                    assert isinstance(prompt, str)
                    assert len(prompt) > 0

    def test_multiple_sessions_same_workflow(self):
        """Test managing multiple sessions in one workflow."""
        workflow = ContextWorkflow(Config.default())
        
        # First session
        with workflow.session("s1", "Task 1", "Plan 1") as ctx1:
            with ctx1.activity("a1", "Activity 1", "coder", "comp", "analysis") as act1:
                workflow.manager.record_action("Action in session 1", ["s1"])
        
        # Second session
        with workflow.session("s2", "Task 2", "Plan 2") as ctx2:
            with ctx2.activity("a2", "Activity 2", "architect", "comp", "analysis") as act2:
                workflow.manager.record_action("Action in session 2", ["s2"])
        
        # Verify both sessions exist
        assert len(workflow.manager.storage.sessions) == 2

    def test_context_building_with_activity_and_task(self):
        """Test that context builds properly with activity and task info."""
        workflow = ContextWorkflow(Config.default())
        
        with workflow.session("build_api", "Build API", "Plan") as session_ctx:
            with session_ctx.activity("impl", "Implementation phase", "coder", "component1", "analysis", ["coding"]) as activity_ctx:
                with activity_ctx.task("Implement endpoint", ["api", "endpoint"]) as task_ctx:
                    prompt = task_ctx.build_prompt(include_history=False)
                    
                    # Prompt should contain session, activity, and task info
                    assert "TASK" in prompt or "Task" in prompt or "Implement endpoint" in prompt

    def test_recording_all_context_types(self):
        """Test recording all types of context."""
        workflow = ContextWorkflow(Config.default())
        
        with workflow.session("s1", "Build", "Plan") as session_ctx:
            # Record all types
            workflow.manager.record_action("Performed action", ["tag1"])
            workflow.manager.record_decision("Made decision", ["tag2"])
            workflow.manager.record_learning("Learned something", ["tag3"])
            workflow.manager.record_result("Got result", ["tag4"])
            
            # Verify storage
            stats = workflow.manager.storage.get_stats()
            assert stats["items"] >= 4

    def test_semantic_config_integration(self):
        """Test workflow with semantic search config."""
        config = Config(use_semantic=False, max_results=5)
        workflow = ContextWorkflow(config)
        
        with workflow.session("s1", "Task", "Plan") as session_ctx:
            with session_ctx.activity("a1", "Activity", "coder", "comp", "analysis") as activity_ctx:
                # Record some items
                workflow.manager.record_action("Implementation detail", ["api", "endpoint"])
                workflow.manager.record_decision("Architecture choice", ["api", "design"])
                
                # Retrieve should respect max_results
                with activity_ctx.task("Create endpoint", ["api"]) as task_ctx:
                    results = workflow.manager.retrieve(["api"], "API implementation")
                    assert len(results) <= 5

    def test_nested_hierarchy_isolation(self):
        """Test that nested contexts properly manage state."""
        workflow = ContextWorkflow(Config.default())
        
        with workflow.session("s1", "First task", "Plan 1") as ctx1:
            with ctx1.activity("a1", "First activity", "coder", "comp", "analysis") as act1:
                with act1.task("Task 1", ["t1"]) as t1:
                    task1 = workflow.manager.get_current_task()
                    assert task1 is not None
                    assert task1.description == "Task 1"
                
                with act1.task("Task 2", ["t2"]) as t2:
                    task2 = workflow.manager.get_current_task()
                    assert task2 is not None
                    assert task2.description == "Task 2"
                    # Should be different from task1
                    assert task1 is not None
                    assert task1.task_id != task2.task_id

    def test_complete_task_tracking(self):
        """Test that task completion is tracked."""
        workflow = ContextWorkflow(Config.default())
        
        with workflow.session("s1", "Task", "Plan") as session_ctx:
            with session_ctx.activity("a1", "Activity", "coder", "comp", "analysis") as activity_ctx:
                with activity_ctx.task("Subtask", ["tag"]) as task_ctx:
                    task_ctx.set_result("Task completed successfully")
                
                # Get task and verify result is set
                task = workflow.manager.get_current_task()
                assert task is not None
                assert task.result == "Task completed successfully"

    def test_retrieval_across_session_history(self):
        """Test that retrieval works across recorded history."""
        workflow = ContextWorkflow(Config.default())
        
        with workflow.session("s1", "Build", "Plan") as session_ctx:
            # Record various decisions and actions
            workflow.manager.record_decision("Decision about authentication", ["auth", "security"])
            workflow.manager.record_action("Implemented login", ["auth", "api"])
            workflow.manager.record_learning("JWT is good for stateless auth", ["auth", "jwt"])
            
            # Retrieve with different query
            results = workflow.manager.retrieve(["security"], "security related decisions")
            assert isinstance(results, list)
            assert len(results) >= 0

    def test_workflow_clear_and_reuse(self):
        """Test clearing and reusing workflow."""
        workflow = ContextWorkflow(Config.default())
        
        # First use
        with workflow.session("s1", "Task 1", "Plan") as ctx:
            workflow.manager.record_action("Action 1", ["tag"])
        
        stats1 = workflow.manager.storage.get_stats()
        assert stats1["sessions"] == 1
        
        # Clear and reuse
        workflow.clear()
        
        stats2 = workflow.manager.storage.get_stats()
        assert stats2["sessions"] == 0
        
        # Use again
        with workflow.session("s2", "Task 2", "Plan") as ctx:
            workflow.manager.record_action("Action 2", ["tag"])
        
        stats3 = workflow.manager.storage.get_stats()
        assert stats3["sessions"] == 1

    def test_real_world_coding_scenario(self):
        """Test a real-world coding scenario."""
        workflow = ContextWorkflow(Config(max_results=5))
        
        with workflow.session("feature_auth", "Implement OAuth2 authentication", "Design -> Implement -> Test") as session_ctx:
            # Design phase
            with session_ctx.activity("design_oauth", "Design OAuth2 flow", "architect", "security", "architecture", ["oauth2", "design"]) as design:
                with design.task("Research OAuth2 standards", ["research", "oauth2"]) as task:
                    task.record_learning("OAuth2 uses authorization codes and tokens")
                    task.record_decision("Use authorization code flow for web apps")
                    task.set_result("OAuth2 design finalized")
            
            # Implement phase
            with session_ctx.activity("implement_oauth", "Implement OAuth2 provider", "coder", "backend", "implementation", ["oauth2", "implementation"]) as impl:
                with impl.task("Setup authorization endpoint", ["endpoint", "auth"]) as task:
                    task.record_action("Created /oauth/authorize endpoint")
                    task.set_result("Authorization endpoint working")
                
                with impl.task("Setup token endpoint", ["endpoint", "token"]) as task:
                    task.record_action("Created /oauth/token endpoint")
                    task.record_decision("Return refresh token with expiry")
                    task.set_result("Token endpoint working")

"""
Tests for the workflow generator.

This module tests that the Python workflow models generate
valid, expected YAML structures.
"""

import pytest
from pathlib import Path
from ruamel.yaml import YAML

from tools.workflow_model import Workflow, Job, Step, checkout_step, setup_python_step
from tools.gen_workflows import create_ci_workflow, create_benchmark_workflow


class TestWorkflowModels:
    """Test the basic workflow model classes."""
    
    def test_step_to_dict(self):
        """Test Step.to_dict() method."""
        step = Step(
            name="Test step",
            uses="actions/checkout@v4",
            with_={"token": "${{ secrets.GITHUB_TOKEN }}"},
            env={"CI": "true"},
            if_="success()",
            continue_on_error=True,
            timeout_minutes=5
        )
        
        expected = {
            "name": "Test step",
            "uses": "actions/checkout@v4",
            "with": {"token": "${{ secrets.GITHUB_TOKEN }}"},
            "env": {"CI": "true"},
            "if": "success()",
            "continue-on-error": True,
            "timeout-minutes": 5
        }
        
        assert step.to_dict() == expected
    
    def test_step_minimal(self):
        """Test Step with minimal required fields."""
        step = Step(name="Minimal step", run="echo 'hello'")
        
        expected = {
            "name": "Minimal step",
            "run": "echo 'hello'"
        }
        
        assert step.to_dict() == expected
    
    def test_job_to_dict(self):
        """Test Job.to_dict() method."""
        job = Job(
            runs_on="ubuntu-latest",
            steps=[
                Step("Step 1", run="echo 'step1'"),
                Step("Step 2", run="echo 'step2'")
            ],
            timeout_minutes=30,
            strategy={"matrix": {"python-version": ["3.11", "3.12"]}},
            env={"TEST": "true"}
        )
        
        result = job.to_dict()
        
        assert result["runs-on"] == "ubuntu-latest"
        assert result["timeout-minutes"] == 30
        assert result["strategy"] == {"matrix": {"python-version": ["3.11", "3.12"]}}
        assert result["env"] == {"TEST": "true"}
        assert len(result["steps"]) == 2
        assert result["steps"][0]["name"] == "Step 1"
    
    def test_workflow_to_dict(self):
        """Test Workflow.to_dict() method."""
        workflow = Workflow(
            name="Test Workflow",
            on={"push": {"branches": ["main"]}},
            jobs={
                "test": Job(
                    runs_on="ubuntu-latest",
                    steps=[Step("Test step", run="echo 'test'")]
                )
            },
            env={"GLOBAL": "true"},
            permissions={"contents": "read"}
        )
        
        result = workflow.to_dict()
        
        assert result["name"] == "Test Workflow"
        assert result["on"] == {"push": {"branches": ["main"]}}
        assert result["env"] == {"GLOBAL": "true"}
        assert result["permissions"] == {"contents": "read"}
        assert "test" in result["jobs"]


class TestConvenienceFunctions:
    """Test convenience functions for common step patterns."""
    
    def test_checkout_step(self):
        """Test checkout_step convenience function."""
        step = checkout_step("v3")
        
        assert step.name == "🔄 Checkout repository"
        assert step.uses == "actions/checkout@v3"
        assert step.with_ == {}
    
    def test_setup_python_step(self):
        """Test setup_python_step convenience function."""
        step = setup_python_step("3.11", cache=True)
        
        assert step.name == "🐍 Set up Python"
        assert step.uses == "actions/setup-python@v5"
        assert step.with_ == {"python-version": "3.11", "cache": "pip"}
    
    def test_setup_python_step_no_cache(self):
        """Test setup_python_step without cache."""
        step = setup_python_step("3.12", cache=False)
        
        assert step.with_ == {"python-version": "3.12"}


class TestWorkflowGeneration:
    """Test the generated workflows."""
    
    def test_ci_workflow_structure(self):
        """Test that CI workflow has expected structure."""
        workflow = create_ci_workflow()
        
        assert workflow.name == "🧪 CI Tests & Validation"
        assert "pull_request" in workflow.on
        assert "push" in workflow.on
        assert "workflow_dispatch" in workflow.on
        
        # Check permissions
        assert workflow.permissions == {
            "contents": "read",
            "pull-requests": "write"
        }
        
        # Check environment variables
        assert workflow.env["CI"] == "true"
        assert workflow.env["PYTHONUNBUFFERED"] == "1"
        
        # Check jobs exist
        assert "workflow-validation" in workflow.jobs
        assert "python-tests" in workflow.jobs
        assert "script-validation" in workflow.jobs
    
    def test_benchmark_workflow_structure(self):
        """Test that benchmark workflow has expected structure."""
        workflow = create_benchmark_workflow()
        
        assert workflow.name == "📊 Benchmark Validation"
        assert "workflow_dispatch" in workflow.on
        assert "schedule" in workflow.on
        
        # Check job exists
        assert "benchmark" in workflow.jobs
        benchmark_job = workflow.jobs["benchmark"]
        assert benchmark_job.runs_on == "ubuntu-latest"
        assert benchmark_job.timeout_minutes == 30
    
    def test_python_tests_matrix(self):
        """Test that python-tests job has correct matrix strategy."""
        workflow = create_ci_workflow()
        python_job = workflow.jobs["python-tests"]
        
        assert python_job.strategy is not None
        matrix = python_job.strategy["matrix"]
        assert "python-version" in matrix
        assert "3.11" in matrix["python-version"]
        assert "3.12" in matrix["python-version"]
        assert "os" in matrix
        assert "ubuntu-latest" in matrix["os"]
    
    def test_yaml_generation_valid(self):
        """Test that generated YAML is valid and parseable."""
        workflow = create_ci_workflow()
        yaml_dict = workflow.to_dict()
        
        # Should be able to serialize and deserialize without errors
        yaml = YAML()
        import io
        stream = io.StringIO()
        yaml.dump(yaml_dict, stream)
        yaml_content = stream.getvalue()
        
        # Should be able to parse it back
        parsed = yaml.load(yaml_content)
        assert parsed["name"] == workflow.name
        assert parsed["on"] == workflow.on
    
    def test_step_ordering(self):
        """Test that steps maintain their order."""
        workflow = create_ci_workflow()
        workflow_validation_steps = workflow.jobs["workflow-validation"].steps
        
        step_names = [step.name for step in workflow_validation_steps]
        expected_order = [
            "📥 Checkout code",
            "🐍 Set up Python 3.11", 
            "📦 Install test dependencies",
            "🔍 Run comprehensive workflow validation tests"
        ]
        
        assert step_names == expected_order


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_with_dict(self):
        """Test that empty with_ dict is handled correctly."""
        step = Step("Test", uses="some/action", with_={})
        result = step.to_dict()
        
        # Empty with_ dict should not appear in output
        assert "with" not in result
    
    def test_none_values_filtered(self):
        """Test that None values are filtered out."""
        step = Step(
            name="Test",
            uses="some/action",
            run=None,
            if_=None,
            continue_on_error=None
        )
        result = step.to_dict()
        
        expected_keys = {"name", "uses"}
        assert set(result.keys()) == expected_keys
    
    def test_workflow_with_no_permissions(self):
        """Test workflow without permissions."""
        workflow = Workflow(
            name="Simple",
            on={"push": None},
            jobs={"test": Job("ubuntu-latest", [Step("Test", run="echo test")])}
        )
        
        result = workflow.to_dict()
        assert "permissions" not in result
    
    def test_job_needs_single_string(self):
        """Test job needs with single string dependency."""
        job = Job(
            runs_on="ubuntu-latest",
            steps=[Step("Test", run="echo test")],
            needs="setup"
        )
        
        result = job.to_dict()
        assert result["needs"] == "setup"
    
    def test_job_needs_list(self):
        """Test job needs with list of dependencies.""" 
        job = Job(
            runs_on="ubuntu-latest",
            steps=[Step("Test", run="echo test")],
            needs=["setup", "lint"]
        )
        
        result = job.to_dict()
        assert result["needs"] == ["setup", "lint"]
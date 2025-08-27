"""
Python models for GitHub Actions workflows.

This module provides dataclasses that represent GitHub Actions workflow
components in a typed, testable way. These models are then converted
to YAML for consumption by GitHub Actions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class Step:
    """Represents a single step in a GitHub Actions job."""
    name: str
    uses: Optional[str] = None
    run: Optional[str] = None
    with_: Dict[str, Any] = field(default_factory=dict)  # maps to YAML key "with"
    env: Dict[str, str] = field(default_factory=dict)
    if_: Optional[str] = None  # maps to YAML key "if"
    continue_on_error: Optional[bool] = None
    timeout_minutes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for YAML generation."""
        result = {"name": self.name}
        
        if self.uses:
            result["uses"] = self.uses
        if self.run:
            result["run"] = self.run
        if self.with_:
            result["with"] = self.with_
        if self.env:
            result["env"] = self.env
        if self.if_:
            result["if"] = self.if_
        if self.continue_on_error is not None:
            result["continue-on-error"] = self.continue_on_error
        if self.timeout_minutes is not None:
            result["timeout-minutes"] = self.timeout_minutes
            
        return result


@dataclass
class Job:
    """Represents a GitHub Actions job."""
    runs_on: Union[str, List[str]]
    steps: List[Step]
    strategy: Optional[Dict[str, Any]] = None
    env: Dict[str, str] = field(default_factory=dict)
    timeout_minutes: Optional[int] = None
    needs: Optional[Union[str, List[str]]] = None
    if_: Optional[str] = None
    permissions: Optional[Dict[str, str]] = None
    outputs: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for YAML generation."""
        result = {
            "runs-on": self.runs_on,
            "steps": [step.to_dict() for step in self.steps]
        }
        
        if self.strategy:
            result["strategy"] = self.strategy
        if self.env:
            result["env"] = self.env
        if self.timeout_minutes is not None:
            result["timeout-minutes"] = self.timeout_minutes
        if self.needs:
            result["needs"] = self.needs
        if self.if_:
            result["if"] = self.if_
        if self.permissions:
            result["permissions"] = self.permissions
        if self.outputs:
            result["outputs"] = self.outputs
            
        return result


@dataclass
class Workflow:
    """Represents a complete GitHub Actions workflow."""
    name: str
    on: Dict[str, Any]
    jobs: Dict[str, Job]
    env: Dict[str, str] = field(default_factory=dict)
    permissions: Optional[Dict[str, str]] = None
    concurrency: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary for YAML generation."""
        result = {
            "name": self.name,
            "on": self.on,
            "jobs": {job_name: job.to_dict() for job_name, job in self.jobs.items()}
        }
        
        if self.env:
            result["env"] = self.env
        if self.permissions:
            result["permissions"] = self.permissions
        if self.concurrency:
            result["concurrency"] = self.concurrency
            
        return result


# Convenience functions for common step patterns
def checkout_step(version: str = "v4") -> Step:
    """Standard checkout step."""
    return Step("🔄 Checkout repository", uses=f"actions/checkout@{version}")


def setup_python_step(version: str = "3.12", cache: bool = True) -> Step:
    """Standard Python setup step."""
    with_params = {"python-version": version}
    if cache:
        with_params["cache"] = "pip"
    return Step("🐍 Set up Python", uses="actions/setup-python@v5", with_=with_params)


def install_deps_step(requirements: str = "requirements.txt") -> Step:
    """Standard dependency installation step."""
    return Step(
        "📦 Install dependencies",
        run=f"pip install --upgrade pip && pip install -r {requirements}"
    )


def run_tests_step(test_command: str = "pytest -v") -> Step:
    """Standard test execution step."""
    return Step("🧪 Run tests", run=test_command)


def benchmark_step(script: str, output_dir: str = "data/results") -> Step:
    """Standard benchmark execution step."""
    return Step(
        f"📊 Run {script}",
        run=f"python {script}",
        env={"BENCHMARK_OUTPUT_DIR": output_dir}
    )
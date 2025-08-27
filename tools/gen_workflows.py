"""
Generate GitHub Actions workflows from Python models.

This script converts typed Python workflow definitions into deterministic
YAML files that GitHub Actions can consume.
"""

import os
from pathlib import Path
from typing import List

from ruamel.yaml import YAML

from .workflow_model import (
    Workflow, Job, Step,
    checkout_step, setup_python_step, install_deps_step, run_tests_step
)


def create_ci_workflow() -> Workflow:
    """Create the main CI workflow with tests and validation."""
    
    # Define test matrix
    matrix = {
        "python-version": ["3.11", "3.12"],
        "os": ["ubuntu-latest"]
    }
    
    return Workflow(
        name="🧪 CI Tests & Validation",
        on={
            "pull_request": {
                "branches": ["main"],
                "paths": [
                    ".github/workflows/**",
                    "scripts/**", 
                    "tests/**",
                    "requirements.txt",
                    "*.py"
                ]
            },
            "push": {
                "branches": ["main"],
                "paths": [
                    ".github/workflows/**",
                    "scripts/**",
                    "tests/**", 
                    "requirements.txt",
                    "*.py"
                ]
            },
            "workflow_dispatch": None
        },
        permissions={
            "contents": "read",
            "pull-requests": "write"
        },
        env={
            "PYTHONUNBUFFERED": "1",
            "CI": "true"
        },
        jobs={
            "workflow-validation": Job(
                runs_on="ubuntu-latest",
                timeout_minutes=10,
                steps=[
                    Step("📥 Checkout code", uses="actions/checkout@v4"),
                    Step(
                        "🐍 Set up Python 3.11", 
                        uses="actions/setup-python@v5",
                        with_={"python-version": "3.11"}
                    ),
                    Step(
                        "📦 Install test dependencies",
                        run="python -m pip install --upgrade pip\npip install pytest pyyaml"
                    ),
                    Step(
                        "🔍 Run comprehensive workflow validation tests",
                        run="echo \"🔍 Running comprehensive workflow validation tests...\"\npython -m pytest tests/test_workflow_integration.py -v --tb=short",
                        env={
                            "PYTHONUNBUFFERED": "1",
                            "CI": "true"
                        }
                    )
                ]
            ),
            "python-tests": Job(
                runs_on="${{ matrix.os }}",
                timeout_minutes=15,
                strategy={
                    "fail-fast": False,
                    "matrix": matrix
                },
                steps=[
                    checkout_step(),
                    Step(
                        "🐍 Set up Python ${{ matrix.python-version }}",
                        uses="actions/setup-python@v5",
                        with_={
                            "python-version": "${{ matrix.python-version }}",
                            "cache": "pip"
                        }
                    ),
                    install_deps_step(),
                    Step(
                        "🧪 Run Python tests",
                        run="python -m pytest tests/ -v --tb=short -x"
                    ),
                    Step(
                        "📊 Test benchmark runners", 
                        run="timeout 30 python scripts/run_benchmarks.py --quick --output data/results/test_ci.json || echo \"⚠️ Benchmark test failed\"",
                        continue_on_error=True
                    )
                ]
            ),
            "script-validation": Job(
                runs_on="ubuntu-latest", 
                timeout_minutes=10,
                steps=[
                    checkout_step(),
                    setup_python_step("3.12"),
                    install_deps_step(),
                    Step(
                        "🔍 Validate script structure",
                        run="for script in scripts/*.py; do python -m py_compile \"$script\"; done"
                    ),
                    Step(
                        "📋 Check script imports",
                        run="python -c \"import sys; sys.path.append('scripts'); import run_benchmarks; print('✅ Scripts validated')\""
                    )
                ]
            )
        }
    )


def create_benchmark_workflow() -> Workflow:
    """Create a simplified benchmark workflow."""
    
    return Workflow(
        name="📊 Benchmark Validation",
        on={
            "workflow_dispatch": {
                "inputs": {
                    "benchmark_type": {
                        "description": "Type of benchmark to run",
                        "required": False,
                        "default": "quick",
                        "type": "choice",
                        "options": ["quick", "complete", "comprehensive"]
                    }
                }
            },
            "schedule": [{"cron": "0 2 * * *"}]  # Daily at 2 AM UTC
        },
        env={
            "PYTHONUNBUFFERED": "1"
        },
        jobs={
            "benchmark": Job(
                runs_on="ubuntu-latest",
                timeout_minutes=30,
                steps=[
                    checkout_step(),
                    setup_python_step("3.12"),
                    Step(
                        "📦 Install benchmark dependencies", 
                        run="pip install --upgrade pip\npip install datason orjson ujson msgpack-python jsonpickle\npip install -r requirements.txt"
                    ),
                    Step(
                        "📊 Run benchmarks",
                        run="BENCHMARK_TYPE=\"${{ github.event.inputs.benchmark_type || 'quick' }}\"\ntimeout 300 python scripts/run_benchmarks.py --quick --output \"data/results/ci_${BENCHMARK_TYPE}_$(date +%Y%m%d_%H%M%S).json\""
                    ),
                    Step(
                        "💾 Upload benchmark results",
                        uses="actions/upload-artifact@v4",
                        with_={
                            "name": "benchmark-results-${{ github.run_id }}",
                            "path": "data/results/ci_*.json"
                        },
                        if_="always()"
                    )
                ]
            )
        }
    )


def write_yaml(workflow: Workflow, filename: str, output_dir: Path) -> None:
    """Write workflow to YAML file with consistent formatting."""
    yaml = YAML()
    yaml.indent(mapping=2, sequence=2, offset=2)
    yaml.preserve_quotes = False
    yaml.width = 4096  # Prevent line wrapping
    yaml.map_indent = 2
    yaml.sequence_indent = 4
    
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        # Add generation notice
        f.write("# This file is generated by tools/gen_workflows.py\n")
        f.write("# DO NOT EDIT MANUALLY - Edit the Python model instead\n\n")
        
        yaml.dump(workflow.to_dict(), f)
    
    print(f"Generated {output_path}")


def main() -> None:
    """Generate all workflows."""
    workflows_dir = Path(".github/workflows")
    
    # Generate main CI workflow
    ci_workflow = create_ci_workflow()
    write_yaml(ci_workflow, "ci.yml", workflows_dir)
    
    # Generate benchmark workflow  
    benchmark_workflow = create_benchmark_workflow()
    write_yaml(benchmark_workflow, "benchmarks.yml", workflows_dir)
    
    print("\\n✅ All workflows generated successfully!")
    print("\\n📋 Next steps:")
    print("1. Review the generated YAML files")
    print("2. Test with: python -m tools.gen_workflows")
    print("3. Add pre-commit hooks for validation")
    print("4. Commit the changes")


if __name__ == "__main__":
    main()
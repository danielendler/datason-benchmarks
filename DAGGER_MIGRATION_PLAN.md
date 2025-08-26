# Dagger Migration Plan: DataSON Benchmark CI/CD

## Overview

Migrating from complex GitHub Actions YAML workflows to a hybrid approach using minimal GitHub Actions triggers with Dagger Python SDK pipelines. This addresses the YAML syntax issues we encountered while maintaining full functionality and improving testability.

## Architecture Design

### Current Problems with GitHub Actions Approach
- ❌ Complex YAML syntax errors with multi-line shell scripts
- ❌ Difficult to test locally
- ❌ Poor IDE support for YAML
- ❌ Long feedback cycles (10+ minutes for simple typos)
- ❌ Hard to debug pipeline logic

### Proposed Hybrid Architecture

```
GitHub Event → Minimal GitHub Actions YAML → Dagger Python Pipeline → Results
     ↓                      ↓                        ↓                    ↓
  push/schedule         Simple trigger        Real pipeline logic    Artifacts/Reports
```

## Implementation Plan

### Phase 1: Foundation Setup

1. **Install Dagger Python SDK**
   ```bash
   pip install dagger-io
   ```

2. **Create Dagger Module Structure**
   ```
   dagger/
   ├── __init__.py
   ├── benchmark_pipeline.py    # Main pipeline functions
   ├── daily_pipeline.py        # Daily benchmark logic
   ├── weekly_pipeline.py       # Weekly benchmark logic
   ├── utils.py                 # Shared utilities
   └── containers.py            # Container configurations
   ```

### Phase 2: Pipeline Implementation

#### Core Pipeline Functions

```python
import dagger
from dagger import dag, function, object_type
from typing import Annotated

@object_type
class BenchmarkPipeline:
    
    @function
    async def daily_benchmarks(
        self,
        source: Annotated[dagger.Directory, "Source code directory"],
        focus_area: str = "api_modes"
    ) -> str:
        """Run daily benchmark analysis with specified focus area."""
        
        # Set up Python environment
        python_container = (
            dag.container()
            .from_("python:3.12-slim")
            .with_directory("/src", source)
            .with_workdir("/src")
            .with_exec(["pip", "install", "--upgrade", "pip"])
            .with_exec(["pip", "install", "datason", "orjson", "ujson", "msgpack-python", "jsonpickle"])
        )
        
        # Run benchmark based on focus area
        timestamp = await (
            python_container
            .with_exec(["date", "+%Y%m%d_%H%M%S"])
            .stdout()
        )
        
        output_file = f"daily_{focus_area}_{timestamp.strip()}.json"
        
        result = await (
            python_container
            .with_exec([
                "python", "scripts/improved_benchmark_runner.py",
                "--suite-type", focus_area,
                "--output-dir", "data/results", 
                "--output-file", output_file,
                "--generate-report"
            ])
            .with_exec([
                "python", "scripts/improved_report_generator.py",
                f"data/results/{output_file}",
                "--output-file", f"docs/results/daily_{focus_area}_{timestamp.strip()}_report.html"
            ])
            .with_exec(["python", "scripts/generate_github_pages.py"])
            .directory("/src")
        )
        
        return await result.export("./results")

    @function
    async def weekly_benchmarks(
        self,
        source: Annotated[dagger.Directory, "Source code directory"],
        benchmark_type: str = "comprehensive"
    ) -> str:
        """Run comprehensive weekly benchmark analysis."""
        
        # Enhanced test data generation
        python_container = (
            dag.container()
            .from_("python:3.12-slim")
            .with_directory("/src", source)
            .with_workdir("/src")
            .with_exec(["pip", "install", "--upgrade", "pip"])
            .with_exec(["pip", "install", "datason", "orjson", "ujson", "msgpack-python", "jsonpickle"])
            .with_exec(["pip", "install", "cbor2", "pickle5", "dill", "cloudpickle"])
        )
        
        # Generate enhanced test scenarios
        await (
            python_container
            .with_exec(["mkdir", "-p", "data/synthetic/weekly"])
            .with_exec([
                "python", "-c",
                """
import json
from pathlib import Path
version_scenarios = {
    'large_nested_structures': {
        'description': 'Complex nested data structures for version comparison',
        'data': {
            'levels': [
                {'level': i, 'data': {f'key_{j}': f'value_{j}' for j in range(10)}}
                for i in range(20)
            ]
        }
    },
    'high_frequency_serialization': {
        'description': 'Repetitive serialization tasks',
        'data': [{'id': i, 'payload': f'data_{i}' * 100} for i in range(100)]
    }
}
Path('data/synthetic/weekly').mkdir(exist_ok=True)
json.dump(version_scenarios, open('data/synthetic/weekly/version_comparison_data.json', 'w'), indent=2)
print('✅ Enhanced test data generated')
                """
            ])
            .stdout()
        )
        
        # Run comprehensive benchmarks
        timestamp = await (
            python_container
            .with_exec(["date", "+%Y%m%d_%H%M%S"])
            .stdout()
        )
        
        output_file = f"weekly_comprehensive_{timestamp.strip()}.json"
        
        result = await (
            python_container
            .with_exec([
                "python", "scripts/improved_benchmark_runner.py",
                "--suite-type", "comprehensive",
                "--output-dir", "data/results/weekly",
                "--output-file", output_file,
                "--generate-report"
            ])
            .with_exec([
                "python", "scripts/improved_report_generator.py",
                f"data/results/weekly/{output_file}",
                "--output-file", f"docs/results/{output_file.replace('.json', '_report.html')}"
            ])
            .with_exec(["python", "scripts/generate_github_pages.py"])
            .directory("/src")
        )
        
        return await result.export("./results")

    @function 
    async def test_pipeline(
        self,
        source: Annotated[dagger.Directory, "Source code directory"]
    ) -> str:
        """Run the test suite to validate pipeline components."""
        
        python_container = (
            dag.container()
            .from_("python:3.12-slim")
            .with_directory("/src", source)
            .with_workdir("/src")
            .with_exec(["pip", "install", "--upgrade", "pip"])
            .with_exec(["pip", "install", "datason", "orjson", "ujson", "msgpack-python", "jsonpickle"])
        )
        
        # Run test suite
        test_result = await (
            python_container
            .with_exec(["python", "-m", "pytest", "tests/test_improved_reporting.py", "-v"])
            .with_exec(["python", "-m", "pytest", "tests/test_github_actions_integration.py", "-v"])
            .stdout()
        )
        
        return test_result
```

### Phase 3: Minimal GitHub Actions Workflows

#### Daily Benchmarks Workflow
```yaml
# .github/workflows/dagger-daily-benchmarks.yml
name: 📅 Daily Benchmarks (Dagger)

on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6:00 AM UTC
  workflow_dispatch:
    inputs:
      focus_area:
        description: 'Focus area for daily benchmarks'
        required: false
        default: 'api_modes'
        type: choice
        options: [api_modes, competitive, versions, comprehensive]

jobs:
  daily-benchmarks:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        
      - name: Run Daily Benchmarks with Dagger
        uses: dagger/dagger-for-github@v5
        with:
          version: "latest"
          verb: call
          args: >-
            daily-benchmarks 
            --source=. 
            --focus-area="${{ github.event.inputs.focus_area || 'api_modes' }}"
          cloud-token: ${{ secrets.DAGGER_CLOUD_TOKEN }}
          
      - name: Commit Results
        run: |
          git config --local user.email "github-actions[bot]@users.noreply.github.com"
          git config --local user.name "github-actions[bot]"
          git add data/results/ docs/results/
          if ! git diff --staged --quiet; then
            git commit -m "📊 Daily ${{ github.event.inputs.focus_area || 'api_modes' }} Benchmarks - $(date -u '+%Y-%m-%d')"
            git push origin HEAD
          fi
```

#### Weekly Benchmarks Workflow  
```yaml
# .github/workflows/dagger-weekly-benchmarks.yml
name: 🗓️ Weekly Benchmarks (Dagger)

on:
  schedule:
    - cron: '30 6 * * 1'  # Weekly on Monday at 6:30 AM UTC
  workflow_dispatch:
    inputs:
      benchmark_type:
        description: 'Benchmark type'
        required: false
        default: 'comprehensive'
        type: choice
        options: [comprehensive, api_modes, competitive, versions]

jobs:
  weekly-benchmarks:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        
      - name: Run Weekly Benchmarks with Dagger
        uses: dagger/dagger-for-github@v5
        with:
          version: "latest"
          verb: call
          args: >-
            weekly-benchmarks 
            --source=. 
            --benchmark-type="${{ github.event.inputs.benchmark_type || 'comprehensive' }}"
          cloud-token: ${{ secrets.DAGGER_CLOUD_TOKEN }}
          
      - name: Commit Results
        run: |
          git config --local user.email "github-actions[bot]@users.noreply.github.com"
          git config --local user.name "github-actions[bot]"
          git add data/results/ docs/results/
          if ! git diff --staged --quiet; then
            git commit -m "📊 Weekly ${{ github.event.inputs.benchmark_type || 'comprehensive' }} Benchmarks - $(date -u '+%Y-%m-%d')"
            git push origin HEAD
          fi
```

## Benefits of This Approach

### ✅ Reliability
- No complex YAML syntax issues
- Type-safe Python pipeline code
- Comprehensive error handling

### ✅ Testability  
- Run pipelines locally with `dagger call`
- Unit test pipeline components
- Quick iteration cycles

### ✅ Maintainability
- IDE support with autocomplete
- Version control for pipeline logic
- Easier debugging and profiling

### ✅ Portability
- Same pipeline code works on any CI provider
- Local development matches production
- Container-based execution ensures consistency

## Migration Steps

1. **Install Dagger SDK** and create basic pipeline structure
2. **Implement daily benchmark pipeline** in Python
3. **Implement weekly benchmark pipeline** in Python  
4. **Create minimal GitHub Actions workflows** to trigger Dagger
5. **Test locally** with `dagger call` commands
6. **Deploy and validate** in CI environment
7. **Remove old complex YAML workflows** once validated

## Local Development Commands

```bash
# Install Dagger
pip install dagger-io

# Test daily pipeline locally
dagger call daily-benchmarks --source=. --focus-area=api_modes

# Test weekly pipeline locally  
dagger call weekly-benchmarks --source=. --benchmark-type=comprehensive

# Run test suite
dagger call test-pipeline --source=.
```

## Expected Outcomes

- 🎯 **Eliminate YAML syntax errors** completely
- ⚡ **Faster development cycles** with local testing
- 🔧 **Better maintainability** with Python instead of YAML
- 📊 **Same benchmark functionality** with improved reliability
- 🧪 **Enhanced testability** of CI pipeline components

This hybrid approach maintains GitHub's native CI integration while solving the complex YAML issues through Dagger's Python SDK, providing the best of both worlds.
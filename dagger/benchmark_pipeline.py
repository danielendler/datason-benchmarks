"""
DataSON Benchmark Pipeline

Main Dagger pipeline functions for DataSON benchmark automation.
Provides daily and weekly benchmark execution with comprehensive reporting.
"""

import dagger
from dagger import dag, function, object_type
from typing import Annotated
import asyncio


@object_type
class BenchmarkPipeline:
    """DataSON Benchmark Pipeline for CI/CD automation."""
    
    @function
    async def daily_benchmarks(
        self,
        source: Annotated[dagger.Directory, "Source code directory"],
        focus_area: str = "api_modes"
    ) -> str:
        """
        Run daily benchmark analysis with specified focus area.
        
        Args:
            source: Source code directory
            focus_area: Benchmark focus (api_modes, competitive, versions, comprehensive)
            
        Returns:
            Status message of the pipeline execution
        """
        
        # Set up Python environment with all dependencies
        python_container = await self._setup_python_environment(source)
        
        # Generate timestamp for unique naming
        timestamp = await (
            python_container
            .with_exec(["date", "+%Y%m%d_%H%M%S"])
            .stdout()
        )
        timestamp = timestamp.strip()
        
        output_file = f"daily_{focus_area}_{timestamp}.json"
        report_file = f"daily_{focus_area}_{timestamp}_report.html"
        
        print(f"🎯 Running daily {focus_area} benchmarks...")
        
        # Execute benchmark runner
        benchmark_result = await (
            python_container
            .with_exec([
                "python", "scripts/improved_benchmark_runner.py",
                "--suite-type", focus_area,
                "--output-dir", "data/results", 
                "--output-file", output_file,
                "--generate-report"
            ])
            .stdout()
        )
        
        # Generate HTML report
        report_result = await (
            python_container
            .with_exec([
                "python", "scripts/improved_report_generator.py",
                f"data/results/{output_file}",
                "--output-file", f"docs/results/{report_file}"
            ])
            .stdout()
        )
        
        # Update GitHub Pages
        pages_result = await (
            python_container
            .with_exec(["python", "scripts/generate_github_pages.py"])
            .stdout()
        )
        
        # Export results back to host
        results_dir = python_container.directory("/src")
        await results_dir.export(".")
        
        return f"✅ Daily {focus_area} benchmarks completed successfully at {timestamp}"

    @function
    async def weekly_benchmarks(
        self,
        source: Annotated[dagger.Directory, "Source code directory"],
        benchmark_type: str = "comprehensive"
    ) -> str:
        """
        Run comprehensive weekly benchmark analysis.
        
        Args:
            source: Source code directory
            benchmark_type: Type of weekly benchmark (comprehensive, api_modes, competitive, versions)
            
        Returns:
            Status message of the pipeline execution
        """
        
        # Set up enhanced Python environment with additional libraries
        python_container = await self._setup_enhanced_python_environment(source)
        
        # Generate enhanced test scenarios
        print("🏗️ Generating enhanced test scenarios...")
        await (
            python_container
            .with_exec(["mkdir", "-p", "data/synthetic/weekly"])
            .with_exec([
                "python", "-c", self._get_test_data_generation_script()
            ])
            .stdout()
        )
        
        # Generate timestamp for unique naming
        timestamp = await (
            python_container
            .with_exec(["date", "+%Y%m%d_%H%M%S"])
            .stdout()
        )
        timestamp = timestamp.strip()
        
        output_file = f"weekly_{benchmark_type}_{timestamp}.json"
        report_file = f"weekly_{benchmark_type}_{timestamp}_report.html"
        
        print(f"🎯 Running weekly {benchmark_type} benchmarks...")
        
        # Execute comprehensive benchmarks
        benchmark_result = await (
            python_container
            .with_exec([
                "python", "scripts/improved_benchmark_runner.py",
                "--suite-type", benchmark_type,
                "--output-dir", "data/results/weekly",
                "--output-file", output_file,
                "--generate-report"
            ])
            .stdout()
        )
        
        # Generate HTML report
        report_result = await (
            python_container
            .with_exec([
                "python", "scripts/improved_report_generator.py",
                f"data/results/weekly/{output_file}",
                "--output-file", f"docs/results/{report_file}"
            ])
            .stdout()
        )
        
        # Update GitHub Pages
        pages_result = await (
            python_container
            .with_exec(["python", "scripts/generate_github_pages.py"])
            .stdout()
        )
        
        # Export results back to host
        results_dir = python_container.directory("/src")
        await results_dir.export(".")
        
        return f"✅ Weekly {benchmark_type} benchmarks completed successfully at {timestamp}"

    @function 
    async def test_pipeline(
        self,
        source: Annotated[dagger.Directory, "Source code directory"]
    ) -> str:
        """
        Run the test suite to validate pipeline components.
        
        Args:
            source: Source code directory
            
        Returns:
            Test results and status
        """
        
        python_container = await self._setup_python_environment(source)
        
        print("🧪 Running test suite...")
        
        # Run improved reporting tests
        test_result_1 = await (
            python_container
            .with_exec(["python", "-m", "pytest", "tests/test_improved_reporting.py", "-v"])
            .stdout()
        )
        
        # Run GitHub Actions integration tests  
        test_result_2 = await (
            python_container
            .with_exec(["python", "-m", "pytest", "tests/test_github_actions_integration.py", "-v"])
            .stdout()
        )
        
        return f"✅ Test suite completed\n{test_result_1}\n{test_result_2}"

    @function
    async def validate_system(
        self,
        source: Annotated[dagger.Directory, "Source code directory"]
    ) -> str:
        """
        Run complete end-to-end validation of the benchmark system.
        
        Args:
            source: Source code directory
            
        Returns:
            Validation results and status
        """
        
        python_container = await self._setup_python_environment(source)
        
        print("🔍 Running complete system validation...")
        
        # Run comprehensive benchmark test
        validation_result = await (
            python_container
            .with_exec([
                "python", "scripts/improved_benchmark_runner.py",
                "--suite-type", "comprehensive",
                "--output-dir", "data/results",
                "--output-file", "validation_test.json",
                "--generate-report"
            ])
            .with_exec([
                "python", "scripts/improved_report_generator.py",
                "data/results/validation_test.json",
                "--output-file", "docs/results/validation_test_report.html"
            ])
            .with_exec(["python", "scripts/generate_github_pages.py"])
            .stdout()
        )
        
        # Clean up test files
        await (
            python_container
            .with_exec(["rm", "-f", "data/results/validation_test.json"])
            .with_exec(["rm", "-f", "docs/results/validation_test_report.html"])
            .stdout()
        )
        
        return "✅ End-to-end validation completed successfully"

    async def _setup_python_environment(self, source: dagger.Directory) -> dagger.Container:
        """Set up basic Python environment with core dependencies."""
        
        return (
            dag.container()
            .from_("python:3.12-slim")
            .with_directory("/src", source)
            .with_workdir("/src")
            .with_exec(["pip", "install", "--upgrade", "pip"])
            .with_exec([
                "pip", "install", 
                "datason", "orjson", "ujson", "msgpack-python", "jsonpickle"
            ])
            .with_env_variable("PYTHONPATH", "/src")
        )

    async def _setup_enhanced_python_environment(self, source: dagger.Directory) -> dagger.Container:
        """Set up enhanced Python environment with additional competitive libraries."""
        
        container = await self._setup_python_environment(source)
        
        return (
            container
            .with_exec([
                "pip", "install", 
                "cbor2", "pickle5", "dill", "cloudpickle"
            ], use_entrypoint=True)  # Allow failures for optional libraries
        )

    def _get_test_data_generation_script(self) -> str:
        """Return the Python script for generating enhanced test data."""
        
        return """
import json
from pathlib import Path

print('🏗️ Generating enhanced test scenarios...')

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

Path('data/synthetic/weekly').mkdir(parents=True, exist_ok=True)
with open('data/synthetic/weekly/version_comparison_data.json', 'w') as f:
    json.dump(version_scenarios, f, indent=2)

print('✅ Enhanced test data generated successfully')
"""
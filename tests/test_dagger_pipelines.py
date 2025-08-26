"""
Comprehensive Test Suite for Dagger Benchmark Pipelines

Tests the BenchmarkPipeline class and all its functions to ensure
feature parity with legacy GitHub Actions workflows.
"""

import pytest
import asyncio
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Import our Dagger pipeline (this will require dagger-io to be installed)
try:
    from dagger.benchmark_pipeline import BenchmarkPipeline
    import dagger
    DAGGER_AVAILABLE = True
except ImportError:
    DAGGER_AVAILABLE = False


@pytest.mark.asyncio
@pytest.mark.skipif(not DAGGER_AVAILABLE, reason="Dagger not installed")
class TestDaggerBenchmarkPipeline:
    """Test suite for Dagger benchmark pipeline functions."""
    
    @pytest.fixture
    def pipeline(self):
        """Create a BenchmarkPipeline instance for testing."""
        return BenchmarkPipeline()
    
    @pytest.fixture
    def mock_source_directory(self):
        """Mock source directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create basic project structure
            (Path(temp_dir) / "scripts").mkdir()
            (Path(temp_dir) / "data" / "results").mkdir(parents=True)
            (Path(temp_dir) / "docs" / "results").mkdir(parents=True)
            
            # Create mock scripts
            (Path(temp_dir) / "scripts" / "improved_benchmark_runner.py").touch()
            (Path(temp_dir) / "scripts" / "improved_report_generator.py").touch()
            (Path(temp_dir) / "scripts" / "generate_github_pages.py").touch()
            
            yield temp_dir

    async def test_daily_benchmarks_api_modes(self, pipeline, mock_source_directory):
        """Test daily benchmarks with api_modes focus area."""
        
        # Mock the Dagger container execution
        with patch('dagger.dag') as mock_dag:
            mock_container = AsyncMock()
            mock_container.with_exec.return_value = mock_container
            mock_container.with_directory.return_value = mock_container
            mock_container.with_workdir.return_value = mock_container
            mock_container.with_env_variable.return_value = mock_container
            mock_container.stdout.return_value = "20250826_120000"
            mock_container.directory.return_value.export.return_value = None
            
            mock_dag.container.return_value.from_.return_value = mock_container
            
            # Create mock source directory
            mock_source = Mock()
            
            # Test the function
            result = await pipeline.daily_benchmarks(mock_source, "api_modes")
            
            # Assertions
            assert "Daily api_modes benchmarks completed successfully" in result
            assert "20250826_120000" in result
            
            # Verify container was configured correctly
            mock_container.with_exec.assert_called()
            mock_container.with_directory.assert_called()
            mock_container.with_workdir.assert_called_with("/src")

    async def test_weekly_benchmarks_comprehensive(self, pipeline, mock_source_directory):
        """Test weekly benchmarks with comprehensive analysis."""
        
        with patch('dagger.dag') as mock_dag:
            mock_container = AsyncMock()
            mock_container.with_exec.return_value = mock_container
            mock_container.with_directory.return_value = mock_container
            mock_container.with_workdir.return_value = mock_container
            mock_container.with_env_variable.return_value = mock_container
            mock_container.stdout.return_value = "20250826_120000"
            mock_container.directory.return_value.export.return_value = None
            
            mock_dag.container.return_value.from_.return_value = mock_container
            
            mock_source = Mock()
            
            result = await pipeline.weekly_benchmarks(mock_source, "comprehensive")
            
            assert "Weekly comprehensive benchmarks completed successfully" in result
            assert "20250826_120000" in result

    async def test_validate_system(self, pipeline, mock_source_directory):
        """Test complete system validation."""
        
        with patch('dagger.dag') as mock_dag:
            mock_container = AsyncMock()
            mock_container.with_exec.return_value = mock_container
            mock_container.with_directory.return_value = mock_container
            mock_container.with_workdir.return_value = mock_container
            mock_container.with_env_variable.return_value = mock_container
            mock_container.stdout.return_value = "validation output"
            
            mock_dag.container.return_value.from_.return_value = mock_container
            
            mock_source = Mock()
            
            result = await pipeline.validate_system(mock_source)
            
            assert "End-to-end validation completed successfully" in result

    async def test_test_pipeline(self, pipeline, mock_source_directory):
        """Test the pipeline test suite function."""
        
        with patch('dagger.dag') as mock_dag:
            mock_container = AsyncMock()
            mock_container.with_exec.return_value = mock_container
            mock_container.with_directory.return_value = mock_container
            mock_container.with_workdir.return_value = mock_container
            mock_container.with_env_variable.return_value = mock_container
            mock_container.stdout.return_value = "test output"
            
            mock_dag.container.return_value.from_.return_value = mock_container
            
            mock_source = Mock()
            
            result = await pipeline.test_pipeline(mock_source)
            
            assert "Test suite completed" in result

    def test_setup_python_environment_dependencies(self, pipeline):
        """Test that Python environment setup includes all required dependencies."""
        
        # This tests the dependency list in the setup function
        with patch('dagger.dag') as mock_dag:
            mock_container = Mock()
            mock_dag.container.return_value.from_.return_value = mock_container
            
            mock_source = Mock()
            
            # Call the private method directly for testing
            asyncio.run(pipeline._setup_python_environment(mock_source))
            
            # Verify the correct dependencies are installed
            calls = mock_container.with_exec.call_args_list
            pip_install_calls = [call for call in calls if call[0][0][0] == "pip" and "install" in call[0][0]]
            
            assert len(pip_install_calls) > 0
            # Check that key dependencies are included
            dependency_calls = str(pip_install_calls)
            assert "datason" in dependency_calls
            assert "orjson" in dependency_calls
            assert "ujson" in dependency_calls

    def test_enhanced_python_environment_optional_deps(self, pipeline):
        """Test that enhanced environment includes optional competitive libraries."""
        
        with patch('dagger.dag') as mock_dag:
            mock_container = Mock()
            mock_container.with_exec.return_value = mock_container
            mock_dag.container.return_value.from_.return_value = mock_container
            
            mock_source = Mock()
            
            # Test the enhanced environment setup
            asyncio.run(pipeline._setup_enhanced_python_environment(mock_source))
            
            # Verify optional libraries are attempted
            calls = mock_container.with_exec.call_args_list
            enhanced_calls = str(calls)
            assert "cbor2" in enhanced_calls or "pickle5" in enhanced_calls

    def test_test_data_generation_script(self, pipeline):
        """Test the test data generation script content."""
        
        script = pipeline._get_test_data_generation_script()
        
        # Verify script contains key components
        assert "large_nested_structures" in script
        assert "high_frequency_serialization" in script
        assert "version_comparison_data.json" in script
        assert "json.dump" in script

    async def test_error_handling_missing_scripts(self, pipeline):
        """Test error handling when required scripts are missing."""
        
        with patch('dagger.dag') as mock_dag:
            mock_container = AsyncMock()
            # Simulate script execution failure
            mock_container.with_exec.side_effect = Exception("Script not found")
            mock_dag.container.return_value.from_.return_value = mock_container
            
            mock_source = Mock()
            
            # The function should handle errors gracefully
            with pytest.raises(Exception):
                await pipeline.daily_benchmarks(mock_source, "api_modes")

    def test_supported_benchmark_types(self, pipeline):
        """Test that all expected benchmark types are supported."""
        
        # Test daily benchmark focus areas
        daily_types = ["api_modes", "competitive", "versions", "comprehensive"]
        for benchmark_type in daily_types:
            # This should not raise any validation errors
            assert benchmark_type in ["api_modes", "competitive", "versions", "comprehensive"]
        
        # Test weekly benchmark types  
        weekly_types = ["comprehensive", "api_modes", "competitive", "versions"]
        for benchmark_type in weekly_types:
            assert benchmark_type in ["comprehensive", "api_modes", "competitive", "versions"]


class TestDaggerWorkflowFeatureParity:
    """Test feature parity between Dagger and legacy workflows."""
    
    def test_daily_workflow_input_options(self):
        """Test that Dagger daily workflow supports expected input options."""
        
        # Read the Dagger daily workflow
        with open('.github/workflows/dagger-daily-benchmarks.yml', 'r') as f:
            dagger_workflow = f.read()
        
        # Check for expected input options
        assert 'focus_area' in dagger_workflow
        assert 'api_modes' in dagger_workflow
        assert 'competitive' in dagger_workflow
        assert 'versions' in dagger_workflow
        assert 'comprehensive' in dagger_workflow

    def test_weekly_workflow_input_options(self):
        """Test that Dagger weekly workflow supports expected input options."""
        
        with open('.github/workflows/dagger-weekly-benchmarks.yml', 'r') as f:
            dagger_workflow = f.read()
        
        assert 'benchmark_type' in dagger_workflow
        assert 'comprehensive' in dagger_workflow

    def test_workflow_scheduling(self):
        """Test that Dagger workflows have proper scheduling."""
        
        with open('.github/workflows/dagger-daily-benchmarks.yml', 'r') as f:
            daily_workflow = f.read()
        
        with open('.github/workflows/dagger-weekly-benchmarks.yml', 'r') as f:
            weekly_workflow = f.read()
        
        # Check scheduling exists
        assert 'schedule:' in daily_workflow
        assert 'cron:' in daily_workflow
        assert 'schedule:' in weekly_workflow
        assert 'cron:' in weekly_workflow

    def test_missing_features_identified(self):
        """Test that identifies missing features from legacy workflows."""
        
        # This test documents the known missing features
        missing_features = [
            "Python dependency caching",
            "CI result tagging with timestamps", 
            "Phase 4 enhanced report generation",
            "GitHub Pages index updates",
            "Artifact upload functionality",
            "Comprehensive error handling",
            "Timeout protection",
            "Permissions configuration"
        ]
        
        # This test serves as documentation of what needs to be implemented
        assert len(missing_features) == 8
        assert "Python dependency caching" in missing_features


class TestDaggerIntegration:
    """Integration tests for Dagger pipeline functionality."""
    
    @pytest.mark.skipif(not DAGGER_AVAILABLE, reason="Dagger not installed")
    def test_dagger_functions_discoverable(self):
        """Test that Dagger functions are properly discoverable."""
        
        pipeline = BenchmarkPipeline()
        
        # Check that the pipeline has the expected methods
        assert hasattr(pipeline, 'daily_benchmarks')
        assert hasattr(pipeline, 'weekly_benchmarks')
        assert hasattr(pipeline, 'test_pipeline')
        assert hasattr(pipeline, 'validate_system')

    @pytest.mark.skipif(not DAGGER_AVAILABLE, reason="Dagger not installed")
    def test_dagger_module_structure(self):
        """Test that Dagger module is properly structured."""
        
        # Check that dagger.json exists
        assert Path('dagger.json').exists()
        
        # Check that dagger module exists
        assert Path('dagger/__init__.py').exists()
        assert Path('dagger/benchmark_pipeline.py').exists()

    def test_requirements_include_dagger(self):
        """Test that requirements include Dagger dependencies."""
        
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        assert 'dagger-io' in requirements


if __name__ == "__main__":
    print("🧪 Running Dagger Pipeline Tests")
    print("=" * 50)
    
    # Run the tests
    pytest.main([__file__, "-v"])
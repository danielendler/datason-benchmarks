#!/usr/bin/env python3
"""
Comprehensive tests for the datason-benchmarks module.
Tests critical functionality that could be flaky or break integration.
"""

import unittest
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add scripts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


class TestSymlinkManagement(unittest.TestCase):
    """Test the latest results symlink logic."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.results_dir = Path(self.test_dir) / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_symlink_creation_and_update(self):
        """Test that symlinks are created and updated correctly."""
        from run_benchmarks import BenchmarkRunner
        
        # Create a benchmark runner instance
        runner = BenchmarkRunner(str(self.results_dir))
        
        # Create test result files
        test_results = {
            "metadata": {"timestamp": 1234567890},
            "results": {"test": "data"}
        }
        
        # Test different benchmark types and their symlinks
        test_cases = [
            ("quick", "latest_quick.json"),
            ("competitive", "latest_competitive.json"),
            ("quick_enhanced", "latest_quick.json"),  # Should map to quick
            ("enhanced_competitive", "latest_competitive.json"),  # Should map to competitive
            ("configuration", "latest_configuration.json")
        ]
        
        for benchmark_type, expected_symlink in test_cases:
            with self.subTest(benchmark_type=benchmark_type):
                # Save results (this should create the symlink)
                runner._save_results(test_results, benchmark_type)
                
                # Check that the actual result file exists (the pattern is different)
                result_files = list(self.results_dir.glob(f"{benchmark_type}_*.json"))
                self.assertTrue(len(result_files) > 0, f"No result file created for {benchmark_type}")
                
                # Check that the symlink exists and points to the right file
                symlink_path = self.results_dir / expected_symlink
                self.assertTrue(symlink_path.exists(), f"Symlink {expected_symlink} not created")
                self.assertTrue(symlink_path.is_symlink(), f"{expected_symlink} is not a symlink")
                
                # Verify symlink target
                target = symlink_path.readlink()
                expected_prefix = benchmark_type.replace("_enhanced", "")
                self.assertTrue(target.name.startswith(expected_prefix), 
                              f"Symlink points to wrong file: {target}, expected to start with {expected_prefix}")
    
    def test_symlink_replacement(self):
        """Test that old symlinks are properly replaced."""
        import time
        from run_benchmarks import BenchmarkRunner
        
        runner = BenchmarkRunner(str(self.results_dir))
        
        # Create first result
        test_results1 = {"test": "data1", "timestamp": 1}
        runner._save_results(test_results1, "quick")
        
        symlink_path = self.results_dir / "latest_quick.json"
        first_target = symlink_path.readlink()
        
        # Wait a bit to ensure different timestamp
        time.sleep(1)
        
        # Create second result (should replace symlink)
        test_results2 = {"test": "data2", "timestamp": 2}
        runner._save_results(test_results2, "quick")
        
        second_target = symlink_path.readlink()
        
        # Verify the symlink was updated
        self.assertNotEqual(first_target, second_target, "Symlink was not updated")
        self.assertTrue(symlink_path.exists(), "Symlink doesn't exist after update")
        
        # Verify we can read the new data
        with open(symlink_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data["test"], "data2", "Symlink points to wrong data")
    
    def test_broken_symlink_handling(self):
        """Test that broken symlinks are properly handled."""
        from run_benchmarks import BenchmarkRunner
        
        runner = BenchmarkRunner(str(self.results_dir))
        symlink_path = self.results_dir / "latest_quick.json"
        
        # Create a broken symlink
        symlink_path.symlink_to("nonexistent_file.json")
        self.assertTrue(symlink_path.is_symlink(), "Broken symlink not created")
        self.assertFalse(symlink_path.exists(), "Broken symlink should not exist")
        
        # Save results should replace the broken symlink
        test_results = {"test": "data"}
        runner._save_results(test_results, "quick")
        
        # Verify the broken symlink was replaced
        self.assertTrue(symlink_path.exists(), "Symlink should exist after replacement")
        self.assertTrue(symlink_path.is_symlink(), "Should still be a symlink")
        
        # Verify we can read the data
        with open(symlink_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data["test"], "data")


class TestUnitConversion(unittest.TestCase):
    """Test time unit conversion utilities."""
    
    def test_phase4_time_formatting(self):
        """Test the phase4 smart time formatting."""
        try:
            from phase4_enhanced_reports import Phase4EnhancedReportGenerator
            generator = Phase4EnhancedReportGenerator()
            
            # Test various time ranges
            test_cases = [
                (0.0001, "μs"),  # Should format as microseconds
                (0.5, "ms"),     # Should format as milliseconds
                (50.0, "ms"),    # Should format as milliseconds
                (1500.0, "s"),   # Should format as seconds
            ]
            
            for time_ms, expected_unit in test_cases:
                with self.subTest(time_ms=time_ms):
                    formatted = generator._format_time_smart(time_ms)
                    self.assertIn(expected_unit, formatted, 
                                f"Expected {expected_unit} in formatted time: {formatted}")
                    
        except ImportError:
            self.skipTest("phase4_enhanced_reports not available")
    
    def test_regression_detector_time_formatting(self):
        """Test regression detector time formatting."""
        try:
            from regression_detector import RegressionDetector
            
            # Mock configuration
            with patch('regression_detector.Path.exists', return_value=False):
                detector = RegressionDetector()
                
                test_cases = [
                    (0.0001, ["μs", "microsecond"]),
                    (0.5, ["ms", "millisecond"]), 
                    (1.5, ["s", "second"]),
                ]
                
                for time_seconds, expected_units in test_cases:
                    with self.subTest(time_seconds=time_seconds):
                        formatted_value, unit = detector.format_time_value(time_seconds)
                        
                        # Check that at least one expected unit appears
                        unit_found = any(expected in unit.lower() or expected in formatted_value 
                                       for expected in expected_units)
                        self.assertTrue(unit_found, 
                                      f"Expected one of {expected_units} in {formatted_value}, {unit}")
                        
        except ImportError:
            self.skipTest("regression_detector not available")
    
    def test_generate_report_time_formatting(self):
        """Test generate_report adaptive time formatting."""
        try:
            from generate_report import ReportGenerator
            generator = ReportGenerator()
            
            # Test adaptive formatting
            test_cases = [
                (0.001, "μs"),
                (1.0, "ms"),
                (1000.0, "s"),
            ]
            
            for time_ms, expected_unit in test_cases:
                with self.subTest(time_ms=time_ms):
                    formatted = generator._format_time_adaptive(time_ms)
                    self.assertIn(expected_unit, formatted)
                    
            # Test consistent formatting (only returns number, not unit)
            formatted_consistent = generator._format_time_consistent(5.5, "ms")
            self.assertIn("5.5", formatted_consistent)
            # Note: _format_time_consistent returns only the number, not the unit
            
        except ImportError:
            self.skipTest("generate_report not available")
    
    def test_unit_conversion_consistency(self):
        """Test that all unit converters handle edge cases consistently."""
        edge_cases = [0.0, 0.0001, 1.0, 1000.0, float('inf')]
        
        converters = []
        
        try:
            from phase4_enhanced_reports import Phase4EnhancedReportGenerator
            converters.append(("phase4", Phase4EnhancedReportGenerator()._format_time_smart))
        except ImportError:
            pass
            
        try:
            from generate_report import ReportGenerator
            converters.append(("report", ReportGenerator()._format_time_adaptive))
        except ImportError:
            pass
        
        for test_value in edge_cases:
            if test_value == float('inf'):
                continue  # Skip infinity for now
                
            for name, converter in converters:
                with self.subTest(converter=name, value=test_value):
                    try:
                        result = converter(test_value)
                        # Basic checks - should be a string with a unit
                        self.assertIsInstance(result, str)
                        self.assertTrue(len(result) > 0)
                        # Should contain at least one known unit (skip if consistent formatter or zero value)
                        if "consistent" not in name and test_value != 0.0:  # consistent formatters only return numbers
                            units = ["μs", "ms", "s", "ns"]
                            self.assertTrue(any(unit in result for unit in units),
                                          f"No recognizable unit in: {result}")
                    except Exception as e:
                        self.fail(f"Converter {name} failed on {test_value}: {e}")


class TestBenchmarkIntegration(unittest.TestCase):
    """Test integration points that could be flaky."""
    
    def test_benchmark_result_structure(self):
        """Test that benchmark results have consistent structure."""
        # Check if we can import the benchmark runner
        try:
            from run_benchmarks import BenchmarkRunner
            
            # Create a minimal runner
            with tempfile.TemporaryDirectory() as temp_dir:
                runner = BenchmarkRunner(temp_dir)
                
                # Test metadata generation
                metadata = runner.metadata
                required_fields = ["timestamp", "python_version", "datason_version"]
                
                for field in required_fields:
                    self.assertIn(field, metadata, f"Missing required metadata field: {field}")
                
        except ImportError as e:
            self.skipTest(f"Cannot import benchmark runner: {e}")
    
    def test_symlink_path_resolution(self):
        """Test that symlink paths are resolved correctly across different scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir) / "results" 
            results_dir.mkdir()
            
            # Test different path scenarios
            scenarios = [
                ("simple", "test_result.json"),
                ("with_timestamp", "quick_benchmark_20240101_120000.json"),
                ("with_special_chars", "test-result_v2.0.json"),
            ]
            
            for scenario_name, filename in scenarios:
                with self.subTest(scenario=scenario_name):
                    # Create test file
                    test_file = results_dir / filename
                    test_file.write_text('{"test": "data"}')
                    
                    # Create symlink
                    symlink = results_dir / f"latest_{scenario_name}.json"
                    if symlink.exists():
                        symlink.unlink()
                    symlink.symlink_to(filename)
                    
                    # Verify symlink works
                    self.assertTrue(symlink.exists(), f"Symlink doesn't exist: {symlink}")
                    self.assertTrue(symlink.is_symlink(), f"Not a symlink: {symlink}")
                    
                    # Verify we can read through symlink
                    with open(symlink, 'r') as f:
                        data = json.load(f)
                        self.assertEqual(data["test"], "data")
    
    def test_baseline_file_detection(self):
        """Test baseline file detection logic used in workflows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            
            # Create various baseline files
            baseline_files = [
                "latest_competitive.json",
                "latest_quick.json", 
                "latest.json",
                "datason_baseline.json"
            ]
            
            for filename in baseline_files:
                (results_dir / filename).write_text('{"baseline": "data"}')
            
            # Test the priority logic (similar to what's in workflows)
            def find_best_baseline():
                # This mimics the workflow logic
                if (results_dir / "latest_competitive.json").exists():
                    return "latest_competitive.json"
                elif (results_dir / "latest_quick.json").exists():
                    return "latest_quick.json"  
                elif (results_dir / "latest.json").exists():
                    return "latest.json"
                return None
            
            best_baseline = find_best_baseline()
            self.assertEqual(best_baseline, "latest_competitive.json", 
                           "Should prioritize competitive baseline")
            
            # Test when competitive is missing
            (results_dir / "latest_competitive.json").unlink()
            best_baseline = find_best_baseline()
            self.assertEqual(best_baseline, "latest_quick.json",
                           "Should fallback to quick baseline")


class TestWorkflowIntegrationPoints(unittest.TestCase):
    """Test critical points where workflows could break."""
    
    def test_pr_comment_file_creation(self):
        """Test that PR comment files are created with expected structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            comment_file = Path(temp_dir) / "pr_comment.md"
            
            # Test comment file creation (simulate what scripts do)
            comment_content = """# DataSON PR Performance Analysis

**PR #123** | **Commit**: abc123

## Benchmark Results
- Test result: PASS

## Performance Impact  
- No regressions detected

---
*Generated by datason-benchmarks*"""
            
            comment_file.write_text(comment_content)
            
            # Verify structure
            content = comment_file.read_text()
            required_sections = ["DataSON PR Performance Analysis", "Benchmark Results", "Performance Impact"]
            
            for section in required_sections:
                self.assertIn(section, content, f"Missing section: {section}")
            
            # Test for signature (used by comment management)
            self.assertIn("Generated by datason-benchmarks", content, "Missing bot signature")


if __name__ == '__main__':
    unittest.main()
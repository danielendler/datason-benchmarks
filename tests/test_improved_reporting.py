#!/usr/bin/env python3
"""
Test Suite for Improved DataSON Benchmark Reporting System
==========================================================

Tests the new benchmark runner and report generator to ensure they work correctly.
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.improved_benchmark_runner import ImprovedBenchmarkRunner, BenchmarkResult
from scripts.improved_report_generator import ImprovedReportGenerator

class TestImprovedBenchmarkRunner(unittest.TestCase):
    """Test the improved benchmark runner"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.runner = ImprovedBenchmarkRunner(output_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_benchmark_runner_initialization(self):
        """Test that benchmark runner initializes correctly"""
        self.assertIsInstance(self.runner.metadata, dict)
        self.assertIn('timestamp', self.runner.metadata)
        self.assertIn('python_version', self.runner.metadata)
        self.assertIn('datason_version', self.runner.metadata)
        self.assertEqual(self.runner.metadata['benchmark_framework'], 'improved_v1')
    
    def test_datason_methods_configuration(self):
        """Test that DataSON methods are properly configured"""
        expected_methods = ['serialize', 'dump_secure', 'save_string', 'deserialize', 
                          'load_basic', 'load_smart', 'dump_json', 'loads_json']
        
        for method in expected_methods:
            self.assertIn(method, self.runner.datason_methods)
            self.assertIn('func', self.runner.datason_methods[method])
            self.assertIn('type', self.runner.datason_methods[method])
    
    def test_competitor_configuration(self):
        """Test that competitor libraries are properly configured"""
        expected_competitors = ['orjson', 'ujson', 'json', 'pickle', 'msgpack']
        
        for competitor in expected_competitors:
            self.assertIn(competitor, self.runner.competitors)
            self.assertIn('notes', self.runner.competitors[competitor])
    
    def test_benchmark_result_structure(self):
        """Test BenchmarkResult dataclass structure"""
        result = BenchmarkResult(
            method="test_method",
            library="test_lib", 
            scenario="test_scenario",
            mean_time=0.001,
            min_time=0.0005,
            max_time=0.002,
            std_time=0.0001,
            successful_runs=5,
            error_count=0
        )
        
        result_dict = result.to_dict()
        self.assertEqual(result_dict['mean'], 0.001)
        self.assertEqual(result_dict['mean_ms'], 1.0)  # Converted to ms
        self.assertEqual(result_dict['successful_runs'], 5)
        self.assertEqual(result_dict['error_count'], 0)
    
    def test_generate_test_scenarios(self):
        """Test that test scenarios are generated correctly"""
        scenarios = self.runner.generate_test_scenarios()
        
        expected_scenarios = [
            'api_response_processing',
            'secure_data_storage', 
            'ml_model_serialization',
            'mobile_app_sync',
            'web_service_integration'
        ]
        
        for scenario in expected_scenarios:
            self.assertIn(scenario, scenarios)
            self.assertIsInstance(scenarios[scenario], dict)
    
    def test_comprehensive_suite_structure(self):
        """Test that comprehensive suite returns proper structure"""
        # This is an integration test that might be slow, so we'll mock it
        try:
            results = self.runner.run_comprehensive_suite("test_suite")
            
            # Check basic structure
            self.assertIn('suite_type', results)
            self.assertIn('metadata', results) 
            self.assertIn('scenarios', results)
            self.assertIn('timestamp', results)
            
            # Check that main sections are present (even if they error)
            self.assertIn('datason_api_comparison', results)
            self.assertIn('competitive_analysis', results)
            self.assertIn('version_evolution', results)
            
        except Exception as e:
            # If DataSON isn't available or there are import issues, 
            # we'll just check that the runner handles errors gracefully
            self.fail(f"Comprehensive suite should handle errors gracefully, but got: {e}")

class TestImprovedReportGenerator(unittest.TestCase):
    """Test the improved report generator"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ImprovedReportGenerator()
        
        # Create sample benchmark data
        self.sample_data = {
            "suite_type": "test_suite",
            "metadata": {
                "timestamp": "2025-08-26T06:49:18.258266",
                "python_version": "3.12.0",
                "datason_version": "0.12.0",
                "benchmark_framework": "improved_v1"
            },
            "scenarios": ["test_scenario"],
            "timestamp": "2025-08-26T06:49:19.014431",
            "datason_api_comparison": {
                "type": "datason_api_comparison",
                "results": {
                    "test_scenario": {
                        "serialize": {
                            "mean": 0.001,
                            "min": 0.0008,
                            "max": 0.0015,
                            "std": 0.0002,
                            "successful_runs": 5,
                            "error_count": 0,
                            "mean_ms": 1.0
                        },
                        "dump_secure": {
                            "mean": 0.002,
                            "min": 0.0018,
                            "max": 0.0025,
                            "std": 0.0003,
                            "successful_runs": 5,
                            "error_count": 0,
                            "mean_ms": 2.0
                        }
                    }
                }
            },
            "competitive_analysis": {
                "results": {
                    "test_scenario": {
                        "serialization": {
                            "datason": {
                                "mean": 0.001,
                                "successful_runs": 5,
                                "error_count": 0,
                                "mean_ms": 1.0
                            },
                            "json": {
                                "mean": 0.0005,
                                "successful_runs": 5,
                                "error_count": 0,
                                "mean_ms": 0.5
                            }
                        },
                        "deserialization": {
                            "datason": {
                                "mean": 0.0008,
                                "successful_runs": 5,
                                "error_count": 0,
                                "mean_ms": 0.8
                            }
                        },
                        "output_size": {
                            "datason": {
                                "size": 100,
                                "size_type": "string chars"
                            }
                        }
                    }
                }
            },
            "version_evolution": {
                "current_version": "0.12.0",
                "baseline_performance": {
                    "test_scenario": {
                        "serialize": {
                            "mean": 0.001,
                            "successful_runs": 5
                        },
                        "deserialize": {
                            "mean": 0.0008,
                            "successful_runs": 5
                        },
                        "output_size": 100
                    }
                }
            }
        }
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_time_formatting(self):
        """Test time formatting with appropriate units"""
        # Test microseconds
        self.assertEqual(self.generator.format_time(0.000001), "1.0μs")
        self.assertEqual(self.generator.format_time(0.000500), "500.0μs")
        
        # Test milliseconds
        self.assertEqual(self.generator.format_time(0.001), "1.0ms")
        self.assertEqual(self.generator.format_time(0.0156), "15.6ms")
        
        # Test seconds
        self.assertEqual(self.generator.format_time(1.5), "1.50s")
        
        # Test zero
        self.assertEqual(self.generator.format_time(0), "0ms")
    
    def test_performance_class_assignment(self):
        """Test performance class assignment for color coding"""
        self.assertEqual(self.generator.get_performance_class(0.0005), "fast")  # < 1ms
        self.assertEqual(self.generator.get_performance_class(0.005), "medium")   # < 10ms
        self.assertEqual(self.generator.get_performance_class(0.05), "slow")     # > 10ms
        self.assertEqual(self.generator.get_performance_class(0), "error")       # 0
    
    def test_datason_api_matrix_generation(self):
        """Test DataSON API matrix HTML generation"""
        html = self.generator.generate_datason_api_matrix(self.sample_data)
        
        # Check that it contains expected elements
        self.assertIn("DataSON API Performance Matrix", html)
        self.assertIn("performance-table", html)
        self.assertIn("Test Scenario", html)  # Title case display name
        self.assertIn("serialize", html)
        self.assertIn("dump_secure", html)
        self.assertIn("1.0ms", html)  # serialize time
        self.assertIn("2.0ms", html)  # dump_secure time
    
    def test_competitive_analysis_generation(self):
        """Test competitive analysis HTML generation"""
        html = self.generator.generate_competitive_analysis(self.sample_data)
        
        # Check that it contains expected elements
        self.assertIn("Competitive Analysis", html)
        self.assertIn("comparison-card", html)
        self.assertIn("Serialization Performance", html)
        self.assertIn("Deserialization Performance", html)
        self.assertIn("datason", html)
        self.assertIn("json", html)
        self.assertIn("performance-bar", html)
    
    def test_version_evolution_generation(self):
        """Test version evolution HTML generation"""
        html = self.generator.generate_version_evolution(self.sample_data)
        
        # Check that it contains expected elements
        self.assertIn("Version Evolution Tracking", html)
        self.assertIn("v0.12.0", html)
        self.assertIn("summary-stats", html)
        self.assertIn("Scenarios Tested", html)
        self.assertIn("Avg Serialize Time", html)
    
    def test_full_report_generation(self):
        """Test complete HTML report generation"""
        output_file = Path(self.temp_dir) / "test_report.html"
        
        self.generator.generate_report(self.sample_data, output_file)
        
        # Check that file was created
        self.assertTrue(output_file.exists())
        
        # Check file content
        with open(output_file, 'r') as f:
            html_content = f.read()
        
        # Check for essential HTML structure
        self.assertIn("<!DOCTYPE html>", html_content)
        self.assertIn("<title>DataSON Benchmark Report", html_content)
        self.assertIn("DataSON API Performance Matrix", html_content)
        self.assertIn("Competitive Analysis", html_content)
        self.assertIn("Version Evolution Tracking", html_content)
        
        # Check for CSS styling
        self.assertIn("performance-table", html_content)
        self.assertIn("comparison-card", html_content)
        
        # Check metadata is displayed
        self.assertIn("Python 3.12.0", html_content)
        self.assertIn("v0.12.0", html_content)
    
    def test_error_handling_missing_data(self):
        """Test report generation handles missing data gracefully"""
        # Test with minimal data
        minimal_data = {
            "suite_type": "test",
            "metadata": {},
            "scenarios": []
        }
        
        # Should not crash with missing sections
        api_html = self.generator.generate_datason_api_matrix(minimal_data)
        self.assertIn("error-message", api_html)
        
        comp_html = self.generator.generate_competitive_analysis(minimal_data)
        self.assertIn("error-message", comp_html)
        
        version_html = self.generator.generate_version_evolution(minimal_data)
        self.assertIn("error-message", version_html)

class TestReportingIntegration(unittest.TestCase):
    """Integration tests for the complete reporting workflow"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up integration test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from benchmark to report"""
        try:
            # Step 1: Run benchmark
            runner = ImprovedBenchmarkRunner(output_dir=self.temp_dir)
            
            # Use a minimal test scenario to avoid DataSON import issues in tests
            test_scenarios = {
                "simple_test": {
                    "data": {"test": "value"},
                    "number": 42
                }
            }
            
            # Mock the comprehensive suite to avoid DataSON dependency
            mock_results = {
                "suite_type": "integration_test",
                "metadata": runner.metadata,
                "scenarios": ["simple_test"],
                "timestamp": "2025-08-26T10:00:00",
                "datason_api_comparison": {
                    "type": "datason_api_comparison",
                    "results": {
                        "simple_test": {
                            "serialize": {
                                "mean": 0.001,
                                "successful_runs": 5,
                                "error_count": 0,
                                "mean_ms": 1.0
                            }
                        }
                    }
                },
                "competitive_analysis": {
                    "results": {
                        "simple_test": {
                            "serialization": {
                                "datason": {"mean": 0.001, "successful_runs": 5, "error_count": 0}
                            }
                        }
                    }
                },
                "version_evolution": {
                    "current_version": "test",
                    "baseline_performance": {
                        "simple_test": {
                            "serialize": {"mean": 0.001, "successful_runs": 5}
                        }
                    }
                }
            }
            
            # Step 2: Save results to JSON
            results_file = Path(self.temp_dir) / "integration_test_results.json"
            with open(results_file, 'w') as f:
                json.dump(mock_results, f, indent=2)
            
            # Step 3: Generate report
            generator = ImprovedReportGenerator()
            report_file = Path(self.temp_dir) / "integration_test_report.html"
            
            generator.generate_report(mock_results, report_file)
            
            # Step 4: Verify report was generated
            self.assertTrue(report_file.exists())
            
            # Step 5: Verify report content
            with open(report_file, 'r') as f:
                report_content = f.read()
            
            self.assertIn("DataSON Benchmark Report", report_content)
            # Check for integration_test in either lowercase or title case
            self.assertTrue(
                "integration_test" in report_content.lower() or "Integration_Test" in report_content,
                "Expected integration_test content not found in report"
            )
            self.assertIn("1.0ms", report_content)  # Performance data
            self.assertIn("Simple Test", report_content)  # Check scenario title case
            
            print(f"✅ Integration test successful - report generated at {report_file}")
            
        except ImportError as e:
            # Skip test if dependencies are missing
            self.skipTest(f"Skipping integration test due to missing dependencies: {e}")
        except Exception as e:
            self.fail(f"Integration test failed: {e}")

class TestReportingValidation(unittest.TestCase):
    """Validation tests for report quality and correctness"""
    
    def test_html_validity_basic_structure(self):
        """Test that generated HTML has valid basic structure"""
        generator = ImprovedReportGenerator()
        
        sample_data = {
            "suite_type": "validation",
            "metadata": {"python_version": "3.12.0", "datason_version": "0.12.0"},
            "scenarios": ["test"],
            "timestamp": "2025-08-26T10:00:00"
        }
        
        temp_file = Path(tempfile.mkdtemp()) / "validation.html"
        generator.generate_report(sample_data, temp_file)
        
        with open(temp_file, 'r') as f:
            html = f.read()
        
        # Basic HTML5 structure validation
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("<html lang=\"en\">", html)
        self.assertIn("<head>", html)
        self.assertIn("<meta charset=\"UTF-8\">", html)
        self.assertIn("<title>", html)
        self.assertIn("</title>", html)
        self.assertIn("<body>", html)
        self.assertIn("</body>", html)
        self.assertIn("</html>", html)
        
        # CSS and responsive design
        self.assertIn("viewport", html)
        self.assertIn("@media", html)  # Responsive CSS
        
        # Clean up
        temp_file.unlink()
        temp_file.parent.rmdir()

def create_test_suite():
    """Create a complete test suite for the improved reporting system"""
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestImprovedBenchmarkRunner,
        TestImprovedReportGenerator,
        TestReportingIntegration,
        TestReportingValidation
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite

if __name__ == '__main__':
    # Run the complete test suite
    print("🧪 Running DataSON Improved Reporting System Tests")
    print("=" * 60)
    
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All tests passed! Improved reporting system is working correctly.")
        exit_code = 0
    else:
        print(f"❌ Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        exit_code = 1
    
    print(f"📊 Test Summary: {result.testsRun} tests run")
    sys.exit(exit_code)
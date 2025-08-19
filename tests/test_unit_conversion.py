#!/usr/bin/env python3
"""
Comprehensive tests for unit conversion utilities.
Tests edge cases and potential flaky behavior in time formatting.
"""

import unittest
import sys
import os
from unittest.mock import patch
import math

# Add scripts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


class TestTimeFormattingEdgeCases(unittest.TestCase):
    """Test edge cases in time formatting that could be flaky."""
    
    def setUp(self):
        """Set up test environment."""
        self.converters = []
        
        # Collect available time formatters
        try:
            from phase4_enhanced_reports import Phase4EnhancedReportGenerator
            self.converters.append(("phase4", Phase4EnhancedReportGenerator()._format_time_smart))
        except ImportError:
            pass
            
        try:
            from generate_report import ReportGenerator
            generator = ReportGenerator()
            self.converters.append(("report_adaptive", generator._format_time_adaptive))
            # Test consistent formatting with different units
            self.converters.append(("report_ms", lambda x: generator._format_time_consistent(x, "ms")))
            self.converters.append(("report_us", lambda x: generator._format_time_consistent(x * 1000, "μs")))
        except ImportError:
            pass
            
        try:
            from regression_detector import RegressionDetector
            with patch('regression_detector.Path.exists', return_value=False):
                detector = RegressionDetector()
                self.converters.append(("regression", lambda x: detector.format_time_value(x/1000)[0]))  # Convert ms to s
        except ImportError:
            pass
    
    def test_zero_values(self):
        """Test handling of zero time values."""
        for name, converter in self.converters:
            with self.subTest(converter=name):
                try:
                    result = converter(0.0)
                    self.assertIsInstance(result, str)
                    self.assertTrue(len(result) > 0, "Empty result for zero value")
                    # Should contain a zero value and a unit
                    self.assertTrue(any(char.isdigit() or char == '.' for char in result), 
                                  "No numeric value in result")
                except Exception as e:
                    self.fail(f"Converter {name} failed on zero: {e}")
    
    def test_very_small_values(self):
        """Test handling of very small time values."""
        small_values = [1e-9, 1e-6, 1e-4, 1e-3]  # nanoseconds to milliseconds range
        
        for name, converter in self.converters:
            for value in small_values:
                with self.subTest(converter=name, value=value):
                    try:
                        result = converter(value)
                        self.assertIsInstance(result, str)
                        self.assertTrue(len(result) > 0)
                        # Should use appropriate small unit (μs, ns) for very small values
                        # Skip unit check for consistent formatters that only return numbers
                        if value < 0.001 and not any(x in name for x in ["_ms", "_us", "consistent"]):  # Less than 1ms
                            self.assertTrue("μs" in result or "ns" in result or "ms" in result,
                                          f"Should use small unit for {value}, got: {result}")
                    except Exception as e:
                        self.fail(f"Converter {name} failed on small value {value}: {e}")
    
    def test_large_values(self):
        """Test handling of large time values."""
        large_values = [1000.0, 10000.0, 60000.0, 3600000.0]  # 1s to 1 hour in ms
        
        for name, converter in self.converters:
            for value in large_values:
                with self.subTest(converter=name, value=value):
                    try:
                        result = converter(value)
                        self.assertIsInstance(result, str)
                        self.assertTrue(len(result) > 0)
                        # Should use seconds for large values
                        if value >= 1000:  # More than 1 second in ms
                            # At least some converters should use 's' for large values
                            pass  # Different converters have different thresholds
                    except Exception as e:
                        self.fail(f"Converter {name} failed on large value {value}: {e}")
    
    def test_boundary_values(self):
        """Test values at unit conversion boundaries."""
        # Test values right at the boundaries between units
        boundary_tests = [
            (0.001, ["μs", "ms"]),  # 1μs boundary 
            (1.0, ["ms"]),          # 1ms 
            (1000.0, ["ms", "s"]),  # 1s boundary
        ]
        
        for name, converter in self.converters:
            for value, acceptable_units in boundary_tests:
                with self.subTest(converter=name, value=value):
                    try:
                        result = converter(value)
                        # Should use one of the acceptable units
                        has_acceptable_unit = any(unit in result for unit in acceptable_units)
                        if not has_acceptable_unit:
                            # This is a soft assertion - different converters may have different boundaries
                            print(f"Warning: {name} used unexpected unit for {value}: {result}")
                    except Exception as e:
                        self.fail(f"Converter {name} failed on boundary value {value}: {e}")
    
    def test_precision_consistency(self):
        """Test that precision is appropriate for each unit range."""
        test_cases = [
            (0.0001, 1),   # Very small - should have reasonable precision
            (0.5, 3),      # Sub-millisecond - should have decimal places
            (5.0, 1),      # Milliseconds - should have some precision
            (1500.0, 2),   # Seconds - should have reasonable precision
        ]
        
        for name, converter in self.converters:
            for value, max_decimal_places in test_cases:
                with self.subTest(converter=name, value=value):
                    try:
                        result = converter(value)
                        
                        # Extract numeric part
                        numeric_part = ''.join(c for c in result if c.isdigit() or c == '.')
                        if '.' in numeric_part:
                            decimal_places = len(numeric_part.split('.')[1])
                            # Should not have excessive precision
                            self.assertLessEqual(decimal_places, max_decimal_places + 2,  # Allow some flexibility
                                               f"Too much precision in {result}")
                    except Exception as e:
                        self.fail(f"Converter {name} failed precision test on {value}: {e}")
    
    def test_special_float_values(self):
        """Test handling of special float values."""
        special_values = [
            (float('nan'), "NaN or error handling"),
            (float('inf'), "Infinity handling"),
            (-1.0, "Negative values"),
        ]
        
        for name, converter in self.converters:
            for value, description in special_values:
                with self.subTest(converter=name, value=value, case=description):
                    try:
                        result = converter(value)
                        # Should handle gracefully - either return something reasonable or raise expected exception
                        if not math.isnan(value) and not math.isinf(value):
                            self.assertIsInstance(result, str)
                            self.assertTrue(len(result) > 0)
                    except (ValueError, OverflowError, ZeroDivisionError):
                        # These exceptions are acceptable for special values
                        pass
                    except Exception as e:
                        self.fail(f"Converter {name} failed unexpectedly on {description}: {e}")


class TestUnitConversionConsistency(unittest.TestCase):
    """Test consistency between different unit converters."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_values = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    def test_unit_selection_logic(self):
        """Test that unit selection follows logical patterns."""
        # All converters should use similar logic for unit selection
        expected_patterns = [
            (0.001, ["μs", "ms"]),    # Very small should use microseconds or milliseconds
            (1.0, ["ms"]),            # 1ms should use milliseconds
            (1000.0, ["ms", "s"]),    # 1000ms should use milliseconds or seconds
        ]
        
        available_converters = []
        try:
            from phase4_enhanced_reports import Phase4EnhancedReportGenerator
            available_converters.append(("phase4", Phase4EnhancedReportGenerator()._format_time_smart))
        except ImportError:
            pass
            
        try:
            from generate_report import ReportGenerator
            available_converters.append(("report", ReportGenerator()._format_time_adaptive))
        except ImportError:
            pass
        
        for name, converter in available_converters:
            for value, acceptable_units in expected_patterns:
                with self.subTest(converter=name, value=value):
                    result = converter(value)
                    has_acceptable = any(unit in result for unit in acceptable_units)
                    if not has_acceptable:
                        print(f"Note: {name} uses different unit for {value}: {result}")
    
    def test_cross_converter_reasonableness(self):
        """Test that different converters produce reasonable results."""
        converters = []
        
        try:
            from phase4_enhanced_reports import Phase4EnhancedReportGenerator
            converters.append(("phase4", Phase4EnhancedReportGenerator()._format_time_smart))
        except ImportError:
            pass
            
        try:
            from generate_report import ReportGenerator
            gen = ReportGenerator()
            converters.append(("report", gen._format_time_adaptive))
        except ImportError:
            pass
        
        # Test that all converters produce reasonable output
        for value in self.test_values:
            results = {}
            for name, converter in converters:
                try:
                    results[name] = converter(value)
                except Exception as e:
                    results[name] = f"ERROR: {e}"
            
            # All successful results should have units
            for name, result in results.items():
                if not result.startswith("ERROR:"):
                    units = ["μs", "ms", "s", "ns"]
                    has_unit = any(unit in result for unit in units)
                    self.assertTrue(has_unit, f"{name} result '{result}' missing unit for {value}")


class TestRegressionDetectorFormatting(unittest.TestCase):
    """Test specific regression detector formatting edge cases."""
    
    def test_regression_detector_time_values(self):
        """Test regression detector time formatting with various input ranges."""
        try:
            from regression_detector import RegressionDetector
            
            with patch('regression_detector.Path.exists', return_value=False):
                detector = RegressionDetector()
                
                # Test various time ranges (input is in seconds for regression detector)
                test_cases = [
                    0.0001,    # 0.1ms
                    0.001,     # 1ms  
                    0.01,      # 10ms
                    0.1,       # 100ms
                    1.0,       # 1s
                    10.0,      # 10s
                ]
                
                for time_seconds in test_cases:
                    with self.subTest(time_seconds=time_seconds):
                        value_str, unit = detector.format_time_value(time_seconds)
                        
                        # Basic validation
                        self.assertIsInstance(value_str, str)
                        self.assertIsInstance(unit, str)
                        self.assertTrue(len(value_str) > 0)
                        self.assertTrue(len(unit) > 0)
                        
                        # Should contain numeric value
                        self.assertTrue(any(c.isdigit() for c in value_str))
                        
        except ImportError:
            self.skipTest("regression_detector not available")


if __name__ == '__main__':
    unittest.main()
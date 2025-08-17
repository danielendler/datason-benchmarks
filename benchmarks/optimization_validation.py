#!/usr/bin/env python3
"""
Optimization Validation Benchmark Suite
=======================================

This benchmark suite specifically tests the performance optimizations implemented
in the performance/investigate-critical-bottlenecks branch:

1. Homogeneous array optimization
2. Nested structure optimization  
3. String interning for repeated values
4. Reduced profiling overhead

These tests are designed to detect performance regressions in the optimization
code paths and validate that the improvements are working as expected.
"""

import time
import statistics
from typing import Dict, List, Any
import json


def time_operation(func, *args, **kwargs):
    """Time a function call and return duration in milliseconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return (end - start) * 1000, result


class OptimizationValidationSuite:
    """Test suite for validation optimization performance improvements."""

    def __init__(self):
        try:
            import datason
            self.datason = datason
            self.available = True
        except ImportError:
            self.available = False

    def generate_homogeneous_array_data(self, size: int) -> List[Dict[str, Any]]:
        """Generate data that should trigger homogeneous array optimization."""
        return [
            {
                "id": i,
                "name": f"user_{i}",
                "active": True,
                "score": i * 1.5,
                "category": "standard"
            }
            for i in range(size)
        ]

    def generate_nested_structure_data(self) -> Dict[str, Any]:
        """Generate data that should trigger nested structure optimization."""
        return {
            "users": [
                {"id": i, "name": f"user_{i}", "role": "member"}
                for i in range(50)
            ],
            "metadata": {
                "total": 50,
                "status": "active",
                "version": "1.0",
                "config": {
                    "max_users": 1000,
                    "timeout": 30,
                    "debug": False
                }
            },
            "api_settings": {
                "endpoint": "https://api.example.com",
                "auth_method": "bearer",
                "rate_limit": 1000,
                "retry_attempts": 3
            }
        }

    def generate_string_interning_data(self, size: int) -> List[Dict[str, Any]]:
        """Generate data with repeated strings that should benefit from interning."""
        return [
            {
                "id": i,
                "status": "active",        # Repeated string
                "type": "user",           # Repeated string
                "plan": "premium",        # Repeated string
                "region": "us-east-1" if i < size // 2 else "us-west-2",  # Semi-repeated
                "tags": ["important", "verified", "premium"],  # Repeated values
                "unique_field": f"unique_value_{i}"
            }
            for i in range(size)
        ]

    def generate_mixed_optimization_data(self) -> Dict[str, Any]:
        """Generate data that should trigger multiple optimizations."""
        return {
            "homogeneous_users": [
                {"id": i, "name": f"user_{i}", "active": True}
                for i in range(100)
            ],
            "homogeneous_products": [
                {"sku": f"PROD-{i:04d}", "price": 19.99, "available": True}
                for i in range(75)
            ],
            "metadata": {
                "api_version": "v2.1",
                "status": "success",
                "timestamp": "2025-01-15T10:30:00Z",
                "request_id": "req_12345",
                "config": {
                    "cache_enabled": True,
                    "timeout": 5000,
                    "retry_policy": "exponential"
                }
            },
            "repeated_config": {
                "environment": "production",  # Repeated across different sections
                "debug": False,               # Repeated
                "log_level": "info"           # Repeated
            }
        }

    def benchmark_homogeneous_arrays(self) -> Dict[str, Any]:
        """Benchmark homogeneous array optimization."""
        if not self.available:
            return {"error": "DataSON not available"}

        results = {}
        
        # Test different array sizes to see scaling
        for size in [10, 50, 100, 200, 500]:
            data = self.generate_homogeneous_array_data(size)
            
            # Run multiple iterations for statistical accuracy
            durations = []
            for _ in range(5):
                duration_ms, json_str = time_operation(self.datason.save_string, data)
                loaded = self.datason.load_basic(json_str)
                assert loaded == data  # Correctness check
                durations.append(duration_ms)
            
            results[f"homogeneous_array_{size}"] = {
                "size": size,
                "mean_ms": statistics.mean(durations),
                "median_ms": statistics.median(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "json_size": len(json_str),
                "optimization_expected": True,
                "performance_target_ms": size * 0.02 + 1.0  # 0.02ms per item + 1ms overhead
            }

        return results

    def benchmark_nested_structures(self) -> Dict[str, Any]:
        """Benchmark nested structure optimization."""
        if not self.available:
            return {"error": "DataSON not available"}

        data = self.generate_nested_structure_data()
        
        # Run multiple iterations
        durations = []
        for _ in range(10):
            duration_ms, json_str = time_operation(self.datason.save_string, data)
            loaded = self.datason.load_basic(json_str)
            assert loaded == data
            durations.append(duration_ms)

        return {
            "nested_structure": {
                "mean_ms": statistics.mean(durations),
                "median_ms": statistics.median(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "json_size": len(json_str),
                "optimization_expected": True,
                "performance_target_ms": 3.0  # Should complete in under 3ms
            }
        }

    def benchmark_string_interning(self) -> Dict[str, Any]:
        """Benchmark string interning optimization."""
        if not self.available:
            return {"error": "DataSON not available"}

        results = {}
        
        # Test different sizes to see string interning benefits
        for size in [25, 50, 100, 200]:
            data = self.generate_string_interning_data(size)
            
            durations = []
            for _ in range(5):
                duration_ms, json_str = time_operation(self.datason.save_string, data)
                loaded = self.datason.load_basic(json_str)
                assert loaded == data
                durations.append(duration_ms)

            results[f"string_interning_{size}"] = {
                "size": size,
                "mean_ms": statistics.mean(durations),
                "median_ms": statistics.median(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "json_size": len(json_str),
                "optimization_expected": True,
                "performance_target_ms": size * 0.03 + 1.5  # Should benefit from string interning
            }

        return results

    def benchmark_mixed_optimizations(self) -> Dict[str, Any]:
        """Benchmark data that should trigger multiple optimizations."""
        if not self.available:
            return {"error": "DataSON not available"}

        data = self.generate_mixed_optimization_data()
        
        durations = []
        for _ in range(10):
            duration_ms, json_str = time_operation(self.datason.save_string, data)
            loaded = self.datason.load_basic(json_str)
            assert loaded == data
            durations.append(duration_ms)

        return {
            "mixed_optimizations": {
                "mean_ms": statistics.mean(durations),
                "median_ms": statistics.median(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "json_size": len(json_str),
                "optimization_expected": True,
                "performance_target_ms": 5.0  # Complex data but should be optimized
            }
        }

    def benchmark_profiling_overhead(self) -> Dict[str, Any]:
        """Benchmark profiling system overhead."""
        if not self.available:
            return {"error": "DataSON not available"}

        # Test data
        data = self.generate_homogeneous_array_data(100)

        # Test without profiling
        import os
        os.environ.pop('DATASON_PROFILE', None)
        
        durations_no_profile = []
        for _ in range(10):
            duration_ms, json_str = time_operation(self.datason.save_string, data)
            durations_no_profile.append(duration_ms)

        # Test with profiling enabled
        os.environ['DATASON_PROFILE'] = '1'
        self.datason.profile_sink = []
        
        durations_with_profile = []
        for _ in range(10):
            duration_ms, json_str = time_operation(self.datason.save_string, data)
            durations_with_profile.append(duration_ms)

        # Calculate overhead
        mean_no_profile = statistics.mean(durations_no_profile)
        mean_with_profile = statistics.mean(durations_with_profile)
        overhead_pct = ((mean_with_profile - mean_no_profile) / mean_no_profile) * 100

        return {
            "profiling_overhead": {
                "mean_ms_no_profile": mean_no_profile,
                "mean_ms_with_profile": mean_with_profile,
                "overhead_percentage": overhead_pct,
                "optimization_expected": True,
                "performance_target_overhead_pct": 30.0,  # Should be under 30% overhead
                "profile_events_captured": len(self.datason.profile_sink)
            }
        }

    def run_full_optimization_suite(self) -> Dict[str, Any]:
        """Run the complete optimization validation benchmark suite."""
        if not self.available:
            return {"error": "DataSON not available"}

        print("🚀 Running Optimization Validation Benchmark Suite...")
        
        results = {
            "suite_type": "optimization_validation",
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "datason_version": getattr(self.datason, '__version__', 'unknown'),
                "purpose": "Validate performance optimizations in critical bottlenecks branch"
            },
            "benchmarks": {}
        }

        # Run each benchmark
        print("  📊 Testing homogeneous array optimization...")
        results["benchmarks"]["homogeneous_arrays"] = self.benchmark_homogeneous_arrays()
        
        print("  📊 Testing nested structure optimization...")
        results["benchmarks"]["nested_structures"] = self.benchmark_nested_structures()
        
        print("  📊 Testing string interning optimization...")
        results["benchmarks"]["string_interning"] = self.benchmark_string_interning()
        
        print("  📊 Testing mixed optimizations...")
        results["benchmarks"]["mixed_optimizations"] = self.benchmark_mixed_optimizations()
        
        print("  📊 Testing profiling overhead...")
        results["benchmarks"]["profiling_overhead"] = self.benchmark_profiling_overhead()

        # Performance analysis
        print("  🔍 Analyzing optimization effectiveness...")
        total_tests = 0
        passed_tests = 0
        
        for benchmark_name, benchmark_data in results["benchmarks"].items():
            if isinstance(benchmark_data, dict) and "error" not in benchmark_data:
                for test_name, test_data in benchmark_data.items():
                    if isinstance(test_data, dict) and "performance_target_ms" in test_data:
                        total_tests += 1
                        if test_data["mean_ms"] <= test_data["performance_target_ms"]:
                            passed_tests += 1
                        else:
                            print(f"    ⚠️  {benchmark_name}.{test_name}: {test_data['mean_ms']:.2f}ms > {test_data['performance_target_ms']:.2f}ms target")
                    elif isinstance(test_data, dict) and "performance_target_overhead_pct" in test_data:
                        total_tests += 1
                        if test_data["overhead_percentage"] <= test_data["performance_target_overhead_pct"]:
                            passed_tests += 1
                        else:
                            print(f"    ⚠️  {benchmark_name}.{test_name}: {test_data['overhead_percentage']:.1f}% > {test_data['performance_target_overhead_pct']:.1f}% target")

        results["optimization_summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "optimization_effectiveness": (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        }

        print(f"  ✅ Optimization validation: {passed_tests}/{total_tests} tests passed ({results['optimization_summary']['optimization_effectiveness']:.1f}%)")
        
        return results


def main():
    """Run the optimization validation suite."""
    suite = OptimizationValidationSuite()
    results = suite.run_full_optimization_suite()
    
    # Save results
    output_file = f"optimization_validation_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📁 Results saved to: {output_file}")
    
    if "optimization_summary" in results:
        effectiveness = results["optimization_summary"]["optimization_effectiveness"]
        if effectiveness >= 80:
            print("🎉 Optimization validation PASSED! Performance improvements working as expected.")
        else:
            print(f"⚠️  Optimization validation PARTIAL: {effectiveness:.1f}% effectiveness")
    
    return results


if __name__ == "__main__":
    main()
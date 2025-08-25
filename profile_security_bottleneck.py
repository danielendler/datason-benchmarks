#!/usr/bin/env python3
"""
DataSON Security Performance Profiler
======================================

Deep analysis of datason_secure performance bottlenecks with nested structures.
"""

import cProfile
import io
import pstats
import time
import datason
from typing import Dict, Any, List


def create_test_data_variants() -> Dict[str, Any]:
    """Create various test data structures to isolate the performance issue."""
    return {
        # Original problematic nested structure
        "json_safe_nested": {
            "level1": {
                "level2": {
                    "level3": {
                        "data": [
                            {"id": i, "nested": {"value": i * 2}}
                            for i in range(10)
                        ]
                    }
                }
            },
            "config": {
                "database": {
                    "host": "localhost",
                    "port": 5432,
                    "credentials": {
                        "username": "user",
                        "encrypted": True
                    }
                }
            }
        },
        
        # Simplified version to test nesting impact
        "simple_nested": {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "simple"
                    }
                }
            }
        },
        
        # List-heavy version to test array impact
        "list_heavy": {
            "data": [{"id": i, "value": i * 2} for i in range(10)]
        },
        
        # Flat structure for baseline
        "flat_structure": {
            "id": 1,
            "name": "test",
            "value": 42,
            "active": True
        }
    }


def profile_method(method_name: str, method_func, data: Any, iterations: int = 100) -> Dict[str, Any]:
    """Profile a specific DataSON method and return detailed metrics."""
    print(f"\n=== Profiling {method_name} ===")
    
    # Warm up
    for _ in range(5):
        try:
            method_func(data)
        except Exception as e:
            return {"error": str(e), "method": method_name}
    
    # Time measurement
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = method_func(data)
        end = time.perf_counter()
        times.append((end - start) * 1000000)  # microseconds
    
    # Detailed profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    for _ in range(10):  # Profile fewer iterations to avoid noise
        result = method_func(data)
    
    profiler.disable()
    
    # Capture profile stats
    stats_buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_buffer)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    profile_output = stats_buffer.getvalue()
    
    return {
        "method": method_name,
        "avg_time_us": sum(times) / len(times),
        "min_time_us": min(times),
        "max_time_us": max(times),
        "std_dev_us": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
        "output_size": len(str(result)),
        "profile_stats": profile_output,
        "error": None
    }


def analyze_security_overhead():
    """Analyze the specific overhead introduced by security features."""
    print("🔍 DataSON Security Performance Analysis")
    print("=" * 50)
    
    test_data_variants = create_test_data_variants()
    
    # DataSON methods to compare
    methods = {
        "serialize": lambda d: datason.serialize(d),
        "dump_api": lambda d: datason.dump_api(d),
        "dump_fast": lambda d: datason.dump_fast(d), 
        "dump_secure": lambda d: datason.dump_secure(d),
        "dump_ml": lambda d: datason.dump_ml(d),
    }
    
    results = {}
    
    for data_name, data in test_data_variants.items():
        print(f"\n{'='*60}")
        print(f"Testing with: {data_name}")
        print(f"Data complexity: {len(str(data))} chars")
        
        data_results = {}
        baseline_time = None
        
        for method_name, method_func in methods.items():
            try:
                profile_result = profile_method(method_name, method_func, data)
                data_results[method_name] = profile_result
                
                # Use serialize as baseline
                if method_name == "serialize":
                    baseline_time = profile_result["avg_time_us"]
                
                # Calculate relative performance
                if baseline_time and profile_result["avg_time_us"]:
                    slowdown = profile_result["avg_time_us"] / baseline_time
                    print(f"{method_name:12}: {profile_result['avg_time_us']:6.1f} μs ({slowdown:4.1f}x) | Output: {profile_result['output_size']:4d} chars")
                else:
                    print(f"{method_name:12}: {profile_result['avg_time_us']:6.1f} μs | Output: {profile_result['output_size']:4d} chars")
                    
            except Exception as e:
                print(f"{method_name:12}: ERROR - {e}")
                data_results[method_name] = {"error": str(e)}
        
        results[data_name] = data_results
    
    return results


def analyze_nested_structure_impact():
    """Test how nesting depth affects dump_secure performance."""
    print(f"\n{'='*60}")
    print("📊 Nested Structure Impact Analysis")
    
    def create_nested_data(depth: int, list_size: int = 5) -> Dict[str, Any]:
        """Create nested data with specified depth and list size."""
        if depth <= 0:
            return {"data": [{"id": i, "value": i} for i in range(list_size)]}
        
        return {"level": create_nested_data(depth - 1, list_size)}
    
    # Test different nesting depths
    depths = [1, 2, 3, 4, 5]
    list_sizes = [1, 5, 10, 20]
    
    print("\nNesting Depth Impact:")
    for depth in depths:
        data = create_nested_data(depth, 5)
        
        # Test both serialize and dump_secure
        serialize_time = profile_method("serialize", datason.serialize, data, 50)["avg_time_us"]
        secure_time = profile_method("dump_secure", datason.dump_secure, data, 50)["avg_time_us"]
        
        overhead = secure_time / serialize_time if serialize_time > 0 else 0
        print(f"Depth {depth}: serialize={serialize_time:5.1f}μs, dump_secure={secure_time:5.1f}μs, overhead={overhead:4.1f}x")
    
    print("\nList Size Impact (depth=3):")
    for size in list_sizes:
        data = create_nested_data(3, size)
        
        serialize_time = profile_method("serialize", datason.serialize, data, 50)["avg_time_us"] 
        secure_time = profile_method("dump_secure", datason.dump_secure, data, 50)["avg_time_us"]
        
        overhead = secure_time / serialize_time if serialize_time > 0 else 0
        print(f"Size {size:2d}: serialize={serialize_time:5.1f}μs, dump_secure={secure_time:5.1f}μs, overhead={overhead:4.1f}x")


def main():
    """Run the complete security performance analysis."""
    print(f"DataSON Version: {getattr(datason, '__version__', 'unknown')}")
    
    # Main performance analysis
    results = analyze_security_overhead()
    
    # Nested structure specific analysis
    analyze_nested_structure_impact()
    
    # Summary of findings
    print(f"\n{'='*60}")
    print("📋 Performance Analysis Summary")
    
    # Find the worst-performing cases
    for data_name, data_results in results.items():
        if "dump_secure" in data_results and "serialize" in data_results:
            secure_result = data_results["dump_secure"]
            base_result = data_results["serialize"]
            
            if not secure_result.get("error") and not base_result.get("error"):
                overhead = secure_result["avg_time_us"] / base_result["avg_time_us"]
                size_overhead = secure_result["output_size"] / base_result["output_size"]
                
                print(f"{data_name:15}: {overhead:4.1f}x slower, {size_overhead:4.1f}x larger output")
    
    # Show detailed profile for the most problematic case
    if "json_safe_nested" in results and "dump_secure" in results["json_safe_nested"]:
        print(f"\n{'='*60}")
        print("🔬 Detailed Profile for dump_secure on json_safe_nested:")
        print(results["json_safe_nested"]["dump_secure"]["profile_stats"])


if __name__ == "__main__":
    main()
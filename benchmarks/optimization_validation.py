#!/usr/bin/env python3
"""
Optimization validation script to ensure DataSON performance improvements are consistent.

This script validates that the performance optimizations implemented in the 
performance/investigate-critical-bottlenecks branch are working correctly and
delivering the expected improvements.
"""

import os
import time
import statistics
from typing import Dict, List
import datason

# Ensure profiling is enabled
os.environ['DATASON_PROFILE'] = '1'


def benchmark_scenario(data, description: str, runs: int = 5) -> Dict:
    """Benchmark a specific data scenario multiple times"""
    
    serialize_times = []
    deserialize_times = []
    event_counts = []
    
    print(f"\n=== {description} ===")
    
    for run in range(runs):
        # Clear profile sink
        datason.profile_sink = []
        
        # Test serialization
        start_time = time.perf_counter()
        json_str = datason.save_string(data)
        serialize_time = time.perf_counter() - start_time
        
        # Count serialization events
        serialize_events = len(datason.profile_sink)
        datason.profile_sink = []
        
        # Test deserialization
        start_time = time.perf_counter()
        loaded = datason.load_basic(json_str)
        deserialize_time = time.perf_counter() - start_time
        
        # Count deserialization events
        deserialize_events = len(datason.profile_sink)
        total_events = serialize_events + deserialize_events
        
        # Validate round-trip
        if loaded != data:
            print(f"  ❌ Run {run+1}: Round-trip failed!")
            continue
            
        serialize_times.append(serialize_time * 1000)  # Convert to ms
        deserialize_times.append(deserialize_time * 1000)  # Convert to ms
        event_counts.append(total_events)
    
    if not serialize_times:
        return None
    
    # Calculate statistics
    serialize_stats = {
        'mean': statistics.mean(serialize_times),
        'median': statistics.median(serialize_times),
        'min': min(serialize_times),
        'max': max(serialize_times),
        'stdev': statistics.stdev(serialize_times) if len(serialize_times) > 1 else 0
    }
    
    deserialize_stats = {
        'mean': statistics.mean(deserialize_times),
        'median': statistics.median(deserialize_times),
        'min': min(deserialize_times),
        'max': max(deserialize_times),
        'stdev': statistics.stdev(deserialize_times) if len(deserialize_times) > 1 else 0
    }
    
    event_stats = {
        'mean': statistics.mean(event_counts),
        'median': statistics.median(event_counts),
        'min': min(event_counts),
        'max': max(event_counts)
    }
    
    print(f"  Serialization: {serialize_stats['median']:.3f}ms median ({serialize_stats['min']:.3f}-{serialize_stats['max']:.3f}ms range)")
    print(f"  Deserialization: {deserialize_stats['median']:.3f}ms median ({deserialize_stats['min']:.3f}-{deserialize_stats['max']:.3f}ms range)")
    print(f"  Events: {event_stats['median']:.0f} median ({event_stats['min']:.0f}-{event_stats['max']:.0f} range)")
    print(f"  Output size: {len(json_str):,} chars")
    
    return {
        'serialize': serialize_stats,
        'deserialize': deserialize_stats,
        'events': event_stats,
        'output_size': len(json_str),
        'runs': len(serialize_times)
    }


def main():
    print(f"DataSON Optimization Validation v{datason.__version__}")
    print("=" * 60)
    
    # Test scenarios that match benchmark data
    scenarios = [
        # API Response scenario (matches benchmark api_response)
        {
            'name': 'API Response',
            'data': {
                'status': 'success',
                'data': {
                    'users': [
                        {
                            'id': i,
                            'username': f'user_{i}',
                            'email': f'user{i}@example.com',
                            'active': i % 2 == 0,
                            'balance': i * 10.5,
                            'created_at': f'2024-01-{(i % 28) + 1:02d}T12:00:00Z'
                        }
                        for i in range(25)
                    ],
                    'pagination': {
                        'page': 1,
                        'per_page': 25,
                        'total': 1000,
                        'has_next': True,
                        'has_prev': False
                    }
                },
                'timestamp': '2024-01-01T12:00:00Z'
            },
            'expected_serialize_max': 1.0,  # ms
            'expected_deserialize_max': 0.5,  # ms
            'expected_events_max': 200
        },
        
        # Simple Objects scenario (matches benchmark simple_objects)
        {
            'name': 'Simple Objects',
            'data': {
                'string_field': 'hello world',
                'int_field': 42,
                'float_field': 3.14159,
                'bool_field': True,
                'null_field': None,
                'list_field': [1, 2, 3, 'four', 5.0],
                'dict_field': {
                    'nested_string': 'nested value',
                    'nested_int': 123,
                    'nested_list': ['a', 'b', 'c']
                }
            },
            'expected_serialize_max': 0.5,  # ms
            'expected_deserialize_max': 0.1,  # ms
            'expected_events_max': 50
        },
        
        # Nested Structures scenario (matches benchmark nested_structures)
        {
            'name': 'Nested Structures',
            'data': {
                'level1': {
                    'data': 'level1_value',
                    'level2': {
                        'data': 'level2_value',
                        'items': [
                            {
                                'id': i,
                                'level3': {
                                    'data': f'level3_value_{i}',
                                    'level4': {
                                        'final_data': f'final_{i}',
                                        'metadata': {
                                            'created': f'2024-01-{i+1:02d}',
                                            'active': i % 2 == 0
                                        }
                                    }
                                }
                            }
                            for i in range(10)
                        ]
                    }
                }
            },
            'expected_serialize_max': 0.8,  # ms
            'expected_deserialize_max': 0.3,  # ms
            'expected_events_max': 150
        }
    ]
    
    results = {}
    validation_passed = True
    
    for scenario in scenarios:
        result = benchmark_scenario(scenario['data'], scenario['name'], runs=5)
        if result is None:
            print(f"  ❌ {scenario['name']}: Benchmark failed")
            validation_passed = False
            continue
            
        results[scenario['name']] = result
        
        # Validate performance expectations
        serialize_ok = result['serialize']['max'] <= scenario['expected_serialize_max']
        deserialize_ok = result['deserialize']['max'] <= scenario['expected_deserialize_max']
        events_ok = result['events']['max'] <= scenario['expected_events_max']
        
        status = "✅" if (serialize_ok and deserialize_ok and events_ok) else "⚠️"
        print(f"  {status} Performance validation: ", end="")
        
        issues = []
        if not serialize_ok:
            issues.append(f"serialize {result['serialize']['max']:.3f}ms > {scenario['expected_serialize_max']}ms")
        if not deserialize_ok:
            issues.append(f"deserialize {result['deserialize']['max']:.3f}ms > {scenario['expected_deserialize_max']}ms")  
        if not events_ok:
            issues.append(f"events {result['events']['max']} > {scenario['expected_events_max']}")
        
        if issues:
            print(f"ISSUES: {', '.join(issues)}")
            validation_passed = False
        else:
            print("PASSED")
    
    # Summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION VALIDATION SUMMARY")
    print("=" * 60)
    
    if validation_passed:
        print("✅ ALL TESTS PASSED - Optimizations are working correctly!")
        print("\nPerformance Summary:")
        for name, result in results.items():
            serialize_ms = result['serialize']['median']
            deserialize_ms = result['deserialize']['median'] 
            events = result['events']['median']
            print(f"  {name:20s}: {serialize_ms:6.3f}ms serialize, {deserialize_ms:6.3f}ms deserialize, {events:3.0f} events")
    else:
        print("❌ VALIDATION FAILED - Some tests did not meet performance expectations")
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)

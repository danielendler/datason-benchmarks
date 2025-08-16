#!/usr/bin/env python3
"""
Comprehensive profiling test for all DataSON APIs.

This script tests the performance of different DataSON APIs to understand
which ones are optimized and which need work.
"""

import time
import datason
from datetime import datetime, timezone
import uuid
import statistics
from typing import List, Dict, Any
from decimal import Decimal

def create_test_datasets():
    """Create test datasets of varying complexity."""
    datasets = {}
    
    # Simple dataset
    datasets['simple'] = {
        'name': 'Simple object',
        'data': {'test': 'value', 'number': 42, 'nested': {'key': 'value'}}
    }
    
    # Complex web API response (like benchmarks use)
    datasets['web_api'] = {
        'name': 'Web API Response',
        'data': {
            'api_response': {
                'status': 'success',
                'timestamp': datetime.now(timezone.utc),
                'request_id': str(uuid.uuid4()),
                'data': {
                    'users': [
                        {
                            'id': i,
                            'username': f'user_{i}',
                            'email': f'user{i}@example.com',
                            'created_at': datetime.now(timezone.utc),
                            'profile': {
                                'age': 25 + i,
                                'preferences': ['pref1', 'pref2', f'custom_{i}'],
                                'settings': {
                                    'notifications': True,
                                    'theme': 'dark' if i % 2 else 'light'
                                }
                            }
                        }
                        for i in range(20)
                    ]
                }
            }
        }
    }
    
    # ML dataset with numeric arrays
    datasets['ml_data'] = {
        'name': 'ML Training Data',
        'data': {
            'experiment': {
                'experiment_id': str(uuid.uuid4()),
                'hyperparameters': {
                    'learning_rate': Decimal('0.001'),
                    'batch_size': 64,
                },
                'training_data': {
                    'features': [[float(i+j*0.1) for j in range(10)] for i in range(100)],
                    'labels': [i % 2 for i in range(100)],
                }
            }
        }
    }
    
    # Financial data with Decimals
    datasets['financial'] = {
        'name': 'Financial Transaction',
        'data': {
            'transaction': {
                'transaction_id': str(uuid.uuid4()),
                'amount': Decimal('1234567.89'),
                'fees': {
                    'base_fee': Decimal('2.50'),
                    'percentage_fee': Decimal('0.0025'),
                },
                'timestamp': datetime.now(timezone.utc),
            }
        }
    }
    
    return datasets


def benchmark_api(api_name: str, serialize_func, deserialize_func, data: Any,
                  iterations: int = 10, warmup: int = 2) -> Dict[str, float]:
    """
    Benchmark a specific API with statistical robustness.

    Args:
        api_name: Name of the API being tested
        serialize_func: Serialization function
        deserialize_func: Deserialization function  
        data: Data to test with
        iterations: Number of test iterations
        warmup: Number of warmup iterations

    Returns:
        Dictionary with timing statistics
    """
    # Warmup runs (not counted)
    for _ in range(warmup):
        serialized = serialize_func(data)
        _ = deserialize_func(serialized)

    serialize_times = []
    deserialize_times = []

    # Actual benchmark runs
    for _ in range(iterations):
        # Serialization
        start = time.perf_counter()
        serialized = serialize_func(data)
        serialize_times.append((time.perf_counter() - start) * 1000)

        # Deserialization
        start = time.perf_counter()
        _ = deserialize_func(serialized)
        deserialize_times.append((time.perf_counter() - start) * 1000)

    # Remove outliers (min and max) if we have enough samples
    if len(serialize_times) > 4:
        serialize_times.remove(max(serialize_times))
        serialize_times.remove(min(serialize_times))
        deserialize_times.remove(max(deserialize_times))
        deserialize_times.remove(min(deserialize_times))

    return {
        'serialize': {
            'mean': statistics.mean(serialize_times),
            'median': statistics.median(serialize_times),
            'stdev': statistics.stdev(serialize_times) if len(serialize_times) > 1 else 0,
            'samples': len(serialize_times)
        },
        'deserialize': {
            'mean': statistics.mean(deserialize_times),
            'median': statistics.median(deserialize_times),
            'stdev': statistics.stdev(deserialize_times) if len(deserialize_times) > 1 else 0,
            'samples': len(deserialize_times)
        }
    }


def profile_with_profiling_enabled(api_name: str, serialize_func, deserialize_func,
                                  data: Any) -> Dict[str, Any]:
    """Run with profiling enabled to get stage breakdown."""
    import os, importlib
    os.environ['DATASON_PROFILE'] = '1'
    importlib.reload(datason)
    datason.profile_sink = []

    # Serialize
    datason.profile_sink.clear()
    _ = serialize_func(data)
    serialize_events = list(datason.profile_sink)

    # Deserialize
    datason.profile_sink.clear()
    serialized = serialize_func(data)
    _ = deserialize_func(serialized)
    deserialize_events = list(datason.profile_sink)

    os.environ.pop('DATASON_PROFILE', None)
    importlib.reload(datason)

    return {
        'serialize_stages': serialize_events,
        'deserialize_stages': deserialize_events
    }


def main():
    """Run comprehensive API profiling."""
    print("🚀 DataSON Comprehensive API Profiling")
    print("=" * 60)
    print(f"DataSON Version: {datason.__version__}")
    print(f"Testing with 10 iterations (removing min/max outliers)")
    print()

    # Define APIs to test
    apis = [
        # Basic/Direct APIs (what profiling demo uses)
        {
            'name': 'save_string/load_basic',
            'serialize': datason.save_string,
            'deserialize': datason.load_basic,
            'description': 'Direct basic API (profiling demo)'
        },
        # Compatibility APIs (what benchmarks use)
        {
            'name': 'dumps_json/loads_json',
            'serialize': datason.dumps_json,
            'deserialize': datason.loads_json,
            'description': 'JSON compatibility layer (benchmarks)'
        },
        # Smart APIs
        {
            'name': 'dumps/loads',
            'serialize': datason.dumps,
            'deserialize': datason.loads,
            'description': 'Smart enhanced API'
        },
        # New modern APIs
        {
            'name': 'dump/load_smart',
            'serialize': datason.dump,
            'deserialize': datason.load_smart,
            'description': 'Modern smart API'
        },
        {
            'name': 'dump_ml/load_smart',
            'serialize': datason.dump_ml,
            'deserialize': datason.load_smart,
            'description': 'ML-optimized API'
        },
        {
            'name': 'dump_api/load_basic',
            'serialize': datason.dump_api,
            'deserialize': datason.load_basic,
            'description': 'API-optimized serialization'
        },
    ]

    # Create test datasets
    datasets = create_test_datasets()

    # Results storage
    results = {}

    # Test each API with each dataset
    for api in apis:
        print(f"\n📊 Testing: {api['name']}")
        print(f"   {api['description']}")
        print("-" * 60)

        api_results = {}

        for dataset_name, dataset_info in datasets.items():
            try:
                # Run benchmark
                stats = benchmark_api(
                    api['name'],
                    api['serialize'],
                    api['deserialize'],
                    dataset_info['data'],
                    iterations=10,
                    warmup=2
                )

                api_results[dataset_name] = stats

                # Print results
                print(f"\n   {dataset_info['name']}:"
                      f"\n     Serialize:   {stats['serialize']['mean']:6.2f}ms (median: {stats['serialize']['median']:.2f}ms, stdev: {stats['serialize']['stdev']:.2f}ms)"
                      f"\n     Deserialize: {stats['deserialize']['mean']:6.2f}ms (median: {stats['deserialize']['median']:.2f}ms, stdev: {stats['deserialize']['stdev']:.2f}ms)"
                      f"\n     Ratio: {stats['deserialize']['mean']/stats['serialize']['mean']:.1f}x")
            except Exception as e:
                print(f"   ❌ Error testing {dataset_name}: {e}")
                api_results[dataset_name] = {'error': str(e)}

        results[api['name']] = api_results

    # Summary comparison
    print("\n" + "=" * 60)
    print("📈 Performance Summary (mean times in ms)")
    print("=" * 60)

    # Create comparison table
    print(f"\n{'API':<25} {'Simple':<15} {'Web API':<15} {'ML Data':<15} {'Financial':<15}")
    print("-" * 85)

    for api in apis:
        api_name = api['name']
        if api_name in results:
            row = f"{api_name:<25}"
            for dataset in ['simple', 'web_api', 'ml_data', 'financial']:
                if dataset in results[api_name] and 'error' not in results[api_name][dataset]:
                    ser = results[api_name][dataset]['serialize']['mean']
                    des = results[api_name][dataset]['deserialize']['mean']
                    row += f" {ser:.1f}/{des:.1f}ms".ljust(15)
                else:
                    row += " ERROR".ljust(15)
            print(row)

    print("\nFormat: serialize/deserialize times")

    # Profile stage analysis for key APIs
    print("\n" + "=" * 60)
    print("🔍 Stage-level Profiling Analysis")
    print("=" * 60)

    # Profile the benchmark API with complex data
    profile_data = profile_with_profiling_enabled(
        'dumps_json/loads_json',
        datason.dumps_json,
        datason.loads_json,
        datasets['web_api']['data']
    )

    print("\ndumps_json/loads_json stage breakdown (Web API data):")
    print("\nSerialization stages:")
    for event in profile_data['serialize_stages'][:10]:  # Show first 10
        print(f"  {event['stage']}: {event['duration']/1_000_000:.3f}ms")

    print("\nDeserialization stages:")
    for event in profile_data['deserialize_stages'][:10]:  # Show first 10
        print(f"  {event['stage']}: {event['duration']/1_000_000:.3f}ms")

    print("\n✅ Profiling complete!")

    return results


if __name__ == "__main__":
    main()

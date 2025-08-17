#!/usr/bin/env python3
"""
Optimized PR Benchmark Suite
============================

Based on learnings from Phase 1-4, this creates the optimal dataset combination
for PR testing that maximizes regression detection while minimizing execution time.

This benchmark suite now includes multiple API tiers to test different aspects of
DataSON's performance and features. It also incorporates statistical robustness
by using warmup runs, more iterations, and outlier removal.

Key Tiers:
1. Basic: Raw performance with save_string/load_basic.
2. API Optimized: For web services using save_api/load_basic.
3. Smart: Advanced type handling with dump/load_smart.
4. ML Optimized: For scientific workloads with dump_ml/load_smart.
5. Compatibility: For stdlib json drop-in replacement.
"""

import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List
import json
import argparse
import statistics

logger = logging.getLogger(__name__)


class OptimizedPRBenchmark:
    """Optimized benchmark suite specifically designed for PR testing."""

    def __init__(self):
        try:
            import datason
            self.datason = datason
            self.datason_available = True
        except ImportError:
            logger.error("DataSON not available")
            self.datason_available = False

        # Optional ML libraries for enhanced testing
        try:
            import numpy as np
            self.numpy = np
            self.numpy_available = True
        except ImportError:
            self.numpy_available = False

    def get_api_tiers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the different API tiers for benchmarking.
        Each tier has a serialize and deserialize function.
        """
        if not self.datason_available:
            return {}
        # Determine available serialization methods for in-memory operations
        basic_ser = self.datason.save_string
        
        # API-optimized: Use serialize with API config
        def api_serialize(obj):
            config = self.datason.get_api_config()
            return self.datason.dumps_json(self.datason.serialize(obj, config=config))
        
        # Smart: Use dumps (in-memory version of dump)
        smart_ser = self.datason.dumps
        
        # ML-optimized: Use serialize with ML config  
        def ml_serialize(obj):
            config = self.datason.get_ml_config()
            return self.datason.serialize(obj, config=config)

        return {
            'basic': {
                'serialize': basic_ser,
                'deserialize': self.datason.load_basic,
                'description': 'Direct basic API - fastest baseline'
            },
            'api_optimized': {
                'serialize': api_serialize,
                'deserialize': self.datason.loads_json,  # Since api_serialize outputs JSON string
                'description': 'API-optimized - best for web services'
            },
            'smart': {
                'serialize': smart_ser,
                'deserialize': self.datason.loads,  # Matching deserializer for dumps
                'description': 'Smart detection - type preservation'
            },
            'ml_optimized': {
                'serialize': ml_serialize,
                'deserialize': lambda x: x if isinstance(x, dict) else self.datason.loads(x),  # Handle dict outputs directly
                'description': 'ML-optimized - NumPy/tensor support'
            },
            'compatibility': {
                'serialize': self.datason.dumps_json,
                'deserialize': self.datason.loads_json,
                'description': 'Stdlib compatible - legacy support'
            }
        }

    def benchmark_with_statistics(self, func, data, iterations=10, warmup=2):
        """
        Run a benchmark with warmup, iterations, and statistical analysis.
        Removes min and max outliers.
        """
        # Warmup
        for _ in range(warmup):
            try:
                func(data)
            except Exception:
                # Ignore errors during warmup, but they might indicate a problem
                pass

        # Measure
        times = []
        errors = 0
        for _ in range(iterations):
            try:
                start = time.perf_counter()
                result = func(data)
                times.append((time.perf_counter() - start) * 1000) # ms
            except Exception as e:
                errors += 1
                logger.debug(f"Benchmarking error: {e}")

        if not times:
            return {"error_count": errors, "status": "failed"}

        # Remove outliers if we have enough samples
        if len(times) > 2:
            times.remove(min(times))
            times.remove(max(times))

        if not times:
            return {"error_count": errors, "status": "failed_after_outlier_removal"}

        return {
            'mean_ms': round(statistics.mean(times), 3),
            'median_ms': round(statistics.median(times), 3),
            'stdev_ms': round(statistics.stdev(times), 3) if len(times) > 1 else 0.0,
            'min_ms': round(min(times), 3),
            'max_ms': round(max(times), 3),
            'iterations': len(times),
            'error_count': errors
        }

    def create_pr_optimized_datasets(self) -> Dict[str, Any]:
        """Create the optimal dataset combination for PR testing."""
        datasets = {}

        # 1. Web API Response (from Phase 3) - Most common regression source
        datasets.update(self._create_web_api_dataset())

        # 2. ML Training Data (from Phase 2) - Complex object handling
        datasets.update(self._create_ml_dataset())

        # 3. Financial Transaction (from Phase 3) - Precision critical
        datasets.update(self._create_financial_dataset())

        # 4. Mixed Types Challenge (from Phase 1) - Edge cases
        datasets.update(self._create_mixed_types_dataset())

        # 5. Security Test Data (from Phase 2) - PII detection
        datasets.update(self._create_security_dataset())

        return datasets

    def _create_web_api_dataset(self) -> Dict[str, Any]:
        """Web API response - catches most serialization regressions."""
        return {
            "web_api_response": {
                "description": "Realistic web API response with nested data",
                "domain": "web_api",
                "size_category": "medium",
                "data": {
                    "api_response": {
                        "status": "success",
                        "timestamp": datetime.now(timezone.utc),
                        "request_id": str(uuid.uuid4()),
                        "data": {
                            "users": [
                                {
                                    "id": i,
                                    "username": f"user_{i}",
                                    "email": f"user{i}@example.com",
                                    "created_at": datetime.now(timezone.utc) - timedelta(days=i*10),
                                    "profile": {
                                        "age": 25 + i,
                                        "preferences": ["pref1", "pref2", f"custom_{i}"],
                                        "settings": {
                                            "notifications": True,
                                            "theme": "dark" if i % 2 else "light"
                                        }
                                    },
                                    "stats": {
                                        "login_count": i * 10,
                                        "last_active": datetime.now(timezone.utc) - timedelta(hours=i)
                                    }
                                }
                                for i in range(20)
                            ],
                            "pagination": {
                                "page": 1,
                                "per_page": 20,
                                "total": 1000,
                                "has_next": True
                            },
                            "metadata": {
                                "query_time_ms": Decimal("12.34"),
                                "cached": True,
                                "api_version": "2.1.0"
                            }
                        }
                    }
                }
            }
        }

    def _create_ml_dataset(self) -> Dict[str, Any]:
        """ML training data - reveals complex object handling issues."""
        data = {
            "ml_training_batch": {
                "description": "ML training batch with numpy arrays and metadata",
                "domain": "machine_learning",
                "size_category": "large",
                "data": {
                    "experiment": {
                        "experiment_id": str(uuid.uuid4()),
                        "model_name": "fraud_detection_v3",
                        "hyperparameters": {
                            "learning_rate": Decimal("0.001"),
                            "batch_size": 64,
                            "l2_reg": Decimal("0.01")
                        },
                        "metrics": {
                            "accuracy": Decimal("0.9876"),
                            "precision": Decimal("0.8456"),
                            "recall": Decimal("0.7912"),
                            "f1_score": Decimal("0.8175")
                        },
                        "training_data": {
                            "features": [[float(i+j*0.1) for j in range(10)] for i in range(50)],
                            "labels": [i % 2 for i in range(50)],
                            "weights": [1.0 + i*0.01 for i in range(50)]
                        }
                    }
                }
            }
        }

        # Add numpy arrays if available
        if self.numpy_available:
            import numpy as np
            data["ml_training_batch"]["data"]["numpy_arrays"] = {
                "feature_matrix": np.random.rand(50, 10),
                "target_vector": np.random.randint(0, 2, size=50),
                "feature_importance": np.random.rand(10),
                "confusion_matrix": np.array([[45, 3], [2, 50]])
            }

        return data

    def _create_financial_dataset(self) -> Dict[str, Any]:
        """Financial transaction data - precision critical."""
        return {
            "financial_transaction": {
                "description": "Financial transaction with high precision decimals",
                "domain": "finance",
                "size_category": "medium",
                "data": {
                    "transaction": {
                        "transaction_id": str(uuid.uuid4()),
                        "account_from": "ACC-2024-001",
                        "account_to": "ACC-2024-002",
                        "amount": Decimal("1234567.89"),
                        "currency": "USD",
                        "exchange_rate": Decimal("1.0000"),
                        "fees": {
                            "base_fee": Decimal("2.50"),
                            "percentage_fee": Decimal("0.0025"),
                            "total_fee": Decimal("3085.92")
                        },
                        "timestamp": datetime.now(timezone.utc),
                        "status": "completed",
                        "risk_assessment": {
                            "score": Decimal("0.023"),
                            "factors": ["amount_high", "new_recipient"],
                            "approved": True
                        },
                        "compliance": {
                            "aml_check": True,
                            "sanctions_check": True,
                            "pep_check": False
                        }
                    }
                }
            }
        }

    def _create_mixed_types_dataset(self) -> Dict[str, Any]:
        """Mixed types challenge - edge cases for type preservation."""
        return {
            "mixed_types_challenge": {
                "description": "Edge cases for type preservation testing",
                "domain": "type_testing",
                "size_category": "small",
                "data": {
                    "type_challenges": {
                        # Numeric edge cases
                        "numbers": {
                            "zero": 0,
                            "negative": -42,
                            "large_int": 9223372036854775807,
                            "decimal": Decimal("123.456789"),
                            "float_precision": 3.141592653589793
                        },
                        # String edge cases
                        "strings": {
                            "empty": "",
                            "unicode": "Hello 世界 🌍",
                            "json_like": "{'key': 'value'}",
                            "special_chars": "Line1\nLine2\tTabbed"
                        },
                        # Container edge cases
                        "containers": {
                            "empty_list": [],
                            "empty_dict": {},
                            "nested_mixed": {
                                "list_in_dict": [1, "two", 3.0],
                                "dict_in_list": [{"a": 1}, {"b": 2}]
                            }
                        },
                        # None and boolean
                        "nulls_and_bools": {
                            "none_value": None,
                            "true_bool": True,
                            "false_bool": False
                        },
                        # Datetime variations
                        "datetimes": {
                            "now": datetime.now(timezone.utc),
                            "past": datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                            "future": datetime(2030, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
                        }
                    }
                }
            }
        }

    def _create_security_dataset(self) -> Dict[str, Any]:
        """Security test data - PII detection testing."""
        return {
            "security_pii_test": {
                "description": "Data with PII for security feature testing",
                "domain": "security",
                "size_category": "small",
                "data": {
                    "user_profile": {
                        "user_id": str(uuid.uuid4()),
                        "personal_info": {
                            # PII that should be detected/redacted
                            "email": "john.doe@company.com",
                            "phone": "+1-555-123-4567",
                            "ssn": "123-45-6789",
                            "credit_card": "4532-1234-5678-9012"
                        },
                        "public_info": {
                            "username": "johndoe123",
                            "display_name": "John D.",
                            "join_date": datetime(2023, 1, 15, tzinfo=timezone.utc),
                            "public_profile": True
                        },
                        "preferences": {
                            "notifications": True,
                            "data_sharing": False,
                            "marketing": False
                        }
                    }
                }
            }
        }

    def run_pr_benchmark(self, iterations: int = 10, warmup: int = 2) -> Dict[str, Any]:
        """Run optimized PR benchmark suite across multiple API tiers."""
        if not self.datason_available:
            return {"error": "DataSON not available"}

        logger.info(f"🚀 Running Optimized PR Benchmark Suite (iterations={iterations}, warmup={warmup})")

        start_time = time.time()
        datasets = self.create_pr_optimized_datasets()
        api_tiers = self.get_api_tiers()

        results = {
            "suite_type": "pr_optimized_tiered",
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "iterations": iterations,
                "warmup": warmup,
                "datasets_count": len(datasets),
                "api_tiers_count": len(api_tiers),
                "numpy_available": self.numpy_available,
                "datason_version": getattr(self.datason, '__version__', 'unknown')
            },
            "results_by_tier": {},
            "performance_summary": {
                "total_tests": 0,
                "successful_tests": 0,
                "failed_tests": 0,
            },
            "execution_time": 0.0
        }

        total_successful_tests = 0
        total_failed_tests = 0

        for tier_name, tier_apis in api_tiers.items():
            logger.info(f"  🔬 Testing Tier: {tier_name} ({tier_apis['description']})")
            tier_results = {
                "description": tier_apis['description'],
                "datasets": {}
            }

            for dataset_name, dataset_info in datasets.items():
                logger.info(f"    📊 Benchmarking {dataset_name}...")

                # Benchmark Serialization
                serialize_func = tier_apis['serialize']
                ser_stats = self.benchmark_with_statistics(
                    serialize_func, dataset_info["data"], iterations, warmup
                )

                serialized_data = None
                if ser_stats.get("status") != "failed":
                    try:
                        serialized_data = serialize_func(dataset_info["data"])
                        total_successful_tests += 1
                    except Exception as e:
                        logger.warning(f"      Serialization failed for {dataset_name} in tier {tier_name}: {e}")
                        ser_stats["status"] = "failed"
                        total_failed_tests +=1


                # Benchmark Deserialization
                des_stats = {"status": "skipped"}
                if serialized_data is not None:
                    deserialize_func = tier_apis['deserialize']
                    des_stats = self.benchmark_with_statistics(
                        deserialize_func, serialized_data, iterations, warmup
                    )
                    if des_stats.get("status") == "failed":
                        total_failed_tests += 1
                    else:
                        total_successful_tests += 1
                else:
                    total_failed_tests += 1


                tier_results["datasets"][dataset_name] = {
                    "description": dataset_info["description"],
                    "domain": dataset_info["domain"],
                    "size_category": dataset_info["size_category"],
                    "serialization": ser_stats,
                    "deserialization": des_stats
                }

            results["results_by_tier"][tier_name] = tier_results

        # Final summary
        results["performance_summary"]["total_tests"] = len(datasets) * len(api_tiers) * 2
        results["performance_summary"]["successful_tests"] = total_successful_tests
        results["performance_summary"]["failed_tests"] = total_failed_tests
        results["execution_time"] = time.time() - start_time

        logger.info(f"✅ Tiered benchmark completed in {results['execution_time']:.2f}s")
        logger.info(f"   PASSED: {total_successful_tests}, FAILED: {total_failed_tests}, TOTAL: {results['performance_summary']['total_tests']}")

        return results


def main():
    """Run optimized PR benchmark."""
    parser = argparse.ArgumentParser(description="Run PR-optimized benchmark suite")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup iterations")
    args = parser.parse_args()

    # Import DataSON for result saving
    try:
        import datason
    except ImportError:
        print("Error: DataSON not available for saving results")
        return 1

    pr_benchmark = OptimizedPRBenchmark()
    results = pr_benchmark.run_pr_benchmark(iterations=args.iterations, warmup=args.warmup)

    if args.output:
        filename = args.output
    else:
        timestamp = int(time.time())
        filename = f"data/results/pr_optimized_tiered_{timestamp}.json"

    with open(filename, 'w') as f:
        # Using compatibility API to save results
        f.write(datason.dumps_json(results))

    print(f"📊 Results saved to {filename}")
    return 0


if __name__ == "__main__":
    main()

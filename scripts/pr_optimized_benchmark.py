#!/usr/bin/env python3
"""
Optimized PR Benchmark Suite
============================

Based on learnings from Phase 1-4, this creates the optimal dataset combination
for PR testing that maximizes regression detection while minimizing execution time.

Key Principles from Our Analysis:
1. Web API scenario catches most serialization regressions (Phase 3)
2. ML data reveals complex object handling issues (Phase 2)
3. Security features need targeted testing (Phase 2)
4. Financial data exposes precision problems (Phase 3)
5. Mixed data types reveal type preservation issues (Phase 1)
"""

import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List
import json

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
                        }
                    },
                    "training_data": {
                        "features": [[float(i+j*0.1) for j in range(10)] for i in range(50)],
                        "labels": [i % 2 for i in range(50)],
                        "weights": [1.0 + i*0.01 for i in range(50)]
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
                            "json_like": '{"key": "value"}',
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
    
    def run_pr_benchmark(self, iterations: int = 5) -> Dict[str, Any]:
        """Run optimized PR benchmark suite with actual performance testing."""
        if not self.datason_available:
            return {"error": "DataSON not available"}
        
        logger.info("🚀 Running Optimized PR Benchmark Suite")
        
        start_time = time.time()
        datasets = self.create_pr_optimized_datasets()
        
        results = {
            "suite_type": "pr_optimized",
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "iterations": iterations,
                "datasets_count": len(datasets),
                "numpy_available": self.numpy_available,
                "datason_version": getattr(self.datason, '__version__', 'unknown')
            },
            "competitive": {
                "tiers": {
                    "pr_optimized": {
                        "description": "PR-optimized scenarios for regression detection",
                        "datasets": {}
                    }
                }
            },
            "performance_summary": {
                "total_tests": 0,
                "successful_tests": 0,
                "failed_tests": 0,
                "avg_serialization_ms": 0.0,
                "avg_deserialization_ms": 0.0
            },
            "execution_time": 0.0
        }
        
        total_serialization_times = []
        total_deserialization_times = []
        successful_tests = 0
        failed_tests = 0
        
        # Benchmark each dataset with actual performance testing
        for dataset_name, dataset_info in datasets.items():
            logger.info(f"  📊 Testing {dataset_name}...")
            
            dataset_result = {
                "description": dataset_info["description"],
                "domain": dataset_info["domain"],
                "size_category": dataset_info["size_category"],
                "serialization": {"datason": {}},
                "deserialization": {"datason": {}}
            }
            
            # Run serialization benchmarks
            serialization_times = []
            serialization_errors = 0
            
            for i in range(iterations):
                try:
                    start = time.perf_counter()
                    serialized = self.datason.dump(dataset_info["data"])
                    end = time.perf_counter()
                    serialization_times.append((end - start) * 1000)  # Convert to ms
                except Exception as e:
                    serialization_errors += 1
                    logger.warning(f"    Serialization error in {dataset_name} iteration {i}: {e}")
            
            # Run deserialization benchmarks
            deserialization_times = []
            deserialization_errors = 0
            
            if serialization_times:  # Only test deserialization if serialization worked
                try:
                    # Get a successful serialization result for deserialization testing
                    test_serialized = self.datason.dumps(dataset_info["data"])
                    
                    for i in range(iterations):
                        try:
                            start = time.perf_counter()
                            deserialized = self.datason.loads(test_serialized)
                            end = time.perf_counter()
                            deserialization_times.append((end - start) * 1000)  # Convert to ms
                        except Exception as e:
                            deserialization_errors += 1
                            logger.warning(f"    Deserialization error in {dataset_name} iteration {i}: {e}")
                except Exception as e:
                    logger.warning(f"    Could not create test data for deserialization in {dataset_name}: {e}")
            
            # Calculate statistics
            if serialization_times:
                dataset_result["serialization"]["datason"] = {
                    "mean_ms": round(sum(serialization_times) / len(serialization_times), 3),
                    "min_ms": round(min(serialization_times), 3),
                    "max_ms": round(max(serialization_times), 3),
                    "iterations": len(serialization_times),
                    "error_count": serialization_errors
                }
                total_serialization_times.extend(serialization_times)
                successful_tests += 1
            else:
                dataset_result["serialization"]["datason"] = {
                    "error_count": serialization_errors,
                    "status": "failed"
                }
                failed_tests += 1
            
            if deserialization_times:
                dataset_result["deserialization"]["datason"] = {
                    "mean_ms": round(sum(deserialization_times) / len(deserialization_times), 3),
                    "min_ms": round(min(deserialization_times), 3),
                    "max_ms": round(max(deserialization_times), 3),
                    "iterations": len(deserialization_times),
                    "error_count": deserialization_errors
                }
                total_deserialization_times.extend(deserialization_times)
            else:
                dataset_result["deserialization"]["datason"] = {
                    "error_count": deserialization_errors,
                    "status": "failed"
                }
            
            results["competitive"]["tiers"]["pr_optimized"]["datasets"][dataset_name] = dataset_result
        
        # Calculate overall performance summary
        results["performance_summary"]["total_tests"] = len(datasets)
        results["performance_summary"]["successful_tests"] = successful_tests
        results["performance_summary"]["failed_tests"] = failed_tests
        
        if total_serialization_times:
            results["performance_summary"]["avg_serialization_ms"] = round(
                sum(total_serialization_times) / len(total_serialization_times), 3
            )
        
        if total_deserialization_times:
            results["performance_summary"]["avg_deserialization_ms"] = round(
                sum(total_deserialization_times) / len(total_deserialization_times), 3
            )
        
        results["execution_time"] = time.time() - start_time
        logger.info(f"✅ PR benchmark completed in {results['execution_time']:.2f}s")
        logger.info(f"   📊 {successful_tests}/{len(datasets)} scenarios passed")
        logger.info(f"   ⚡ Avg serialization: {results['performance_summary']['avg_serialization_ms']:.3f}ms")
        logger.info(f"   ⚡ Avg deserialization: {results['performance_summary']['avg_deserialization_ms']:.3f}ms")
        
        return results


def main():
    """Run optimized PR benchmark."""
    pr_benchmark = OptimizedPRBenchmark()
    results = pr_benchmark.run_pr_benchmark(iterations=5)
    
    timestamp = int(time.time())
    filename = f"data/results/pr_optimized_{timestamp}.json"
    
    with open(filename, 'w') as f:
        # Dogfood DataSON for serialization (no indent param in DataSON)
        f.write(datason.dumps(results))
    
    print(f"📊 Results saved to {filename}")


if __name__ == "__main__":
    main() 
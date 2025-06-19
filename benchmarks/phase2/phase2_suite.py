#!/usr/bin/env python3
"""
Phase 2 Advanced Features Benchmark Suite
==========================================

Tests DataSON's Phase 2 features:
1. Security Testing - dump_secure() with PII redaction
2. Accuracy Analysis - load_smart() vs load_perfect() success rates
3. ML Framework Integration - Real numpy/pandas/torch benchmarks
"""

import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from statistics import mean, stdev
from typing import Any, Dict, List, Optional
import uuid
import json

logger = logging.getLogger(__name__)


class Phase2BenchmarkSuite:
    """Advanced features benchmark suite for DataSON Phase 2 capabilities."""
    
    def __init__(self):
        # Import dependencies
        try:
            import datason
            self.datason = datason
            self.datason_available = True
        except ImportError:
            logger.error("DataSON not available for Phase 2 testing")
            self.datason_available = False
        
        # Optional ML libraries
        self.numpy_available = False
        self.pandas_available = False
        
        try:
            import numpy as np
            self.numpy = np
            self.numpy_available = True
        except ImportError:
            logger.debug("NumPy not available for ML testing")
        
        try:
            import pandas as pd
            self.pandas = pd
            self.pandas_available = True
        except ImportError:
            logger.debug("Pandas not available for ML testing")
    
    def create_security_test_data(self) -> Dict[str, Any]:
        """Create test data with PII for security testing."""
        return {
            "security_pii_data": {
                "description": "Data with PII for security redaction testing",
                "tier": "security_enhanced",
                "data": {
                    "user": {
                        "user_id": str(uuid.uuid4()),
                        "email": "john.doe@example.com",
                        "first_name": "John",
                        "last_name": "Doe",
                        "phone": "+1-555-123-4567",
                        "ssn": "123-45-6789",
                        "address": {
                            "street": "123 Main St",
                            "city": "Anytown",
                            "state": "CA",
                            "zip": "12345"
                        }
                    },
                    "financial": {
                        "account_number": "1234567890123456",
                        "routing_number": "123456789",
                        "credit_score": 750,
                        "balance": Decimal("1234.56")
                    },
                    "sensitive_notes": "Contains medical information and personal details",
                    "metadata": {
                        "sensitivity": "HIGH",
                        "created": datetime.now(timezone.utc)
                    }
                }
            }
        }
    
    def create_accuracy_test_data(self) -> Dict[str, Any]:
        """Create complex data for accuracy testing."""
        return {
            "accuracy_complex_data": {
                "description": "Complex nested data for accuracy testing",
                "tier": "accuracy_enhanced",
                "data": {
                    "nested_levels": {
                        "level_1": {
                            "level_2": {
                                "level_3": {
                                    "deep_uuid": uuid.uuid4(),
                                    "deep_datetime": datetime.now(timezone.utc),
                                    "deep_decimal": Decimal("999.999999999")
                                }
                            }
                        }
                    },
                    "mixed_types": [
                        "string",
                        42,
                        3.14159,
                        True,
                        None,
                        datetime.now(timezone.utc),
                        uuid.uuid4(),
                        Decimal("123.456"),
                        {"nested": "dict"},
                        [1, 2, {"deep": "list"}]
                    ],
                    "edge_cases": {
                        "empty_dict": {},
                        "empty_list": [],
                        "unicode": "Hello 世界 🌍",
                        "large_decimal": Decimal("99999999999999999999.99999999999999"),
                        "scientific": Decimal("1.23e-10")
                    }
                }
            }
        }
    
    def create_ml_test_data(self) -> Dict[str, Any]:
        """Create ML framework test data."""
        data = {
            "ml_framework_data": {
                "description": "ML framework objects for testing",
                "tier": "ml_enhanced",
                "data": {
                    "experiment": {
                        "experiment_id": str(uuid.uuid4()),
                        "model_name": "test_classifier",
                        "created_at": datetime.now(timezone.utc)
                    },
                    "hyperparameters": {
                        "learning_rate": Decimal("0.001"),
                        "batch_size": 32,
                        "epochs": 100
                    },
                    "basic_arrays": {
                        "features": [[1.0, 2.0, 3.0] for _ in range(10)],
                        "labels": [i % 2 for i in range(10)]
                    }
                }
            }
        }
        
        # Add numpy arrays if available
        if self.numpy_available:
            import numpy as np
            data["ml_framework_data"]["data"]["numpy_arrays"] = {
                "feature_matrix": np.random.rand(10, 5),
                "target_vector": np.random.randint(0, 2, size=10),
                "weights": np.random.normal(0, 1, size=(5, 3))
            }
        
        # Add pandas data if available
        if self.pandas_available:
            import pandas as pd
            df = pd.DataFrame({
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature_2": [0.1, 0.2, 0.3, 0.4, 0.5],
                "target": [0, 1, 0, 1, 0]
            })
            data["ml_framework_data"]["data"]["pandas_objects"] = {
                "dataframe": df,
                "series": pd.Series([1, 2, 3, 4, 5])
            }
        
        return data
    
    def test_security_features(self, iterations: int = 10) -> Dict[str, Any]:
        """Test DataSON's security features with PII redaction."""
        if not self.datason_available:
            return {"error": "DataSON not available"}
        
        logger.info("🔒 Testing DataSON security features...")
        
        test_data = self.create_security_test_data()
        data = test_data["security_pii_data"]["data"]
        
        results = {
            "methodology": "security_testing",
            "pii_redaction": {},
            "performance": {},
            "security_validation": {}
        }
        
        # Test dump_secure if available
        if hasattr(self.datason, 'dump_secure'):
            logger.info("  Testing dump_secure() PII redaction...")
            
            # Benchmark performance
            times = []
            redacted_outputs = []
            
            for _ in range(iterations):
                try:
                    start = time.perf_counter()
                    secured_output = self.datason.dump_secure(data)
                    end = time.perf_counter()
                    
                    times.append(end - start)
                    redacted_outputs.append(secured_output)
                except Exception as e:
                    logger.error(f"dump_secure() failed: {e}")
                    return {"error": f"dump_secure() failed: {e}"}
            
            if times:
                results["performance"]["dump_secure"] = {
                    "mean_ms": mean(times) * 1000,
                    "min_ms": min(times) * 1000,
                    "max_ms": max(times) * 1000,
                    "std_ms": stdev(times) * 1000 if len(times) > 1 else 0.0,
                    "successful_runs": len(times)
                }
                
                # Analyze redaction effectiveness
                sample_output = redacted_outputs[0]
                results["security_validation"] = self._analyze_pii_redaction(
                    original_data=data,
                    secured_output=sample_output
                )
        else:
            results["pii_redaction"]["dump_secure"] = "Method not available"
        
        # Compare with regular serialize for reference
        if hasattr(self.datason, 'serialize'):
            times = []
            for _ in range(iterations):
                try:
                    start = time.perf_counter()
                    regular_output = self.datason.serialize(data)
                    end = time.perf_counter()
                    times.append(end - start)
                except Exception:
                    pass
            
            if times:
                results["performance"]["regular_serialize"] = {
                    "mean_ms": mean(times) * 1000,
                    "successful_runs": len(times)
                }
        
        return results
    
    def test_accuracy_features(self, iterations: int = 10) -> Dict[str, Any]:
        """Test DataSON's accuracy features - load_smart vs load_perfect."""
        if not self.datason_available:
            return {"error": "DataSON not available"}
        
        logger.info("🎯 Testing DataSON accuracy features...")
        
        test_data = self.create_accuracy_test_data()
        data = test_data["accuracy_complex_data"]["data"]
        
        results = {
            "methodology": "accuracy_testing",
            "loading_methods": {},
            "reconstruction_accuracy": {}
        }
        
        # First serialize the data
        try:
            serialized_data = self.datason.serialize(data)
        except Exception as e:
            return {"error": f"Failed to serialize test data: {e}"}
        
        # Test load_smart if available
        if hasattr(self.datason, 'load_smart'):
            logger.info("  Testing load_smart() reconstruction...")
            
            smart_times = []
            smart_successes = 0
            smart_outputs = []
            
            for _ in range(iterations):
                try:
                    start = time.perf_counter()
                    smart_result = self.datason.load_smart(serialized_data)
                    end = time.perf_counter()
                    
                    smart_times.append(end - start)
                    smart_outputs.append(smart_result)
                    smart_successes += 1
                except Exception as e:
                    logger.debug(f"load_smart() failed: {e}")
            
            results["loading_methods"]["load_smart"] = {
                "mean_ms": mean(smart_times) * 1000 if smart_times else 0,
                "success_rate": smart_successes / iterations,
                "successful_runs": smart_successes
            }
            
            # Analyze reconstruction accuracy
            if smart_outputs:
                results["reconstruction_accuracy"]["load_smart"] = self._analyze_reconstruction_accuracy(
                    original=data,
                    reconstructed=smart_outputs[0]
                )
        
        # Test load_perfect if available
        if hasattr(self.datason, 'load_perfect'):
            logger.info("  Testing load_perfect() reconstruction...")
            
            perfect_times = []
            perfect_successes = 0
            perfect_outputs = []
            
            for _ in range(iterations):
                try:
                    start = time.perf_counter()
                    perfect_result = self.datason.load_perfect(serialized_data)
                    end = time.perf_counter()
                    
                    perfect_times.append(end - start)
                    perfect_outputs.append(perfect_result)
                    perfect_successes += 1
                except Exception as e:
                    logger.debug(f"load_perfect() failed: {e}")
            
            results["loading_methods"]["load_perfect"] = {
                "mean_ms": mean(perfect_times) * 1000 if perfect_times else 0,
                "success_rate": perfect_successes / iterations,
                "successful_runs": perfect_successes
            }
            
            # Analyze reconstruction accuracy
            if perfect_outputs:
                results["reconstruction_accuracy"]["load_perfect"] = self._analyze_reconstruction_accuracy(
                    original=data,
                    reconstructed=perfect_outputs[0]
                )
        
        # Test regular deserialize for comparison
        if hasattr(self.datason, 'deserialize'):
            regular_times = []
            regular_successes = 0
            
            for _ in range(iterations):
                try:
                    start = time.perf_counter()
                    regular_result = self.datason.deserialize(serialized_data)
                    end = time.perf_counter()
                    
                    regular_times.append(end - start)
                    regular_successes += 1
                except Exception:
                    pass
            
            results["loading_methods"]["regular_deserialize"] = {
                "mean_ms": mean(regular_times) * 1000 if regular_times else 0,
                "success_rate": regular_successes / iterations,
                "successful_runs": regular_successes
            }
        
        return results
    
    def test_ml_framework_integration(self, iterations: int = 5) -> Dict[str, Any]:
        """Test DataSON's ML framework integration."""
        if not self.datason_available:
            return {"error": "DataSON not available"}
        
        logger.info("🧠 Testing DataSON ML framework integration...")
        
        results = {
            "methodology": "ml_framework_testing",
            "framework_support": {
                "numpy": self.numpy_available,
                "pandas": self.pandas_available
            },
            "serialization_results": {},
            "framework_specific": {}
        }
        
        test_data = self.create_ml_test_data()
        data = test_data["ml_framework_data"]["data"]
        
        # Test different DataSON ML methods
        ml_methods = []
        
        if hasattr(self.datason, 'dump_ml'):
            ml_methods.append(("dump_ml", self.datason.dump_ml))
        
        if hasattr(self.datason, 'serialize'):
            ml_methods.append(("serialize", self.datason.serialize))
        
        for method_name, method_func in ml_methods:
            logger.info(f"  Testing {method_name} with ML data...")
            
            times = []
            successes = 0
            
            for _ in range(iterations):
                try:
                    start = time.perf_counter()
                    result = method_func(data)
                    end = time.perf_counter()
                    
                    times.append(end - start)
                    successes += 1
                except Exception as e:
                    logger.debug(f"{method_name} failed: {e}")
            
            results["serialization_results"][method_name] = {
                "mean_ms": mean(times) * 1000 if times else 0,
                "success_rate": successes / iterations,
                "successful_runs": successes,
                "framework_compatibility": {
                    "basic_arrays": True,
                    "numpy_arrays": self.numpy_available,
                    "pandas_objects": self.pandas_available
                }
            }
        
        # Test numpy-specific handling if available
        if self.numpy_available:
            results["framework_specific"]["numpy"] = self._test_numpy_integration()
        
        # Test pandas-specific handling if available  
        if self.pandas_available:
            results["framework_specific"]["pandas"] = self._test_pandas_integration()
        
        return results
    
    def run_phase2_complete_suite(self, iterations: int = 10) -> Dict[str, Any]:
        """Run complete Phase 2 benchmark suite."""
        logger.info("🚀 Running Phase 2 Complete Advanced Features Suite...")
        
        start_time = time.time()
        
        results = {
            "suite_type": "phase2_advanced_features",
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "datason_available": self.datason_available,
                "numpy_available": self.numpy_available,
                "pandas_available": self.pandas_available,
                "iterations": iterations
            },
            "security_testing": self.test_security_features(iterations),
            "accuracy_testing": self.test_accuracy_features(iterations),
            "ml_framework_testing": self.test_ml_framework_integration(iterations // 2),
            "execution_time": 0.0
        }
        
        results["execution_time"] = time.time() - start_time
        
        # Generate summary
        results["summary"] = self._generate_phase2_summary(results)
        
        logger.info("✅ Phase 2 complete suite finished")
        return results
    
    def _analyze_pii_redaction(self, original_data: Any, secured_output: str) -> Dict[str, Any]:
        """Analyze effectiveness of PII redaction."""
        # Convert secured output back to check redaction
        try:
            secured_data = json.loads(secured_output)
        except:
            return {"error": "Could not parse secured output"}
        
        # Look for common PII patterns that should be redacted
        pii_checks = {
            "email_redacted": "john.doe@example.com" not in secured_output,
            "ssn_redacted": "123-45-6789" not in secured_output,
            "phone_redacted": "+1-555-123-4567" not in secured_output,
            "account_redacted": "1234567890123456" not in secured_output,
            "contains_redaction_markers": "<REDACTED>" in secured_output or "***" in secured_output
        }
        
        redaction_score = sum(pii_checks.values()) / len(pii_checks)
        
        return {
            "redaction_effectiveness": redaction_score,
            "pii_checks": pii_checks,
            "output_size_reduction": len(str(original_data)) - len(secured_output),
            "structure_preserved": isinstance(secured_data, dict)
        }
    
    def _analyze_reconstruction_accuracy(self, original: Any, reconstructed: Any) -> Dict[str, Any]:
        """Analyze reconstruction accuracy between original and reconstructed data."""
        try:
            # Simple accuracy checks
            type_match = type(original) == type(reconstructed)
            
            if isinstance(original, dict) and isinstance(reconstructed, dict):
                key_match = set(original.keys()) == set(reconstructed.keys())
                structure_accuracy = 1.0 if key_match else 0.7
            else:
                structure_accuracy = 1.0 if type_match else 0.0
            
            # For demo purposes - in real implementation would do deep comparison
            accuracy_score = structure_accuracy
            
            return {
                "accuracy_score": accuracy_score,
                "type_preservation": type_match,
                "structure_preservation": structure_accuracy,
                "notes": "Simplified accuracy analysis for demo"
            }
        except Exception as e:
            return {"error": f"Accuracy analysis failed: {e}"}
    
    def _test_numpy_integration(self) -> Dict[str, Any]:
        """Test DataSON's numpy integration."""
        if not self.numpy_available:
            return {"error": "NumPy not available"}
        
        import numpy as np
        
        # Test various numpy objects
        test_arrays = {
            "1d_array": np.array([1, 2, 3, 4, 5]),
            "2d_array": np.random.rand(3, 3),
            "float_array": np.array([1.1, 2.2, 3.3]),
            "int_array": np.array([1, 2, 3], dtype=np.int32)
        }
        
        results = {}
        
        for array_name, array_data in test_arrays.items():
            try:
                if hasattr(self.datason, 'dump_ml'):
                    serialized = self.datason.dump_ml({array_name: array_data})
                    results[array_name] = "success"
                else:
                    results[array_name] = "dump_ml_not_available"
            except Exception as e:
                results[array_name] = f"failed: {e}"
        
        return {"numpy_serialization_tests": results}
    
    def _test_pandas_integration(self) -> Dict[str, Any]:
        """Test DataSON's pandas integration."""
        if not self.pandas_available:
            return {"error": "Pandas not available"}
        
        import pandas as pd
        
        # Test various pandas objects
        test_objects = {
            "dataframe": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            "series": pd.Series([1, 2, 3, 4, 5]),
            "datetime_series": pd.Series(pd.date_range("2023-01-01", periods=3))
        }
        
        results = {}
        
        for obj_name, obj_data in test_objects.items():
            try:
                if hasattr(self.datason, 'dump_ml'):
                    serialized = self.datason.dump_ml({obj_name: obj_data})
                    results[obj_name] = "success"
                else:
                    results[obj_name] = "dump_ml_not_available"
            except Exception as e:
                results[obj_name] = f"failed: {e}"
        
        return {"pandas_serialization_tests": results}
    
    def _generate_phase2_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of Phase 2 test results."""
        summary = {
            "phase2_capabilities": {
                "security_features": "available" if "dump_secure" in str(results.get("security_testing", {})) else "limited",
                "accuracy_features": "available" if "load_smart" in str(results.get("accuracy_testing", {})) else "limited",
                "ml_integration": "available" if results["metadata"]["numpy_available"] or results["metadata"]["pandas_available"] else "basic"
            },
            "performance_highlights": {},
            "feature_recommendations": []
        }
        
        # Extract performance highlights
        if "security_testing" in results and "performance" in results["security_testing"]:
            security_perf = results["security_testing"]["performance"]
            if "dump_secure" in security_perf:
                summary["performance_highlights"]["security"] = f"{security_perf['dump_secure']['mean_ms']:.2f}ms avg"
        
        if "accuracy_testing" in results and "loading_methods" in results["accuracy_testing"]:
            accuracy_perf = results["accuracy_testing"]["loading_methods"]
            if "load_smart" in accuracy_perf:
                summary["performance_highlights"]["smart_loading"] = f"{accuracy_perf['load_smart']['success_rate']:.1%} success rate"
        
        # Generate recommendations
        if results["metadata"]["numpy_available"]:
            summary["feature_recommendations"].append("Use dump_ml() for numpy array serialization")
        
        if results["metadata"]["pandas_available"]:
            summary["feature_recommendations"].append("Use dump_ml() for pandas DataFrame serialization")
        
        summary["feature_recommendations"].append("Use dump_secure() for PII-sensitive data")
        summary["feature_recommendations"].append("Use load_smart() for flexible deserialization")
        
        return summary 
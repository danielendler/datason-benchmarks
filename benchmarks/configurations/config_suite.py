#!/usr/bin/env python3
"""
Configuration Benchmark Suite
==============================

Tests DataSON's various configuration options to identify optimal settings for different use cases.
Follows the strategy of testing 8-12 realistic configurations instead of exhaustive combinations.
"""

import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from statistics import mean, stdev
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ConfigurationBenchmarkSuite:
    """Tests DataSON configuration performance across realistic use cases."""
    
    def __init__(self):
        # Import datason here to avoid issues if not available
        try:
            import datason
            self.datason = datason
            self.available = True
        except ImportError:
            logger.error("DataSON not available for configuration testing")
            self.available = False
    
    def get_test_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get realistic configuration scenarios for testing."""
        if not self.available:
            return {}
        
        return {
            "default": {
                "description": "Out-of-box experience",
                "config": {}  # Default settings
            },
            
            "api_fast": {
                "description": "Fast API responses",
                "config": self.datason.get_performance_config()
            },
            
            "ml_training": {
                "description": "ML model serialization", 
                "config": self.datason.get_ml_config()
            },
            
            "secure_storage": {
                "description": "Secure data storage",
                "config": self.datason.get_strict_config()
            },
            
            "api_consistent": {
                "description": "Consistent API responses",
                "config": self.datason.get_api_config()
            }
        }
    
    def create_configuration_test_data(self) -> Dict[str, Any]:
        """Create test data for configuration benchmarking."""
        return {
            "small_objects": {
                "description": "Small API-style objects",
                "data": {
                    "id": 12345,
                    "name": "Test Object",
                    "active": True,
                    "value": 99.99,
                    "tags": ["tag1", "tag2"]
                }
            },
            
            "complex_types": {
                "description": "Complex type handling",
                "data": {
                    "decimals": [Decimal("19.99"), Decimal("0.01")],
                    "timestamps": [
                        datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                        datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
                    ],
                    "nested": {
                        "level1": {
                            "level2": {"data": "deep value"}
                        }
                    }
                }
            },
            
            "medium_dataset": {
                "description": "Medium-sized realistic dataset",
                "data": {
                    "users": [
                        {
                            "id": i,
                            "email": f"user{i}@example.com",
                            "created": datetime(2024, 1, 1, 12, i % 60, 0, tzinfo=timezone.utc),
                            "profile": {
                                "name": f"User {i}",
                                "age": 20 + (i % 50),
                                "preferences": {
                                    "notifications": i % 2 == 0,
                                    "theme": "dark" if i % 3 == 0 else "light"
                                }
                            }
                        }
                        for i in range(50)
                    ]
                }
            }
        }
    
    def benchmark_configuration(self, config_name: str, config: Dict[str, Any], 
                              test_data: Any, iterations: int = 10) -> Dict[str, Any]:
        """Benchmark a specific configuration."""
        if not self.available:
            return {"error": "DataSON not available"}
        
        try:
            # Test serialization
            ser_times = []
            deser_times = []
            
            # Warmup
            for _ in range(2):
                try:
                    serialized = self.datason.serialize(test_data, config=config)
                    self.datason.deserialize(serialized)
                except Exception:
                    pass
            
            # Actual benchmarks
            for _ in range(iterations):
                # Serialization
                try:
                    start = time.perf_counter()
                    serialized = self.datason.serialize(test_data, config=config)
                    end = time.perf_counter()
                    ser_times.append(end - start)
                    
                    # Deserialization
                    start = time.perf_counter()
                    result = self.datason.deserialize(serialized)
                    end = time.perf_counter()
                    deser_times.append(end - start)
                    
                except Exception as e:
                    logger.warning(f"Configuration {config_name} failed: {e}")
                    return {
                        "error": str(e),
                        "config": config
                    }
            
            if not ser_times:
                return {"error": "No successful runs"}
            
            return {
                "serialization": {
                    "mean": mean(ser_times),
                    "min": min(ser_times),
                    "max": max(ser_times),
                    "std": stdev(ser_times) if len(ser_times) > 1 else 0.0,
                    "mean_ms": mean(ser_times) * 1000
                },
                "deserialization": {
                    "mean": mean(deser_times),
                    "min": min(deser_times),
                    "max": max(deser_times),
                    "std": stdev(deser_times) if len(deser_times) > 1 else 0.0,
                    "mean_ms": mean(deser_times) * 1000
                },
                "total_time_ms": (mean(ser_times) + mean(deser_times)) * 1000,
                "successful_runs": len(ser_times),
                "config": config
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "config": config
            }
    
    def run_configuration_tests(self, iterations: int = 10) -> Dict[str, Any]:
        """Run all configuration tests."""
        if not self.available:
            return {"error": "DataSON not available for configuration testing"}
        
        configurations = self.get_test_configurations()
        test_datasets = self.create_configuration_test_data()
        
        results = {}
        
        for config_name, config_info in configurations.items():
            logger.info(f"Testing configuration: {config_name}")
            
            config_results = {}
            
            for dataset_name, dataset_info in test_datasets.items():
                logger.info(f"  Dataset: {dataset_name}")
                
                test_result = self.benchmark_configuration(
                    config_name,
                    config_info["config"],
                    dataset_info["data"],
                    iterations
                )
                
                config_results[dataset_name] = {
                    "description": dataset_info["description"],
                    "results": test_result
                }
            
            results[config_name] = {
                "description": config_info["description"],
                "datasets": config_results
            }
        
        # Add summary analysis
        results["summary"] = self._generate_config_summary(results)
        
        return results
    
    def _generate_config_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of configuration performance."""
        summary = {
            "configurations_tested": [],
            "fastest_configuration": {},
            "best_for_use_case": {},
            "performance_comparison": {}
        }
        
        try:
            # Extract configuration names (excluding summary)
            config_names = [name for name in results.keys() if name != "summary"]
            summary["configurations_tested"] = config_names
            
            # Find fastest configuration for each dataset
            dataset_names = []
            if config_names:
                first_config = results[config_names[0]]
                if "datasets" in first_config:
                    dataset_names = list(first_config["datasets"].keys())
            
            for dataset_name in dataset_names:
                fastest_config = None
                fastest_time = float("inf")
                
                for config_name in config_names:
                    config_data = results.get(config_name, {})
                    dataset_data = config_data.get("datasets", {}).get(dataset_name, {})
                    result_data = dataset_data.get("results", {})
                    
                    if "total_time_ms" in result_data:
                        total_time = result_data["total_time_ms"]
                        if total_time < fastest_time:
                            fastest_time = total_time
                            fastest_config = config_name
                
                if fastest_config:
                    summary["fastest_configuration"][dataset_name] = {
                        "config": fastest_config,
                        "time_ms": fastest_time
                    }
            
            # Best configuration recommendations
            summary["best_for_use_case"] = {
                "speed_critical": summary["fastest_configuration"].get("small_objects", {}).get("config", "api_fast"),
                "complex_types": summary["fastest_configuration"].get("complex_types", {}).get("config", "ml_training"),
                "large_datasets": summary["fastest_configuration"].get("medium_dataset", {}).get("config", "default")
            }
            
            # Performance comparison matrix
            comparison = {}
            for config_name in config_names:
                comparison[config_name] = {}
                config_data = results.get(config_name, {})
                
                for dataset_name in dataset_names:
                    dataset_data = config_data.get("datasets", {}).get(dataset_name, {})
                    result_data = dataset_data.get("results", {})
                    
                    if "total_time_ms" in result_data:
                        comparison[config_name][dataset_name] = result_data["total_time_ms"]
                    else:
                        comparison[config_name][dataset_name] = None
            
            summary["performance_comparison"] = comparison
            
        except Exception as e:
            logger.warning(f"Could not generate configuration summary: {e}")
            summary["error"] = str(e)
        
        return summary 
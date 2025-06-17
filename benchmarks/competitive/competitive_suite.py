#!/usr/bin/env python3
"""
Competitive Benchmark Suite
============================

Benchmarks DataSON against other serialization libraries using realistic test scenarios.
Follows the strategy of testing 6-8 key competitors with practical data patterns.
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CompetitiveBenchmarkSuite:
    """Runs competitive benchmarks between DataSON and other serialization libraries."""
    
    def __init__(self):
        # Import here to avoid circular imports
        import sys
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        from competitors.adapter_registry import CompetitorRegistry
        self.registry = CompetitorRegistry()
    
    def create_benchmark_datasets(self) -> Dict[str, Any]:
        """Create realistic test datasets for competitive benchmarking."""
        return {
            "api_response": {
                "description": "Typical API response with metadata",
                "data": {
                    "status": "success",
                    "timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                    "request_id": str(uuid.UUID("12345678-1234-5678-9012-123456789012")),
                    "items": [
                        {
                            "id": i,
                            "name": f"Item {i:03d}",
                            "price": Decimal(f"{19.99 + i * 0.50:.2f}"),
                            "created": datetime(2024, 1, 1, 12, min(i, 59), 0, tzinfo=timezone.utc),
                            "active": i % 2 == 0,
                            "tags": [f"tag{j}" for j in range(i % 3 + 1)]
                        }
                        for i in range(20)  # Moderate size for comparison
                    ]
                }
            },
            
            "simple_objects": {
                "description": "Simple data types common in APIs",
                "data": {
                    "strings": ["hello", "world", "test", "data"],
                    "numbers": [1, 2, 3, 42, 100],
                    "floats": [3.14, 2.71, 1.41, 0.577],
                    "booleans": [True, False, True],
                    "null_values": [None, None],
                    "mixed_list": ["string", 42, True, None, 3.14]
                }
            },
            
            "nested_structures": {
                "description": "Deeply nested data structures",
                "data": {
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
                }
            },
            
            "datetime_heavy": {
                "description": "Data with many datetime and UUID objects",
                "data": {
                    "events": [
                        {
                            "id": str(uuid.UUID(int=i)),
                            "timestamp": datetime(2024, 1, 1, 12, i % 60, 0, tzinfo=timezone.utc),
                            "type": "event_type",
                            "metadata": {
                                "created": datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                                "updated": datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
                            }
                        }
                        for i in range(15)
                    ]
                }
            }
        }
    
    def benchmark_serialization_speed(self, competitors: List[str], data: Any, 
                                    iterations: int = 10) -> Dict[str, Dict[str, float]]:
        """Benchmark serialization speed across competitors."""
        results = {}
        
        for competitor_name in competitors:
            adapter = self.registry.get_adapter(competitor_name)
            if not adapter:
                logger.warning(f"Skipping {competitor_name} - not available")
                continue
            
            times = []
            error_count = 0
            
            # Warmup
            for _ in range(2):
                try:
                    adapter.serialize(data)
                except Exception:
                    pass
            
            # Actual benchmarks
            for _ in range(iterations):
                try:
                    start = time.perf_counter()
                    serialized = adapter.serialize(data)
                    end = time.perf_counter()
                    times.append(end - start)
                except Exception as e:
                    error_count += 1
                    logger.debug(f"Serialization error for {competitor_name}: {e}")
            
            if times:
                results[competitor_name] = {
                    "mean": mean(times),
                    "min": min(times),
                    "max": max(times),
                    "std": stdev(times) if len(times) > 1 else 0.0,
                    "successful_runs": len(times),
                    "error_count": error_count,
                    "mean_ms": mean(times) * 1000
                }
            else:
                results[competitor_name] = {
                    "error": "All serialization attempts failed",
                    "error_count": error_count
                }
        
        return results
    
    def benchmark_deserialization_speed(self, competitors: List[str], data: Any,
                                      iterations: int = 10) -> Dict[str, Dict[str, float]]:
        """Benchmark deserialization speed across competitors."""
        results = {}
        
        # First serialize the data with each competitor
        serialized_data = {}
        for competitor_name in competitors:
            adapter = self.registry.get_adapter(competitor_name)
            if adapter:
                try:
                    serialized_data[competitor_name] = adapter.serialize(data)
                except Exception as e:
                    logger.warning(f"Could not pre-serialize for {competitor_name}: {e}")
        
        # Now benchmark deserialization
        for competitor_name in competitors:
            if competitor_name not in serialized_data:
                continue
                
            adapter = self.registry.get_adapter(competitor_name)
            if not adapter:
                continue
            
            serialized = serialized_data[competitor_name]
            times = []
            error_count = 0
            
            # Warmup
            for _ in range(2):
                try:
                    adapter.deserialize(serialized)
                except Exception:
                    pass
            
            # Actual benchmarks
            for _ in range(iterations):
                try:
                    start = time.perf_counter()
                    result = adapter.deserialize(serialized)
                    end = time.perf_counter()
                    times.append(end - start)
                except Exception as e:
                    error_count += 1
                    logger.debug(f"Deserialization error for {competitor_name}: {e}")
            
            if times:
                results[competitor_name] = {
                    "mean": mean(times),
                    "min": min(times),
                    "max": max(times),
                    "std": stdev(times) if len(times) > 1 else 0.0,
                    "successful_runs": len(times),
                    "error_count": error_count,
                    "mean_ms": mean(times) * 1000
                }
            else:
                results[competitor_name] = {
                    "error": "All deserialization attempts failed",
                    "error_count": error_count
                }
        
        return results
    
    def benchmark_output_size(self, competitors: List[str], data: Any) -> Dict[str, Dict[str, Any]]:
        """Compare output size across competitors."""
        results = {}
        
        for competitor_name in competitors:
            adapter = self.registry.get_adapter(competitor_name)
            if not adapter:
                continue
            
            try:
                serialized = adapter.serialize(data)
                
                if isinstance(serialized, bytes):
                    size = len(serialized)
                    size_type = "bytes"
                elif isinstance(serialized, str):
                    size = len(serialized.encode('utf-8'))
                    size_type = "utf-8 bytes"
                else:
                    size = len(str(serialized))
                    size_type = "string chars"
                
                results[competitor_name] = {
                    "size": size,
                    "size_type": size_type,
                    "supports_binary": adapter.supports_binary()
                }
            except Exception as e:
                results[competitor_name] = {
                    "error": str(e)
                }
        
        return results
    
    def run_competitive_comparison(self, competitors: Optional[List[str]] = None,
                                 iterations: int = 10) -> Dict[str, Any]:
        """Run complete competitive comparison."""
        if competitors is None:
            competitors = self.registry.list_available_names()
        
        # Filter to only available competitors
        available_competitors = [c for c in competitors if self.registry.get_adapter(c)]
        
        logger.info(f"Running competitive benchmark with: {available_competitors}")
        
        datasets = self.create_benchmark_datasets()
        results = {}
        
        for dataset_name, dataset_info in datasets.items():
            logger.info(f"Benchmarking dataset: {dataset_name}")
            
            data = dataset_info["data"]
            
            # Run all benchmark types
            serialization_results = self.benchmark_serialization_speed(
                available_competitors, data, iterations
            )
            deserialization_results = self.benchmark_deserialization_speed(
                available_competitors, data, iterations
            )
            size_results = self.benchmark_output_size(available_competitors, data)
            
            results[dataset_name] = {
                "description": dataset_info["description"],
                "serialization": serialization_results,
                "deserialization": deserialization_results,
                "output_size": size_results,
                "competitors_tested": available_competitors
            }
        
        # Add summary
        results["summary"] = self._generate_competitive_summary(results, available_competitors)
        
        return results
    
    def _generate_competitive_summary(self, results: Dict[str, Any], 
                                    competitors: List[str]) -> Dict[str, Any]:
        """Generate summary of competitive results."""
        summary = {
            "competitors_tested": competitors,
            "datasets_tested": [k for k in results.keys() if k != "summary"],
            "fastest_serialization": {},
            "fastest_deserialization": {},
            "smallest_output": {},
            "datason_performance": {}
        }
        
        # Find fastest performers for each dataset
        for dataset_name, dataset_results in results.items():
            if dataset_name == "summary":
                continue
            
            # Fastest serialization
            ser_results = dataset_results.get("serialization", {})
            if ser_results:
                fastest_ser = min(
                    [(name, metrics.get("mean", float("inf"))) 
                     for name, metrics in ser_results.items() 
                     if isinstance(metrics, dict) and "mean" in metrics],
                    key=lambda x: x[1],
                    default=(None, None)
                )
                if fastest_ser[0]:
                    summary["fastest_serialization"][dataset_name] = {
                        "library": fastest_ser[0],
                        "time_ms": fastest_ser[1] * 1000
                    }
            
            # Fastest deserialization
            deser_results = dataset_results.get("deserialization", {})
            if deser_results:
                fastest_deser = min(
                    [(name, metrics.get("mean", float("inf"))) 
                     for name, metrics in deser_results.items() 
                     if isinstance(metrics, dict) and "mean" in metrics],
                    key=lambda x: x[1],
                    default=(None, None)
                )
                if fastest_deser[0]:
                    summary["fastest_deserialization"][dataset_name] = {
                        "library": fastest_deser[0],
                        "time_ms": fastest_deser[1] * 1000
                    }
            
            # Smallest output
            size_results = dataset_results.get("output_size", {})
            if size_results:
                smallest = min(
                    [(name, metrics.get("size", float("inf"))) 
                     for name, metrics in size_results.items() 
                     if isinstance(metrics, dict) and "size" in metrics],
                    key=lambda x: x[1],
                    default=(None, None)
                )
                if smallest[0]:
                    summary["smallest_output"][dataset_name] = {
                        "library": smallest[0],
                        "size_bytes": smallest[1]
                    }
            
            # DataSON specific performance
            if "datason" in ser_results and isinstance(ser_results["datason"], dict):
                datason_ser = ser_results["datason"].get("mean", 0) * 1000
                datason_deser = 0
                if "datason" in deser_results and isinstance(deser_results["datason"], dict):
                    datason_deser = deser_results["datason"].get("mean", 0) * 1000
                
                summary["datason_performance"][dataset_name] = {
                    "serialization_ms": datason_ser,
                    "deserialization_ms": datason_deser
                }
        
        return summary 
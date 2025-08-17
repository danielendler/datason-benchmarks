#!/usr/bin/env python3
"""
Competitive Benchmark Suite
============================

Benchmarks DataSON against other serialization libraries using realistic test scenarios.
Enhanced with multi-tier testing for fair capability-based comparisons.
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
    
    def create_multi_tier_datasets(self) -> Dict[str, Any]:
        """Create tiered test datasets for fair capability-based comparisons."""
        return {
            # Tier 1: JSON-Safe Data (ALL libraries can handle this)
            "json_safe_simple": {
                "description": "Simple data types - fair comparison for all libraries",
                "tier": "json_safe",
                "data": {
                    "strings": ["hello", "world", "test", "data"],
                    "numbers": [1, 2, 3, 42, 100],
                    "floats": [3.14, 2.71, 1.41, 0.577],
                    "booleans": [True, False, True],
                    "null_values": [None, None],
                    "mixed_list": ["string", 42, True, None, 3.14]
                }
            },
            
            "json_safe_nested": {
                "description": "Nested structures with basic types only",
                "tier": "json_safe", 
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

            # Tier 2: Object-Enhanced Data (DataSON, jsonpickle, pickle can handle)
            "object_datetime_heavy": {
                "description": "Rich datetime and UUID objects - object-capable libraries",
                "tier": "object_enhanced",
                "data": {
                    "events": [
                        {
                            "id": uuid.UUID(int=i),
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
            },
            
            "object_api_response": {
                "description": "Realistic API response with objects - object handling comparison", 
                "tier": "object_enhanced",
                "data": {
                    "status": "success",
                    "timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                    "request_id": uuid.UUID("12345678-1234-5678-9012-123456789012"),
                    "items": [
                        {
                            "id": i,
                            "name": f"Item {i:03d}",
                            "price": Decimal(f"{19.99 + i * 0.50:.2f}"),
                            "created": datetime(2024, 1, 1, 12, min(i, 59), 0, tzinfo=timezone.utc),
                            "active": i % 2 == 0,
                            "tags": [f"tag{j}" for j in range(i % 3 + 1)]
                        }
                        for i in range(20)
                    ]
                }
            },

            # Tier 3: ML/Complex Data (DataSON, pickle excel here)
            "ml_complex_data": {
                "description": "Complex ML-like data structures - specialized libraries",
                "tier": "ml_complex",
                "data": {
                    "model_metadata": {
                        "name": "classifier_v1",
                        "version": "1.0.0",
                        "created": datetime.now(timezone.utc),
                        "hyperparameters": {
                            "learning_rate": Decimal("0.001"),
                            "batch_size": 32,
                            "epochs": 100
                        }
                    },
                    "training_data": {
                        "features": [[1.0, 2.0, 3.0] for _ in range(100)],
                        "labels": [i % 2 for i in range(100)],
                        "metadata": {
                            "source": "synthetic",
                            "generated_at": datetime.now(timezone.utc)
                        }
                    },
                    "performance_metrics": {
                        "accuracy": Decimal("0.95"),
                        "precision": Decimal("0.93"),
                        "recall": Decimal("0.97"),
                        "f1_score": Decimal("0.95")
                    }
                }
            }
        }
    
    def create_benchmark_datasets(self) -> Dict[str, Any]:
        """Legacy method - now calls multi-tier datasets for backward compatibility."""
        legacy_datasets = self.create_multi_tier_datasets()
        
        # Add legacy mappings for backward compatibility
        legacy_datasets.update({
            "api_response": legacy_datasets["object_api_response"],
            "simple_objects": legacy_datasets["json_safe_simple"], 
            "nested_structures": legacy_datasets["json_safe_nested"],
            "datetime_heavy": legacy_datasets["object_datetime_heavy"]
        })
        
        return legacy_datasets

    def run_capability_based_comparison(self, iterations: int = 10) -> Dict[str, Any]:
        """Run benchmarks segmented by capability tiers for fair comparisons."""
        datasets = self.create_multi_tier_datasets()
        results = {
            "methodology": "capability_based_comparison",
            "tiers": {},
            "summary": {}
        }
        
        # Group datasets by tier
        tiers = {}
        for dataset_name, dataset_info in datasets.items():
            tier = dataset_info.get("tier", "unknown")
            if tier not in tiers:
                tiers[tier] = []
            tiers[tier].append((dataset_name, dataset_info))
        
        # Run benchmarks for each tier
        for tier_name, tier_datasets in tiers.items():
            logger.info(f"🎯 Running {tier_name} tier benchmarks...")
            
            # Get competitors for this tier
            competitors = self.registry.get_competitors_by_capability(tier_name)
            if not competitors:
                logger.warning(f"No competitors available for {tier_name} tier")
                continue
            
            logger.info(f"Testing {len(competitors)} libraries: {competitors}")
            
            tier_results = {}
            
            for dataset_name, dataset_info in tier_datasets:
                logger.info(f"  Dataset: {dataset_name}")
                
                data = dataset_info["data"]
                
                # Run benchmarks only with capable competitors
                serialization_results = self.benchmark_serialization_speed(
                    competitors, data, iterations
                )
                deserialization_results = self.benchmark_deserialization_speed(
                    competitors, data, iterations
                )
                size_results = self.benchmark_output_size(competitors, data)
                
                tier_results[dataset_name] = {
                    "description": dataset_info["description"],
                    "serialization": serialization_results,
                    "deserialization": deserialization_results,
                    "output_size": size_results,
                    "competitors_tested": competitors,
                    "tier": tier_name
                }
            
            results["tiers"][tier_name] = {
                "datasets": tier_results,
                "competitors": competitors,
                "summary": self._generate_tier_summary(tier_results, competitors)
            }
        
        # Generate overall summary
        results["summary"] = self._generate_capability_summary(results["tiers"])
        
        return results

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
                    "mean_ms": mean(times) * 1000,
                    "success_rate": len(times) / (len(times) + error_count) if (len(times) + error_count) > 0 else 0.0
                }
            else:
                results[competitor_name] = {
                    "error": "All serialization attempts failed",
                    "error_count": error_count,
                    "success_rate": 0.0
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
                    "mean_ms": mean(times) * 1000,
                    "success_rate": len(times) / (len(times) + error_count) if (len(times) + error_count) > 0 else 0.0
                }
            else:
                results[competitor_name] = {
                    "error": "All deserialization attempts failed",
                    "error_count": error_count,
                    "success_rate": 0.0
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
        """Run complete competitive comparison - enhanced with capability awareness."""
        if competitors is None:
            # Use capability-based comparison by default for fairer results
            return self.run_capability_based_comparison(iterations)
        
        # Legacy mode - test specific competitors if requested
        available_competitors = [c for c in competitors if self.registry.get_adapter(c)]
        
        logger.info(f"Running legacy competitive benchmark with: {available_competitors}")
        
        datasets = self.create_benchmark_datasets()
        results = {}
        
        for dataset_name, dataset_info in datasets.items():
            if dataset_name in ["api_response", "simple_objects", "nested_structures", "datetime_heavy"]:
                # Skip legacy duplicates in favor of tier-based datasets
                continue
                
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
                "competitors_tested": available_competitors,
                "tier": dataset_info.get("tier", "mixed")
            }
        
        # Add summary
        results["summary"] = self._generate_competitive_summary(results, available_competitors)
        
        return results

    def _generate_tier_summary(self, tier_results: Dict[str, Any], competitors: List[str]) -> Dict[str, Any]:
        """Generate summary for a specific capability tier."""
        summary = {
            "fastest_serialization": {},
            "fastest_deserialization": {},
            "smallest_output": {},
            "datason_performance": {},
            "success_rates": {}
        }
        
        for dataset_name, dataset_data in tier_results.items():
            # Fastest serialization
            ser_results = dataset_data.get("serialization", {})
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
            
            # Success rates
            for lib_name, lib_results in ser_results.items():
                if isinstance(lib_results, dict) and "success_rate" in lib_results:
                    if lib_name not in summary["success_rates"]:
                        summary["success_rates"][lib_name] = []
                    summary["success_rates"][lib_name].append(lib_results["success_rate"])
        
        # Average success rates
        for lib_name, rates in summary["success_rates"].items():
            summary["success_rates"][lib_name] = {
                "average_success_rate": mean(rates),
                "datasets_tested": len(rates)
            }
        
        return summary

    def _generate_capability_summary(self, tier_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary across all capability tiers."""
        return {
            "methodology": "Multi-tier capability-based testing for fair comparisons",
            "tiers_tested": list(tier_results.keys()),
            "key_insights": {
                "json_safe": "All libraries compared on equal footing - basic JSON types only",
                "object_enhanced": "Object-capable libraries compared - datetime, UUID, Decimal handling",
                "ml_complex": "Specialized libraries compared - complex ML workflows and custom objects"
            },
            "fairness_improvement": "Eliminated unfair comparisons of object handling vs string conversion",
            "total_datasets": sum(len(tier["datasets"]) for tier in tier_results.values()),
            "datason_variants_tested": [
                name for tier in tier_results.values() 
                for name in tier.get("competitors", []) 
                if name.startswith("datason")
            ]
        }

    def _generate_competitive_summary(self, results: Dict[str, Any], 
                                    competitors: List[str]) -> Dict[str, Any]:
        """Generate competitive summary - legacy method."""
        summary = {
            "fastest_serialization": {},
            "fastest_deserialization": {},
            "smallest_output": {},
            "datason_performance": {}
        }
        
        for dataset_name, dataset_results in results.items():
            if dataset_name == "summary":
                continue
                
            # Extract serialization results
            ser_results = dataset_results.get("serialization", {})
            deser_results = dataset_results.get("deserialization", {})
            
            # Find fastest serialization
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
            
            # Find fastest deserialization
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
            
            # Enhanced DataSON API variants analysis
            datason_variants = [name for name in ser_results.keys() if name.startswith("datason")]
            if datason_variants and "datason_api_analysis" not in summary:
                summary["datason_api_analysis"] = {
                    "variants_tested": len(datason_variants),
                    "performance_by_variant": {},
                    "api_recommendations": {},
                    "performance_ranking": {},
                    "api_showcase_summary": {
                        "methodology": "DataSON API variants tested as separate competitors",
                        "fairness": "Each API variant optimized for different use cases",
                        "variants_description": {variant: self._get_api_description(variant) for variant in datason_variants},
                        "use_case_mapping": {variant: self._get_api_use_cases(variant) for variant in datason_variants}
                    }
                }
            
            for variant in datason_variants:
                if variant in ser_results and isinstance(ser_results[variant], dict):
                    datason_ser = ser_results[variant].get("mean", 0) * 1000
                    datason_deser = 0
                    if variant in deser_results and isinstance(deser_results[variant], dict):
                        datason_deser = deser_results[variant].get("mean", 0) * 1000
                    
                    # Legacy format for compatibility
                    if variant not in summary["datason_performance"]:
                        summary["datason_performance"][variant] = {}
                    summary["datason_performance"][variant][dataset_name] = {
                        "serialization_ms": datason_ser,
                        "deserialization_ms": datason_deser
                    }
                    
                    # Enhanced API analysis
                    if variant not in summary["datason_api_analysis"]["performance_by_variant"]:
                        summary["datason_api_analysis"]["performance_by_variant"][variant] = {
                            "description": self._get_api_description(variant),
                            "use_cases": self._get_api_use_cases(variant),
                            "datasets": {}
                        }
                    
                    summary["datason_api_analysis"]["performance_by_variant"][variant]["datasets"][dataset_name] = {
                        "serialization_ms": datason_ser,
                        "deserialization_ms": datason_deser,
                        "total_ms": datason_ser + datason_deser
                    }
        
        # Generate API rankings and recommendations
        if "datason_api_analysis" in summary:
            summary["datason_api_analysis"]["performance_ranking"] = self._rank_datason_apis(summary["datason_api_analysis"]["performance_by_variant"])
            summary["datason_api_analysis"]["api_recommendations"] = self._generate_api_recommendations(summary["datason_api_analysis"]["performance_by_variant"])
        
        return summary 
    
    def _get_api_description(self, variant: str) -> str:
        """Get description for DataSON API variant."""
        descriptions = {
            "datason": "Standard DataSON API - balanced features and performance",
            "datason_api": "API-optimized - designed for web services and REST APIs", 
            "datason_ml": "ML-optimized - enhanced support for NumPy, tensors, and ML objects",
            "datason_fast": "Performance-optimized - maximum speed with minimal features",
            "datason_secure": "Security-enhanced - includes PII redaction and data sanitization",
            "datason_smart": "Smart detection - advanced type inference and preservation",
            "datason_perfect": "Perfect fidelity - maximum type preservation and round-trip accuracy"
        }
        return descriptions.get(variant, "DataSON variant with specialized optimizations")
    
    def _get_api_use_cases(self, variant: str):
        """Get recommended use cases for DataSON API variant.""" 
        use_cases = {
            "datason": ["General purpose", "Balanced workloads", "Default choice"],
            "datason_api": ["REST APIs", "Web services", "API responses", "Client-server communication"],
            "datason_ml": ["Machine learning", "Data science", "NumPy arrays", "Model serialization"],
            "datason_fast": ["High-throughput systems", "Performance-critical applications", "Batch processing"],
            "datason_secure": ["Sensitive data", "PII handling", "Compliance requirements", "Data anonymization"],
            "datason_smart": ["Type preservation", "Complex objects", "Schema inference", "Data migration"],
            "datason_perfect": ["Exact round-trips", "Critical data integrity", "Schema validation", "Data archival"]
        }
        return use_cases.get(variant, ["Specialized applications"])
    
    def _rank_datason_apis(self, performance_data):
        """Rank DataSON APIs by performance across datasets."""
        variant_averages = {}
        
        for variant, data in performance_data.items():
            total_times = []
            for dataset_name, metrics in data["datasets"].items():
                total_times.append(metrics["total_ms"])
            
            if total_times:
                variant_averages[variant] = {
                    "average_total_ms": sum(total_times) / len(total_times),
                    "datasets_tested": len(total_times)
                }
        
        # Rank by average performance
        sorted_variants = sorted(
            variant_averages.items(),
            key=lambda x: x[1]["average_total_ms"]
        )
        
        ranking = {}
        for i, (variant, metrics) in enumerate(sorted_variants, 1):
            ranking[variant] = {
                "rank": i,
                "average_total_ms": metrics["average_total_ms"],
                "performance_category": self._get_performance_category(i, len(sorted_variants))
            }
        
        return ranking
    
    def _get_performance_category(self, rank: int, total: int) -> str:
        """Categorize performance based on ranking."""
        if rank == 1:
            return "🥇 Fastest"
        elif rank <= total * 0.3:
            return "🚀 High Performance"
        elif rank <= total * 0.7:
            return "⚖️ Balanced" 
        else:
            return "🔧 Feature-Rich"
    
    def _generate_api_recommendations(self, performance_data):
        """Generate recommendations for when to use each DataSON API."""
        recommendations = {
            "speed_focused": [],
            "feature_focused": [], 
            "balanced": [],
            "specialized": []
        }
        
        # Analyze performance characteristics
        speed_ranking = []
        for variant, data in performance_data.items():
            avg_times = []
            for dataset_metrics in data["datasets"].values():
                avg_times.append(dataset_metrics["total_ms"])
            
            if avg_times:
                avg_total = sum(avg_times) / len(avg_times)
                speed_ranking.append((variant, avg_total))
        
        speed_ranking.sort(key=lambda x: x[1])
        
        # Categorize based on performance and features
        for i, (variant, avg_time) in enumerate(speed_ranking):
            if i < len(speed_ranking) * 0.3:  # Top 30%
                recommendations["speed_focused"].append({
                    "variant": variant,
                    "description": self._get_api_description(variant),
                    "avg_time_ms": avg_time,
                    "use_when": "Performance is the primary concern"
                })
            elif variant in ["datason_secure", "datason_perfect", "datason_smart"]:
                recommendations["feature_focused"].append({
                    "variant": variant,
                    "description": self._get_api_description(variant),
                    "avg_time_ms": avg_time,
                    "use_when": "Advanced features outweigh performance considerations"
                })
            elif variant in ["datason", "datason_api"]:
                recommendations["balanced"].append({
                    "variant": variant,
                    "description": self._get_api_description(variant),
                    "avg_time_ms": avg_time,
                    "use_when": "Good balance of features and performance needed"
                })
            else:
                recommendations["specialized"].append({
                    "variant": variant,
                    "description": self._get_api_description(variant),
                    "avg_time_ms": avg_time,
                    "use_when": "Specific use case requirements"
                })
        
        return recommendations

    def print_enhanced_api_summary(self, results: Dict[str, Any]) -> None:
        """Print an enhanced summary highlighting DataSON API variants."""
        print("\n" + "="*80)
        print("🚀 DATASON API VARIANTS PERFORMANCE ANALYSIS")
        print("="*80)
        
        if "summary" not in results or "datason_api_analysis" not in results["summary"]:
            print("❌ No DataSON API analysis available in results")
            return
        
        api_analysis = results["summary"]["datason_api_analysis"]
        
        # Performance ranking
        if "performance_ranking" in api_analysis:
            print("\n🏆 PERFORMANCE RANKING (Fastest to Slowest):")
            print("-" * 50)
            ranking = api_analysis["performance_ranking"]
            for variant, data in sorted(ranking.items(), key=lambda x: x[1]["rank"]):
                rank = data["rank"]
                category = data["performance_category"]
                avg_ms = data["average_total_ms"]
                description = self._get_api_description(variant)
                
                print(f"{rank:2}. {variant:15} {avg_ms:8.3f}ms  {category}")
                print(f"    {description}")
                print()
        
        # API recommendations by category
        if "api_recommendations" in api_analysis:
            print("\n💡 RECOMMENDED USE CASES:")
            print("-" * 50)
            recommendations = api_analysis["api_recommendations"]
            
            for category, recs in recommendations.items():
                if recs:
                    print(f"\n{category.replace('_', ' ').title()}:")
                    for rec in recs:
                        variant = rec["variant"]
                        use_when = rec["use_when"]
                        avg_time = rec["avg_time_ms"]
                        use_cases = self._get_api_use_cases(variant)
                        
                        print(f"  • {variant} ({avg_time:.3f}ms) - {use_when}")
                        print(f"    Best for: {', '.join(use_cases[:3])}")
        
        # Performance insights
        print(f"\n📊 PERFORMANCE INSIGHTS:")
        print("-" * 50)
        if "performance_ranking" in api_analysis:
            ranking = api_analysis["performance_ranking"]
            if ranking:
                fastest = min(ranking.values(), key=lambda x: x["average_total_ms"])
                slowest = max(ranking.values(), key=lambda x: x["average_total_ms"])
                
                speed_ratio = slowest["average_total_ms"] / fastest["average_total_ms"]
                
                print(f"⚡ Performance spread: {speed_ratio:.1f}x difference between fastest and slowest")
                print(f"🥇 Fastest API: {[k for k, v in ranking.items() if v['rank'] == 1][0]}")
                print(f"🎯 Most balanced: APIs in 'Balanced' category offer good speed/feature trade-offs")
                
                # Specific recommendations
                print(f"\n🎯 QUICK RECOMMENDATIONS:")
                speed_focused = [k for k, v in ranking.items() if v["performance_category"] == "🥇 Fastest"]
                balanced = [k for k, v in ranking.items() if v["performance_category"] == "⚖️ Balanced"]
                
                if speed_focused:
                    print(f"   For speed: {', '.join(speed_focused)}")
                if balanced:
                    print(f"   For general use: {', '.join(balanced)}")
                print(f"   For security: datason_secure (includes PII redaction)")
                print(f"   For ML workflows: datason_ml (NumPy/Pandas optimized)")
        
        print("\n" + "="*80)

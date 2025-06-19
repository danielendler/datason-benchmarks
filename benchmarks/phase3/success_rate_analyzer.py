#!/usr/bin/env python3
"""
Success Rate Analyzer
======================

Comprehensive analysis of serialization/deserialization success rates,
type preservation, and data accuracy across different scenarios.
"""

import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid
import json

logger = logging.getLogger(__name__)


class SuccessRateAnalyzer:
    """Analyzes success rates and accuracy metrics for serialization libraries."""
    
    def __init__(self):
        # Import DataSON with fallback
        try:
            import datason
            self.datason = datason
            self.datason_available = True
        except ImportError:
            logger.error("DataSON not available for success rate analysis")
            self.datason_available = False
        
        # Import competitor registry
        try:
            from competitors.adapter_registry import CompetitorRegistry
            self.registry = CompetitorRegistry()
        except ImportError:
            logger.error("Competitor registry not available")
            self.registry = None
    
    def analyze_success_rates(self, test_data: Dict[str, Any], 
                             iterations: int = 50) -> Dict[str, Any]:
        """Comprehensive success rate analysis across all competitors."""
        if not self.registry:
            return {"error": "Competitor registry not available"}
        
        logger.info("🎯 Analyzing success rates and accuracy metrics...")
        
        results = {
            "methodology": "comprehensive_success_rate_analysis",
            "test_iterations": iterations,
            "metrics": {},
            "accuracy_analysis": {},
            "type_preservation": {},
            "reliability_scores": {}
        }
        
        available_competitors = self.registry.get_available_competitors()
        
        for competitor_name, competitor_info in available_competitors.items():
            logger.info(f"  📊 Testing {competitor_name}...")
            
            competitor_results = self._analyze_competitor_success_rate(
                competitor_name, test_data, iterations
            )
            
            results["metrics"][competitor_name] = competitor_results
        
        # Generate cross-competitor analysis
        results["accuracy_analysis"] = self._analyze_accuracy_patterns(results["metrics"])
        results["type_preservation"] = self._analyze_type_preservation(results["metrics"])
        results["reliability_scores"] = self._calculate_reliability_scores(results["metrics"])
        
        return results
    
    def _analyze_competitor_success_rate(self, competitor_name: str, 
                                       test_data: Dict[str, Any], 
                                       iterations: int) -> Dict[str, Any]:
        """Analyze success rate for a single competitor."""
        adapter = self.registry.get_adapter(competitor_name)
        if not adapter or not adapter.available:
            return {"error": f"Adapter {competitor_name} not available"}
        
        results = {
            "serialization_success_rate": 0.0,
            "deserialization_success_rate": 0.0,
            "type_preservation_score": 0.0,
            "data_accuracy_score": 0.0,
            "performance_reliability": {},
            "failure_patterns": [],
            "detailed_metrics": {}
        }
        
        serialization_successes = 0
        deserialization_successes = 0
        type_preservation_successes = 0
        accuracy_scores = []
        performance_times = []
        
        for iteration in range(iterations):
            try:
                # Test serialization
                start_time = time.perf_counter()
                serialized = adapter.serialize(test_data)
                ser_time = time.perf_counter() - start_time
                
                serialization_successes += 1
                performance_times.append(ser_time)
                
                try:
                    # Test deserialization
                    start_time = time.perf_counter()
                    deserialized = adapter.deserialize(serialized)
                    deser_time = time.perf_counter() - start_time
                    
                    deserialization_successes += 1
                    performance_times.append(deser_time)
                    
                    # Analyze accuracy and type preservation
                    accuracy_score, type_score = self._analyze_reconstruction_quality(
                        original=test_data,
                        reconstructed=deserialized
                    )
                    
                    accuracy_scores.append(accuracy_score)
                    if type_score > 0.9:  # 90% threshold for type preservation
                        type_preservation_successes += 1
                        
                except Exception as deser_error:
                    results["failure_patterns"].append({
                        "type": "deserialization_failure",
                        "iteration": iteration,
                        "error": str(deser_error)
                    })
                    
            except Exception as ser_error:
                results["failure_patterns"].append({
                    "type": "serialization_failure",
                    "iteration": iteration,
                    "error": str(ser_error)
                })
        
        # Calculate success rates
        results["serialization_success_rate"] = serialization_successes / iterations
        results["deserialization_success_rate"] = deserialization_successes / iterations
        results["type_preservation_score"] = type_preservation_successes / iterations
        results["data_accuracy_score"] = mean(accuracy_scores) if accuracy_scores else 0.0
        
        # Performance reliability
        if performance_times:
            results["performance_reliability"] = {
                "mean_time_ms": mean(performance_times) * 1000,
                "std_time_ms": stdev(performance_times) * 1000 if len(performance_times) > 1 else 0.0,
                "coefficient_of_variation": (stdev(performance_times) / mean(performance_times)) if len(performance_times) > 1 and mean(performance_times) > 0 else 0.0
            }
        
        # Detailed metrics
        results["detailed_metrics"] = {
            "total_iterations": iterations,
            "serialization_attempts": iterations,
            "deserialization_attempts": serialization_successes,
            "successful_round_trips": deserialization_successes,
            "failure_rate": 1.0 - (deserialization_successes / iterations)
        }
        
        return results
    
    def _analyze_reconstruction_quality(self, original: Any, 
                                      reconstructed: Any) -> Tuple[float, float]:
        """Analyze the quality of data reconstruction."""
        try:
            accuracy_score = self._calculate_accuracy_score(original, reconstructed)
            type_score = self._calculate_type_preservation_score(original, reconstructed)
            return accuracy_score, type_score
        except Exception as e:
            logger.debug(f"Reconstruction quality analysis failed: {e}")
            return 0.0, 0.0
    
    def _calculate_accuracy_score(self, original: Any, reconstructed: Any) -> float:
        """Calculate data accuracy score (0.0 to 1.0)."""
        try:
            # For simple comparison, convert both to JSON-serializable format
            if original == reconstructed:
                return 1.0
            
            # If types don't match, lower the score
            if type(original) != type(reconstructed):
                return 0.5
            
            # For complex objects, compare string representations
            if str(original) == str(reconstructed):
                return 0.9
                
            # For dictionaries, compare key-by-key
            if isinstance(original, dict) and isinstance(reconstructed, dict):
                total_keys = len(set(original.keys()) | set(reconstructed.keys()))
                if total_keys == 0:
                    return 1.0
                
                matching_keys = 0
                for key in original.keys():
                    if key in reconstructed and original[key] == reconstructed[key]:
                        matching_keys += 1
                
                return matching_keys / total_keys
            
            # For lists, compare element-by-element
            if isinstance(original, list) and isinstance(reconstructed, list):
                if len(original) != len(reconstructed):
                    return 0.3
                
                if len(original) == 0:
                    return 1.0
                
                matching_elements = sum(
                    1 for o, r in zip(original, reconstructed) if o == r
                )
                return matching_elements / len(original)
            
            return 0.3  # Partial match for other cases
            
        except Exception:
            return 0.0
    
    def _calculate_type_preservation_score(self, original: Any, reconstructed: Any) -> float:
        """Calculate type preservation score (0.0 to 1.0)."""
        try:
            score = 0.0
            total_checks = 0
            
            def check_type_preservation(orig, recon, weight=1.0):
                nonlocal score, total_checks
                total_checks += weight
                
                if type(orig) == type(recon):
                    score += weight
                elif isinstance(orig, (int, float)) and isinstance(recon, (int, float)):
                    score += weight * 0.8  # Numeric types are similar
                elif orig is None and recon is None:
                    score += weight
                
                # For containers, recursively check contents
                if isinstance(orig, dict) and isinstance(recon, dict):
                    for key in orig.keys():
                        if key in recon:
                            check_type_preservation(orig[key], recon[key], weight * 0.1)
                
                elif isinstance(orig, list) and isinstance(recon, list):
                    for o, r in zip(orig[:5], recon[:5]):  # Check first 5 elements
                        check_type_preservation(o, r, weight * 0.1)
            
            check_type_preservation(original, reconstructed)
            
            return score / total_checks if total_checks > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_accuracy_patterns(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze accuracy patterns across competitors."""
        analysis = {
            "accuracy_rankings": {},
            "patterns": [],
            "insights": []
        }
        
        # Rank competitors by accuracy
        accuracy_scores = {}
        for competitor, data in metrics.items():
            if isinstance(data, dict) and "data_accuracy_score" in data:
                accuracy_scores[competitor] = data["data_accuracy_score"]
        
        analysis["accuracy_rankings"] = dict(
            sorted(accuracy_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Identify patterns
        high_accuracy = [name for name, score in accuracy_scores.items() if score > 0.95]
        medium_accuracy = [name for name, score in accuracy_scores.items() if 0.8 <= score <= 0.95]
        low_accuracy = [name for name, score in accuracy_scores.items() if score < 0.8]
        
        if high_accuracy:
            analysis["patterns"].append(f"High accuracy (>95%): {', '.join(high_accuracy)}")
        if medium_accuracy:
            analysis["patterns"].append(f"Medium accuracy (80-95%): {', '.join(medium_accuracy)}")
        if low_accuracy:
            analysis["patterns"].append(f"Low accuracy (<80%): {', '.join(low_accuracy)}")
        
        # Generate insights
        if high_accuracy and "datason" in [c.lower() for c in high_accuracy]:
            analysis["insights"].append("DataSON maintains high accuracy across test scenarios")
        
        if len(high_accuracy) < 3:
            analysis["insights"].append("Most libraries struggle with complex data reconstruction")
        
        return analysis
    
    def _analyze_type_preservation(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze type preservation patterns."""
        analysis = {
            "type_preservation_rankings": {},
            "preservation_insights": [],
            "critical_findings": []
        }
        
        # Rank by type preservation
        type_scores = {}
        for competitor, data in metrics.items():
            if isinstance(data, dict) and "type_preservation_score" in data:
                type_scores[competitor] = data["type_preservation_score"]
        
        analysis["type_preservation_rankings"] = dict(
            sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Identify preservation patterns
        perfect_preservation = [name for name, score in type_scores.items() if score > 0.99]
        good_preservation = [name for name, score in type_scores.items() if 0.9 <= score <= 0.99]
        poor_preservation = [name for name, score in type_scores.items() if score < 0.9]
        
        if perfect_preservation:
            analysis["preservation_insights"].append(
                f"Perfect type preservation: {', '.join(perfect_preservation)}"
            )
        
        if poor_preservation:
            analysis["critical_findings"].append(
                f"Poor type preservation detected: {', '.join(poor_preservation)}"
            )
        
        # Critical analysis for datetime/UUID/Decimal preservation
        datason_variants = [name for name in type_scores.keys() if name.lower().startswith('datason')]
        if datason_variants:
            avg_datason_score = mean([type_scores[name] for name in datason_variants])
            if avg_datason_score > 0.95:
                analysis["preservation_insights"].append(
                    "DataSON variants excel at preserving complex Python types"
                )
        
        return analysis
    
    def _calculate_reliability_scores(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall reliability scores for each competitor."""
        reliability = {}
        
        for competitor, data in metrics.items():
            if not isinstance(data, dict):
                continue
            
            # Composite reliability score
            factors = {
                "serialization_success": data.get("serialization_success_rate", 0.0),
                "deserialization_success": data.get("deserialization_success_rate", 0.0),
                "data_accuracy": data.get("data_accuracy_score", 0.0),
                "type_preservation": data.get("type_preservation_score", 0.0)
            }
            
            # Weighted score (higher weight on round-trip success)
            weights = {
                "serialization_success": 0.2,
                "deserialization_success": 0.3,
                "data_accuracy": 0.3,
                "type_preservation": 0.2
            }
            
            reliability_score = sum(
                factors[factor] * weights[factor] 
                for factor in factors
            )
            
            # Performance consistency bonus
            perf_data = data.get("performance_reliability", {})
            if "coefficient_of_variation" in perf_data:
                cv = perf_data["coefficient_of_variation"]
                consistency_bonus = max(0, (1.0 - cv) * 0.1)  # Up to 10% bonus
                reliability_score += consistency_bonus
            
            reliability[competitor] = {
                "overall_score": min(reliability_score, 1.0),  # Cap at 1.0
                "component_scores": factors,
                "grade": self._assign_reliability_grade(reliability_score)
            }
        
        return dict(sorted(reliability.items(), key=lambda x: x[1]["overall_score"], reverse=True))
    
    def _assign_reliability_grade(self, score: float) -> str:
        """Assign letter grade based on reliability score."""
        if score >= 0.95:
            return "A+"
        elif score >= 0.90:
            return "A"
        elif score >= 0.85:
            return "A-"
        elif score >= 0.80:
            return "B+"
        elif score >= 0.75:
            return "B"
        elif score >= 0.70:
            return "B-"
        elif score >= 0.65:
            return "C+"
        elif score >= 0.60:
            return "C"
        else:
            return "F"
    
    def run_comprehensive_success_analysis(self, iterations: int = 100) -> Dict[str, Any]:
        """Run comprehensive success rate analysis with multiple test datasets."""
        if not self.datason_available:
            return {"error": "DataSON not available"}
        
        logger.info("🎯 Running comprehensive success rate analysis...")
        
        # Create diverse test datasets
        test_datasets = {
            "basic_types": {
                "strings": ["hello", "world"],
                "numbers": [1, 2, 3, 42],
                "floats": [3.14, 2.71],
                "booleans": [True, False],
                "null": None
            },
            "datetime_objects": {
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                "timestamps": [
                    datetime.now(timezone.utc) for _ in range(3)
                ]
            },
            "uuid_decimal": {
                "id": uuid.uuid4(),
                "price": Decimal("19.99"),
                "amounts": [Decimal(f"{i}.99") for i in range(5)]
            },
            "complex_nested": {
                "user": {
                    "id": uuid.uuid4(),
                    "created": datetime.now(timezone.utc),
                    "balance": Decimal("1000.50"),
                    "settings": {
                        "notifications": True,
                        "theme": "dark"
                    }
                },
                "metadata": {
                    "version": "1.0",
                    "tags": ["important", "user"],
                    "last_modified": datetime.now(timezone.utc)
                }
            }
        }
        
        results = {
            "methodology": "comprehensive_success_analysis",
            "datasets": {},
            "overall_analysis": {}
        }
        
        # Test each dataset
        for dataset_name, dataset in test_datasets.items():
            logger.info(f"  📊 Testing dataset: {dataset_name}")
            
            dataset_results = self.analyze_success_rates(dataset, iterations)
            results["datasets"][dataset_name] = dataset_results
        
        # Generate overall analysis
        results["overall_analysis"] = self._generate_overall_success_analysis(
            results["datasets"]
        )
        
        return results
    
    def _generate_overall_success_analysis(self, dataset_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall analysis across all datasets."""
        analysis = {
            "cross_dataset_performance": {},
            "consistency_analysis": {},
            "recommendations": []
        }
        
        # Aggregate performance across datasets
        competitor_aggregates = {}
        
        for dataset_name, dataset_data in dataset_results.items():
            if "metrics" in dataset_data:
                for competitor, metrics in dataset_data["metrics"].items():
                    if competitor not in competitor_aggregates:
                        competitor_aggregates[competitor] = {
                            "success_rates": [],
                            "accuracy_scores": [],
                            "type_scores": []
                        }
                    
                    if isinstance(metrics, dict):
                        competitor_aggregates[competitor]["success_rates"].append(
                            metrics.get("deserialization_success_rate", 0.0)
                        )
                        competitor_aggregates[competitor]["accuracy_scores"].append(
                            metrics.get("data_accuracy_score", 0.0)
                        )
                        competitor_aggregates[competitor]["type_scores"].append(
                            metrics.get("type_preservation_score", 0.0)
                        )
        
        # Calculate averages and consistency
        for competitor, data in competitor_aggregates.items():
            if data["success_rates"]:
                analysis["cross_dataset_performance"][competitor] = {
                    "avg_success_rate": mean(data["success_rates"]),
                    "avg_accuracy": mean(data["accuracy_scores"]),
                    "avg_type_preservation": mean(data["type_scores"]),
                    "consistency_score": 1.0 - (stdev(data["success_rates"]) if len(data["success_rates"]) > 1 else 0.0)
                }
        
        # Generate recommendations
        top_performers = sorted(
            analysis["cross_dataset_performance"].items(),
            key=lambda x: x[1]["avg_success_rate"],
            reverse=True
        )[:3]
        
        if top_performers:
            analysis["recommendations"].append(
                f"Top reliability: {', '.join([name for name, _ in top_performers])}"
            )
        
        # DataSON-specific analysis
        datason_variants = [
            name for name in analysis["cross_dataset_performance"].keys()
            if name.lower().startswith('datason')
        ]
        
        if datason_variants:
            avg_datason_success = mean([
                analysis["cross_dataset_performance"][name]["avg_success_rate"]
                for name in datason_variants
            ])
            
            if avg_datason_success > 0.95:
                analysis["recommendations"].append(
                    "DataSON variants provide excellent reliability across data types"
                )
        
        return analysis 
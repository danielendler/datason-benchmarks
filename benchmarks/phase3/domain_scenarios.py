#!/usr/bin/env python3
"""
Domain-Specific Benchmark Scenarios
====================================

Realistic use case scenarios for testing DataSON against competitors
in real-world contexts like Web APIs, ML pipelines, and data processing.
"""

import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from statistics import mean, stdev
from typing import Any, Dict, List, Optional
import random

logger = logging.getLogger(__name__)


class DomainScenarioBenchmarkSuite:
    """Domain-specific realistic benchmark scenarios."""
    
    def __init__(self):
        # Check for DataSON availability
        try:
            import datason
            self.datason = datason
            self.datason_available = True
        except ImportError:
            logger.error("DataSON not available for domain scenario testing")
            self.datason_available = False
        
        # Optional ML libraries
        self.numpy_available = False
        self.pandas_available = False
        
        try:
            import numpy as np
            self.numpy = np
            self.numpy_available = True
        except ImportError:
            logger.debug("NumPy not available for ML scenarios")
        
        try:
            import pandas as pd
            self.pandas = pd
            self.pandas_available = True
        except ImportError:
            logger.debug("Pandas not available for ML scenarios")
    
    def create_web_api_scenario(self) -> Dict[str, Any]:
        """Create realistic Web API response scenario."""
        return {
            "web_api_response": {
                "description": "E-commerce API user profile response",
                "domain": "web_api",
                "data": {
                    "status": "success",
                    "timestamp": datetime.now(timezone.utc),
                    "request_id": uuid.uuid4(),
                    "user": {
                        "user_id": uuid.uuid4(),
                        "email": "john.doe@example.com",
                        "created_at": datetime(2023, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
                        "last_login": datetime.now(timezone.utc) - timedelta(hours=2),
                        "profile": {
                            "first_name": "John",
                            "last_name": "Doe",
                            "age": 32,
                            "phone": "+1-555-123-4567",
                            "preferences": {
                                "newsletter": True,
                                "notifications": False,
                                "theme": "dark"
                            }
                        },
                        "subscription": {
                            "plan": "premium",
                            "price": Decimal("29.99"),
                            "expires_at": datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
                            "auto_renew": True
                        }
                    },
                    "recommendations": [
                        {
                            "item_id": uuid.uuid4(),
                            "name": f"Product {i}",
                            "price": Decimal(f"{19.99 + i * 5.00:.2f}"),
                            "category": random.choice(["electronics", "books", "clothing", "sports"]),
                            "rating": round(random.uniform(3.5, 5.0), 1),
                            "available": i % 3 != 0
                        }
                        for i in range(10)
                    ],
                    "session": {
                        "session_id": str(uuid.uuid4()),
                        "expires_at": datetime.now(timezone.utc) + timedelta(hours=24),
                        "csrf_token": f"csrf_{uuid.uuid4().hex[:16]}"
                    }
                }
            }
        }
    
    def create_ml_training_scenario(self) -> Dict[str, Any]:
        """Create realistic ML training pipeline scenario."""
        scenario_data = {
            "ml_training_pipeline": {
                "description": "ML model training pipeline data",
                "domain": "machine_learning",
                "data": {
                    "experiment": {
                        "experiment_id": uuid.uuid4(),
                        "model_name": "fraud_detection_v2",
                        "created_at": datetime.now(timezone.utc),
                        "hyperparameters": {
                            "learning_rate": Decimal("0.001"),
                            "batch_size": 64,
                            "epochs": 150,
                            "dropout_rate": Decimal("0.3"),
                            "l2_regularization": Decimal("0.01")
                        },
                        "optimizer": "adam",
                        "loss_function": "binary_crossentropy"
                    },
                    "dataset_metadata": {
                        "source": "fraud_detection_2024",
                        "total_samples": 50000,
                        "features": 47,
                        "positive_class_ratio": Decimal("0.023"),
                        "train_split": Decimal("0.7"),
                        "validation_split": Decimal("0.15"),
                        "test_split": Decimal("0.15"),
                        "created_at": datetime(2024, 1, 10, 8, 0, 0, tzinfo=timezone.utc)
                    },
                    "training_metrics": {
                        "epoch": 150,
                        "train_loss": Decimal("0.0234"),
                        "val_loss": Decimal("0.0267"),
                        "train_accuracy": Decimal("0.9876"),
                        "val_accuracy": Decimal("0.9834"),
                        "precision": Decimal("0.8456"),
                        "recall": Decimal("0.7912"),
                        "f1_score": Decimal("0.8175"),
                        "auc_roc": Decimal("0.9234")
                    },
                    "model_artifacts": {
                        "model_size_bytes": 2456789,
                        "checkpoint_path": "/models/fraud_detection_v2/checkpoint_150.ckpt",
                        "saved_at": datetime.now(timezone.utc),
                        "version": "2.1.0"
                    }
                }
            }
        }
        
        # Add numpy arrays if available
        if self.numpy_available:
            import numpy as np
            scenario_data["ml_training_pipeline"]["data"]["sample_data"] = {
                "feature_matrix": np.random.rand(100, 47),
                "target_vector": np.random.randint(0, 2, size=100),
                "feature_importance": np.random.rand(47),
                "confusion_matrix": np.array([[9876, 124], [89, 911]])
            }
        
        # Add pandas DataFrames if available
        if self.pandas_available:
            import pandas as pd
            scenario_data["ml_training_pipeline"]["data"]["training_history"] = {
                "metrics_df": pd.DataFrame({
                    "epoch": range(1, 11),
                    "train_loss": np.random.uniform(0.02, 0.1, 10),
                    "val_loss": np.random.uniform(0.025, 0.12, 10),
                    "train_acc": np.random.uniform(0.95, 0.99, 10),
                    "val_acc": np.random.uniform(0.93, 0.98, 10)
                })
            }
        
        return scenario_data
    
    def create_data_pipeline_scenario(self) -> Dict[str, Any]:
        """Create realistic data processing pipeline scenario."""
        return {
            "data_pipeline_batch": {
                "description": "ETL pipeline batch processing job",
                "domain": "data_engineering",
                "data": {
                    "batch_info": {
                        "batch_id": uuid.uuid4(),
                        "job_name": "daily_user_analytics",
                        "started_at": datetime.now(timezone.utc),
                        "expected_completion": datetime.now(timezone.utc) + timedelta(hours=2),
                        "priority": "high",
                        "retries": 0,
                        "max_retries": 3
                    },
                    "data_sources": [
                        {
                            "source_id": f"src_{i}",
                            "name": f"database_{i}",
                            "type": random.choice(["postgresql", "mysql", "mongodb", "redis"]),
                            "connection_string": f"db://server{i}:5432/analytics",
                            "last_updated": datetime.now(timezone.utc) - timedelta(hours=random.randint(1, 48)),
                            "record_count": random.randint(10000, 1000000),
                            "size_bytes": random.randint(1024*1024, 1024*1024*1024)
                        }
                        for i in range(5)
                    ],
                    "processing_stats": {
                        "records_processed": 847563,
                        "records_failed": 234,
                        "success_rate": Decimal("0.9997"),
                        "avg_processing_time_ms": Decimal("12.34"),
                        "peak_memory_mb": 2048,
                        "cpu_utilization": Decimal("0.87")
                    },
                    "output_destinations": [
                        {
                            "destination_id": uuid.uuid4(),
                            "type": "data_warehouse",
                            "location": "s3://analytics-bucket/daily/2024/06/19/",
                            "format": "parquet",
                            "compression": "snappy",
                            "partitioned_by": ["date", "region"],
                            "size_estimate_gb": Decimal("45.7")
                        },
                        {
                            "destination_id": uuid.uuid4(),
                            "type": "real_time_stream",
                            "topic": "user_analytics_stream",
                            "kafka_cluster": "analytics-cluster-prod",
                            "throughput_msg_per_sec": 1250
                        }
                    ],
                    "data_quality": {
                        "completeness_score": Decimal("0.987"),
                        "accuracy_score": Decimal("0.994"),
                        "consistency_score": Decimal("0.991"),
                        "validity_score": Decimal("0.996"),
                        "anomalies_detected": 12,
                        "quality_rules_passed": 47,
                        "quality_rules_total": 50
                    }
                }
            }
        }
    
    def create_financial_api_scenario(self) -> Dict[str, Any]:
        """Create realistic financial services API scenario."""
        return {
            "financial_transaction_api": {
                "description": "Banking API transaction processing",
                "domain": "financial_services",
                "data": {
                    "transaction": {
                        "transaction_id": uuid.uuid4(),
                        "account_id": uuid.uuid4(),
                        "type": "transfer",
                        "amount": Decimal("1250.75"),
                        "currency": "USD",
                        "timestamp": datetime.now(timezone.utc),
                        "description": "Online transfer to savings",
                        "reference": f"TXN{random.randint(100000, 999999)}",
                        "status": "completed"
                    },
                    "account": {
                        "account_id": uuid.uuid4(),
                        "account_number": "****7890",
                        "account_type": "checking",
                        "balance": Decimal("15847.32"),
                        "available_balance": Decimal("15597.32"),
                        "currency": "USD",
                        "last_updated": datetime.now(timezone.utc),
                        "interest_rate": Decimal("0.0125")
                    },
                    "customer": {
                        "customer_id": uuid.uuid4(),
                        "masked_ssn": "***-**-1234",
                        "email_hash": "sha256:a1b2c3d4...",
                        "credit_score": 785,
                        "risk_level": "low",
                        "account_created": datetime(2019, 3, 15, tzinfo=timezone.utc),
                        "last_login": datetime.now(timezone.utc) - timedelta(hours=6)
                    },
                    "compliance": {
                        "aml_check": "passed",
                        "kyc_status": "verified",
                        "sanctions_check": "clear",
                        "pep_check": "clear",
                        "risk_score": Decimal("2.3"),
                        "compliance_flags": [],
                        "last_reviewed": datetime(2024, 6, 1, tzinfo=timezone.utc)
                    },
                    "audit_trail": [
                        {
                            "event_id": uuid.uuid4(),
                            "action": "transaction_initiated",
                            "timestamp": datetime.now(timezone.utc) - timedelta(seconds=30),
                            "user_agent": "BankingApp/iOS 2.1.0",
                            "ip_address": "192.168.1.100",
                            "location": "New York, NY"
                        },
                        {
                            "event_id": uuid.uuid4(),
                            "action": "fraud_check_completed",
                            "timestamp": datetime.now(timezone.utc) - timedelta(seconds=25),
                            "result": "approved",
                            "confidence_score": Decimal("0.97")
                        }
                    ]
                }
            }
        }
    
    def run_domain_scenario_benchmarks(self, iterations: int = 10) -> Dict[str, Any]:
        """Run benchmarks across all domain scenarios."""
        if not self.datason_available:
            return {"error": "DataSON not available"}
        
        logger.info("🌍 Running domain-specific scenario benchmarks...")
        
        # Create all scenarios
        scenarios = {}
        scenarios.update(self.create_web_api_scenario())
        scenarios.update(self.create_ml_training_scenario())
        scenarios.update(self.create_data_pipeline_scenario())
        scenarios.update(self.create_financial_api_scenario())
        
        results = {
            "methodology": "domain_scenario_benchmarking",
            "scenarios": {},
            "cross_scenario_analysis": {},
            "recommendations": {}
        }
        
        # Benchmark each scenario
        for scenario_name, scenario_info in scenarios.items():
            logger.info(f"  📊 Benchmarking {scenario_name}...")
            
            scenario_results = self._benchmark_scenario(
                scenario_name=scenario_name,
                scenario_data=scenario_info["data"],
                domain=scenario_info["domain"],
                iterations=iterations
            )
            
            results["scenarios"][scenario_name] = {
                "description": scenario_info["description"],
                "domain": scenario_info["domain"],
                "results": scenario_results
            }
        
        # Generate cross-scenario analysis
        results["cross_scenario_analysis"] = self._analyze_cross_scenario_performance(
            results["scenarios"]
        )
        
        # Generate domain-specific recommendations
        results["recommendations"] = self._generate_domain_recommendations(
            results["scenarios"]
        )
        
        return results
    
    def _benchmark_scenario(self, scenario_name: str, scenario_data: Any, 
                          domain: str, iterations: int) -> Dict[str, Any]:
        """Benchmark a single domain scenario."""
        results = {
            "serialization": {},
            "deserialization": {},
            "accuracy_metrics": {},
            "domain_specific_analysis": {}
        }
        
        # Test different DataSON methods based on domain
        datason_methods = self._get_domain_appropriate_methods(domain)
        
        for method_name, method_func in datason_methods.items():
            logger.debug(f"    Testing {method_name} for {domain}...")
            
            # Serialization benchmarking
            ser_times = []
            serialized_outputs = []
            
            for _ in range(iterations):
                try:
                    start = time.perf_counter()
                    serialized = method_func(scenario_data)
                    end = time.perf_counter()
                    
                    ser_times.append(end - start)
                    serialized_outputs.append(serialized)
                except Exception as e:
                    logger.debug(f"Serialization failed for {method_name}: {e}")
            
            if ser_times:
                results["serialization"][method_name] = {
                    "mean_ms": mean(ser_times) * 1000,
                    "min_ms": min(ser_times) * 1000,
                    "max_ms": max(ser_times) * 1000,
                    "std_ms": stdev(ser_times) * 1000 if len(ser_times) > 1 else 0.0,
                    "successful_runs": len(ser_times),
                    "output_size": len(serialized_outputs[0]) if serialized_outputs else 0
                }
                
                # Deserialization benchmarking
                if serialized_outputs:
                    deser_times = []
                    reconstruction_successes = 0
                    
                    for serialized in serialized_outputs[:5]:  # Test first 5 outputs
                        try:
                            start = time.perf_counter()
                            reconstructed = self.datason.deserialize(serialized)
                            end = time.perf_counter()
                            
                            deser_times.append(end - start)
                            reconstruction_successes += 1
                        except Exception as e:
                            logger.debug(f"Deserialization failed for {method_name}: {e}")
                    
                    if deser_times:
                        results["deserialization"][method_name] = {
                            "mean_ms": mean(deser_times) * 1000,
                            "success_rate": reconstruction_successes / len(serialized_outputs[:5]),
                            "successful_runs": len(deser_times)
                        }
        
        # Domain-specific analysis
        results["domain_specific_analysis"] = self._analyze_domain_performance(
            domain, results["serialization"], scenario_data
        )
        
        return results
    
    def _get_domain_appropriate_methods(self, domain: str) -> Dict[str, Any]:
        """Get DataSON methods appropriate for specific domain."""
        methods = {}
        
        # Always test default serialize
        if hasattr(self.datason, 'serialize'):
            methods["serialize"] = self.datason.serialize
        
        # Domain-specific methods
        if domain == "web_api" and hasattr(self.datason, 'dump_api'):
            methods["dump_api"] = self.datason.dump_api
        elif domain == "machine_learning" and hasattr(self.datason, 'dump_ml'):
            methods["dump_ml"] = self.datason.dump_ml
        elif domain == "financial_services" and hasattr(self.datason, 'dump_secure'):
            methods["dump_secure"] = self.datason.dump_secure
        
        # Always test fast method if available
        if hasattr(self.datason, 'dump_fast'):
            methods["dump_fast"] = self.datason.dump_fast
        
        return methods
    
    def _analyze_domain_performance(self, domain: str, serialization_results: Dict[str, Any], 
                                  scenario_data: Any) -> Dict[str, Any]:
        """Analyze performance characteristics specific to domain."""
        analysis = {
            "domain": domain,
            "best_method": None,
            "domain_insights": []
        }
        
        if not serialization_results:
            return analysis
        
        # Find best performing method
        best_time = float('inf')
        best_method = None
        
        for method, results in serialization_results.items():
            if results["mean_ms"] < best_time:
                best_time = results["mean_ms"]
                best_method = method
        
        analysis["best_method"] = best_method
        
        # Domain-specific insights
        if domain == "web_api":
            analysis["domain_insights"].append("API responses benefit from consistent UUID/datetime formatting")
            if "dump_api" in serialization_results:
                analysis["domain_insights"].append("dump_api() optimized for web API compatibility")
        
        elif domain == "machine_learning":
            analysis["domain_insights"].append("ML workflows need framework-specific optimizations")
            if "dump_ml" in serialization_results:
                analysis["domain_insights"].append("dump_ml() handles ML framework objects efficiently")
        
        elif domain == "financial_services":
            analysis["domain_insights"].append("Financial data requires security and compliance features")
            if "dump_secure" in serialization_results:
                analysis["domain_insights"].append("dump_secure() provides PII protection for sensitive data")
        
        elif domain == "data_engineering":
            analysis["domain_insights"].append("Data pipelines prioritize speed and reliability")
            analysis["domain_insights"].append("Large batch processing benefits from fast serialization")
        
        return analysis
    
    def _analyze_cross_scenario_performance(self, scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance patterns across different scenarios."""
        return {
            "methodology": "Cross-scenario performance analysis",
            "performance_consistency": self._calculate_performance_consistency(scenarios),
            "method_rankings": self._rank_methods_across_scenarios(scenarios),
            "domain_patterns": self._identify_domain_patterns(scenarios)
        }
    
    def _calculate_performance_consistency(self, scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate how consistently each method performs across scenarios."""
        method_performances = {}
        
        for scenario_name, scenario_data in scenarios.items():
            if "results" in scenario_data and "serialization" in scenario_data["results"]:
                for method, perf in scenario_data["results"]["serialization"].items():
                    if method not in method_performances:
                        method_performances[method] = []
                    method_performances[method].append(perf["mean_ms"])
        
        consistency = {}
        for method, times in method_performances.items():
            if len(times) > 1:
                consistency[method] = {
                    "avg_performance_ms": mean(times),
                    "std_deviation": stdev(times),
                    "coefficient_of_variation": stdev(times) / mean(times) if mean(times) > 0 else 0,
                    "scenarios_tested": len(times)
                }
        
        return consistency
    
    def _rank_methods_across_scenarios(self, scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """Rank DataSON methods across all scenarios."""
        method_scores = {}
        
        for scenario_name, scenario_data in scenarios.items():
            if "results" in scenario_data and "serialization" in scenario_data["results"]:
                # Rank methods in this scenario (1 = fastest)
                methods = scenario_data["results"]["serialization"]
                sorted_methods = sorted(methods.items(), key=lambda x: x[1]["mean_ms"])
                
                for rank, (method, _) in enumerate(sorted_methods, 1):
                    if method not in method_scores:
                        method_scores[method] = []
                    method_scores[method].append(rank)
        
        # Calculate average rankings
        average_rankings = {}
        for method, ranks in method_scores.items():
            average_rankings[method] = {
                "average_rank": mean(ranks),
                "best_rank": min(ranks),
                "worst_rank": max(ranks),
                "scenarios_tested": len(ranks)
            }
        
        return dict(sorted(average_rankings.items(), key=lambda x: x[1]["average_rank"]))
    
    def _identify_domain_patterns(self, scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """Identify performance patterns by domain."""
        domain_patterns = {}
        
        for scenario_name, scenario_data in scenarios.items():
            domain = scenario_data.get("domain", "unknown")
            if domain not in domain_patterns:
                domain_patterns[domain] = {
                    "scenarios": [],
                    "recommended_methods": [],
                    "performance_characteristics": []
                }
            
            domain_patterns[domain]["scenarios"].append(scenario_name)
            
            if "results" in scenario_data and "domain_specific_analysis" in scenario_data["results"]:
                analysis = scenario_data["results"]["domain_specific_analysis"]
                if "best_method" in analysis and analysis["best_method"]:
                    domain_patterns[domain]["recommended_methods"].append(analysis["best_method"])
                
                if "domain_insights" in analysis:
                    domain_patterns[domain]["performance_characteristics"].extend(
                        analysis["domain_insights"]
                    )
        
        # Deduplicate and summarize
        for domain in domain_patterns:
            # Find most common recommended method
            methods = domain_patterns[domain]["recommended_methods"]
            if methods:
                method_counts = {}
                for method in methods:
                    method_counts[method] = method_counts.get(method, 0) + 1
                domain_patterns[domain]["primary_recommendation"] = max(
                    method_counts.items(), key=lambda x: x[1]
                )[0]
            
            # Deduplicate characteristics
            domain_patterns[domain]["performance_characteristics"] = list(set(
                domain_patterns[domain]["performance_characteristics"]
            ))
        
        return domain_patterns
    
    def _generate_domain_recommendations(self, scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable recommendations for each domain."""
        recommendations = {
            "by_domain": {},
            "general": [],
            "performance_optimization": {}
        }
        
        # Domain-specific recommendations
        for scenario_name, scenario_data in scenarios.items():
            domain = scenario_data.get("domain", "unknown")
            
            if domain not in recommendations["by_domain"]:
                recommendations["by_domain"][domain] = {
                    "recommended_method": None,
                    "use_cases": [],
                    "performance_notes": []
                }
            
            if "results" in scenario_data and "domain_specific_analysis" in scenario_data["results"]:
                analysis = scenario_data["results"]["domain_specific_analysis"]
                
                if "best_method" in analysis:
                    recommendations["by_domain"][domain]["recommended_method"] = analysis["best_method"]
                
                if "domain_insights" in analysis:
                    recommendations["by_domain"][domain]["performance_notes"].extend(
                        analysis["domain_insights"]
                    )
            
            recommendations["by_domain"][domain]["use_cases"].append(scenario_data["description"])
        
        # General recommendations
        recommendations["general"] = [
            "Use dump_api() for web APIs and REST services",
            "Use dump_ml() for machine learning and data science workflows",
            "Use dump_secure() for financial and healthcare data",
            "Use dump_fast() for high-throughput data processing",
            "Default serialize() provides good balance for general use"
        ]
        
        return recommendations 
#!/usr/bin/env python3
"""
Phase 3 Benchmark Suite: Realistic Use Case Scenarios
======================================================

Comprehensive testing of DataSON and competitors across realistic domain scenarios
with detailed success rate analysis and accuracy metrics.

Phase 3 Features:
- Domain-Specific Benchmarks (Web API, ML, Financial, Data Pipeline)
- Success Rate Analysis (Accuracy, Type Preservation, Reliability)
- Cross-Scenario Performance Analysis
- Actionable Recommendations
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .domain_scenarios import DomainScenarioBenchmarkSuite
from .success_rate_analyzer import SuccessRateAnalyzer

logger = logging.getLogger(__name__)


class Phase3BenchmarkSuite:
    """Phase 3: Realistic Use Case Scenarios benchmark suite."""
    
    def __init__(self):
        self.domain_suite = DomainScenarioBenchmarkSuite()
        self.success_analyzer = SuccessRateAnalyzer()
        
        # Check availability
        self.available = (
            self.domain_suite.datason_available and 
            self.success_analyzer.datason_available
        )
    
    def run_phase3_complete(self, iterations: int = 10) -> Dict[str, Any]:
        """Run complete Phase 3 testing: Domain scenarios + Success analysis."""
        if not self.available:
            return {"error": "Phase 3 components not available"}
        
        logger.info("🚀 Running Phase 3: Realistic Use Case Scenarios")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        results = {
            "suite_type": "phase3_realistic_scenarios",
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "iterations": iterations,
                "phase": 3,
                "description": "Realistic domain scenarios with comprehensive success analysis"
            },
            "domain_scenarios": {},
            "success_analysis": {},
            "cross_analysis": {},
            "execution_time": 0.0,
            "summary": {}
        }
        
        # 1. Run Domain-Specific Scenarios
        logger.info("🌍 Phase 3.1: Domain-Specific Benchmarks")
        domain_results = self.domain_suite.run_domain_scenario_benchmarks(iterations)
        results["domain_scenarios"] = domain_results
        
        # 2. Run Success Rate Analysis
        logger.info("🎯 Phase 3.2: Comprehensive Success Rate Analysis")
        success_results = self.success_analyzer.run_comprehensive_success_analysis(iterations)
        results["success_analysis"] = success_results
        
        # 3. Cross-Analysis Integration
        logger.info("📊 Phase 3.3: Cross-Scenario Integration Analysis")
        cross_results = self._integrate_domain_and_success_analysis(
            domain_results, success_results
        )
        results["cross_analysis"] = cross_results
        
        # 4. Generate Phase 3 Summary
        results["execution_time"] = time.time() - start_time
        results["summary"] = self._generate_phase3_summary(results)
        
        logger.info(f"✅ Phase 3 completed in {results['execution_time']:.2f}s")
        
        return results
    
    def run_domain_focused(self, domain: str = "all", iterations: int = 10) -> Dict[str, Any]:
        """Run domain-focused testing for specific use cases."""
        if not self.available:
            return {"error": "Phase 3 components not available"}
        
        logger.info(f"🎯 Running domain-focused testing: {domain}")
        
        if domain == "all":
            return self.domain_suite.run_domain_scenario_benchmarks(iterations)
        
        # Single domain testing (future enhancement)
        return {"error": f"Single domain testing for {domain} not yet implemented"}
    
    def run_success_focused(self, iterations: int = 50) -> Dict[str, Any]:
        """Run success rate focused analysis with higher iteration count."""
        if not self.available:
            return {"error": "Success analyzer not available"}
        
        logger.info("🎯 Running success-focused analysis")
        
        return self.success_analyzer.run_comprehensive_success_analysis(iterations)
    
    def _integrate_domain_and_success_analysis(self, domain_results: Dict[str, Any], 
                                             success_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate domain scenarios with success rate analysis."""
        integration = {
            "methodology": "domain_success_integration",
            "performance_consistency": {},
            "domain_reliability_mapping": {},
            "optimization_opportunities": [],
            "critical_insights": []
        }
        
        # Map domain performance to success metrics
        if "scenarios" in domain_results and "datasets" in success_results:
            domain_perf = domain_results["scenarios"]
            success_metrics = success_results["datasets"]
            
            # Performance consistency analysis
            integration["performance_consistency"] = self._analyze_performance_consistency(
                domain_perf, success_metrics
            )
            
            # Domain-reliability mapping
            integration["domain_reliability_mapping"] = self._map_domain_to_reliability(
                domain_perf, success_metrics
            )
        
        # Optimization opportunities
        integration["optimization_opportunities"] = self._identify_optimization_opportunities(
            domain_results, success_results
        )
        
        # Critical insights
        integration["critical_insights"] = self._extract_critical_insights(
            domain_results, success_results
        )
        
        return integration
    
    def _analyze_performance_consistency(self, domain_perf: Dict[str, Any], 
                                       success_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance consistency between domain scenarios and success tests."""
        consistency = {
            "methodology": "performance_consistency_analysis",
            "datason_variants": {},
            "competitor_comparison": {},
            "consistency_scores": {}
        }
        
        # Focus on DataSON variants across different test types
        datason_methods = ["datason", "datason_api", "datason_ml", "datason_fast", "datason_secure"]
        
        for method in datason_methods:
            method_consistency = {
                "domain_performance": {},
                "success_metrics": {},
                "consistency_score": 0.0
            }
            
            # Extract domain performance
            for scenario_name, scenario_data in domain_perf.items():
                if "results" in scenario_data and "serialization" in scenario_data["results"]:
                    ser_results = scenario_data["results"]["serialization"]
                    if method in ser_results:
                        method_consistency["domain_performance"][scenario_name] = {
                            "mean_ms": ser_results[method]["mean_ms"],
                            "success_rate": ser_results[method].get("successful_runs", 0) / 10  # Assuming 10 iterations
                        }
            
            # Extract success metrics
            for dataset_name, dataset_data in success_metrics.items():
                if "metrics" in dataset_data and method in dataset_data["metrics"]:
                    metrics = dataset_data["metrics"][method]
                    if isinstance(metrics, dict):
                        method_consistency["success_metrics"][dataset_name] = {
                            "success_rate": metrics.get("deserialization_success_rate", 0.0),
                            "accuracy": metrics.get("data_accuracy_score", 0.0)
                        }
            
            # Calculate consistency score
            if method_consistency["domain_performance"] and method_consistency["success_metrics"]:
                # Simple consistency measure: variance in success rates
                all_success_rates = []
                
                for data in method_consistency["domain_performance"].values():
                    all_success_rates.append(data["success_rate"])
                
                for data in method_consistency["success_metrics"].values():
                    all_success_rates.append(data["success_rate"])
                
                if len(all_success_rates) > 1:
                    import statistics
                    variance = statistics.variance(all_success_rates)
                    method_consistency["consistency_score"] = max(0.0, 1.0 - variance)
            
            consistency["datason_variants"][method] = method_consistency
        
        return consistency
    
    def _map_domain_to_reliability(self, domain_perf: Dict[str, Any], 
                                 success_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Map domain scenarios to reliability characteristics."""
        mapping = {
            "domain_reliability_profiles": {},
            "recommended_methods_by_domain": {},
            "reliability_insights": []
        }
        
        # Analyze each domain
        for scenario_name, scenario_data in domain_perf.items():
            domain = scenario_data.get("domain", "unknown")
            
            if domain not in mapping["domain_reliability_profiles"]:
                mapping["domain_reliability_profiles"][domain] = {
                    "scenarios": [],
                    "avg_performance": {},
                    "reliability_characteristics": []
                }
            
            mapping["domain_reliability_profiles"][domain]["scenarios"].append(scenario_name)
            
            # Find best performing method for this domain
            if "results" in scenario_data and "serialization" in scenario_data["results"]:
                ser_results = scenario_data["results"]["serialization"]
                
                best_method = None
                best_time = float('inf')
                
                for method, results in ser_results.items():
                    if results["mean_ms"] < best_time:
                        best_time = results["mean_ms"]
                        best_method = method
                
                if best_method:
                    if domain not in mapping["recommended_methods_by_domain"]:
                        mapping["recommended_methods_by_domain"][domain] = []
                    mapping["recommended_methods_by_domain"][domain].append(best_method)
        
        # Generate insights
        for domain, data in mapping["recommended_methods_by_domain"].items():
            if data:
                most_common = max(set(data), key=data.count)
                mapping["reliability_insights"].append(
                    f"Domain '{domain}' performs best with {most_common}"
                )
        
        return mapping
    
    def _identify_optimization_opportunities(self, domain_results: Dict[str, Any], 
                                           success_results: Dict[str, Any]) -> List[str]:
        """Identify specific optimization opportunities."""
        opportunities = []
        
        # Check for underperforming scenarios
        if "scenarios" in domain_results:
            for scenario_name, scenario_data in domain_results["scenarios"].items():
                if "results" in scenario_data and "serialization" in scenario_data["results"]:
                    ser_results = scenario_data["results"]["serialization"]
                    
                    # Look for scenarios where default perform worse than optimized methods
                    if "datason" in ser_results and "datason_fast" in ser_results:
                        default_time = ser_results["datason"]["mean_ms"]
                        fast_time = ser_results["datason_fast"]["mean_ms"]
                        
                        if fast_time < default_time * 0.8:  # 20% improvement
                            opportunities.append(
                                f"Use dump_fast() for {scenario_name} - {((default_time - fast_time) / default_time * 100):.1f}% faster"
                            )
        
        # Check for accuracy issues
        if "datasets" in success_results:
            low_accuracy_competitors = []
            
            for dataset_name, dataset_data in success_results["datasets"].items():
                if "metrics" in dataset_data:
                    for competitor, metrics in dataset_data["metrics"].items():
                        if isinstance(metrics, dict):
                            accuracy = metrics.get("data_accuracy_score", 0.0)
                            if accuracy < 0.8 and competitor not in low_accuracy_competitors:
                                low_accuracy_competitors.append(competitor)
            
            if low_accuracy_competitors:
                opportunities.append(
                    f"Accuracy improvements needed for: {', '.join(low_accuracy_competitors)}"
                )
        
        return opportunities
    
    def _extract_critical_insights(self, domain_results: Dict[str, Any], 
                                 success_results: Dict[str, Any]) -> List[str]:
        """Extract critical insights from combined analysis."""
        insights = []
        
        # Domain-specific insights
        if "recommendations" in domain_results:
            recommendations = domain_results["recommendations"]
            
            if "by_domain" in recommendations:
                for domain, domain_rec in recommendations["by_domain"].items():
                    if "recommended_method" in domain_rec and domain_rec["recommended_method"]:
                        insights.append(
                            f"✅ {domain.replace('_', ' ').title()}: Use {domain_rec['recommended_method']} for optimal performance"
                        )
        
        # Success rate insights
        if "overall_analysis" in success_results:
            overall = success_results["overall_analysis"]
            
            if "cross_dataset_performance" in overall:
                top_performer = max(
                    overall["cross_dataset_performance"].items(),
                    key=lambda x: x[1]["avg_success_rate"],
                    default=(None, None)
                )
                
                if top_performer[0]:
                    insights.append(
                        f"🏆 Most reliable overall: {top_performer[0]} ({top_performer[1]['avg_success_rate']:.1%} success rate)"
                    )
        
        # Cross-analysis insights
        datason_variants = ["datason", "datason_api", "datason_ml", "datason_fast", "datason_secure"]
        
        # Check if DataSON variants dominate performance
        domain_best_methods = []
        if "scenarios" in domain_results:
            for scenario_data in domain_results["scenarios"].values():
                if "results" in scenario_data and "domain_specific_analysis" in scenario_data["results"]:
                    best_method = scenario_data["results"]["domain_specific_analysis"].get("best_method")
                    if best_method:
                        domain_best_methods.append(best_method)
        
        datason_dominance = sum(1 for method in domain_best_methods if method in datason_variants)
        if len(domain_best_methods) > 0:
            dominance_ratio = datason_dominance / len(domain_best_methods)
            if dominance_ratio > 0.7:
                insights.append(
                    f"🚀 DataSON variants lead in {dominance_ratio:.1%} of domain scenarios"
                )
        
        return insights
    
    def _generate_phase3_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive Phase 3 summary."""
        summary = {
            "phase3_highlights": {
                "domains_tested": 0,
                "success_datasets": 0,
                "total_scenarios": 0,
                "datason_variants_tested": []
            },
            "key_findings": [],
            "performance_recommendations": [],
            "reliability_assessment": {},
            "next_steps": []
        }
        
        # Extract highlights
        if "domain_scenarios" in results and "scenarios" in results["domain_scenarios"]:
            domains = set()
            for scenario_data in results["domain_scenarios"]["scenarios"].values():
                if "domain" in scenario_data:
                    domains.add(scenario_data["domain"])
            
            summary["phase3_highlights"]["domains_tested"] = len(domains)
            summary["phase3_highlights"]["total_scenarios"] = len(results["domain_scenarios"]["scenarios"])
        
        if "success_analysis" in results and "datasets" in results["success_analysis"]:
            summary["phase3_highlights"]["success_datasets"] = len(results["success_analysis"]["datasets"])
        
        # Extract DataSON variants tested
        if "domain_scenarios" in results and "scenarios" in results["domain_scenarios"]:
            datason_variants = set()
            for scenario_data in results["domain_scenarios"]["scenarios"].values():
                if "results" in scenario_data and "serialization" in scenario_data["results"]:
                    for method in scenario_data["results"]["serialization"].keys():
                        if method.lower().startswith("datason"):
                            datason_variants.add(method)
            
            summary["phase3_highlights"]["datason_variants_tested"] = list(datason_variants)
        
        # Key findings from cross-analysis
        if "cross_analysis" in results:
            cross_analysis = results["cross_analysis"]
            
            if "critical_insights" in cross_analysis:
                summary["key_findings"] = cross_analysis["critical_insights"]
            
            if "optimization_opportunities" in cross_analysis:
                summary["performance_recommendations"] = cross_analysis["optimization_opportunities"]
        
        # Reliability assessment
        if "success_analysis" in results and "overall_analysis" in results["success_analysis"]:
            overall_analysis = results["success_analysis"]["overall_analysis"]
            
            if "cross_dataset_performance" in overall_analysis:
                top_3_reliable = sorted(
                    overall_analysis["cross_dataset_performance"].items(),
                    key=lambda x: x[1]["avg_success_rate"],
                    reverse=True
                )[:3]
                
                summary["reliability_assessment"] = {
                    "top_3_most_reliable": [
                        {
                            "name": name,
                            "success_rate": f"{data['avg_success_rate']:.1%}",
                            "accuracy": f"{data['avg_accuracy']:.1%}"
                        }
                        for name, data in top_3_reliable
                    ]
                }
        
        # Next steps
        summary["next_steps"] = [
            "Implement domain-specific optimization recommendations",
            "Address identified accuracy issues in low-performing libraries",
            "Develop use-case specific configuration guides",
            "Create automated recommendation system based on data characteristics"
        ]
        
        return summary 
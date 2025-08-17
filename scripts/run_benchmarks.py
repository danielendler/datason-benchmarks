#!/usr/bin/env python3
"""
DataSON Benchmark Runner
========================

Enhanced benchmark runner with multi-tier capability-based testing.
Tests different DataSON API methods for fairer comparisons.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any
import datason

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Enhanced benchmark runner with capability-based testing."""
    
    def __init__(self, output_dir: str = "data/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata for all benchmarks
        self.metadata = {
            "timestamp": time.time(),
            "python_version": self._get_python_version(),
            "datason_version": self._get_datason_version(),
            "enhancement": "capability_based_testing_v1"
        }
    
    def _get_python_version(self) -> str:
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_datason_version(self) -> str:
        try:
            import datason
            return getattr(datason, '__version__', 'unknown')
        except ImportError:
            return 'not_installed'
    
    def run_enhanced_competitive_benchmark(self) -> Dict[str, Any]:
        """Run enhanced capability-based competitive benchmark."""
        logger.info("🚀 Running Enhanced Competitive Benchmark with Multi-Tier Testing...")
        
        from benchmarks.competitive.competitive_suite import CompetitiveBenchmarkSuite
        competitive_suite = CompetitiveBenchmarkSuite()
        
        # Use the new capability-based comparison
        results = {
            "suite_type": "enhanced_competitive",
            "metadata": self.metadata,
            "competitive": competitive_suite.run_capability_based_comparison(iterations=10)
        }
        
        self._save_results(results, "enhanced_competitive")
        
        # Print summary
        self._print_enhanced_summary(results["competitive"])
        
        logger.info("✅ Enhanced competitive benchmark completed")
        return results
    
    def run_datason_api_showcase(self) -> Dict[str, Any]:
        """Showcase different DataSON API methods on appropriate data."""
        logger.info("🎯 Running DataSON API Methods Showcase...")
        
        from benchmarks.competitive.competitive_suite import CompetitiveBenchmarkSuite
        competitive_suite = CompetitiveBenchmarkSuite()
        
        # Test comprehensive DataSON variants including new APIs
        datason_variants = ["datason", "datason_api", "datason_ml", "datason_fast", 
                           "datason_secure", "datason_smart", "datason_perfect"]
        
        api_results = competitive_suite.run_competitive_comparison(
            competitors=datason_variants, iterations=10
        )
        
        results = {
            "suite_type": "datason_api_showcase",
            "metadata": self.metadata,
            "api_comparison": api_results
        }
        
        # Print enhanced API analysis summary
        if api_results and "summary" in api_results and "datason_api_analysis" in api_results["summary"]:
            competitive_suite.print_enhanced_api_summary(api_results)
        
        self._save_results(results, "datason_api_showcase")
        logger.info("✅ DataSON API showcase completed")
        return results

    def run_quick_benchmark(self) -> Dict[str, Any]:
        """Quick benchmark using enhanced methodology."""
        logger.info("⚡ Running Quick Enhanced Benchmark...")
        
        from benchmarks.competitive.competitive_suite import CompetitiveBenchmarkSuite
        competitive_suite = CompetitiveBenchmarkSuite()
        
        # Use capability-based comparison with fewer iterations
        results = {
            "suite_type": "quick_enhanced",
            "metadata": self.metadata,
            "competitive": competitive_suite.run_capability_based_comparison(iterations=5)
        }
        
        self._save_results(results, "quick_enhanced")
        logger.info("✅ Quick benchmark completed")
        return results

    def run_configuration_benchmark(self) -> Dict[str, Any]:
        """DataSON configuration optimization benchmark."""
        logger.info("🔧 Running DataSON configuration benchmarks...")
        
        from benchmarks.configurations.config_suite import ConfigurationBenchmarkSuite
        config_suite = ConfigurationBenchmarkSuite()
        
        results = {
            "suite_type": "configurations",
            "metadata": self.metadata,
            "configurations": config_suite.run_configuration_tests(iterations=10)
        }
        
        self._save_results(results, "configurations")
        logger.info("✅ Configuration benchmarks completed")
        return results
    
    def run_version_benchmark(self) -> Dict[str, Any]:
        """DataSON version comparison benchmark."""
        logger.info("📈 Running DataSON version comparison...")
        
        from benchmarks.versioning.version_suite import DataSONVersionBenchmarkSuite
        version_suite = DataSONVersionBenchmarkSuite()
        
        results = {
            "suite_type": "versioning",
            "metadata": self.metadata,
            "versioning": version_suite.run_version_comparison(iterations=5)
        }
        
        self._save_results(results, "versioning")
        logger.info("✅ Version comparison completed")
        return results
    
    def run_phase2_benchmark(self, test_type: str = "complete") -> Dict[str, Any]:
        """Run Phase 2 advanced features benchmark."""
        logger.info("🚀 Running Phase 2 Advanced Features Benchmark...")
        
        from benchmarks.phase2.phase2_suite import Phase2BenchmarkSuite
        phase2_suite = Phase2BenchmarkSuite()
        
        if test_type == "security":
            logger.info("🔒 Testing security features only...")
            results = {
                "suite_type": "phase2_security",
                "metadata": self.metadata,
                "security_testing": phase2_suite.test_security_features(iterations=10)
            }
        elif test_type == "accuracy":
            logger.info("🎯 Testing accuracy features only...")
            results = {
                "suite_type": "phase2_accuracy", 
                "metadata": self.metadata,
                "accuracy_testing": phase2_suite.test_accuracy_features(iterations=10)
            }
        elif test_type == "ml":
            logger.info("🧠 Testing ML framework integration only...")
            results = {
                "suite_type": "phase2_ml",
                "metadata": self.metadata,
                "ml_testing": phase2_suite.test_ml_framework_integration(iterations=5)
            }
        else:  # complete
            logger.info("🎯 Running complete Phase 2 suite...")
            results = {
                "suite_type": "phase2_complete",
                "metadata": self.metadata,
                "phase2": phase2_suite.run_phase2_complete_suite(iterations=10)
            }
        
        self._save_results(results, f"phase2_{test_type}")
        
        # Print summary
        self._print_phase2_summary(results)
        
        logger.info("✅ Phase 2 benchmark completed")
        return results

    def run_phase3_benchmark(self, test_type: str = "complete") -> Dict[str, Any]:
        """Run Phase 3 realistic use case scenarios benchmark."""
        logger.info("🌍 Running Phase 3 Realistic Use Case Scenarios...")
        
        from benchmarks.phase3.phase3_suite import Phase3BenchmarkSuite
        phase3_suite = Phase3BenchmarkSuite()
        
        if test_type == "domain":
            logger.info("🎯 Testing domain-specific scenarios only...")
            results = {
                "suite_type": "phase3_domain",
                "metadata": self.metadata,
                "domain_scenarios": phase3_suite.run_domain_focused(iterations=10)
            }
        elif test_type == "success":
            logger.info("📊 Testing success rate analysis only...")
            results = {
                "suite_type": "phase3_success",
                "metadata": self.metadata,
                "success_analysis": phase3_suite.run_success_focused(iterations=50)
            }
        else:  # complete
            logger.info("🎯 Running complete Phase 3 suite...")
            results = {
                "suite_type": "phase3_complete",
                "metadata": self.metadata,
                "phase3": phase3_suite.run_phase3_complete(iterations=10)
            }
        
        self._save_results(results, f"phase3_{test_type}")
        
        # Print summary
        self._print_phase3_summary(results)
        
        logger.info("✅ Phase 3 benchmark completed")
        return results

    def run_phase4_enhanced_report(self, result_file: str) -> str:
        """Generate Phase 4 enhanced HTML report."""
        logger.info(f"🎨 Generating Phase 4 enhanced report for {result_file}")
        
        from scripts.phase4_enhanced_reports import Phase4ReportGenerator
        generator = Phase4ReportGenerator()
        
        try:
            report_path = generator.generate_comprehensive_report(result_file)
            logger.info(f"✅ Phase 4 enhanced report generated: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Failed to generate Phase 4 report: {e}")
            raise
    
    def run_phase4_decision_engine(self, domain: str) -> Dict[str, Any]:
        """Run Phase 4 decision engine for library recommendation."""
        logger.info(f"🤖 Running Phase 4 decision engine for {domain} domain")
        
        from scripts.phase4_decision_engine import DecisionEngine, create_requirements_from_questionnaire, Domain
        
        # Map CLI domain to questionnaire format
        domain_map = {
            'web': 'web',
            'ml': 'ml', 
            'finance': 'finance',
            'data': 'data',
            'enterprise': 'enterprise',
            'performance': 'performance'
        }
        
        # Create sample requirements for the domain
        questionnaire_answers = {
            "domain": domain_map.get(domain, 'general'),
            "speed_priority": "high" if domain == 'performance' else "medium",
            "accuracy_priority": "high",
            "security_priority": "high" if domain == 'finance' else "medium",
            "compatibility_priority": "high",
            "data_types": self._get_typical_data_types_for_domain(domain),
            "volume_level": "high" if domain == 'performance' else "medium",
            "team_expertise": "intermediate",
            "existing_stack": [],
            "compliance_needs": domain == 'finance'
        }
        
        requirements = create_requirements_from_questionnaire(questionnaire_answers)
        engine = DecisionEngine()
        
        recommendations = engine.recommend_library(requirements)
        
        # Print recommendations
        if recommendations:
            print(f"\n🎯 Top Recommendations for {domain} domain:")
            for i, rec in enumerate(recommendations[:3], 1):
                method_str = f".{rec.method_name}" if rec.method_name else ""
                print(f"\n{i}. {rec.library_name}{method_str}")
                print(f"   Score: {rec.total_score:.2f} | Confidence: {rec.confidence:.2f}")
                print(f"   Fit: {rec.use_case_fit}")
                if rec.pros:
                    print(f"   ✅ {rec.pros[0]}")
                if rec.cons:
                    print(f"   ⚠️  {rec.cons[0]}")
            
            # Detailed explanation for top recommendation
            explanation = engine.explain_recommendation(recommendations[0], requirements)
            print(f"\n📋 Why {recommendations[0].library_name}?")
            print(f"   {explanation['implementation_guidance']['basic_usage'][:100]}...")
        
        return {
            "domain": domain,
            "recommendations": [
                {
                    "library": rec.library_name,
                    "method": rec.method_name,
                    "score": rec.total_score,
                    "confidence": rec.confidence,
                    "fit": rec.use_case_fit,
                    "pros": rec.pros,
                    "cons": rec.cons
                }
                for rec in recommendations
            ]
        }
    
    def run_phase4_trend_analysis(self) -> Dict[str, Any]:
        """Run Phase 4 trend analysis and regression detection."""
        logger.info("📈 Running Phase 4 trend analysis...")
        
        from scripts.phase4_trend_analyzer import TrendAnalyzer
        
        analyzer = TrendAnalyzer()
        
        # Find recent result files to ingest
        recent_files = list(self.output_dir.glob("*.json"))
        
        # Ingest recent results
        ingested_count = 0
        for result_file in recent_files[-5:]:  # Last 5 files
            try:
                analyzer.ingest_benchmark_results(result_file.name)
                ingested_count += 1
            except Exception as e:
                logger.warning(f"Failed to ingest {result_file.name}: {e}")
        
        logger.info(f"📊 Ingested {ingested_count} result files")
        
        # Detect regressions
        regressions = analyzer.detect_performance_regressions(lookback_days=30)
        
        # Generate trend report
        trend_report = analyzer.generate_trend_report(lookback_days=30)
        
        # Print summary
        if regressions:
            print(f"\n🚨 Detected {len(regressions)} performance regressions:")
            for reg in regressions[:3]:  # Show first 3
                print(f"   • {reg['severity'].upper()}: {reg['description']}")
        else:
            print("\n✅ No performance regressions detected")
        
        print(f"\n📈 Trend Report Summary:")
        summary = trend_report['summary']
        print(f"   • {summary['total_benchmark_runs']} benchmark runs analyzed")
        print(f"   • {summary['unique_libraries_tested']} libraries tested")
        print(f"   • {summary['recent_alerts']} recent alerts")
        
        if trend_report['performance_insights']:
            print(f"\n💡 Key Insights:")
            for insight in trend_report['performance_insights'][:3]:
                print(f"   • {insight}")
        
        return trend_report
    
    def run_comprehensive_api_profiling(self) -> Dict[str, Any]:
        """Run the comprehensive API profiling harness across all DataSON APIs."""
        logger.info("🔬 Running comprehensive API profiling across all DataSON APIs...")
        try:
            from scripts.comprehensive_api_profiling import main as comprehensive_main
        except ImportError as e:
            logger.error(f"Failed to import comprehensive API profiling script: {e}")
            return {"error": "comprehensive profiling script not available"}
        results = comprehensive_main()
        # Save results to JSON
        out_file = self.output_dir / f"comprehensive_api_profile_{int(time.time())}.json"
        try:
            import datason
            out_file.write_text(datason.dumps_json(results))
            logger.info(f"✅ Comprehensive API profiling results saved to: {out_file}")
        except Exception as e:
            logger.error(f"Failed to save comprehensive profiling results: {e}")
        return results
    
    def _get_typical_data_types_for_domain(self, domain: str) -> list:
        """Get typical data types used in each domain."""
        domain_data_types = {
            'web': ['datetime', 'uuid'],
            'ml': ['numpy', 'pandas', 'datetime'],
            'finance': ['decimal', 'datetime', 'uuid'],
            'data': ['datetime', 'decimal', 'numpy'],
            'enterprise': ['datetime', 'uuid', 'decimal'],
            'performance': ['datetime']
        }
        return domain_data_types.get(domain, ['datetime'])

    def run_complete_benchmark(self) -> Dict[str, Any]:
        """Complete enhanced benchmark suite with all tests including Phase 2."""
        logger.info("🎯 Running Complete Enhanced Benchmark Suite with Phase 2...")
        
        start_time = time.time()
        
        # Run all benchmark types with enhanced methodology
        competitive_results = self.run_enhanced_competitive_benchmark()
        api_results = self.run_datason_api_showcase()
        config_results = self.run_configuration_benchmark()
        version_results = self.run_version_benchmark()
        phase2_results = self.run_phase2_benchmark("complete")
        
        # Combine results
        results = {
            "suite_type": "complete_enhanced_phase2",
            "metadata": self.metadata,
            "execution_time": time.time() - start_time,
            "competitive": competitive_results["competitive"],
            "datason_api_showcase": api_results["api_comparison"],
            "configurations": config_results["configurations"],
            "versioning": version_results["versioning"],
            "phase2_advanced": phase2_results.get("phase2", {}),
            "summary": self._generate_enhanced_summary(competitive_results, config_results, phase2_results)
        }
        
        self._save_results(results, "complete_enhanced_phase2")
        logger.info("✅ Complete enhanced benchmark suite with Phase 2 finished")
        return results
    
    def _print_enhanced_summary(self, results: Dict[str, Any]):
        """Print a user-friendly summary of enhanced results."""
        print("\n" + "="*60)
        print("🎯 ENHANCED BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        if "summary" in results:
            summary = results["summary"]
            print(f"📊 Methodology: {summary.get('methodology', 'Unknown')}")
            print(f"🔧 Tiers Tested: {', '.join(summary.get('tiers_tested', []))}")
            print(f"📈 Total Datasets: {summary.get('total_datasets', 0)}")
            
            if "datason_variants_tested" in summary:
                variants = summary["datason_variants_tested"]
                print(f"🚀 DataSON Variants: {', '.join(variants)}")
        
        if "tiers" in results:
            for tier_name, tier_data in results["tiers"].items():
                print(f"\n📋 {tier_name.upper()} TIER:")
                competitors = tier_data.get("competitors", [])
                print(f"   Libraries: {', '.join(competitors)}")
                
                tier_summary = tier_data.get("summary", {})
                if "success_rates" in tier_summary:
                    print("   Success Rates:")
                    for lib, rate_data in tier_summary["success_rates"].items():
                        rate = rate_data.get("average_success_rate", 0) * 100
                        print(f"     {lib}: {rate:.1f}%")
        
        print("\n" + "="*60)
    
    def _generate_enhanced_summary(self, competitive: Dict[str, Any], config: Dict[str, Any], phase2: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary for enhanced benchmark results."""
        summary = {
            "methodology": "Enhanced multi-tier capability-based testing",
            "fairness_improvements": [
                "Eliminated unfair object vs string conversion comparisons",
                "Segmented tests by library capabilities",
                "Added DataSON API method variants testing",
                "Included success rate analysis"
            ],
            "datason_optimizations_tested": [],
            "key_insights": {}
        }
        
        # Extract DataSON optimization insights
        if "configurations" in config and isinstance(config["configurations"], dict):
            config_data = config["configurations"]
            if "summary" in config_data:
                summary["datason_optimizations_tested"] = list(config_data["summary"].keys())
        
        # Extract competitive insights
        if "summary" in competitive:
            comp_summary = competitive["summary"]
            if "key_insights" in comp_summary:
                summary["key_insights"] = comp_summary["key_insights"]
        
        # Extract Phase 2 insights
        if "summary" in phase2:
            phase2_summary = phase2["summary"]
            if "key_insights" in phase2_summary:
                summary["key_insights"] = phase2_summary["key_insights"]
        
        return summary

    def _print_phase2_summary(self, results: Dict[str, Any]):
        """Print Phase 2 benchmark summary."""
        print("\n" + "="*80)
        print("📊 PHASE 2 ADVANCED FEATURES SUMMARY")
        print("="*80)
        
        if "security_testing" in results:
            print("\n🔒 SECURITY FEATURES")
            security = results["security_testing"]
            if "performance" in security and "dump_secure" in security["performance"]:
                perf = security["performance"]["dump_secure"]
                print(f"   dump_secure(): {perf['mean_ms']:.2f}ms avg ({perf['successful_runs']} runs)")
            if "security_validation" in security:
                validation = security["security_validation"]
                if "redaction_effectiveness" in validation:
                    score = validation["redaction_effectiveness"]
                    print(f"   PII Redaction: {score:.1%} effectiveness")
        
        if "accuracy_testing" in results:
            print("\n🎯 ACCURACY FEATURES")
            accuracy = results["accuracy_testing"]
            if "loading_methods" in accuracy:
                methods = accuracy["loading_methods"]
                for method, data in methods.items():
                    if "success_rate" in data:
                        rate = data["success_rate"]
                        time_ms = data.get("mean_ms", 0)
                        print(f"   {method}: {rate:.1%} success rate, {time_ms:.2f}ms avg")
        
        if "ml_testing" in results:
            print("\n🧠 ML FRAMEWORK INTEGRATION")
            ml = results["ml_testing"]
            if "framework_support" in ml:
                support = ml["framework_support"]
                print(f"   NumPy: {'✅ Available' if support.get('numpy') else '❌ Not available'}")
                print(f"   Pandas: {'✅ Available' if support.get('pandas') else '❌ Not available'}")
            
            if "serialization_results" in ml:
                for method, data in ml["serialization_results"].items():
                    if "success_rate" in data:
                        rate = data["success_rate"]
                        time_ms = data.get("mean_ms", 0)
                        print(f"   {method}: {rate:.1%} success rate, {time_ms:.2f}ms avg")
        
        if "phase2" in results:
            phase2 = results["phase2"]
            if "summary" in phase2:
                summary = phase2["summary"]
                
                print("\n🎯 PHASE 2 RECOMMENDATIONS")
                if "feature_recommendations" in summary:
                    for rec in summary["feature_recommendations"]:
                        print(f"   • {rec}")
        
        print("\n" + "="*80)

    def _print_phase3_summary(self, results: Dict[str, Any]):
        """Print Phase 3 benchmark summary."""
        print("\n" + "="*80)
        print("🌍 PHASE 3 REALISTIC USE CASE SCENARIOS SUMMARY")
        print("="*80)
        
        if "domain_scenarios" in results:
            print("\n🎯 DOMAIN-SPECIFIC SCENARIOS")
            domain = results["domain_scenarios"]
            if "scenarios" in domain:
                scenario_count = len(domain["scenarios"])
                print(f"   Tested {scenario_count} realistic scenarios across multiple domains")
                
                # Show domain breakdown
                domains = set()
                for scenario_data in domain["scenarios"].values():
                    if "domain" in scenario_data:
                        domains.add(scenario_data["domain"])
                
                for domain_name in sorted(domains):
                    print(f"   • {domain_name.replace('_', ' ').title()}")
        
        if "success_analysis" in results:
            print("\n📊 SUCCESS RATE ANALYSIS")
            success = results["success_analysis"]
            if "overall_analysis" in success and "cross_dataset_performance" in success["overall_analysis"]:
                performance = success["overall_analysis"]["cross_dataset_performance"]
                
                # Show top 3 most reliable
                top_performers = sorted(
                    performance.items(),
                    key=lambda x: x[1]["avg_success_rate"],
                    reverse=True
                )[:3]
                
                print("   Top 3 Most Reliable Libraries:")
                for i, (name, data) in enumerate(top_performers, 1):
                    rate = data["avg_success_rate"] * 100
                    acc = data["avg_accuracy"] * 100
                    print(f"   {i}. {name}: {rate:.1f}% success, {acc:.1f}% accuracy")
        
        if "phase3" in results:
            phase3 = results["phase3"]
            if "summary" in phase3:
                summary = phase3["summary"]
                
                print("\n🎯 PHASE 3 KEY INSIGHTS")
                if "key_findings" in summary:
                    for finding in summary["key_findings"]:
                        print(f"   • {finding}")
                
                if "performance_recommendations" in summary:
                    print("\n🚀 OPTIMIZATION RECOMMENDATIONS")
                    for rec in summary["performance_recommendations"]:
                        print(f"   • {rec}")
        
        print("\n" + "="*80)

    def _save_results(self, results: Dict[str, Any], benchmark_type: str):
        """Save benchmark results to file using DataSON (dogfooding our own product)."""
        timestamp = int(time.time())
        filename = f"{benchmark_type}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                # Dogfood DataSON v0.11.2 for JSON serialization
                f.write(datason.dumps_json(results))
            logger.info(f"💾 Results saved to: {filepath}")
            
            # Also create/update latest_* symlink for PR workflow compatibility
            # Map benchmark types to expected latest_* filenames
            latest_type_mapping = {
                "quick_enhanced": "quick",  # Map quick_enhanced -> latest_quick.json
                "enhanced_competitive": "competitive",  # Map enhanced_competitive -> latest_competitive.json
            }
            latest_type = latest_type_mapping.get(benchmark_type, benchmark_type)
            latest_filepath = self.output_dir / f"latest_{latest_type}.json"
            
            if latest_filepath.exists() or latest_filepath.is_symlink():
                latest_filepath.unlink()  # Remove existing file/symlink
            latest_filepath.symlink_to(filename)  # Create relative symlink
            logger.info(f"🔗 Latest symlink updated: {latest_filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _save_results_to_file(self, results: Dict[str, Any], output_file: str):
        """Save benchmark results to a specific file using DataSON."""
        filepath = Path(output_file)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w') as f:
                # Dogfood DataSON v0.11.2 for JSON serialization
                f.write(datason.dumps_json(results))
            logger.info(f"💾 Results saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def main():
    """Main entry point with enhanced options."""
    parser = argparse.ArgumentParser(description='Enhanced DataSON Benchmarking Suite')
    
    # Benchmark type selection
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick enhanced benchmark (5 iterations)')
    parser.add_argument('--competitive', action='store_true',
                       help='Run enhanced competitive comparison')
    parser.add_argument('--datason-showcase', action='store_true',
                       help='Showcase DataSON API methods')
    parser.add_argument('--configurations', action='store_true',
                       help='Run DataSON configuration benchmarks')
    parser.add_argument('--versioning', action='store_true',
                       help='Run DataSON version comparison')
    parser.add_argument('--complete', action='store_true',
                       help='Run complete enhanced benchmark suite with Phase 2')
    
    # Phase 2 specific options
    parser.add_argument('--phase2', action='store_true',
                       help='Run Phase 2 advanced features benchmark')
    parser.add_argument('--phase2-security', action='store_true',
                       help='Run Phase 2 security features only')
    parser.add_argument('--phase2-accuracy', action='store_true',
                       help='Run Phase 2 accuracy features only')
    parser.add_argument('--phase2-ml', action='store_true',
                       help='Run Phase 2 ML framework integration only')
    
    # Phase 3 specific options
    parser.add_argument('--phase3', action='store_true',
                       help='Run Phase 3 realistic use case scenarios benchmark')
    parser.add_argument('--phase3-domain', action='store_true',
                       help='Run Phase 3 domain-specific scenarios only')
    parser.add_argument('--phase3-success', action='store_true',
                       help='Run Phase 3 success rate analysis only')
    
    # Phase 4 enhanced reporting options
    parser.add_argument('--phase4-report', type=str, metavar='RESULT_FILE',
                       help='Generate Phase 4 enhanced HTML report from result file')
    parser.add_argument('--phase4-decide', type=str, metavar='DOMAIN',
                       choices=['web', 'ml', 'finance', 'data', 'enterprise', 'performance'],
                       help='Get Phase 4 library recommendation for domain')
    parser.add_argument('--phase4-trends', action='store_true',
                       help='Run Phase 4 trend analysis and regression detection')
    
    # Comprehensive API profiling
    parser.add_argument('--profile-apis', action='store_true',
                       help='Run comprehensive API profiling across all DataSON APIs')
    # Output options
    parser.add_argument('--output', type=str,
                       help='Output file for results (optional)')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate HTML report after running benchmarks (legacy compatibility)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    runner = BenchmarkRunner()
    
    try:
        # Comprehensive API profiling
        if args.profile_apis:
            runner.run_comprehensive_api_profiling()
            return 0
        # Phase 4 options
        if args.phase4_report:
            report_path = runner.run_phase4_enhanced_report(args.phase4_report)
            print(f"✅ Phase 4 enhanced report generated: {report_path}")
            return 0
        elif args.phase4_decide:
            results = runner.run_phase4_decision_engine(args.phase4_decide)
            return 0
        elif args.phase4_trends:
            results = runner.run_phase4_trend_analysis()
            return 0
        
        # Phase 3 options
        elif args.phase3_domain:
            results = runner.run_phase3_benchmark("domain")
        elif args.phase3_success:
            results = runner.run_phase3_benchmark("success")
        elif args.phase3:
            results = runner.run_phase3_benchmark("complete")
        
        # Phase 2 options
        elif args.phase2_security:
            results = runner.run_phase2_benchmark("security")
        elif args.phase2_accuracy:
            results = runner.run_phase2_benchmark("accuracy")
        elif args.phase2_ml:
            results = runner.run_phase2_benchmark("ml")
        elif args.phase2:
            results = runner.run_phase2_benchmark("complete")
        
        # Enhanced benchmark options
        # (deprecated --enhanced-competitive flag removed)
        elif args.datason_showcase:
            results = runner.run_datason_api_showcase()
        elif args.quick:
            results = runner.run_quick_benchmark()
        elif args.competitive:
            results = runner.run_enhanced_competitive_benchmark()  # Use enhanced by default
        elif args.configurations:
            results = runner.run_configuration_benchmark()
        elif args.versioning:
            results = runner.run_version_benchmark()
        elif args.complete:
            results = runner.run_complete_benchmark()
        else:
            # Default: run quick enhanced benchmark
            logger.info("No specific benchmark selected, running quick enhanced benchmark...")
            results = runner.run_quick_benchmark()
        
        # Save to custom output file if specified
        if args.output:
            runner._save_results_to_file(results, args.output)
            print(f"Results saved to: {args.output}")
        
        # Handle legacy --generate-report flag
        if args.generate_report:
            logger.info("🎨 --generate-report flag detected (results already saved automatically)")
            # The script already saves results, this flag is for workflow compatibility
            # Results are saved in _save_results() method called by each benchmark
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    main() 

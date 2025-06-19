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
        
        # Test only DataSON variants on different data types
        datason_variants = ["datason", "datason_api", "datason_ml", "datason_fast"]
        
        results = {
            "suite_type": "datason_api_showcase",
            "metadata": self.metadata,
            "api_comparison": competitive_suite.run_competitive_comparison(
                competitors=datason_variants, iterations=10
            )
        }
        
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

    def _save_results(self, results: Dict[str, Any], benchmark_type: str):
        """Save benchmark results to file."""
        timestamp = int(time.time())
        filename = f"{benchmark_type}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
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
    parser.add_argument('--enhanced-competitive', action='store_true',
                       help='Run enhanced multi-tier competitive comparison')
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
    
    # Output options
    parser.add_argument('--output', type=str,
                       help='Output file for results (optional)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    runner = BenchmarkRunner()
    
    try:
        # Phase 2 options
        if args.phase2_security:
            results = runner.run_phase2_benchmark("security")
        elif args.phase2_accuracy:
            results = runner.run_phase2_benchmark("accuracy")
        elif args.phase2_ml:
            results = runner.run_phase2_benchmark("ml")
        elif args.phase2:
            results = runner.run_phase2_benchmark("complete")
        
        # Enhanced benchmark options
        elif args.enhanced_competitive:
            results = runner.run_enhanced_competitive_benchmark()
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
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    main() 
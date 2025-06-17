#!/usr/bin/env python3
"""
DataSON Benchmarks - Main Benchmark Runner
==========================================

Main script to run all benchmark suites following the open source benchmarking strategy.
Supports competitive analysis, configuration testing, and automated reporting.

Usage:
    python scripts/run_benchmarks.py --quick          # Quick comparison (3-4 competitors)
    python scripts/run_benchmarks.py --competitive    # Full competitive suite
    python scripts/run_benchmarks.py --configurations # DataSON config testing
    python scripts/run_benchmarks.py --all           # Complete benchmark suite
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from competitors.adapter_registry import CompetitorRegistry
from benchmarks.competitive.competitive_suite import CompetitiveBenchmarkSuite
from benchmarks.configurations.config_suite import ConfigurationBenchmarkSuite
from scripts.generate_report import ReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Main benchmark coordination and execution."""
    
    def __init__(self, results_dir: str = "data/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.competitor_registry = CompetitorRegistry()
        self.competitive_suite = CompetitiveBenchmarkSuite()
        self.config_suite = ConfigurationBenchmarkSuite()
        self.report_generator = ReportGenerator()
        
        # Metadata
        self.metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "runner_info": {
                "os": os.name,
                "github_sha": os.environ.get("GITHUB_SHA", "local"),
                "github_ref": os.environ.get("GITHUB_REF", "local"),
                "ci_run_id": os.environ.get("GITHUB_RUN_ID", "local")
            }
        }
    
    def run_quick_benchmark(self) -> Dict[str, Any]:
        """Quick benchmark with core competitors for rapid feedback."""
        logger.info("🚀 Running quick benchmark suite...")
        
        # Core competitors for quick testing
        quick_competitors = ["datason", "orjson", "ujson", "json"]
        
        results = {
            "suite_type": "quick",
            "metadata": self.metadata,
            "competitive": self.competitive_suite.run_competitive_comparison(
                competitors=quick_competitors,
                iterations=5  # Reduced iterations for speed
            )
        }
        
        self._save_results(results, "quick")
        logger.info("✅ Quick benchmark completed")
        return results
    
    def run_competitive_benchmark(self) -> Dict[str, Any]:
        """Full competitive benchmark against all available competitors."""
        logger.info("🏆 Running competitive benchmark suite...")
        
        # Get all available competitors
        available_competitors = self.competitor_registry.get_available_competitors()
        logger.info(f"Testing against {len(available_competitors)} competitors: {list(available_competitors.keys())}")
        
        results = {
            "suite_type": "competitive",
            "metadata": self.metadata,
            "competitive": self.competitive_suite.run_competitive_comparison(
                competitors=list(available_competitors.keys())
            ),
            "competitor_versions": {
                name: info.get("version", "unknown") 
                for name, info in available_competitors.items()
            }
        }
        
        self._save_results(results, "competitive")
        logger.info("✅ Competitive benchmark completed")
        return results
    
    def run_configuration_benchmark(self) -> Dict[str, Any]:
        """DataSON configuration performance testing."""
        logger.info("⚙️ Running configuration benchmark suite...")
        
        results = {
            "suite_type": "configuration",
            "metadata": self.metadata,
            "configurations": self.config_suite.run_configuration_tests()
        }
        
        self._save_results(results, "configuration")
        logger.info("✅ Configuration benchmark completed")
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
    
    def run_complete_benchmark(self) -> Dict[str, Any]:
        """Complete benchmark suite with all tests."""
        logger.info("🎯 Running complete benchmark suite...")
        
        start_time = time.time()
        
        # Run all benchmark types
        competitive_results = self.run_competitive_benchmark()
        config_results = self.run_configuration_benchmark()
        version_results = self.run_version_benchmark()
        
        # Combine results
        results = {
            "suite_type": "complete",
            "metadata": self.metadata,
            "execution_time": time.time() - start_time,
            "competitive": competitive_results["competitive"],
            "configurations": config_results["configurations"],
            "versioning": version_results["versioning"],
            "summary": self._generate_summary(competitive_results, config_results)
        }
        
        self._save_results(results, "complete")
        logger.info("✅ Complete benchmark suite finished")
        return results
    
    def _generate_summary(self, competitive: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level summary of benchmark results."""
        try:
            # Extract key metrics
            competitive_data = competitive.get("competitive", {})
            config_data = config.get("configurations", {})
            
            # Find DataSON performance relative to best competitor
            best_performance = {}
            datason_vs_best = {}
            
            for test_name, test_results in competitive_data.items():
                if isinstance(test_results, dict) and "results" in test_results:
                    results = test_results["results"]
                    if "datason" in results:
                        # Find fastest competitor (excluding DataSON)
                        fastest_time = float("inf")
                        fastest_lib = None
                        
                        for lib, metrics in results.items():
                            if lib != "datason" and isinstance(metrics, dict):
                                time_ms = metrics.get("mean", float("inf")) * 1000
                                if time_ms < fastest_time:
                                    fastest_time = time_ms
                                    fastest_lib = lib
                        
                        if fastest_lib:
                            datason_time = results["datason"].get("mean", 0) * 1000
                            ratio = datason_time / fastest_time if fastest_time > 0 else 0
                            
                            best_performance[test_name] = {
                                "fastest_competitor": fastest_lib,
                                "fastest_time_ms": fastest_time,
                                "datason_time_ms": datason_time,
                                "datason_vs_fastest_ratio": ratio,
                                "datason_faster": ratio < 1.0
                            }
            
            return {
                "total_tests": len(competitive_data) + len(config_data),
                "competitive_tests": len(competitive_data),
                "configuration_tests": len(config_data),
                "datason_vs_best": best_performance,
                "datason_wins": sum(1 for v in best_performance.values() if v["datason_faster"]),
                "competitor_wins": sum(1 for v in best_performance.values() if not v["datason_faster"])
            }
        except Exception as e:
            logger.warning(f"Could not generate summary: {e}")
            return {"error": str(e)}
    
    def _save_results(self, results: Dict[str, Any], suite_type: str) -> None:
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{suite_type}_benchmark_{timestamp}.json"
        filepath = self.results_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Also save as latest
            latest_filepath = self.results_dir / f"latest_{suite_type}.json"
            with open(latest_filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
            logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(description="DataSON Benchmarks Runner")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick benchmark with core competitors")
    parser.add_argument("--competitive", action="store_true", 
                       help="Run full competitive benchmark")
    parser.add_argument("--configurations", action="store_true", 
                       help="Run DataSON configuration tests")
    parser.add_argument("--versioning", action="store_true", 
                       help="Run DataSON version comparison tests")
    parser.add_argument("--all", action="store_true", 
                       help="Run complete benchmark suite")
    parser.add_argument("--output-dir", default="data/results", 
                       help="Output directory for results")
    parser.add_argument("--generate-report", action="store_true", 
                       help="Generate HTML report after benchmarks")
    
    args = parser.parse_args()
    
    # Default to quick if no specific suite specified
    if not any([args.quick, args.competitive, args.configurations, args.versioning, args.all]):
        args.quick = True
        logger.info("No specific suite specified, running quick benchmark")
    
    runner = BenchmarkRunner(results_dir=args.output_dir)
    results = None
    
    try:
        if args.all:
            results = runner.run_complete_benchmark()
        elif args.competitive:
            results = runner.run_competitive_benchmark()
        elif args.configurations:
            results = runner.run_configuration_benchmark()
        elif args.versioning:
            results = runner.run_version_benchmark()
        elif args.quick:
            results = runner.run_quick_benchmark()
        
        # Generate report if requested
        if args.generate_report and results:
            logger.info("📊 Generating benchmark report...")
            runner.report_generator.generate_html_report(results)
            logger.info("✅ Report generated")
            
    except KeyboardInterrupt:
        logger.info("❌ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        sys.exit(1)
    
    logger.info("🎉 Benchmark run completed successfully!")


if __name__ == "__main__":
    main() 
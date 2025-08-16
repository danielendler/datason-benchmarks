#!/usr/bin/env python3
"""
Generate Informative PR Comment
==============================

Parses benchmark results and generates a detailed PR comment with actual performance data,
baseline comparisons, and regression analysis.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse


def load_results(result_file: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(result_file, 'r') as f:
        return json.load(f)


def load_baseline(baseline_file: str) -> Optional[Dict[str, Any]]:
    """Load baseline results if available."""
    if os.path.exists(baseline_file):
        with open(baseline_file, 'r') as f:
            return json.load(f)
    return None


def extract_datason_performance(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract DataSON performance metrics from results."""
    performance: Dict[str, Any] = {
        "tests_run": 0,
        "total_scenarios": 0,
        "serialization_avg_ms": 0.0,
        "deserialization_avg_ms": 0.0,
        "success_rate": 0.0,
        "fastest_scenario": 0.0,
        "slowest_scenario": 0.0,
        "scenarios": []
    }
    
    # Handle different result structures
    test_data = None
    if "competitive" in results:
        # Handle both new PR optimized structure and legacy competitive structure
        competitive = results["competitive"]
        if "tiers" in competitive:
            # New structure: competitive.tiers.tier_name.datasets
            test_data = {}
            for tier_name, tier_info in competitive["tiers"].items():
                if "datasets" in tier_info:
                    test_data.update(tier_info["datasets"])
        else:
            # Legacy structure: competitive.dataset_name
            test_data = competitive
    elif "phase2" in results:
        test_data = results["phase2"]
    elif "phase3" in results:
        test_data = results["phase3"]
    else:
        # Try to find test data in top level
        for key, value in results.items():
            if isinstance(value, dict) and any(k in value for k in ["serialization", "deserialization"]):
                test_data = {key: value}
                break
    
    if not test_data:
        return performance
    
    serialization_times = []
    deserialization_times = []
    successful_tests = 0
    total_tests = 0
    
    for scenario_name, scenario_data in test_data.items():
        if not isinstance(scenario_data, dict):
            continue
            
        performance["total_scenarios"] += 1
        scenario_info = {"name": scenario_name, "status": "❌ Failed"}
        
        # Check serialization performance
        if "serialization" in scenario_data and "datason" in scenario_data["serialization"]:
            ser_data = scenario_data["serialization"]["datason"]
            if "mean_ms" in ser_data and ser_data.get("error_count", 0) == 0:
                serialization_times.append(ser_data["mean_ms"])
                scenario_info["serialization_ms"] = round(ser_data["mean_ms"], 3)
                scenario_info["status"] = "✅ Passed"
                successful_tests += 1
        
        # Check deserialization performance  
        if "deserialization" in scenario_data and "datason" in scenario_data["deserialization"]:
            deser_data = scenario_data["deserialization"]["datason"]
            if "mean_ms" in deser_data and deser_data.get("error_count", 0) == 0:
                deserialization_times.append(deser_data["mean_ms"])
                scenario_info["deserialization_ms"] = round(deser_data["mean_ms"], 3)
        
        total_tests += 1
        performance["scenarios"].append(scenario_info)
    
    # Calculate averages
    if serialization_times:
        performance["serialization_avg_ms"] = round(sum(serialization_times) / len(serialization_times), 3)
        performance["fastest_scenario"] = min(serialization_times)
        performance["slowest_scenario"] = max(serialization_times)
    
    if deserialization_times:
        performance["deserialization_avg_ms"] = round(sum(deserialization_times) / len(deserialization_times), 3)
    
    performance["tests_run"] = total_tests
    performance["success_rate"] = round((successful_tests / total_tests * 100), 1) if total_tests > 0 else 0
    
    return performance


def compare_with_baseline(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Compare current results with baseline."""
    current_perf = extract_datason_performance(current)
    baseline_perf = extract_datason_performance(baseline)
    
    comparison = {
        "has_baseline": True,
        "serialization_change": 0,
        "deserialization_change": 0,
        "success_rate_change": 0,
        "regression_detected": False,
        "improvements": [],
        "regressions": []
    }
    
    # Compare serialization performance
    if baseline_perf["serialization_avg_ms"] > 0:
        change = ((current_perf["serialization_avg_ms"] - baseline_perf["serialization_avg_ms"]) 
                 / baseline_perf["serialization_avg_ms"] * 100)
        comparison["serialization_change"] = round(change, 1)
        
        if change > 10:  # Regression threshold (10% slower)
            comparison["regressions"].append(f"Serialization {change:.1f}% slower")
            comparison["regression_detected"] = True
        elif change < -2:  # Improvement threshold (2% faster - more sensitive)
            comparison["improvements"].append(f"Serialization {abs(change):.1f}% faster")
    
    # Compare deserialization performance
    if baseline_perf["deserialization_avg_ms"] > 0:
        change = ((current_perf["deserialization_avg_ms"] - baseline_perf["deserialization_avg_ms"]) 
                 / baseline_perf["deserialization_avg_ms"] * 100)
        comparison["deserialization_change"] = round(change, 1)
        
        if change > 10:  # Regression threshold (10% slower)
            comparison["regressions"].append(f"Deserialization {change:.1f}% slower")
            comparison["regression_detected"] = True
        elif change < -2:  # Improvement threshold (2% faster - more sensitive)
            comparison["improvements"].append(f"Deserialization {abs(change):.1f}% faster")
    
    # Compare success rate
    success_change = current_perf["success_rate"] - baseline_perf["success_rate"]
    comparison["success_rate_change"] = round(success_change, 1)
    
    if success_change < -5:  # Success rate regression
        comparison["regressions"].append(f"Success rate dropped {abs(success_change):.1f}%")
        comparison["regression_detected"] = True
    elif success_change > 5:  # Success rate improvement
        comparison["improvements"].append(f"Success rate improved {success_change:.1f}%")
    
    return comparison


def generate_pr_comment(pr_number: str, commit_sha: str, benchmark_type: str, 
                       result_file: str, baseline_file: Optional[str] = None) -> str:
    """Generate comprehensive PR comment with actual performance data."""
    
    # Load results
    results = load_results(result_file)
    baseline = load_baseline(baseline_file) if baseline_file else None
    
    # Extract performance data
    performance = extract_datason_performance(results)
    
    # Compare with baseline if available
    comparison = None
    if baseline:
        comparison = compare_with_baseline(results, baseline)
    
    # Start building comment
    comment_lines = [
        "# 🚀 DataSON PR Performance Analysis",
        "",
        f"**PR #{pr_number}** | **Commit**: `{commit_sha}`",
        "",
        "## 📊 Benchmark Results",
        "",
        f"**Suite**: {benchmark_type} | **Tests Run**: {performance['tests_run']} | **Success Rate**: {performance['success_rate']}%",
        ""
    ]
    
    # Performance summary table
    comment_lines.extend([
        "### 🎯 DataSON Performance Summary",
        "",
        "| Metric | Result | Benchmark Details | Status |",
        "|--------|--------|-------------------|--------|",
        f"| Serialization (avg) | **{performance['serialization_avg_ms']:.3f} ms** | {len([s for s in performance['scenarios'] if 'serialization_ms' in s])} measurements across {performance['tests_run']} scenarios | {'✅' if performance['serialization_avg_ms'] > 0 else '❌'} |",
        f"| Deserialization (avg) | **{performance['deserialization_avg_ms']:.3f} ms** | {len([s for s in performance['scenarios'] if 'deserialization_ms' in s])} measurements | {'✅' if performance['deserialization_avg_ms'] > 0 else '❌'} |",
        f"| Success Rate | **{performance['success_rate']:.1f}%** | {performance['tests_run']} scenarios tested with 5 iterations each | {'✅' if performance['success_rate'] > 90 else '⚠️' if performance['success_rate'] > 70 else '❌'} |",
        f"| Performance Range | {performance['fastest_scenario']:.3f} - {performance['slowest_scenario']:.3f} ms | Min to max serialization times | {'✅' if performance['slowest_scenario'] < 100 else '⚠️'} |",
        ""
    ])
    
    # Scenario breakdown
    if performance["scenarios"]:
        comment_lines.extend([
            "### 📋 Test Scenarios",
            "",
            "| Scenario | Status | Serialization | Deserialization |",
            "|----------|--------|---------------|-----------------|"
        ])
        
        for scenario in performance["scenarios"][:5]:  # Show top 5
            ser_time = f"{scenario.get('serialization_ms', 'N/A')} ms" if 'serialization_ms' in scenario else "N/A"
            deser_time = f"{scenario.get('deserialization_ms', 'N/A')} ms" if 'deserialization_ms' in scenario else "N/A"
            name = scenario["name"].replace("_", " ").title()[:30]
            comment_lines.append(f"| {name} | {scenario['status']} | {ser_time} | {deser_time} |")
        
        comment_lines.append("")
    
    # Baseline comparison
    if comparison and comparison["has_baseline"]:
        comment_lines.extend([
            "## 📈 Baseline Comparison",
            ""
        ])
        
        if comparison["improvements"]:
            comment_lines.extend([
                "### ✅ Performance Improvements",
                ""
            ])
            for improvement in comparison["improvements"]:
                comment_lines.append(f"- 🚀 {improvement}")
            comment_lines.append("")
        
        if comparison["regressions"]:
            comment_lines.extend([
                "### ⚠️ Performance Regressions",
                ""
            ])
            for regression in comparison["regressions"]:
                comment_lines.append(f"- 🐌 {regression}")
            comment_lines.append("")
        
        if not comparison["improvements"] and not comparison["regressions"]:
            # Show actual change percentages even if below thresholds
            ser_change = comparison.get("serialization_change", 0)
            deser_change = comparison.get("deserialization_change", 0)
            
            if ser_change != 0 or deser_change != 0:
                comment_lines.extend([
                    "### 📊 Performance Changes (Below Threshold)",
                    ""
                ])
                if ser_change != 0:
                    direction = "faster" if ser_change < 0 else "slower"
                    comment_lines.append(f"- Serialization: {abs(ser_change):.1f}% {direction}")
                if deser_change != 0:
                    direction = "faster" if deser_change < 0 else "slower"
                    comment_lines.append(f"- Deserialization: {abs(deser_change):.1f}% {direction}")
                comment_lines.append("")
            else:
                comment_lines.extend([
                    "✅ **Performance unchanged from baseline**",
                    ""
                ])
    else:
        comment_lines.extend([
            "## ℹ️ Baseline Status",
            "",
            "No baseline available for comparison. Future PRs will compare against this baseline.",
            ""
        ])
    
    # Final status
    if comparison and comparison["regression_detected"]:
        comment_lines.extend([
            "## ⚠️ Status: Performance Alert",
            "",
            "Significant performance regression detected. Please review the changes above.",
            ""
        ])
    else:
        comment_lines.extend([
            "## ✅ Status: Ready for Review",
            "",
            "All benchmarks passed! No significant performance regressions detected.",
            ""
        ])
    
    comment_lines.extend([
        "---",
        "*Generated by [datason-benchmarks](https://github.com/danielendler/datason-benchmarks) • Comprehensive Performance Analysis*"
    ])
    
    return "\n".join(comment_lines)


def main():
    parser = argparse.ArgumentParser(description='Generate informative PR comment')
    parser.add_argument('--pr-number', required=True, help='PR number')
    parser.add_argument('--commit-sha', required=True, help='Commit SHA')
    parser.add_argument('--benchmark-type', required=True, help='Benchmark type')
    parser.add_argument('--result-file', required=True, help='Benchmark result file')
    parser.add_argument('--baseline-file', help='Baseline file for comparison')
    parser.add_argument('--output', default='comment.md', help='Output file')
    
    args = parser.parse_args()
    
    try:
        comment = generate_pr_comment(
            args.pr_number, 
            args.commit_sha, 
            args.benchmark_type,
            args.result_file,
            args.baseline_file
        )
        
        with open(args.output, 'w') as f:
            f.write(comment)
        
        print(f"✅ PR comment generated: {args.output}")
        
    except Exception as e:
        print(f"❌ Failed to generate PR comment: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 
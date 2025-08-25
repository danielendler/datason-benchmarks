#!/usr/bin/env python3
"""
Performance regression detection for DataSON benchmarks.
Compares PR performance against baseline and flags significant regressions.
"""

import json
import os
import statistics
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import datetime
import datason

@dataclass
class RegressionThresholds:
    """Thresholds for regression detection"""
    fail_threshold: float = 0.25  # 25% degradation fails PR
    warn_threshold: float = 0.10  # 10% degradation warns
    notice_threshold: float = 0.05  # 5% degradation notices
    improvement_threshold: float = 0.05  # 5% improvement worth noting

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    higher_is_better: bool = False  # True for throughput, False for time/memory

@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    name: str
    metrics: Dict[str, PerformanceMetric]
    metadata: Dict[str, Any]

@dataclass
class RegressionResult:
    """Result of regression analysis"""
    metric_name: str
    baseline_value: float
    current_value: float
    change_percent: float
    status: str  # 'fail', 'warn', 'notice', 'improvement', 'stable'
    message: str

class PerformanceRegressionDetector:
    """
    Detects performance regressions by comparing current results against baseline.
    Implements the simple regression detection from the strategy document.
    """
    
    def __init__(self, thresholds: Optional[RegressionThresholds] = None):
        self.thresholds = thresholds or RegressionThresholds()
    
    def format_time_value(self, value_seconds: float) -> Tuple[str, str]:
        """
        Format a time value in seconds to the most appropriate unit.
        Returns (formatted_value, unit_name) tuple.
        """
        # Convert to different units and find the most readable one
        if value_seconds >= 1.0:
            return f"{value_seconds:.3f}", "s"
        elif value_seconds >= 0.001:
            return f"{value_seconds * 1000:.3f}", "ms"
        elif value_seconds >= 0.000001:
            return f"{value_seconds * 1000000:.3f}", "μs"
        else:
            return f"{value_seconds * 1000000000:.3f}", "ns"
    
    def format_comparison_values(self, baseline_seconds: float, current_seconds: float) -> Tuple[str, str]:
        """
        Format baseline and current values using the same unit for comparison.
        Chooses the unit based on the larger of the two values for better readability.
        """
        # Choose unit based on the larger value
        larger_value = max(abs(baseline_seconds), abs(current_seconds))
        
        if larger_value >= 1.0:
            # Use seconds
            baseline_str = f"{baseline_seconds:.3f}s"
            current_str = f"{current_seconds:.3f}s"
        elif larger_value >= 0.001:
            # Use milliseconds
            baseline_str = f"{baseline_seconds * 1000:.3f}ms"
            current_str = f"{current_seconds * 1000:.3f}ms"
        elif larger_value >= 0.000001:
            # Use microseconds
            baseline_str = f"{baseline_seconds * 1000000:.3f}μs"
            current_str = f"{current_seconds * 1000000:.3f}μs"
        else:
            # Use nanoseconds
            baseline_str = f"{baseline_seconds * 1000000000:.3f}ns"
            current_str = f"{current_seconds * 1000000000:.3f}ns"
        
        return baseline_str, current_str
    
    def load_benchmark_results(self, file_path: str) -> List[BenchmarkResult]:
        """Load benchmark results from JSON file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Benchmark results file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        results = []
        
        # Handle different result file formats
        if 'benchmarks' in data:
            # Competitive benchmark format
            for bench_name, bench_data in data['benchmarks'].items():
                metrics = {}
                
                # Extract metrics from benchmark data
                if 'results' in bench_data:
                    for lib_name, lib_results in bench_data['results'].items():
                        if lib_name == 'datason':  # Focus on DataSON performance
                            if 'serialize_time' in lib_results:
                                metrics['serialize_time'] = PerformanceMetric(
                                    name='serialize_time',
                                    value=lib_results['serialize_time'],
                                    unit='seconds',
                                    higher_is_better=False
                                )
                            if 'deserialize_time' in lib_results:
                                metrics['deserialize_time'] = PerformanceMetric(
                                    name='deserialize_time',
                                    value=lib_results['deserialize_time'],
                                    unit='seconds',
                                    higher_is_better=False
                                )
                            if 'memory_usage' in lib_results:
                                metrics['memory_usage'] = PerformanceMetric(
                                    name='memory_usage',
                                    value=lib_results['memory_usage'],
                                    unit='MB',
                                    higher_is_better=False
                                )
                            if 'throughput' in lib_results:
                                metrics['throughput'] = PerformanceMetric(
                                    name='throughput',
                                    value=lib_results['throughput'],
                                    unit='ops/sec',
                                    higher_is_better=True
                                )
                
                if metrics:
                    results.append(BenchmarkResult(
                        name=bench_name,
                        metrics=metrics,
                        metadata=bench_data.get('metadata', {})
                    ))
        
        elif 'results' in data:
            # Version comparison format
            for result in data['results']:
                if result.get('library') == 'datason':
                    metrics = {}
                    
                    if 'serialize_time' in result:
                        metrics['serialize_time'] = PerformanceMetric(
                            name='serialize_time',
                            value=result['serialize_time'],
                            unit='seconds',
                            higher_is_better=False
                        )
                    if 'deserialize_time' in result:
                        metrics['deserialize_time'] = PerformanceMetric(
                            name='deserialize_time',
                            value=result['deserialize_time'],
                            unit='seconds',
                            higher_is_better=False
                        )
                    if 'memory_usage' in result:
                        metrics['memory_usage'] = PerformanceMetric(
                            name='memory_usage',
                            value=result['memory_usage'],
                            unit='MB',
                            higher_is_better=False
                        )
                    
                    if metrics:
                        results.append(BenchmarkResult(
                            name=result.get('benchmark', 'unknown'),
                            metrics=metrics,
                            metadata=result
                        ))
        
        elif 'competitive' in data:
            # Quick benchmark format (both old and new)
            if 'tiers' in data['competitive']:
                # New quick enhanced format with tiers
                for tier_name, tier_data in data['competitive']['tiers'].items():
                    if 'datasets' in tier_data:
                        for dataset_name, dataset_data in tier_data['datasets'].items():
                            if 'serialization' in dataset_data and 'datason' in dataset_data['serialization']:
                                metrics = {}
                                
                                # Extract DataSON serialization metrics
                                datason_ser = dataset_data['serialization']['datason']
                                if 'mean' in datason_ser:
                                    serialize_metric_name = f"{dataset_name}_serialize_time"
                                    metrics[serialize_metric_name] = PerformanceMetric(
                                        name=serialize_metric_name,
                                        value=datason_ser['mean'],
                                        unit='seconds',
                                        higher_is_better=False
                                    )
                                
                                # Extract DataSON deserialization metrics if available
                                if 'deserialization' in dataset_data and 'datason' in dataset_data['deserialization']:
                                    datason_deser = dataset_data['deserialization']['datason']
                                    if 'mean' in datason_deser:
                                        deserialize_metric_name = f"{dataset_name}_deserialize_time"
                                        metrics[deserialize_metric_name] = PerformanceMetric(
                                            name=deserialize_metric_name,
                                            value=datason_deser['mean'],
                                            unit='seconds',
                                            higher_is_better=False
                                        )
                                
                                if metrics:
                                    results.append(BenchmarkResult(
                                        name=f"{tier_name}_{dataset_name}",
                                        metrics=metrics,
                                        metadata=dataset_data
                                    ))
            else:
                # Old quick format with direct datasets in competitive
                for dataset_name, dataset_data in data['competitive'].items():
                    if dataset_name == 'summary':  # Skip summary section
                        continue
                    if 'serialization' in dataset_data and 'datason' in dataset_data['serialization']:
                        metrics = {}
                        
                        # Extract DataSON serialization metrics
                        datason_ser = dataset_data['serialization']['datason']
                        if 'mean' in datason_ser:
                            serialize_metric_name = f"{dataset_name}_serialize_time"
                            metrics[serialize_metric_name] = PerformanceMetric(
                                name=serialize_metric_name,
                                value=datason_ser['mean'],
                                unit='seconds',
                                higher_is_better=False
                            )
                        
                        # Extract DataSON deserialization metrics if available
                        if 'deserialization' in dataset_data and 'datason' in dataset_data['deserialization']:
                            datason_deser = dataset_data['deserialization']['datason']
                            if 'mean' in datason_deser:
                                deserialize_metric_name = f"{dataset_name}_deserialize_time"
                                metrics[deserialize_metric_name] = PerformanceMetric(
                                    name=deserialize_metric_name,
                                    value=datason_deser['mean'],
                                    unit='seconds',
                                    higher_is_better=False
                                )
                        
                        if metrics:
                            results.append(BenchmarkResult(
                                name=dataset_name,
                                metrics=metrics,
                                metadata=dataset_data
                            ))
        
        elif 'results_by_tier' in data:
            # Tiered benchmark format (baseline format)
            for tier_name, tier_data in data['results_by_tier'].items():
                if 'datasets' in tier_data:
                    for dataset_name, dataset_data in tier_data['datasets'].items():
                        metrics = {}
                        
                        # Extract serialization metrics
                        if 'serialization' in dataset_data:
                            ser_data = dataset_data['serialization']
                            if 'mean_ms' in ser_data:
                                metrics['serialize_time'] = PerformanceMetric(
                                    name='serialize_time',
                                    value=ser_data['mean_ms'] / 1000.0,  # Convert ms to seconds
                                    unit='seconds',
                                    higher_is_better=False
                                )
                        
                        # Extract deserialization metrics
                        if 'deserialization' in dataset_data:
                            deser_data = dataset_data['deserialization']
                            if 'mean_ms' in deser_data:
                                metrics['deserialize_time'] = PerformanceMetric(
                                    name='deserialize_time',
                                    value=deser_data['mean_ms'] / 1000.0,  # Convert ms to seconds
                                    unit='seconds',
                                    higher_is_better=False
                                )
                        
                        if metrics:
                            results.append(BenchmarkResult(
                                name=f"{tier_name}_{dataset_name}",
                                metrics=metrics,
                                metadata=dataset_data
                            ))
        
        return results
    
    def find_baseline_file(self, results_dir: str = 'data/results') -> Optional[str]:
        """Find the most recent baseline results file"""
        if not os.path.exists(results_dir):
            return None
        
        # Look for latest results files first (e.g., latest.json or latest_*.json)
        latest_file = os.path.join(results_dir, 'latest.json')
        if os.path.exists(latest_file):
            return latest_file
        for fname in os.listdir(results_dir):
            if fname.startswith('latest_') and fname.endswith('.json'):
                return os.path.join(results_dir, fname)
        
        # Fall back to most recent full-run result (complete or comprehensive)
        result_files = []
        for file in os.listdir(results_dir):
            if file.endswith('.json') and ('comprehensive' in file or 'complete' in file):
                result_files.append(os.path.join(results_dir, file))
        
        if result_files:
            # Sort by modification time and return most recent
            result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return result_files[0]
        
        return None
    
    def calculate_change_percent(self, baseline: float, current: float, higher_is_better: bool = False) -> float:
        """Calculate percentage change between baseline and current values"""
        if baseline == 0:
            return 0.0
        
        change_percent = (current - baseline) / baseline
        
        # For metrics where higher is better (like throughput), invert the change
        # so that positive change_percent always means worse performance
        if higher_is_better:
            change_percent = -change_percent
        
        return change_percent
    
    def analyze_regression(self, baseline_value: float, current_value: float, 
                          metric: PerformanceMetric) -> RegressionResult:
        """Analyze a single metric for regression"""
        change_percent = self.calculate_change_percent(
            baseline_value, current_value, metric.higher_is_better
        )
        
        # Determine status based on thresholds
        if change_percent >= self.thresholds.fail_threshold:
            status = 'fail'
            message = f"FAIL: {metric.name} degraded by {change_percent:.1%} (>{self.thresholds.fail_threshold:.1%})"
        elif change_percent >= self.thresholds.warn_threshold:
            status = 'warn'
            message = f"WARN: {metric.name} degraded by {change_percent:.1%} (>{self.thresholds.warn_threshold:.1%})"
        elif change_percent >= self.thresholds.notice_threshold:
            status = 'notice'
            message = f"NOTICE: {metric.name} degraded by {change_percent:.1%} (>{self.thresholds.notice_threshold:.1%})"
        elif change_percent <= -self.thresholds.improvement_threshold:
            status = 'improvement'
            message = f"IMPROVEMENT: {metric.name} improved by {abs(change_percent):.1%}"
        else:
            status = 'stable'
            message = f"STABLE: {metric.name} changed by {change_percent:.1%} (within tolerance)"
        
        return RegressionResult(
            metric_name=metric.name,
            baseline_value=baseline_value,
            current_value=current_value,
            change_percent=change_percent,
            status=status,
            message=message
        )
    
    def find_matching_baseline(self, current_name: str, baseline_results: List[BenchmarkResult]) -> Optional[BenchmarkResult]:
        """Find the best matching baseline result for a current result"""
        # Direct name match first
        for baseline in baseline_results:
            if baseline.name == current_name:
                return baseline
        
        # Try to match dataset names across different formats
        # Extract dataset name from different formats
        current_dataset = current_name
        if '_' in current_name:
            # For formats like "json_safe_json_safe_simple" or "tier_dataset"
            parts = current_name.split('_')
            if len(parts) >= 2:
                # Try last part first (most specific)
                current_dataset = parts[-1]
                # Also try combinations
                if len(parts) >= 3:
                    # Try last two parts joined
                    current_dataset_alt = '_'.join(parts[-2:])
                else:
                    current_dataset_alt = None
            else:
                current_dataset_alt = None
        else:
            current_dataset_alt = None
        
        # Look for partial matches
        for baseline in baseline_results:
            baseline_dataset = baseline.name
            
            # Check if dataset names match (with some tolerance)
            if (current_dataset in baseline_dataset or 
                baseline_dataset in current_dataset or
                (current_dataset_alt and (current_dataset_alt in baseline_dataset or baseline_dataset in current_dataset_alt))):
                return baseline
            
            # Try fuzzy matching for common patterns
            if ('simple' in current_dataset and 'simple' in baseline_dataset) or \
               ('nested' in current_dataset and 'nested' in baseline_dataset) or \
               ('datetime' in current_dataset and 'datetime' in baseline_dataset) or \
               ('api' in current_dataset and 'api' in baseline_dataset) or \
               ('response' in current_dataset and 'response' in baseline_dataset):
                return baseline
        
        return None

    def compare_results(self, baseline_results: List[BenchmarkResult], 
                       current_results: List[BenchmarkResult]) -> List[RegressionResult]:
        """Compare current results against baseline"""
        regressions = []
        
        for current_result in current_results:
            baseline_result = self.find_matching_baseline(current_result.name, baseline_results)
            if not baseline_result:
                continue  # Skip if no baseline data
            
            # Compare each metric
            for metric_name, current_metric in current_result.metrics.items():
                baseline_metric = baseline_result.metrics.get(metric_name)
                if not baseline_metric:
                    continue  # Skip if baseline doesn't have this metric
                
                regression = self.analyze_regression(
                    baseline_metric.value,
                    current_metric.value,
                    current_metric
                )
                regressions.append(regression)
        
        return regressions
    
    def generate_pr_comment(self, regressions: List[RegressionResult]) -> str:
        """Generate GitHub PR comment text based on regression results"""
        if not regressions:
            return "🤷 **No performance data available for comparison**\n\nNo baseline or current benchmark results found."
        
        # Categorize results
        failures = [r for r in regressions if r.status == 'fail']
        warnings = [r for r in regressions if r.status == 'warn']
        notices = [r for r in regressions if r.status == 'notice']
        improvements = [r for r in regressions if r.status == 'improvement']
        stable = [r for r in regressions if r.status == 'stable']
        
        comment_parts = []
        
        # Header
        if failures:
            comment_parts.append("❌ **Performance Regression Detected**")
        elif warnings:
            comment_parts.append("⚠️ **Performance Warning**")
        elif improvements:
            comment_parts.append("✅ **Performance Check - Improvements Detected**")
        else:
            comment_parts.append("✅ **Performance Check - No Regressions**")
        
        comment_parts.append("")  # Empty line
        
        # Failures (block PR)
        if failures:
            comment_parts.append("### ❌ Critical Regressions (PR should not be merged)")
            comment_parts.append("| Metric | Baseline | Current | Change |")
            comment_parts.append("|--------|----------|---------|--------|")
            for r in failures:
                # Use smart unit formatting for time values
                baseline_str, current_str = self.format_comparison_values(r.baseline_value, r.current_value)
                comment_parts.append(
                    f"| {r.metric_name} | {baseline_str} | {current_str} | **{r.change_percent:+.1%}** |"
                )
            comment_parts.append("")
        
        # Warnings
        if warnings:
            comment_parts.append("### ⚠️ Performance Warnings")
            comment_parts.append("| Metric | Baseline | Current | Change |")
            comment_parts.append("|--------|----------|---------|--------|")
            for r in warnings:
                # Use smart unit formatting for time values
                baseline_str, current_str = self.format_comparison_values(r.baseline_value, r.current_value)
                comment_parts.append(
                    f"| {r.metric_name} | {baseline_str} | {current_str} | **{r.change_percent:+.1%}** |"
                )
            comment_parts.append("")
        
        # Improvements
        if improvements:
            comment_parts.append("### 🚀 Performance Improvements")
            comment_parts.append("| Metric | Baseline | Current | Change |")
            comment_parts.append("|--------|----------|---------|--------|")
            for r in improvements:
                # Use smart unit formatting for time values
                baseline_str, current_str = self.format_comparison_values(r.baseline_value, r.current_value)
                comment_parts.append(
                    f"| {r.metric_name} | {baseline_str} | {current_str} | **{r.change_percent:+.1%}** |"
                )
            comment_parts.append("")
        
        # Stable metrics (summary only)
        if stable:
            comment_parts.append(f"### ✅ Stable Metrics ({len(stable)} metrics within tolerance)")
            comment_parts.append("")
        
        # Footer with details
        comment_parts.append("---")
        comment_parts.append("**Regression Detection Thresholds:**")
        comment_parts.append(f"- 🚫 Fail: >{self.thresholds.fail_threshold:.0%} degradation")
        comment_parts.append(f"- ⚠️ Warn: >{self.thresholds.warn_threshold:.0%} degradation")
        comment_parts.append(f"- 📋 Notice: >{self.thresholds.notice_threshold:.0%} degradation")
        comment_parts.append("")
        comment_parts.append(f"*Generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*")
        
        return "\n".join(comment_parts)
    
    def should_fail_pr(self, regressions: List[RegressionResult]) -> bool:
        """Determine if PR should fail based on regression results"""
        return any(r.status == 'fail' for r in regressions)
    
    def detect_regressions(self, current_file: str, baseline_file: Optional[str] = None) -> Tuple[List[RegressionResult], bool]:
        """
        Main entry point for regression detection.
        Returns (regressions, should_fail_pr)
        """
        try:
            # Find baseline file if not provided
            if baseline_file is None:
                baseline_file = self.find_baseline_file()
                if baseline_file is None:
                    return [], False  # No baseline to compare against
            
            # Load results
            baseline_results = self.load_benchmark_results(baseline_file)
            current_results = self.load_benchmark_results(current_file)
            
            # Compare results
            regressions = self.compare_results(baseline_results, current_results)
            
            # Determine if PR should fail
            should_fail = self.should_fail_pr(regressions)
            
            return regressions, should_fail
            
        except Exception as e:
            print(f"Error during regression detection: {e}")
            return [], False
    
    def save_regression_report(self, regressions: List[RegressionResult], output_file: str):
        """Save regression analysis to JSON file"""
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'thresholds': {
                'fail': self.thresholds.fail_threshold,
                'warn': self.thresholds.warn_threshold,
                'notice': self.thresholds.notice_threshold,
                'improvement': self.thresholds.improvement_threshold,
            },
            'summary': {
                'total_metrics': len(regressions),
                'failures': len([r for r in regressions if r.status == 'fail']),
                'warnings': len([r for r in regressions if r.status == 'warn']),
                'notices': len([r for r in regressions if r.status == 'notice']),
                'improvements': len([r for r in regressions if r.status == 'improvement']),
                'stable': len([r for r in regressions if r.status == 'stable']),
                'should_fail_pr': self.should_fail_pr(regressions),
            },
            'regressions': [
                {
                    'metric_name': r.metric_name,
                    'baseline_value': r.baseline_value,
                    'current_value': r.current_value,
                    'change_percent': r.change_percent,
                    'status': r.status,
                    'message': r.message,
                }
                for r in regressions
            ]
        }
        
        # Create directory if output_file has a directory component
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            # Dogfood DataSON v0.11.2 for JSON serialization
            f.write(datason.dumps_json(report))

def main():
    """Command line interface for regression detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect performance regressions')
    parser.add_argument('current_file', help='Current benchmark results file')
    parser.add_argument('--baseline', help='Baseline results file (auto-detect if not provided)')
    parser.add_argument('--output', help='Output file for regression report')
    parser.add_argument('--pr-comment', help='Output file for GitHub PR comment')
    parser.add_argument('--fail-threshold', type=float, default=0.25, help='Failure threshold (default: 0.25)')
    parser.add_argument('--warn-threshold', type=float, default=0.10, help='Warning threshold (default: 0.10)')
    
    args = parser.parse_args()
    
    # Set up detector with custom thresholds
    thresholds = RegressionThresholds(
        fail_threshold=args.fail_threshold,
        warn_threshold=args.warn_threshold,
    )
    detector = PerformanceRegressionDetector(thresholds)
    
    # Detect regressions
    regressions, should_fail = detector.detect_regressions(args.current_file, args.baseline)
    
    # Generate outputs
    if args.output:
        detector.save_regression_report(regressions, args.output)
        print(f"Regression report saved to: {args.output}")
    
    if args.pr_comment:
        comment = detector.generate_pr_comment(regressions)
        # Only create directory if there's a directory component
        comment_dir = os.path.dirname(args.pr_comment)
        if comment_dir:
            os.makedirs(comment_dir, exist_ok=True)
        with open(args.pr_comment, 'w') as f:
            f.write(comment)
        print(f"PR comment saved to: {args.pr_comment}")
    
    # Print summary
    print(f"\nRegression Detection Summary:")
    print(f"- Total metrics analyzed: {len(regressions)}")
    print(f"- Failures: {len([r for r in regressions if r.status == 'fail'])}")
    print(f"- Warnings: {len([r for r in regressions if r.status == 'warn'])}")
    print(f"- Improvements: {len([r for r in regressions if r.status == 'improvement'])}")
    print(f"- Should fail PR: {should_fail}")
    
    # Exit with error code if PR should fail
    if should_fail:
        print("\n❌ Critical regressions detected - PR should not be merged!")
        exit(1)
    else:
        print("\n✅ No critical regressions detected")
        exit(0)

if __name__ == '__main__':
    main() 

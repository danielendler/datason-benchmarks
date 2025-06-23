#!/usr/bin/env python3
"""
Historical trend analysis for DataSON benchmarks.
Tracks performance evolution over time and identifies patterns.
"""

import json
import os
import glob
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import statistics
from pathlib import Path
import argparse
import datason

@dataclass
class TrendDataPoint:
    """Single data point in performance trend"""
    timestamp: str
    date: str
    version: Optional[str]
    commit_hash: Optional[str]
    value: float
    metadata: Dict[str, Any]

@dataclass
class TrendMetric:
    """Performance metric trend analysis"""
    metric_name: str
    library: str
    benchmark: str
    data_points: List[TrendDataPoint]
    trend_direction: str  # 'improving', 'degrading', 'stable', 'volatile'
    trend_strength: float  # 0-1, how strong the trend is
    recent_change: float  # percentage change in recent period
    statistics: Dict[str, float]

@dataclass
class TrendSummary:
    """Summary of trend analysis"""
    analysis_date: str
    lookback_days: int
    total_metrics: int
    improving_metrics: int
    degrading_metrics: int
    stable_metrics: int
    volatile_metrics: int
    key_findings: List[str]
    metrics: List[TrendMetric]

class PerformanceTrendAnalyzer:
    """
    Analyzes historical performance trends from benchmark results.
    Implements simple trend tracking without complex statistics.
    """
    
    def __init__(self, lookback_days: int = 90):
        self.lookback_days = lookback_days
        self.cutoff_date = datetime.now() - timedelta(days=lookback_days)
    
    def parse_filename_timestamp(self, filename: str) -> Optional[datetime]:
        """Extract timestamp from benchmark result filenames"""
        # Look for patterns like: YYYYMMDD_HHMMSS or YYYY-MM-DD_HH-MM-SS
        patterns = [
            r'(\d{8}_\d{6})',  # 20240102_123456
            r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})',  # 2024-01-02_12-34-56
            r'(\d{4}\d{2}\d{2})',  # 20240102
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                timestamp_str = match.group(1)
                try:
                    if '_' in timestamp_str and len(timestamp_str) == 15:  # YYYYMMDD_HHMMSS
                        return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    elif '-' in timestamp_str:  # YYYY-MM-DD_HH-MM-SS
                        return datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
                    elif len(timestamp_str) == 8:  # YYYYMMDD
                        return datetime.strptime(timestamp_str, '%Y%m%d')
                except ValueError:
                    continue
        
        # Fallback to file modification time
        try:
            return datetime.fromtimestamp(os.path.getmtime(filename))
        except:
            return None
    
    def extract_version_info(self, filepath: str, data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Extract version and commit hash from benchmark data"""
        version = None
        commit_hash = None
        
        # Try to extract from metadata
        if 'metadata' in data:
            metadata = data['metadata']
            version = metadata.get('datason_version') or metadata.get('version')
            commit_hash = metadata.get('commit_hash') or metadata.get('git_hash')
        
        # Try to extract from filename
        if not version:
            version_match = re.search(r'_(\d+\.\d+\.\d+)_', filepath)
            if version_match:
                version = version_match.group(1)
        
        if not commit_hash:
            commit_match = re.search(r'_([a-f0-9]{8,})_', filepath)
            if commit_match:
                commit_hash = commit_match.group(1)
        
        return version, commit_hash
    
    def load_historical_data(self, results_dir: str) -> List[Tuple[str, datetime, Dict[str, Any]]]:
        """Load all historical benchmark results"""
        results = []
        
        # Find all JSON result files
        patterns = [
            os.path.join(results_dir, '*.json'),
            os.path.join(results_dir, '**/*.json'),
            os.path.join(results_dir, 'weekly/**/*.json'),
        ]
        
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern, recursive=True))
        
        # Remove duplicates and sort
        files = list(set(files))
        
        for filepath in files:
            # Skip non-benchmark files
            filename = os.path.basename(filepath)
            if filename in ['latest.json', 'generation_summary.json']:
                continue
            
            try:
                timestamp = self.parse_filename_timestamp(filepath)
                if not timestamp or timestamp < self.cutoff_date:
                    continue
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                results.append((filepath, timestamp, data))
                
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {filepath}: {e}")
                continue
        
        # Sort by timestamp
        results.sort(key=lambda x: x[1])
        
        return results
    
    def extract_metrics_from_result(self, filepath: str, timestamp: datetime, 
                                   data: Dict[str, Any]) -> List[TrendDataPoint]:
        """Extract performance metrics from a single result file"""
        data_points = []
        version, commit_hash = self.extract_version_info(filepath, data)
        
        # Handle different result formats
        if 'benchmarks' in data:
            # Competitive benchmark format
            for bench_name, bench_data in data['benchmarks'].items():
                if 'results' in bench_data:
                    for lib_name, lib_results in bench_data['results'].items():
                        # Extract metrics for each library
                        for metric_name, value in lib_results.items():
                            if isinstance(value, (int, float)) and metric_name in [
                                'serialize_time', 'deserialize_time', 'memory_usage', 
                                'throughput', 'size', 'cpu_usage'
                            ]:
                                data_points.append(TrendDataPoint(
                                    timestamp=timestamp.isoformat(),
                                    date=timestamp.strftime('%Y-%m-%d'),
                                    version=version,
                                    commit_hash=commit_hash,
                                    value=float(value),
                                    metadata={
                                        'metric': metric_name,
                                        'library': lib_name,
                                        'benchmark': bench_name,
                                        'filepath': filepath,
                                    }
                                ))
        
        elif 'results' in data:
            # Version comparison or simple results format
            results = data['results']
            if isinstance(results, list):
                for result in results:
                    library = result.get('library', 'unknown')
                    benchmark = result.get('benchmark', 'unknown')
                    
                    for metric_name, value in result.items():
                        if isinstance(value, (int, float)) and metric_name in [
                            'serialize_time', 'deserialize_time', 'memory_usage', 
                            'throughput', 'size', 'cpu_usage'
                        ]:
                            data_points.append(TrendDataPoint(
                                timestamp=timestamp.isoformat(),
                                date=timestamp.strftime('%Y-%m-%d'),
                                version=version,
                                commit_hash=commit_hash,
                                value=float(value),
                                metadata={
                                    'metric': metric_name,
                                    'library': library,
                                    'benchmark': benchmark,
                                    'filepath': filepath,
                                }
                            ))
        
        return data_points
    
    def group_data_points(self, all_data_points: List[TrendDataPoint]) -> Dict[str, List[TrendDataPoint]]:
        """Group data points by metric/library/benchmark combination"""
        groups = {}
        
        for point in all_data_points:
            key = f"{point.metadata['library']}_{point.metadata['benchmark']}_{point.metadata['metric']}"
            if key not in groups:
                groups[key] = []
            groups[key].append(point)
        
        # Sort each group by timestamp
        for key in groups:
            groups[key].sort(key=lambda x: x.timestamp)
        
        return groups
    
    def calculate_trend_direction(self, values: List[float]) -> Tuple[str, float]:
        """
        Calculate trend direction and strength using simple linear approximation.
        Returns (direction, strength) where direction is 'improving'/'degrading'/'stable'/'volatile'
        and strength is 0-1.
        """
        if len(values) < 3:
            return 'stable', 0.0
        
        # Calculate simple linear trend using first and last values
        first_third = values[:len(values)//3]
        last_third = values[-len(values)//3:]
        
        first_avg = statistics.mean(first_third)
        last_avg = statistics.mean(last_third)
        
        if first_avg == 0:
            return 'stable', 0.0
        
        change_percent = (last_avg - first_avg) / first_avg
        
        # Calculate volatility (coefficient of variation)
        mean_val = statistics.mean(values)
        if mean_val == 0:
            volatility = 0
        else:
            std_dev = statistics.stdev(values) if len(values) > 1 else 0
            volatility = std_dev / mean_val
        
        # Determine direction and strength
        abs_change = abs(change_percent)
        
        if volatility > 0.3:  # High volatility threshold
            return 'volatile', min(volatility, 1.0)
        elif abs_change < 0.05:  # Less than 5% change
            return 'stable', abs_change * 4  # Scale to 0-0.2
        elif change_percent < 0:
            # For most metrics, lower is better (improving)
            return 'improving', min(abs_change * 2, 1.0)
        else:
            # Higher values are generally degrading
            return 'degrading', min(abs_change * 2, 1.0)
    
    def analyze_metric_trend(self, key: str, data_points: List[TrendDataPoint]) -> TrendMetric:
        """Analyze trend for a single metric"""
        if not data_points:
            return None
        
        # Extract values and metadata
        values = [point.value for point in data_points]
        library = data_points[0].metadata['library']
        benchmark = data_points[0].metadata['benchmark']
        metric_name = data_points[0].metadata['metric']
        
        # Calculate trend
        trend_direction, trend_strength = self.calculate_trend_direction(values)
        
        # Calculate recent change (last vs previous period)
        recent_change = 0.0
        if len(values) >= 2:
            recent_period = max(1, len(values) // 4)  # Last 25% of data points
            recent_avg = statistics.mean(values[-recent_period:])
            previous_avg = statistics.mean(values[-2*recent_period:-recent_period] if len(values) >= 2*recent_period else values[:-recent_period])
            
            if previous_avg != 0:
                recent_change = (recent_avg - previous_avg) / previous_avg
        
        # Calculate statistics
        stats = {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'first_value': values[0],
            'last_value': values[-1],
            'total_change': (values[-1] - values[0]) / values[0] if values[0] != 0 else 0,
        }
        
        return TrendMetric(
            metric_name=metric_name,
            library=library,
            benchmark=benchmark,
            data_points=data_points,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            recent_change=recent_change,
            statistics=stats
        )
    
    def identify_key_findings(self, trends: List[TrendMetric]) -> List[str]:
        """Identify key findings from trend analysis"""
        findings = []
        
        # Focus on DataSON performance
        datason_trends = [t for t in trends if t.library == 'datason']
        
        if not datason_trends:
            findings.append("⚠️ No DataSON performance trends found in the data")
            return findings
        
        # Analyze DataSON trends
        improving = [t for t in datason_trends if t.trend_direction == 'improving']
        degrading = [t for t in datason_trends if t.trend_direction == 'degrading']
        volatile = [t for t in datason_trends if t.trend_direction == 'volatile']
        
        # Overall performance direction
        if len(improving) > len(degrading):
            findings.append(f"✅ DataSON performance is generally improving ({len(improving)} metrics up vs {len(degrading)} down)")
        elif len(degrading) > len(improving):
            findings.append(f"⚠️ DataSON performance is generally degrading ({len(degrading)} metrics down vs {len(improving)} up)")
        else:
            findings.append(f"📊 DataSON performance is mixed ({len(improving)} improving, {len(degrading)} degrading)")
        
        # Significant improvements
        strong_improvements = [t for t in improving if t.trend_strength > 0.3]
        if strong_improvements:
            best_improvement = max(strong_improvements, key=lambda x: x.trend_strength)
            findings.append(f"🚀 Best improvement: {best_improvement.metric_name} in {best_improvement.benchmark} (+{abs(best_improvement.statistics['total_change']):.1%})")
        
        # Significant degradations
        strong_degradations = [t for t in degrading if t.trend_strength > 0.3]
        if strong_degradations:
            worst_degradation = max(strong_degradations, key=lambda x: x.trend_strength)
            findings.append(f"🔥 Concerning degradation: {worst_degradation.metric_name} in {worst_degradation.benchmark} ({worst_degradation.statistics['total_change']:+.1%})")
        
        # Volatile metrics
        if volatile:
            most_volatile = max(volatile, key=lambda x: x.trend_strength)
            findings.append(f"📈 Most volatile metric: {most_volatile.metric_name} in {most_volatile.benchmark} (high variance)")
        
        # Recent changes
        recent_improvements = [t for t in datason_trends if t.recent_change < -0.05]  # 5% improvement
        recent_degradations = [t for t in datason_trends if t.recent_change > 0.05]   # 5% degradation
        
        if recent_improvements:
            findings.append(f"🆕 Recent improvements in {len(recent_improvements)} metrics")
        if recent_degradations:
            findings.append(f"🆕 Recent degradations in {len(recent_degradations)} metrics")
        
        # Data quality
        avg_data_points = statistics.mean([len(t.data_points) for t in datason_trends])
        findings.append(f"📊 Average {avg_data_points:.1f} data points per metric over {self.lookback_days} days")
        
        return findings
    
    def analyze_trends(self, results_dir: str) -> TrendSummary:
        """Main trend analysis entry point"""
        print(f"Loading historical data from {results_dir} (last {self.lookback_days} days)")
        
        # Load historical data
        historical_data = self.load_historical_data(results_dir)
        print(f"Found {len(historical_data)} result files")
        
        if not historical_data:
            return TrendSummary(
                analysis_date=datetime.now().isoformat(),
                lookback_days=self.lookback_days,
                total_metrics=0,
                improving_metrics=0,
                degrading_metrics=0,
                stable_metrics=0,
                volatile_metrics=0,
                key_findings=["No historical data found for trend analysis"],
                metrics=[]
            )
        
        # Extract all data points
        all_data_points = []
        for filepath, timestamp, data in historical_data:
            data_points = self.extract_metrics_from_result(filepath, timestamp, data)
            all_data_points.extend(data_points)
        
        print(f"Extracted {len(all_data_points)} metric data points")
        
        # Group by metric/library/benchmark
        grouped_data = self.group_data_points(all_data_points)
        print(f"Analyzing trends for {len(grouped_data)} metric combinations")
        
        # Analyze each metric trend
        trends = []
        for key, data_points in grouped_data.items():
            # Only analyze if we have enough data points
            if len(data_points) >= 3:
                trend = self.analyze_metric_trend(key, data_points)
                if trend:
                    trends.append(trend)
        
        # Categorize trends
        improving = len([t for t in trends if t.trend_direction == 'improving'])
        degrading = len([t for t in trends if t.trend_direction == 'degrading'])
        stable = len([t for t in trends if t.trend_direction == 'stable'])
        volatile = len([t for t in trends if t.trend_direction == 'volatile'])
        
        # Identify key findings
        key_findings = self.identify_key_findings(trends)
        
        return TrendSummary(
            analysis_date=datetime.now().isoformat(),
            lookback_days=self.lookback_days,
            total_metrics=len(trends),
            improving_metrics=improving,
            degrading_metrics=degrading,
            stable_metrics=stable,
            volatile_metrics=volatile,
            key_findings=key_findings,
            metrics=trends
        )
    
    def save_trend_analysis(self, summary: TrendSummary, output_file: str):
        """Save trend analysis to JSON file using DataSON (dogfooding)"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert to serializable format
        output_data = asdict(summary)
        
        with open(output_file, 'w') as f:
            # Dogfood DataSON for serialization (no indent param in DataSON)
            f.write(datason.dumps(output_data))
        
        print(f"Trend analysis saved to: {output_file}")
    
    def generate_trend_report(self, summary: TrendSummary) -> str:
        """Generate human-readable trend report"""
        report_lines = []
        
        # Header
        report_lines.append(f"# DataSON Performance Trend Analysis")
        report_lines.append(f"**Analysis Date:** {summary.analysis_date}")
        report_lines.append(f"**Lookback Period:** {summary.lookback_days} days")
        report_lines.append("")
        
        # Summary
        report_lines.append("## Summary")
        report_lines.append(f"- **Total Metrics Analyzed:** {summary.total_metrics}")
        report_lines.append(f"- **Improving:** {summary.improving_metrics} ({summary.improving_metrics/summary.total_metrics*100:.1f}%)")
        report_lines.append(f"- **Degrading:** {summary.degrading_metrics} ({summary.degrading_metrics/summary.total_metrics*100:.1f}%)")
        report_lines.append(f"- **Stable:** {summary.stable_metrics} ({summary.stable_metrics/summary.total_metrics*100:.1f}%)")
        report_lines.append(f"- **Volatile:** {summary.volatile_metrics} ({summary.volatile_metrics/summary.total_metrics*100:.1f}%)")
        report_lines.append("")
        
        # Key Findings
        report_lines.append("## Key Findings")
        for finding in summary.key_findings:
            report_lines.append(f"- {finding}")
        report_lines.append("")
        
        # DataSON specific trends
        datason_metrics = [m for m in summary.metrics if m.library == 'datason']
        if datason_metrics:
            report_lines.append("## DataSON Performance Trends")
            report_lines.append("| Metric | Benchmark | Trend | Change | Strength |")
            report_lines.append("|--------|-----------|-------|--------|----------|")
            
            for metric in sorted(datason_metrics, key=lambda x: x.trend_strength, reverse=True):
                trend_icon = {
                    'improving': '📈',
                    'degrading': '📉',
                    'stable': '➡️',
                    'volatile': '📊'
                }.get(metric.trend_direction, '❓')
                
                report_lines.append(
                    f"| {metric.metric_name} | {metric.benchmark} | {trend_icon} {metric.trend_direction} | "
                    f"{metric.statistics['total_change']:+.1%} | {metric.trend_strength:.2f} |"
                )
            report_lines.append("")
        
        # Competitive comparison trends
        competitive_libs = set(m.library for m in summary.metrics if m.library != 'datason')
        if competitive_libs:
            report_lines.append("## Competitive Library Trends")
            for lib in sorted(competitive_libs):
                lib_metrics = [m for m in summary.metrics if m.library == lib]
                improving = len([m for m in lib_metrics if m.trend_direction == 'improving'])
                total = len(lib_metrics)
                if total > 0:
                    report_lines.append(f"- **{lib}:** {improving}/{total} metrics improving ({improving/total*100:.1f}%)")
            report_lines.append("")
        
        return "\n".join(report_lines)

def main():
    """Command line interface for trend analysis"""
    parser = argparse.ArgumentParser(description='Analyze performance trends')
    parser.add_argument('--input-dir', default='data/results', help='Input directory with benchmark results')
    parser.add_argument('--output', help='Output file for trend analysis JSON')
    parser.add_argument('--report', help='Output file for human-readable report')
    parser.add_argument('--lookback-days', type=int, default=90, help='Days to look back for analysis')
    parser.add_argument('--lookback-weeks', type=int, help='Weeks to look back (overrides days)')
    
    args = parser.parse_args()
    
    # Convert weeks to days if specified
    lookback_days = args.lookback_days
    if args.lookback_weeks:
        lookback_days = args.lookback_weeks * 7
    
    # Initialize analyzer
    analyzer = PerformanceTrendAnalyzer(lookback_days=lookback_days)
    
    # Run analysis
    print(f"Starting trend analysis...")
    summary = analyzer.analyze_trends(args.input_dir)
    
    # Save results
    if args.output:
        analyzer.save_trend_analysis(summary, args.output)
    
    if args.report:
        report = analyzer.generate_trend_report(summary)
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"Trend report saved to: {args.report}")
    
    # Print summary
    print(f"\n📊 Trend Analysis Summary:")
    print(f"- Analyzed {summary.total_metrics} metrics over {lookback_days} days")
    print(f"- Improving: {summary.improving_metrics}")
    print(f"- Degrading: {summary.degrading_metrics}")
    print(f"- Stable: {summary.stable_metrics}")
    print(f"- Volatile: {summary.volatile_metrics}")
    
    print(f"\n🔍 Key Findings:")
    for finding in summary.key_findings[:5]:  # Show top 5 findings
        print(f"  {finding}")

if __name__ == '__main__':
    main() 
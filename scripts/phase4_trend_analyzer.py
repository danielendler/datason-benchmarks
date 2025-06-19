#!/usr/bin/env python3
"""
Phase 4 Trend Analyzer
======================

Advanced trend analysis system for tracking DataSON performance over time,
detecting regressions, and providing historical insights.
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """Phase 4 trend analysis and regression detection system."""
    
    def __init__(self, db_path: str = "data/trends.db", results_dir: str = "data/results"):
        self.db_path = Path(db_path)
        self.results_dir = Path(results_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize trend tracking database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    suite_type TEXT NOT NULL,
                    datason_version TEXT,
                    python_version TEXT,
                    file_path TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    library_name TEXT NOT NULL,
                    method_name TEXT,
                    dataset_name TEXT NOT NULL,
                    metric_type TEXT NOT NULL,  -- 'serialization', 'deserialization', 'success_rate'
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,  -- 'ms', 'percent', 'bytes'
                    tier TEXT,  -- 'json_safe', 'object_enhanced', 'ml_complex'
                    FOREIGN KEY (run_id) REFERENCES benchmark_runs (id)
                );
                
                CREATE TABLE IF NOT EXISTS trend_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    alert_type TEXT NOT NULL,  -- 'regression', 'improvement', 'anomaly'
                    library_name TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    severity TEXT NOT NULL,  -- 'low', 'medium', 'high', 'critical'
                    description TEXT NOT NULL,
                    baseline_value REAL,
                    current_value REAL,
                    change_percent REAL
                );
                
                CREATE INDEX IF NOT EXISTS idx_performance_metrics_library 
                ON performance_metrics (library_name, metric_type, dataset_name);
                
                CREATE INDEX IF NOT EXISTS idx_benchmark_runs_timestamp 
                ON benchmark_runs (timestamp);
            """)
    
    def ingest_benchmark_results(self, result_file: str) -> int:
        """Ingest benchmark results into trend database."""
        result_path = self.results_dir / result_file
        
        if not result_path.exists():
            raise FileNotFoundError(f"Result file not found: {result_path}")
        
        logger.info(f"📊 Ingesting benchmark results: {result_file}")
        
        with open(result_path, 'r') as f:
            results = json.load(f)
        
        # Extract metadata
        metadata = results.get("metadata", {})
        suite_type = results.get("suite_type", "unknown")
        
        # Insert benchmark run record
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO benchmark_runs (timestamp, suite_type, datason_version, 
                                          python_version, file_path)
                VALUES (?, ?, ?, ?, ?)
            """, (
                metadata.get("timestamp", time.time()),
                suite_type,
                metadata.get("datason_version", "unknown"),
                metadata.get("python_version", "unknown"),
                str(result_path)
            ))
            
            run_id = cursor.lastrowid
            
            # Extract and insert performance metrics
            metrics_extracted = self._extract_performance_metrics(results, run_id)
            
            if metrics_extracted:
                cursor.executemany("""
                    INSERT INTO performance_metrics (run_id, library_name, method_name,
                                                   dataset_name, metric_type, value, unit, tier)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, metrics_extracted)
                
                logger.info(f"✅ Ingested {len(metrics_extracted)} performance metrics")
            
            conn.commit()
            
        return run_id
    
    def _extract_performance_metrics(self, results: Dict[str, Any], 
                                   run_id: int) -> List[Tuple]:
        """Extract performance metrics from benchmark results."""
        metrics = []
        
        # Handle competitive benchmark results
        if "competitive" in results:
            competitive = results["competitive"]
            
            if "tiers" in competitive:
                for tier_name, tier_data in competitive["tiers"].items():
                    if "datasets" in tier_data:
                        for dataset_name, dataset in tier_data["datasets"].items():
                            # Serialization metrics
                            if "serialization" in dataset:
                                for library, perf in dataset["serialization"].items():
                                    if isinstance(perf, dict) and "mean_ms" in perf:
                                        metrics.append((
                                            run_id, library, None, dataset_name,
                                            "serialization", perf["mean_ms"], "ms", tier_name
                                        ))
                                        
                                        # Success rate
                                        if "successful_runs" in perf:
                                            success_rate = perf["successful_runs"] / 10.0 * 100  # Assuming 10 iterations
                                            metrics.append((
                                                run_id, library, None, dataset_name,
                                                "success_rate", success_rate, "percent", tier_name
                                            ))
                            
                            # Deserialization metrics
                            if "deserialization" in dataset:
                                for library, perf in dataset["deserialization"].items():
                                    if isinstance(perf, dict) and "mean_ms" in perf:
                                        metrics.append((
                                            run_id, library, None, dataset_name,
                                            "deserialization", perf["mean_ms"], "ms", tier_name
                                        ))
                            
                            # Output size metrics
                            if "output_size" in dataset:
                                for library, size_info in dataset["output_size"].items():
                                    if isinstance(size_info, dict) and "size" in size_info:
                                        metrics.append((
                                            run_id, library, None, dataset_name,
                                            "output_size", size_info["size"], 
                                            size_info.get("size_type", "bytes"), tier_name
                                        ))
        
        # Handle Phase 2 results
        if "phase2" in results:
            phase2 = results["phase2"]
            
            # Security testing metrics
            if "security_testing" in phase2 and "performance" in phase2["security_testing"]:
                perf = phase2["security_testing"]["performance"]
                for method, data in perf.items():
                    if isinstance(data, dict) and "mean_ms" in data:
                        metrics.append((
                            run_id, "datason", method, "security_test",
                            "serialization", data["mean_ms"], "ms", "security"
                        ))
            
            # Accuracy testing metrics
            if "accuracy_testing" in phase2 and "loading_methods" in phase2["accuracy_testing"]:
                methods = phase2["accuracy_testing"]["loading_methods"]
                for method, data in methods.items():
                    if isinstance(data, dict):
                        if "mean_ms" in data:
                            metrics.append((
                                run_id, "datason", method, "accuracy_test",
                                "deserialization", data["mean_ms"], "ms", "accuracy"
                            ))
                        if "success_rate" in data:
                            metrics.append((
                                run_id, "datason", method, "accuracy_test",
                                "success_rate", data["success_rate"] * 100, "percent", "accuracy"
                            ))
        
        # Handle Phase 3 results
        if "phase3" in results:
            phase3 = results["phase3"]
            
            # Domain scenarios
            if "domain_scenarios" in phase3 and "scenarios" in phase3["domain_scenarios"]:
                scenarios = phase3["domain_scenarios"]["scenarios"]
                for scenario_name, scenario_data in scenarios.items():
                    if "results" in scenario_data and "serialization" in scenario_data["results"]:
                        domain = scenario_data.get("domain", "unknown")
                        for method, perf in scenario_data["results"]["serialization"].items():
                            if isinstance(perf, dict) and "mean_ms" in perf:
                                metrics.append((
                                    run_id, "datason", method, scenario_name,
                                    "serialization", perf["mean_ms"], "ms", domain
                                ))
            
            # Success analysis
            if "success_analysis" in phase3:
                success = phase3["success_analysis"]
                if "datasets" in success:
                    for dataset_name, dataset_data in success["datasets"].items():
                        if "metrics" in dataset_data:
                            for library, metrics_data in dataset_data["metrics"].items():
                                if isinstance(metrics_data, dict):
                                    # Success rates
                                    if "deserialization_success_rate" in metrics_data:
                                        metrics.append((
                                            run_id, library, None, dataset_name,
                                            "success_rate", metrics_data["deserialization_success_rate"] * 100,
                                            "percent", "success_analysis"
                                        ))
                                    
                                    # Accuracy scores
                                    if "data_accuracy_score" in metrics_data:
                                        metrics.append((
                                            run_id, library, None, dataset_name,
                                            "accuracy_score", metrics_data["data_accuracy_score"] * 100,
                                            "percent", "success_analysis"
                                        ))
        
        return metrics
    
    def detect_performance_regressions(self, lookback_days: int = 30) -> List[Dict[str, Any]]:
        """Detect performance regressions in recent benchmark runs."""
        logger.info(f"🔍 Detecting performance regressions (last {lookback_days} days)")
        
        cutoff_time = time.time() - (lookback_days * 24 * 60 * 60)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get recent performance data
            query = """
                SELECT pm.library_name, pm.method_name, pm.dataset_name, pm.metric_type,
                       pm.value, pm.unit, pm.tier, br.timestamp
                FROM performance_metrics pm
                JOIN benchmark_runs br ON pm.run_id = br.id
                WHERE br.timestamp > ?
                ORDER BY pm.library_name, pm.metric_type, pm.dataset_name, br.timestamp
            """
            
            cursor = conn.cursor()
            cursor.execute(query, (cutoff_time,))
            results = cursor.fetchall()
        
        # Group by library/metric/dataset and analyze trends
        grouped_data = {}
        for row in results:
            library, method, dataset, metric_type, value, unit, tier, timestamp = row
            key = (library, method or "", dataset, metric_type)
            
            if key not in grouped_data:
                grouped_data[key] = []
            
            grouped_data[key].append((timestamp, value))
        
        # Detect regressions
        regressions = []
        
        for key, data_points in grouped_data.items():
            if len(data_points) < 3:  # Need at least 3 points for trend analysis
                continue
            
            library, method, dataset, metric_type = key
            
            # Sort by timestamp
            data_points.sort(key=lambda x: x[0])
            
            # Calculate trend
            regression = self._analyze_performance_trend(
                data_points, library, method, dataset, metric_type
            )
            
            if regression:
                regressions.append(regression)
        
        # Store regression alerts
        if regressions:
            self._store_regression_alerts(regressions)
        
        logger.info(f"🚨 Detected {len(regressions)} performance regressions")
        return regressions
    
    def _analyze_performance_trend(self, data_points: List[Tuple[float, float]], 
                                 library: str, method: str, dataset: str, 
                                 metric_type: str) -> Optional[Dict[str, Any]]:
        """Analyze performance trend for regression detection."""
        if len(data_points) < 3:
            return None
        
        # Extract values
        timestamps, values = zip(*data_points)
        
        # Calculate baseline (first 3 values) vs recent (last 3 values)
        baseline_values = values[:3]
        recent_values = values[-3:]
        
        baseline_mean = mean(baseline_values)
        recent_mean = mean(recent_values)
        
        # Skip if baseline is zero (avoid division by zero)
        if baseline_mean == 0:
            return None
        
        # Calculate percentage change
        change_percent = ((recent_mean - baseline_mean) / baseline_mean) * 100
        
        # Determine if this is a regression
        is_regression = False
        severity = "low"
        
        # For time-based metrics (ms), increase is bad
        if metric_type in ["serialization", "deserialization"] and change_percent > 0:
            if change_percent > 50:
                is_regression = True
                severity = "critical"
            elif change_percent > 25:
                is_regression = True
                severity = "high"
            elif change_percent > 10:
                is_regression = True
                severity = "medium"
            elif change_percent > 5:
                is_regression = True
                severity = "low"
        
        # For success rates, decrease is bad
        elif metric_type in ["success_rate", "accuracy_score"] and change_percent < 0:
            if abs(change_percent) > 20:
                is_regression = True
                severity = "critical"
            elif abs(change_percent) > 10:
                is_regression = True
                severity = "high"
            elif abs(change_percent) > 5:
                is_regression = True
                severity = "medium"
            elif abs(change_percent) > 2:
                is_regression = True
                severity = "low"
        
        if not is_regression:
            return None
        
        # Create regression alert
        description = f"{library}"
        if method:
            description += f".{method}"
        description += f" {metric_type} on {dataset}: {change_percent:+.1f}% change"
        
        return {
            "alert_type": "regression",
            "library_name": library,
            "method_name": method,
            "dataset_name": dataset,
            "metric_type": metric_type,
            "severity": severity,
            "description": description,
            "baseline_value": baseline_mean,
            "current_value": recent_mean,
            "change_percent": change_percent,
            "data_points": len(data_points)
        }
    
    def _store_regression_alerts(self, regressions: List[Dict[str, Any]]):
        """Store regression alerts in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for regression in regressions:
                cursor.execute("""
                    INSERT INTO trend_alerts (alert_type, library_name, metric_type,
                                            severity, description, baseline_value,
                                            current_value, change_percent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    regression["alert_type"],
                    regression["library_name"],
                    regression["metric_type"],
                    regression["severity"],
                    regression["description"],
                    regression["baseline_value"],
                    regression["current_value"],
                    regression["change_percent"]
                ))
            
            conn.commit()
    
    def generate_trend_report(self, lookback_days: int = 90) -> Dict[str, Any]:
        """Generate comprehensive trend analysis report."""
        logger.info(f"📈 Generating trend report (last {lookback_days} days)")
        
        cutoff_time = time.time() - (lookback_days * 24 * 60 * 60)
        
        report = {
            "title": f"DataSON Performance Trends ({lookback_days} days)",
            "generated_at": datetime.now().isoformat(),
            "summary": {},
            "library_trends": {},
            "regression_alerts": [],
            "performance_insights": [],
            "recommendations": []
        }
        
        with sqlite3.connect(self.db_path) as conn:
            # Get summary statistics
            cursor = conn.cursor()
            
            # Total runs in period
            cursor.execute("SELECT COUNT(*) FROM benchmark_runs WHERE timestamp > ?", (cutoff_time,))
            total_runs = cursor.fetchone()[0]
            
            # Unique libraries tested
            cursor.execute("""
                SELECT COUNT(DISTINCT pm.library_name)
                FROM performance_metrics pm
                JOIN benchmark_runs br ON pm.run_id = br.id
                WHERE br.timestamp > ?
            """, (cutoff_time,))
            unique_libraries = cursor.fetchone()[0]
            
            # Recent regressions
            cursor.execute("""
                SELECT COUNT(*) FROM trend_alerts 
                WHERE detected_at > datetime('now', '-7 days')
            """, ())
            recent_alerts = cursor.fetchone()[0]
            
            report["summary"] = {
                "total_benchmark_runs": total_runs,
                "unique_libraries_tested": unique_libraries,
                "recent_alerts": recent_alerts,
                "analysis_period_days": lookback_days
            }
            
            # Get library-specific trends
            report["library_trends"] = self._get_library_trends(conn, cutoff_time)
            
            # Get recent regression alerts
            cursor.execute("""
                SELECT alert_type, library_name, metric_type, severity, description,
                       baseline_value, current_value, change_percent, detected_at
                FROM trend_alerts
                WHERE detected_at > datetime('now', '-30 days')
                ORDER BY detected_at DESC
                LIMIT 20
            """)
            
            alerts = cursor.fetchall()
            report["regression_alerts"] = [
                {
                    "type": alert[0],
                    "library": alert[1],
                    "metric": alert[2],
                    "severity": alert[3],
                    "description": alert[4],
                    "baseline": alert[5],
                    "current": alert[6],
                    "change_percent": alert[7],
                    "detected_at": alert[8]
                }
                for alert in alerts
            ]
        
        # Generate insights and recommendations
        report["performance_insights"] = self._generate_performance_insights(report)
        report["recommendations"] = self._generate_trend_recommendations(report)
        
        return report
    
    def _get_library_trends(self, conn: sqlite3.Connection, cutoff_time: float) -> Dict[str, Any]:
        """Get performance trends for each library."""
        cursor = conn.cursor()
        
        # Get average performance metrics by library
        query = """
            SELECT pm.library_name, pm.metric_type,
                   AVG(pm.value) as avg_value,
                   COUNT(*) as data_points,
                   MIN(br.timestamp) as first_run,
                   MAX(br.timestamp) as last_run
            FROM performance_metrics pm
            JOIN benchmark_runs br ON pm.run_id = br.id
            WHERE br.timestamp > ? AND pm.metric_type IN ('serialization', 'deserialization', 'success_rate')
            GROUP BY pm.library_name, pm.metric_type
            HAVING COUNT(*) >= 3
            ORDER BY pm.library_name, pm.metric_type
        """
        
        cursor.execute(query, (cutoff_time,))
        results = cursor.fetchall()
        
        # Group by library
        library_trends = {}
        for row in results:
            library, metric_type, avg_value, data_points, first_run, last_run = row
            
            if library not in library_trends:
                library_trends[library] = {
                    "metrics": {},
                    "data_points": 0,
                    "period_days": (last_run - first_run) / (24 * 60 * 60)
                }
            
            library_trends[library]["metrics"][metric_type] = {
                "average": avg_value,
                "data_points": data_points
            }
            library_trends[library]["data_points"] += data_points
        
        return library_trends
    
    def _generate_performance_insights(self, report: Dict[str, Any]) -> List[str]:
        """Generate performance insights from trend data."""
        insights = []
        
        # Analyze regression alerts
        alerts = report["regression_alerts"]
        if alerts:
            critical_alerts = [a for a in alerts if a["severity"] == "critical"]
            if critical_alerts:
                insights.append(f"🚨 {len(critical_alerts)} critical performance regressions detected")
            
            # Most affected library
            library_alert_counts = {}
            for alert in alerts:
                lib = alert["library"]
                library_alert_counts[lib] = library_alert_counts.get(lib, 0) + 1
            
            if library_alert_counts:
                most_affected = max(library_alert_counts.items(), key=lambda x: x[1])
                insights.append(f"📉 {most_affected[0]} has {most_affected[1]} performance alerts")
        
        # Analyze library trends
        trends = report["library_trends"]
        if trends:
            # Find most stable library
            stability_scores = {}
            for library, data in trends.items():
                if "serialization" in data["metrics"]:
                    # Simple stability score based on data points
                    stability_scores[library] = data["data_points"]
            
            if stability_scores:
                most_stable = max(stability_scores.items(), key=lambda x: x[1])
                insights.append(f"✅ {most_stable[0]} shows most consistent performance with {most_stable[1]} data points")
        
        # General insights
        summary = report["summary"]
        if summary["total_benchmark_runs"] > 10:
            insights.append(f"📊 Comprehensive analysis with {summary['total_benchmark_runs']} benchmark runs")
        
        if summary["recent_alerts"] == 0:
            insights.append("🎯 No performance regressions detected in the last 7 days")
        
        return insights
    
    def _generate_trend_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations from trend analysis."""
        recommendations = []
        
        # Recommendations based on alerts
        alerts = report["regression_alerts"]
        critical_alerts = [a for a in alerts if a["severity"] == "critical"]
        
        if critical_alerts:
            recommendations.append("🚨 Investigate critical performance regressions immediately")
            
            # Specific library recommendations
            critical_libraries = set(a["library"] for a in critical_alerts)
            for library in critical_libraries:
                recommendations.append(f"🔍 Review {library} implementation for performance issues")
        
        # Recommendations based on trends
        trends = report["library_trends"]
        datason_trends = {k: v for k, v in trends.items() if k.startswith("datason")}
        
        if datason_trends:
            recommendations.append("📈 Monitor DataSON variants for optimization opportunities")
        
        # General recommendations
        if report["summary"]["total_benchmark_runs"] < 5:
            recommendations.append("📊 Increase benchmark frequency for better trend analysis")
        
        recommendations.extend([
            "🔄 Set up automated regression alerts for CI/CD integration",
            "📋 Review performance baselines quarterly",
            "🎯 Focus optimization efforts on most-used library variants"
        ])
        
        return recommendations
    
    def get_historical_performance(self, library: str, metric_type: str = "serialization", 
                                 days: int = 30) -> Dict[str, Any]:
        """Get historical performance data for a specific library."""
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT pm.value, br.timestamp, pm.dataset_name, pm.unit
                FROM performance_metrics pm
                JOIN benchmark_runs br ON pm.run_id = br.id
                WHERE pm.library_name = ? AND pm.metric_type = ? AND br.timestamp > ?
                ORDER BY br.timestamp
            """
            
            cursor.execute(query, (library, metric_type, cutoff_time))
            results = cursor.fetchall()
        
        if not results:
            return {"error": f"No data found for {library} {metric_type}"}
        
        # Process results
        data_points = []
        for value, timestamp, dataset, unit in results:
            data_points.append({
                "value": value,
                "timestamp": timestamp,
                "date": datetime.fromtimestamp(timestamp).isoformat(),
                "dataset": dataset,
                "unit": unit
            })
        
        # Calculate statistics
        values = [dp["value"] for dp in data_points]
        
        return {
            "library": library,
            "metric_type": metric_type,
            "period_days": days,
            "data_points": data_points,
            "statistics": {
                "count": len(values),
                "mean": mean(values),
                "min": min(values),
                "max": max(values),
                "std": stdev(values) if len(values) > 1 else 0,
                "latest": values[-1] if values else None,
                "trend": "improving" if len(values) > 1 and values[-1] < values[0] else "stable"
            }
        }


def main():
    """CLI entry point for trend analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='DataSON Trend Analysis System')
    parser.add_argument('--ingest', help='Ingest benchmark result file')
    parser.add_argument('--detect-regressions', action='store_true',
                       help='Detect performance regressions')
    parser.add_argument('--trend-report', action='store_true',
                       help='Generate trend analysis report')
    parser.add_argument('--lookback-days', type=int, default=30,
                       help='Days to look back for analysis')
    parser.add_argument('--library', help='Get historical data for specific library')
    parser.add_argument('--metric', default='serialization',
                       help='Metric type for historical analysis')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    analyzer = TrendAnalyzer()
    
    try:
        if args.ingest:
            run_id = analyzer.ingest_benchmark_results(args.ingest)
            print(f"✅ Ingested benchmark results with run ID: {run_id}")
        
        elif args.detect_regressions:
            regressions = analyzer.detect_performance_regressions(args.lookback_days)
            if regressions:
                print(f"🚨 Detected {len(regressions)} performance regressions:")
                for reg in regressions[:5]:  # Show first 5
                    print(f"  • {reg['severity'].upper()}: {reg['description']}")
            else:
                print("✅ No performance regressions detected")
        
        elif args.trend_report:
            report = analyzer.generate_trend_report(args.lookback_days)
            print(f"📈 Trend Report ({args.lookback_days} days)")
            print(f"  • {report['summary']['total_benchmark_runs']} benchmark runs")
            print(f"  • {report['summary']['unique_libraries_tested']} libraries tested")
            print(f"  • {len(report['regression_alerts'])} recent alerts")
        
        elif args.library:
            data = analyzer.get_historical_performance(args.library, args.metric, args.lookback_days)
            if "error" in data:
                print(f"❌ {data['error']}")
            else:
                stats = data["statistics"]
                print(f"📊 {args.library} {args.metric} performance:")
                print(f"  • Mean: {stats['mean']:.3f}")
                print(f"  • Latest: {stats['latest']:.3f}")
                print(f"  • Trend: {stats['trend']}")
        
        else:
            parser.print_help()
        
        return 0
        
    except Exception as e:
        logger.error(f"Trend analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 
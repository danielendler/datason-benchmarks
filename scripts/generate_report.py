#!/usr/bin/env python3
"""
Report Generator for DataSON Benchmarks
========================================

Generates HTML and markdown reports with interactive charts from benchmark results.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates reports with interactive charts from benchmark results."""
    
    def __init__(self, output_dir: str = "docs/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.is_ci = self._is_ci_environment()
    
    def _is_ci_environment(self) -> bool:
        """Check if running in CI environment."""
        ci_indicators = [
            'GITHUB_ACTIONS',
            'CI',
            'CONTINUOUS_INTEGRATION',
            'GITHUB_RUN_ID'
        ]
        return any(os.environ.get(indicator) for indicator in ci_indicators)
    
    def _get_ci_filename_prefix(self) -> str:
        """Get CI-specific filename prefix to differentiate from local runs."""
        if self.is_ci:
            run_id = os.environ.get('GITHUB_RUN_ID', 'unknown')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return f"ci_{timestamp}_{run_id}"
        else:
            # Local runs get temp prefix that will be gitignored
            return f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate an interactive HTML report with charts from benchmark results."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            suite_type = results.get("suite_type", "unknown")
            
            # Generate filename with CI prefix
            filename_prefix = self._get_ci_filename_prefix()
            report_file = self.output_dir / f"{filename_prefix}_{suite_type}_report.html"
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>DataSON Benchmark Report - {suite_type}</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{ 
                        font-family: 'Segoe UI', Arial, sans-serif; 
                        margin: 0; 
                        padding: 20px; 
                        background-color: #f8f9fa;
                    }}
                    .container {{ 
                        max-width: 1400px; 
                        margin: 0 auto; 
                        background: white; 
                        padding: 30px; 
                        border-radius: 8px; 
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    .header {{ 
                        text-align: center; 
                        margin-bottom: 30px; 
                        padding-bottom: 20px; 
                        border-bottom: 2px solid #e9ecef;
                    }}
                    .summary {{ 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; 
                        padding: 25px; 
                        border-radius: 8px; 
                        margin: 20px 0; 
                    }}
                    .chart-container {{ 
                        margin: 30px 0; 
                        padding: 20px; 
                        border: 1px solid #dee2e6; 
                        border-radius: 8px; 
                        background: #ffffff;
                    }}
                    table {{ 
                        border-collapse: collapse; 
                        width: 100%; 
                        margin: 20px 0; 
                        background: white;
                    }}
                    th, td {{ 
                        border: 1px solid #dee2e6; 
                        padding: 12px; 
                        text-align: left; 
                    }}
                    th {{ 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; 
                        font-weight: 600;
                    }}
                    tr:nth-child(even) {{ 
                        background-color: #f8f9fa; 
                    }}
                    .metric {{ 
                        display: inline-block; 
                        margin: 10px 15px; 
                        padding: 15px; 
                        background: #e9ecef; 
                        border-radius: 5px; 
                        text-align: center;
                    }}
                    .metric-value {{ 
                        font-size: 24px; 
                        font-weight: bold; 
                        color: #495057; 
                    }}
                    .metric-label {{ 
                        font-size: 12px; 
                        color: #6c757d; 
                        text-transform: uppercase; 
                    }}
                    .section {{ 
                        margin: 40px 0; 
                        padding: 20px; 
                        border-left: 4px solid #667eea;
                    }}
                    .warning {{ 
                        background-color: #fff3cd; 
                        border: 1px solid #ffeeba; 
                        color: #856404; 
                        padding: 15px; 
                        border-radius: 5px; 
                        margin: 20px 0;
                    }}
                    .ci-badge {{ 
                        background: #28a745; 
                        color: white; 
                        padding: 5px 15px; 
                        border-radius: 20px; 
                        font-size: 12px; 
                        text-transform: uppercase; 
                        font-weight: bold;
                    }}
                    .local-badge {{ 
                        background: #ffc107; 
                        color: #212529; 
                        padding: 5px 15px; 
                        border-radius: 20px; 
                        font-size: 12px; 
                        text-transform: uppercase; 
                        font-weight: bold;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>📊 DataSON Benchmark Report</h1>
                        <span class="{'ci-badge' if self.is_ci else 'local-badge'}">
                            {'CI Environment' if self.is_ci else 'Local Environment'}
                        </span>
                    </div>
                    
                    <div class="summary">
                        <h2>🎯 Execution Summary</h2>
                        <div class="metric">
                            <div class="metric-value">{suite_type.title()}</div>
                            <div class="metric-label">Suite Type</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{timestamp}</div>
                            <div class="metric-label">Generated</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{'CI' if self.is_ci else 'Local'}</div>
                            <div class="metric-label">Environment</div>
                        </div>
                    </div>
            """
            
            if not self.is_ci:
                html += """
                <div class="warning">
                    ⚠️ <strong>Warning:</strong> This report was generated in a local environment. 
                    Results may vary from CI environment and won't be committed to the repository. 
                    Only CI-generated results are stored for consistency.
                </div>
                """
            
            # Add competitive results with charts
            if "competitive" in results:
                html += self._generate_competitive_html_with_charts(results["competitive"])
            
            # Add configuration results with charts
            if "configurations" in results:
                html += self._generate_config_html_with_charts(results["configurations"])
                
            # Add version comparison if available
            if "versioning" in results:
                html += self._generate_versioning_html_with_charts(results["versioning"])
            
            html += """
                    <div class="section">
                        <h2>📋 Raw Data</h2>
                        <details>
                            <summary>Click to view raw JSON data</summary>
                            <pre style="background: #f8f9fa; padding: 20px; border-radius: 5px; overflow-x: auto;">
            """ + json.dumps(results, indent=2) + """
                            </pre>
                        </details>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Save HTML file
            with open(report_file, 'w') as f:
                f.write(html)
            
            logger.info(f"{'CI' if self.is_ci else 'Local'} HTML report generated: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return ""
    
    def _generate_competitive_html_with_charts(self, competitive_data: Dict[str, Any]) -> str:
        """Generate HTML with interactive charts for competitive analysis."""
        html = '<div class="section"><h2>🏆 Competitive Analysis</h2>'
        
        summary = competitive_data.get("summary", {})
        competitors = summary.get("competitors_tested", [])
        
        if competitors:
            html += f"<p><strong>Tested Competitors:</strong> {', '.join(competitors)}</p>"
        
        # Performance comparison chart
        chart_data = self._extract_competitive_chart_data(competitive_data)
        if chart_data:
            html += self._generate_performance_comparison_chart(chart_data)
        
        # Detailed tables for each dataset
        for dataset_name, dataset_data in competitive_data.items():
            if dataset_name == "summary":
                continue
                
            html += f"<h3>📋 {dataset_name} - Detailed Results</h3>"
            
            if "description" in dataset_data:
                html += f"<p><em>{dataset_data['description']}</em></p>"
            
            # Serialization results table
            if "serialization" in dataset_data:
                html += "<h4>Serialization Performance</h4>"
                html += "<table><tr><th>Library</th><th>Mean (ms)</th><th>Min (ms)</th><th>Max (ms)</th><th>Std Dev (ms)</th><th>Success Rate</th></tr>"
                
                for lib, metrics in dataset_data["serialization"].items():
                    if isinstance(metrics, dict) and "mean_ms" in metrics:
                        mean_ms = metrics["mean_ms"]
                        min_ms = metrics.get("min_ms", 0)
                        max_ms = metrics.get("max_ms", 0)
                        std_ms = metrics.get("std_ms", 0)
                        success = metrics.get("successful_runs", 0)
                        errors = metrics.get("error_count", 0)
                        total = success + errors
                        rate = f"{success}/{total}" if total > 0 else "N/A"
                        
                        # Color code performance relative to fastest
                        style = ""
                        if lib == "datason":
                            style = "style='background-color: #e3f2fd; font-weight: bold;'"
                        
                        html += f"<tr {style}><td>{lib}</td><td>{mean_ms:.3f}</td><td>{min_ms:.3f}</td><td>{max_ms:.3f}</td><td>{std_ms:.3f}</td><td>{rate}</td></tr>"
                
                html += "</table>"
        
        html += "</div>"
        return html
    
    def _extract_competitive_chart_data(self, competitive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data for competitive performance charts."""
        datasets = []
        libraries = set()
        
        for dataset_name, dataset_data in competitive_data.items():
            if dataset_name == "summary":
                continue
                
            if "serialization" in dataset_data:
                dataset_info = {
                    "name": dataset_name,
                    "results": {}
                }
                
                for lib, metrics in dataset_data["serialization"].items():
                    if isinstance(metrics, dict) and "mean_ms" in metrics:
                        dataset_info["results"][lib] = metrics["mean_ms"]
                        libraries.add(lib)
                
                if dataset_info["results"]:
                    datasets.append(dataset_info)
        
        return {
            "datasets": datasets,
            "libraries": sorted(list(libraries))
        }
    
    def _generate_performance_comparison_chart(self, chart_data: Dict[str, Any]) -> str:
        """Generate interactive performance comparison chart."""
        datasets = chart_data["datasets"]
        libraries = chart_data["libraries"]
        
        if not datasets or not libraries:
            return ""
        
        # Prepare data for grouped bar chart
        chart_id = f"perf_chart_{abs(hash(str(datasets)))}"
        
        traces = []
        for lib in libraries:
            x_values = []
            y_values = []
            
            for dataset in datasets:
                if lib in dataset["results"]:
                    x_values.append(dataset["name"])
                    y_values.append(dataset["results"][lib])
            
            # Special styling for DataSON
            color = "#667eea" if lib == "datason" else None
            
            trace = {
                "x": x_values,
                "y": y_values,
                "name": lib,
                "type": "bar",
                "marker": {"color": color} if color else {}
            }
            traces.append(trace)
        
        chart_html = f"""
        <div class="chart-container">
            <h3>📈 Performance Comparison Chart</h3>
            <div id="{chart_id}" style="width:100%;height:500px;"></div>
            <script>
                var traces = {json.dumps(traces)};
                
                var layout = {{
                    title: 'Serialization Performance by Dataset (Lower is Better)',
                    xaxis: {{
                        title: 'Dataset',
                        tickangle: -45
                    }},
                    yaxis: {{
                        title: 'Time (milliseconds)',
                        type: 'log'
                    }},
                    barmode: 'group',
                    showlegend: true,
                    legend: {{
                        x: 1.02,
                        y: 1
                    }},
                    margin: {{
                        l: 60,
                        r: 120,
                        t: 80,
                        b: 120
                    }}
                }};
                
                var config = {{
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToAdd: ['pan2d', 'select2d', 'lasso2d']
                }};
                
                Plotly.newPlot('{chart_id}', traces, layout, config);
            </script>
        </div>
        """
        
        return chart_html
    
    def _generate_config_html_with_charts(self, config_data: Dict[str, Any]) -> str:
        """Generate HTML with charts for configuration results."""
        html = '<div class="section"><h2>⚙️ Configuration Testing</h2>'
        
        # Configuration comparison chart
        chart_data = self._extract_config_chart_data(config_data)
        if chart_data:
            html += self._generate_config_comparison_chart(chart_data)
        
        # Detailed results
        summary = config_data.get("summary", {})
        best_use_case = summary.get("best_for_use_case", {})
        
        if best_use_case:
            html += "<h3>🎯 Recommended Configurations</h3>"
            html += "<table><tr><th>Use Case</th><th>Best Configuration</th></tr>"
            for use_case, config in best_use_case.items():
                html += f"<tr><td>{use_case.replace('_', ' ').title()}</td><td>{config}</td></tr>"
            html += "</table>"
        
        html += "</div>"
        return html
    
    def _extract_config_chart_data(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data for configuration performance charts."""
        configs = []
        
        for config_name, config_info in config_data.items():
            if config_name == "summary":
                continue
                
            if "datasets" in config_info:
                config_perf = {
                    "name": config_name,
                    "total_time": 0,
                    "datasets": {}
                }
                
                for dataset_name, dataset_data in config_info["datasets"].items():
                    results = dataset_data.get("results", {})
                    if "total_time_ms" in results:
                        config_perf["datasets"][dataset_name] = results["total_time_ms"]
                        config_perf["total_time"] += results["total_time_ms"]
                
                if config_perf["datasets"]:
                    configs.append(config_perf)
        
        return {"configs": configs}
    
    def _generate_config_comparison_chart(self, chart_data: Dict[str, Any]) -> str:
        """Generate configuration comparison chart."""
        configs = chart_data["configs"]
        
        if not configs:
            return ""
        
        chart_id = f"config_chart_{abs(hash(str(configs)))}"
        
        # Total time comparison
        config_names = [config["name"] for config in configs]
        total_times = [config["total_time"] for config in configs]
        
        chart_html = f"""
        <div class="chart-container">
            <h3>⚙️ Configuration Performance Comparison</h3>
            <div id="{chart_id}" style="width:100%;height:400px;"></div>
            <script>
                var trace = {{
                    x: {json.dumps(config_names)},
                    y: {json.dumps(total_times)},
                    type: 'bar',
                    marker: {{
                        color: '#764ba2'
                    }}
                }};
                
                var layout = {{
                    title: 'Total Processing Time by Configuration',
                    xaxis: {{
                        title: 'Configuration',
                        tickangle: -45
                    }},
                    yaxis: {{
                        title: 'Total Time (milliseconds)'
                    }},
                    margin: {{
                        l: 60,
                        r: 40,
                        t: 80,
                        b: 120
                    }}
                }};
                
                var config = {{
                    responsive: true,
                    displayModeBar: true
                }};
                
                Plotly.newPlot('{chart_id}', [trace], layout, config);
            </script>
        </div>
        """
        
        return chart_html
    
    def _generate_versioning_html_with_charts(self, versioning_data: Dict[str, Any]) -> str:
        """Generate HTML with charts for version comparison results."""
        html = '<div class="section"><h2>📈 DataSON Version Evolution</h2>'
        
        summary = versioning_data.get("summary", {})
        perf_evolution = summary.get("performance_evolution", {})
        
        if perf_evolution:
            html += self._generate_version_evolution_chart(perf_evolution)
        
        # Version compatibility table
        version_results = versioning_data.get("version_results", {})
        if version_results:
            html += "<h3>🔧 Version Compatibility</h3>"
            html += "<table><tr><th>Version</th><th>Available Features</th><th>Config Methods</th><th>Status</th></tr>"
            
            for version, data in version_results.items():
                if isinstance(data, dict) and "available_features" in data:
                    features = ", ".join(data["available_features"])
                    configs = ", ".join(data["available_configs"])
                    status = "✅ Available" if data.get("available", True) else "❌ Failed"
                    html += f"<tr><td>{version}</td><td>{features}</td><td>{configs}</td><td>{status}</td></tr>"
                else:
                    error = data.get("error", "Unknown error")
                    html += f"<tr><td>{version}</td><td colspan='2'>❌ {error}</td><td>❌ Failed</td></tr>"
            
            html += "</table>"
        
        html += "</div>"
        return html
    
    def _generate_version_evolution_chart(self, perf_evolution: Dict[str, Any]) -> str:
        """Generate version evolution performance chart."""
        chart_id = f"version_chart_{abs(hash(str(perf_evolution)))}"
        
        # Prepare data for line chart
        versions = set()
        for dataset_data in perf_evolution.values():
            versions.update(dataset_data.keys())
        
        versions = sorted(list(versions))
        
        traces = []
        for dataset, version_data in perf_evolution.items():
            x_values = []
            y_values = []
            
            for version in versions:
                if version in version_data:
                    x_values.append(version)
                    y_values.append(version_data[version])
            
            trace = {
                "x": x_values,
                "y": y_values,
                "name": dataset,
                "type": "scatter",
                "mode": "lines+markers",
                "line": {"width": 3},
                "marker": {"size": 8}
            }
            traces.append(trace)
        
        chart_html = f"""
        <div class="chart-container">
            <h3>📊 Performance Evolution Across Versions</h3>
            <div id="{chart_id}" style="width:100%;height:500px;"></div>
            <script>
                var traces = {json.dumps(traces)};
                
                var layout = {{
                    title: 'DataSON Performance Evolution (Lower is Better)',
                    xaxis: {{
                        title: 'Version',
                        type: 'category'
                    }},
                    yaxis: {{
                        title: 'Time (milliseconds)',
                        type: 'log'
                    }},
                    showlegend: true,
                    legend: {{
                        x: 1.02,
                        y: 1
                    }},
                    margin: {{
                        l: 60,
                        r: 120,
                        t: 80,
                        b: 60
                    }}
                }};
                
                var config = {{
                    responsive: true,
                    displayModeBar: true
                }};
                
                Plotly.newPlot('{chart_id}', traces, layout, config);
            </script>
        </div>
        """
        
        return chart_html 
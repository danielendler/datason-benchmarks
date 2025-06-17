#!/usr/bin/env python3
"""
Report Generator for DataSON Benchmarks
========================================

Generates HTML and markdown reports from benchmark results.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates reports from benchmark results."""
    
    def __init__(self, output_dir: str = "docs/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate an HTML report from benchmark results."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            suite_type = results.get("suite_type", "unknown")
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>DataSON Benchmark Report - {suite_type}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .summary {{ background-color: #f9f9f9; padding: 20px; margin: 20px 0; }}
                    .competitor {{ margin: 20px 0; }}
                    .dataset {{ margin: 15px 0; }}
                </style>
            </head>
            <body>
                <h1>DataSON Benchmark Report</h1>
                <div class="summary">
                    <h2>Summary</h2>
                    <p><strong>Suite Type:</strong> {suite_type}</p>
                    <p><strong>Generated:</strong> {timestamp}</p>
                    <p><strong>Metadata:</strong> {json.dumps(results.get('metadata', {}), indent=2)}</p>
                </div>
            """
            
            # Add competitive results if available
            if "competitive" in results:
                html += self._generate_competitive_html(results["competitive"])
            
            # Add configuration results if available  
            if "configurations" in results:
                html += self._generate_config_html(results["configurations"])
            
            html += """
            </body>
            </html>
            """
            
            # Save HTML file
            report_file = self.output_dir / f"{suite_type}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(report_file, 'w') as f:
                f.write(html)
            
            logger.info(f"HTML report generated: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return ""
    
    def _generate_competitive_html(self, competitive_data: Dict[str, Any]) -> str:
        """Generate HTML for competitive analysis results."""
        html = "<h2>Competitive Analysis</h2>"
        
        summary = competitive_data.get("summary", {})
        competitors = summary.get("competitors_tested", [])
        
        if competitors:
            html += f"<p><strong>Tested Competitors:</strong> {', '.join(competitors)}</p>"
        
        # Performance tables for each dataset
        for dataset_name, dataset_data in competitive_data.items():
            if dataset_name == "summary":
                continue
                
            html += f"<div class='dataset'><h3>{dataset_name}</h3>"
            
            if "description" in dataset_data:
                html += f"<p><em>{dataset_data['description']}</em></p>"
            
            # Serialization results
            if "serialization" in dataset_data:
                html += "<h4>Serialization Performance</h4>"
                html += "<table><tr><th>Library</th><th>Mean (ms)</th><th>Min (ms)</th><th>Max (ms)</th><th>Success Rate</th></tr>"
                
                for lib, metrics in dataset_data["serialization"].items():
                    if isinstance(metrics, dict) and "mean_ms" in metrics:
                        mean_ms = metrics["mean_ms"]
                        min_ms = metrics.get("min", 0) * 1000
                        max_ms = metrics.get("max", 0) * 1000
                        success = metrics.get("successful_runs", 0)
                        errors = metrics.get("error_count", 0)
                        total = success + errors
                        rate = f"{success}/{total}" if total > 0 else "N/A"
                        
                        html += f"<tr><td>{lib}</td><td>{mean_ms:.2f}</td><td>{min_ms:.2f}</td><td>{max_ms:.2f}</td><td>{rate}</td></tr>"
                
                html += "</table>"
            
            html += "</div>"
        
        return html
    
    def _generate_config_html(self, config_data: Dict[str, Any]) -> str:
        """Generate HTML for configuration results."""
        html = "<h2>Configuration Testing</h2>"
        
        summary = config_data.get("summary", {})
        
        # Best configurations
        best_use_case = summary.get("best_for_use_case", {})
        if best_use_case:
            html += "<h3>Recommended Configurations</h3><ul>"
            for use_case, config in best_use_case.items():
                html += f"<li><strong>{use_case.replace('_', ' ').title()}:</strong> {config}</li>"
            html += "</ul>"
        
        # Configuration comparison
        for config_name, config_info in config_data.items():
            if config_name == "summary":
                continue
                
            html += f"<div class='competitor'><h3>{config_name}</h3>"
            
            if "description" in config_info:
                html += f"<p><em>{config_info['description']}</em></p>"
            
            if "datasets" in config_info:
                html += "<table><tr><th>Dataset</th><th>Total Time (ms)</th><th>Serialization (ms)</th><th>Deserialization (ms)</th></tr>"
                
                for dataset_name, dataset_data in config_info["datasets"].items():
                    results = dataset_data.get("results", {})
                    if "total_time_ms" in results:
                        total_ms = results["total_time_ms"]
                        ser_ms = results.get("serialization", {}).get("mean_ms", 0)
                        deser_ms = results.get("deserialization", {}).get("mean_ms", 0)
                        
                        html += f"<tr><td>{dataset_name}</td><td>{total_ms:.2f}</td><td>{ser_ms:.2f}</td><td>{deser_ms:.2f}</td></tr>"
                
                html += "</table>"
            
            html += "</div>"
        
        return html 
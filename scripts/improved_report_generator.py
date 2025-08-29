#!/usr/bin/env python3
"""
Improved DataSON Benchmark Report Generator
==========================================

Generates clear, useful HTML reports focusing on:
1. DataSON API Performance Matrix
2. Competitive Analysis Tables  
3. Version Evolution Charts
4. Real-world Scenario Results

Removes confusing "page 1-4" terminology.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import statistics

class ImprovedReportGenerator:
    """Generate improved HTML benchmark reports"""
    
    def __init__(self):
        self.css_styles = """
        <style>
        * { box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            line-height: 1.6; margin: 0; padding: 20px; background: #f8f9fa; color: #333;
        }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 40px; padding-bottom: 20px; border-bottom: 2px solid #e9ecef; }
        .header h1 { color: #2c3e50; font-size: 2.5em; margin: 0; }
        .header .subtitle { color: #6c757d; font-size: 1.1em; margin: 10px 0; }
        .metadata { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0; font-size: 0.9em; }
        .metadata strong { color: #495057; }
        
        .section { margin: 40px 0; }
        .section-header { display: flex; align-items: center; margin-bottom: 20px; }
        .section-header h2 { margin: 0; color: #2c3e50; }
        .section-header .badge { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 6px 12px; border-radius: 20px; font-size: 0.8em; margin-left: 12px; }
        
        .performance-table { width: 100%; border-collapse: collapse; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .performance-table th, .performance-table td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #dee2e6; }
        .performance-table th { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-weight: 600; }
        .performance-table tbody tr:hover { background-color: #f8f9fa; }
        .performance-table .numeric { text-align: right; font-family: Monaco, 'Courier New', monospace; }
        
        .comparison-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin: 20px 0; }
        .comparison-card { background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .comparison-card h3 { margin: 0 0 15px 0; color: #495057; font-size: 1.2em; }
        .comparison-card .scenario-description { color: #6c757d; font-size: 0.9em; margin-bottom: 15px; font-style: italic; }
        
        .metric-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #f1f3f4; }
        .metric-row:last-child { border-bottom: none; }
        .metric-label { font-weight: 500; color: #495057; }
        .metric-value { font-family: Monaco, 'Courier New', monospace; color: #28a745; }
        .metric-value.slow { color: #dc3545; }
        .metric-value.medium { color: #ffc107; }
        
        .error-message { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 4px; border-left: 4px solid #dc3545; }
        .success-message { background: #d4edda; color: #155724; padding: 10px; border-radius: 4px; border-left: 4px solid #28a745; }
        
        .performance-bars { margin: 15px 0; }
        .performance-bar { background: #e9ecef; height: 20px; border-radius: 10px; margin: 8px 0; position: relative; overflow: hidden; }
        .performance-bar-fill { height: 100%; border-radius: 10px; transition: width 0.3s ease; }
        .performance-bar-label { position: absolute; left: 8px; top: 2px; font-size: 0.8em; font-weight: 500; color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); }
        
        .datason-fill { background: linear-gradient(90deg, #28a745, #20c997); }
        .competitor-fill { background: linear-gradient(90deg, #007bff, #6610f2); }
        .fast-fill { background: linear-gradient(90deg, #28a745, #20c997); }
        .medium-fill { background: linear-gradient(90deg, #ffc107, #fd7e14); }
        .slow-fill { background: linear-gradient(90deg, #dc3545, #e83e8c); }
        
        .summary-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .stat-card { background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 20px; border-radius: 8px; text-align: center; }
        .stat-number { font-size: 2.5em; font-weight: bold; color: #495057; display: block; }
        .stat-label { color: #6c757d; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }
        
        .footer { margin-top: 60px; padding-top: 20px; border-top: 1px solid #dee2e6; text-align: center; color: #6c757d; font-size: 0.9em; }
        
        @media (max-width: 768px) {
            .container { padding: 15px; }
            .header h1 { font-size: 2em; }
            .comparison-grid { grid-template-columns: 1fr; }
            .performance-table { font-size: 0.9em; }
            .performance-table th, .performance-table td { padding: 8px 10px; }
        }
        </style>
        """
    
    def format_time(self, time_seconds: float) -> str:
        """Format time with appropriate units and color coding"""
        if time_seconds == 0:
            return "0ms"
        
        # Convert to appropriate units
        if time_seconds >= 1.0:
            return f"{time_seconds:.2f}s"
        elif time_seconds >= 0.001:
            return f"{time_seconds * 1000:.1f}ms"
        else:
            return f"{time_seconds * 1000000:.1f}μs"
    
    def get_performance_class(self, time_seconds: float) -> str:
        """Get CSS class for performance coloring"""
        if time_seconds == 0:
            return "error"
        elif time_seconds < 0.001:  # < 1ms
            return "fast"
        elif time_seconds < 0.01:   # < 10ms
            return "medium"
        else:
            return "slow"
    
    def generate_datason_api_matrix(self, data: Dict[str, Any]) -> str:
        """Generate DataSON API performance comparison matrix"""
        if 'datason_api_comparison' not in data:
            return "<p class='error-message'>No DataSON API comparison data available</p>"
        
        api_data = data['datason_api_comparison']['results']
        scenarios = list(api_data.keys())
        
        if not scenarios:
            return "<p class='error-message'>No scenarios found in DataSON API data</p>"
        
        # Get all API methods tested
        methods = set()
        for scenario in scenarios:
            methods.update(api_data[scenario].keys())
        methods = sorted(methods)
        
        html = f"""
        <div class="section">
            <div class="section-header">
                <h2>📊 DataSON API Performance Matrix</h2>
                <span class="badge">Method Comparison</span>
            </div>
            <p>Performance comparison across different DataSON API methods for real-world scenarios.</p>
            
            <table class="performance-table">
                <thead>
                    <tr>
                        <th>Scenario</th>
                        {''.join(f'<th>{method}</th>' for method in methods)}
                    </tr>
                </thead>
                <tbody>
        """
        
        for scenario in scenarios:
            scenario_data = api_data[scenario]
            scenario_display = scenario.replace('_', ' ').title()
            
            html += f"<tr><td><strong>{scenario_display}</strong></td>"
            
            for method in methods:
                if method in scenario_data and not scenario_data[method].get('error'):
                    mean_time = scenario_data[method]['mean']
                    formatted_time = self.format_time(mean_time)
                    perf_class = self.get_performance_class(mean_time)
                    html += f'<td class="numeric metric-value {perf_class}">{formatted_time}</td>'
                else:
                    error_msg = scenario_data.get(method, {}).get('error', 'N/A')
                    if error_msg and error_msg != 'N/A':
                        error_msg = str(error_msg)[:30]
                    else:
                        error_msg = 'N/A'
                    html += f'<td class="error-message" title="{error_msg}">Error</td>'
            
            html += "</tr>"
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html
    
    def generate_competitive_analysis(self, data: Dict[str, Any]) -> str:
        """Generate competitive analysis section"""
        if 'competitive_analysis' not in data:
            return "<p class='error-message'>No competitive analysis data available</p>"
        
        comp_data = data['competitive_analysis']['results']
        scenarios = list(comp_data.keys())
        
        if not scenarios:
            return "<p class='error-message'>No competitive scenarios found</p>"
        
        html = f"""
        <div class="section">
            <div class="section-header">
                <h2>🏁 Competitive Analysis</h2>
                <span class="badge">vs Other Libraries</span>
            </div>
            <p>DataSON performance compared to other popular serialization libraries.</p>
            
            <div class="comparison-grid">
        """
        
        for scenario in scenarios:
            scenario_data = comp_data[scenario]
            scenario_display = scenario.replace('_', ' ').title()
            scenario_description = scenario_data.get('description', '')
            
            html += f"""
            <div class="comparison-card">
                <h3>{scenario_display}</h3>
                <div class="scenario-description">{scenario_description}</div>
            """
            
            # Serialization performance
            if 'serialization' in scenario_data:
                html += "<h4>Serialization Performance:</h4><div class='performance-bars'>"
                
                serialization_data = scenario_data['serialization']
                if serialization_data:
                    # Calculate relative performance for bar chart
                    times = []
                    for lib, result in serialization_data.items():
                        if 'error' not in result and 'mean' in result:
                            times.append(result['mean'])
                    
                    if times:
                        max_time = max(times)
                        
                        for lib, result in serialization_data.items():
                            if 'error' in result:
                                html += f"""
                                <div class="error-message" style="margin: 4px 0; padding: 5px;">
                                    {lib}: {result['error'][:50]}...
                                </div>"""
                            elif 'mean' in result:
                                mean_time = result['mean']
                                width = (mean_time / max_time * 100) if max_time > 0 else 0
                                formatted_time = self.format_time(mean_time)
                                fill_class = 'datason-fill' if lib == 'datason' else 'competitor-fill'
                                
                                html += f"""
                                <div class="performance-bar">
                                    <div class="performance-bar-fill {fill_class}" style="width: {width}%">
                                        <span class="performance-bar-label">{lib}: {formatted_time}</span>
                                    </div>
                                </div>"""
                
                html += "</div>"
            
            # Deserialization performance
            if 'deserialization' in scenario_data:
                html += "<h4>Deserialization Performance:</h4><div class='performance-bars'>"
                
                deserialization_data = scenario_data['deserialization']
                if deserialization_data:
                    # Calculate relative performance for bar chart
                    times = []
                    for lib, result in deserialization_data.items():
                        if 'error' not in result and 'mean' in result:
                            times.append(result['mean'])
                    
                    if times:
                        max_time = max(times)
                        
                        for lib, result in deserialization_data.items():
                            if 'error' in result:
                                html += f"""
                                <div class="error-message" style="margin: 4px 0; padding: 5px;">
                                    {lib}: {result['error'][:50]}...
                                </div>"""
                            elif 'mean' in result:
                                mean_time = result['mean']
                                width = (mean_time / max_time * 100) if max_time > 0 else 0
                                formatted_time = self.format_time(mean_time)
                                fill_class = 'datason-fill' if lib == 'datason' else 'competitor-fill'
                                
                                html += f"""
                                <div class="performance-bar">
                                    <div class="performance-bar-fill {fill_class}" style="width: {width}%">
                                        <span class="performance-bar-label">{lib}: {formatted_time}</span>
                                    </div>
                                </div>"""
                
                html += "</div>"
            
            # Output size comparison
            if 'output_size' in scenario_data:
                html += "<h4>Output Size Comparison:</h4><div class='metric-row'>"
                
                for lib, size_data in scenario_data['output_size'].items():
                    if 'error' not in size_data and 'size' in size_data:
                        size = size_data['size']
                        size_type = size_data.get('size_type', 'bytes')
                        html += f"""
                        <div class="metric-row">
                            <span class="metric-label">{lib}:</span>
                            <span class="metric-value">{size:,} {size_type}</span>
                        </div>"""
                
                html += "</div>"
            
            html += "</div>"
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def generate_version_evolution(self, data: Dict[str, Any]) -> str:
        """Generate version evolution tracking section"""
        if 'version_evolution' not in data:
            return "<p class='error-message'>No version evolution data available</p>"
        
        version_data = data['version_evolution']
        current_version = version_data.get('current_version', 'unknown')
        baseline_performance = version_data.get('baseline_performance', {})
        
        html = f"""
        <div class="section">
            <div class="section-header">
                <h2>📈 Version Evolution Tracking</h2>
                <span class="badge">v{current_version}</span>
            </div>
            <p>Performance baseline for DataSON v{current_version} - future versions will be compared against these metrics.</p>
            
            <div class="summary-stats">
        """
        
        # Calculate summary statistics
        total_scenarios = len(baseline_performance)
        successful_serializations = 0
        total_serialize_time = 0
        total_deserialize_time = 0
        
        for scenario, perf_data in baseline_performance.items():
            if 'serialize' in perf_data and 'error' not in perf_data['serialize']:
                successful_serializations += 1
                total_serialize_time += perf_data['serialize']['mean']
            if 'deserialize' in perf_data and 'error' not in perf_data['deserialize']:
                total_deserialize_time += perf_data['deserialize']['mean']
        
        avg_serialize_time = total_serialize_time / successful_serializations if successful_serializations > 0 else 0
        avg_deserialize_time = total_deserialize_time / successful_serializations if successful_serializations > 0 else 0
        
        html += f"""
            <div class="stat-card">
                <span class="stat-number">{total_scenarios}</span>
                <span class="stat-label">Scenarios Tested</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">{self.format_time(avg_serialize_time)}</span>
                <span class="stat-label">Avg Serialize Time</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">{self.format_time(avg_deserialize_time)}</span>
                <span class="stat-label">Avg Deserialize Time</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">{successful_serializations}/{total_scenarios}</span>
                <span class="stat-label">Success Rate</span>
            </div>
        </div>
        
        <table class="performance-table">
            <thead>
                <tr>
                    <th>Scenario</th>
                    <th>Serialize Time</th>
                    <th>Deserialize Time</th>
                    <th>Output Size</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for scenario, perf_data in baseline_performance.items():
            scenario_display = scenario.replace('_', ' ').title()
            
            # Serialize data
            if 'serialize' in perf_data and 'error' not in perf_data['serialize']:
                serialize_time = self.format_time(perf_data['serialize']['mean'])
                serialize_class = self.get_performance_class(perf_data['serialize']['mean'])
            else:
                serialize_time = "Error"
                serialize_class = "error"
            
            # Deserialize data
            if 'deserialize' in perf_data and 'error' not in perf_data['deserialize']:
                deserialize_time = self.format_time(perf_data['deserialize']['mean'])
                deserialize_class = self.get_performance_class(perf_data['deserialize']['mean'])
            else:
                deserialize_time = "Error"
                deserialize_class = "error"
            
            # Output size
            output_size = perf_data.get('output_size', 'N/A')
            if isinstance(output_size, (int, float)):
                output_size = f"{output_size:,} chars"
            
            html += f"""
            <tr>
                <td><strong>{scenario_display}</strong></td>
                <td class="numeric metric-value {serialize_class}">{serialize_time}</td>
                <td class="numeric metric-value {deserialize_class}">{deserialize_time}</td>
                <td class="numeric">{output_size}</td>
            </tr>
            """
        
        html += """
            </tbody>
        </table>
        </div>
        """
        
        return html
    
    def generate_report(self, data: Dict[str, Any], output_file: Path) -> None:
        """Generate complete HTML report"""
        
        # Extract metadata
        metadata = data.get('metadata', {})
        suite_type = data.get('suite_type', 'comprehensive')
        timestamp = data.get('timestamp', datetime.now().isoformat())
        scenarios = data.get('scenarios', [])
        
        # Parse timestamp for display
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_date = dt.strftime('%B %d, %Y at %I:%M %p UTC')
        except:
            formatted_date = timestamp
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataSON Benchmark Report - {suite_type.title()}</title>
    {self.css_styles}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 DataSON Benchmark Report</h1>
            <div class="subtitle">Comprehensive Performance Analysis - {suite_type.title()} Suite</div>
            <div class="subtitle">Generated on {formatted_date}</div>
        </div>
        
        <div class="metadata">
            <strong>Environment:</strong> Python {metadata.get('python_version', 'unknown')} | 
            <strong>DataSON:</strong> v{metadata.get('datason_version', 'unknown')} | 
            <strong>Framework:</strong> {metadata.get('benchmark_framework', 'standard')} |
            <strong>Scenarios:</strong> {len(scenarios)} tested
        </div>
"""
        
        # Add sections based on available data
        html += self.generate_datason_api_matrix(data)
        html += self.generate_competitive_analysis(data)
        html += self.generate_version_evolution(data)
        
        # Add footer
        html += f"""
        <div class="footer">
            <p>Generated by DataSON Improved Benchmark Runner | <a href="https://github.com/danielendler/datason">DataSON Project</a></p>
            <p>Report generated on {formatted_date}</p>
        </div>
    </div>
</body>
</html>"""
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ HTML report generated: {output_file}")

def main():
    """Main entry point for report generator"""
    parser = argparse.ArgumentParser(description='Generate improved DataSON benchmark reports')
    parser.add_argument('input_file', help='Input JSON benchmark results file')
    parser.add_argument('--output-file', help='Output HTML file path (optional)')
    parser.add_argument('--output-dir', default='docs/results', help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Load benchmark data
    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return 1
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Determine output file
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename based on input
        base_name = input_file.stem
        output_file = output_dir / f"{base_name}_report.html"
    
    # Generate report
    generator = ImprovedReportGenerator()
    generator.generate_report(data, output_file)
    
    return 0

if __name__ == '__main__':
    exit(main())
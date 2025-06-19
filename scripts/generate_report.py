#!/usr/bin/env python3
"""
Report Generator for DataSON Benchmarks
========================================

Generates HTML and markdown reports with interactive charts from benchmark results.
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings

# Suppress DataSON datetime parsing warnings for cleaner logs
# These warnings are caused by overly aggressive datetime detection in DataSON v0.9.0
# See DATASON_DATETIME_PARSING_ISSUES.md for detailed analysis
warnings.filterwarnings('ignore', message='Failed to parse datetime string', module='datason')

try:
    import datason
    import json
    
    # Create a wrapper to match json.dumps interface
    def serialize_with_datason(obj, **kwargs):
        """DataSON serialize + json.dumps for string output."""
        serialized_obj = datason.serialize(obj)
        return json.dumps(serialized_obj, **kwargs)
    
    # Use DataSON for serialization but json.dumps for final string conversion
    datason_serialize = serialize_with_datason
except ImportError:
    # Fallback to standard json if DataSON not available
    import json
    datason_serialize = json.dumps

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates reports with interactive charts from benchmark results."""
    
    def __init__(self, output_dir: str = "docs/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.is_ci = self._is_ci_environment()
    
    def _clean_data_for_json(self, data: Any) -> Any:
        """Clean data for JSON serialization by converting enum strings."""
        if isinstance(data, dict):
            return {k: self._clean_data_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_data_for_json(item) for item in data]
        elif isinstance(data, str):
            # Convert enum-like strings to just the name
            if "." in data and (data.startswith("CacheScope.") or 
                               data.startswith("DataFrameOrient.") or 
                               data.startswith("OutputType.") or 
                               data.startswith("DateFormat.") or 
                               data.startswith("NanHandling.") or 
                               data.startswith("TypeCoercion.")):
                return data.split(".")[-1]
            return data
        return data
    
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
    
    def _format_time_adaptive(self, time_ms: float) -> str:
        """
        Format time with adaptive units for best readability.
        
        Args:
            time_ms: Time in milliseconds
            
        Returns:
            Formatted string with appropriate units
        """
        if time_ms == 0:
            return "0"
        
        # Convert to different units and find the most readable one
        time_s = time_ms / 1000
        time_us = time_ms * 1000  # microseconds
        time_ns = time_ms * 1_000_000  # nanoseconds
        
        # Choose the most appropriate unit (aim for 1-999 range when possible)
        if time_s >= 1:
            return f"{time_s:.3f}s"
        elif time_ms >= 1:
            return f"{time_ms:.3f}ms"
        elif time_us >= 1:
            return f"{time_us:.1f}μs"
        else:
            return f"{time_ns:.0f}ns"
    
    def _format_time_adaptive_header(self, time_values: List[float]) -> str:
        """
        Determine the best unit for a set of time values and return header suffix.
        
        Args:
            time_values: List of time values in milliseconds
            
        Returns:
            Unit string like "(ms)", "(μs)", etc.
        """
        if not time_values:
            return "(ms)"
        
        # Use median for better representation of typical values
        sorted_values = sorted([v for v in time_values if v > 0])
        if not sorted_values:
            return "(ms)"
        
        # Use median as representative value to avoid outliers skewing units
        median_val = sorted_values[len(sorted_values) // 2]
        
        # Also check if max value would create very large numbers
        max_val = max(sorted_values)
        
        if median_val >= 1000 or max_val >= 10000:  # >= 1 second or very large
            return "(s)"
        elif median_val >= 1 or max_val >= 0.015:   # >= 1 millisecond or would be > 15 μs
            return "(ms)"
        elif median_val >= 0.001:  # >= 1 microsecond
            return "(μs)"
        else:
            return "(ns)"
    
    def _format_time_consistent(self, time_ms: float, unit: str) -> str:
        """
        Format time in a consistent unit for tables.
        
        Args:
            time_ms: Time in milliseconds
            unit: Target unit ("s", "ms", "μs", or "ns")
            
        Returns:
            Formatted number string (without unit)
        """
        if time_ms == 0:
            return "0"
        
        if unit == "(s)":
            return f"{time_ms / 1000:.3f}"
        elif unit == "(ms)":
            return f"{time_ms:.3f}"
        elif unit == "(μs)":
            # Use more precision for microseconds to avoid rounding small values to 0
            converted_val = time_ms * 1000
            if converted_val < 0.01:
                return f"{converted_val:.3f}"  # Use 3 decimal places for very small values
            elif converted_val < 1:
                return f"{converted_val:.2f}"  # Use 2 decimal places for small values
            else:
                return f"{converted_val:.1f}"  # Use 1 decimal place for larger values
        elif unit == "(ns)":
            return f"{time_ms * 1_000_000:.0f}"
        else:
            return f"{time_ms:.3f}"  # fallback to ms
    
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
            """ + datason_serialize(results, indent=2) + """
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
        """Generate competitive results with adaptive time formatting."""
        html = '<div class="section"><h2>🏆 Competitive Benchmarks</h2>'
        
        # Performance comparison chart
        chart_data = self._extract_competitive_chart_data(competitive_data)
        if chart_data:
            html += self._generate_performance_comparison_chart(chart_data)
        
        # Detailed results for each dataset
        for dataset_name, dataset_data in competitive_data.items():
            if dataset_name == "summary":
                continue
                
            html += f"<h3>📋 {dataset_name} - Detailed Results</h3>"
            
            # Enhanced description with sample data
            if "description" in dataset_data:
                html += f"<p><em>{dataset_data['description']}</em></p>"
            
            # Add sample data structure based on dataset name
            sample_html = self._generate_sample_data_html(dataset_name)
            if sample_html:
                html += sample_html
            
            # Serialization results table with adaptive formatting
            if "serialization" in dataset_data:
                html += "<h4>Serialization Performance</h4>"
                
                # Collect all time values to determine best unit
                all_times = []
                for lib, metrics in dataset_data["serialization"].items():
                    if isinstance(metrics, dict) and "mean_ms" in metrics:
                        all_times.extend([
                            metrics["mean_ms"],
                            metrics.get("min_ms", 0),
                            metrics.get("max_ms", 0),
                            metrics.get("std_ms", 0)
                        ])
                
                # Determine best unit for this table
                unit = self._format_time_adaptive_header(all_times)
                
                html += f"<table><tr><th>Library</th><th>Mean {unit}</th><th>Min {unit}</th><th>Max {unit}</th><th>Std Dev {unit}</th><th>Success Rate</th></tr>"
                
                for lib, metrics in dataset_data["serialization"].items():
                    if isinstance(metrics, dict) and "mean_ms" in metrics:
                        mean_ms = metrics["mean_ms"]
                        # Convert raw values (in seconds) to milliseconds
                        min_ms = metrics.get("min", 0) * 1000 if "min" in metrics else 0
                        max_ms = metrics.get("max", 0) * 1000 if "max" in metrics else 0
                        std_ms = metrics.get("std", 0) * 1000 if "std" in metrics else 0
                        success = metrics.get("successful_runs", 0)
                        errors = metrics.get("error_count", 0)
                        total = success + errors
                        rate = f"{success}/{total}" if total > 0 else "N/A"
                        
                        # Format times consistently in chosen unit
                        mean_formatted = self._format_time_consistent(mean_ms, unit)
                        min_formatted = self._format_time_consistent(min_ms, unit)
                        max_formatted = self._format_time_consistent(max_ms, unit)
                        std_formatted = self._format_time_consistent(std_ms, unit)
                        
                        # Color code performance relative to fastest
                        style = ""
                        if lib == "datason":
                            style = "style='background-color: #e3f2fd; font-weight: bold;'"
                        
                        html += f"<tr {style}><td>{lib}</td><td>{mean_formatted}</td><td>{min_formatted}</td><td>{max_formatted}</td><td>{std_formatted}</td><td>{rate}</td></tr>"
                
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
                var traces = {datason_serialize(traces)};
                
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
    
    def _generate_sample_data_html(self, dataset_name: str) -> str:
        """Generate HTML showing sample data structure for a dataset."""
        
        # Define sample data structures for each common dataset type
        sample_data_map = {
            "api_response": {
                "description": "REST API response with metadata, pagination, and business objects",
                "sample": {
                    "status": "success",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "request_id": "12345678-1234-5678-9012-123456789012",
                    "data": {
                        "items": [
                            {
                                "id": 1,
                                "name": "Product A",
                                "price": 19.99,
                                "created": "2024-01-01T10:00:00Z",
                                "active": True,
                                "tags": ["electronics", "featured"]
                            },
                            "... 19 more items"
                        ],
                        "pagination": {
                            "total": 1000,
                            "page": 1,
                            "per_page": 20
                        }
                    }
                },
                "key_features": ["UUIDs", "timestamps", "decimal prices", "nested objects", "arrays"]
            },
            
            "simple_objects": {
                "description": "Basic JSON-compatible data types (strings, numbers, booleans, arrays)",
                "sample": {
                    "strings": ["hello", "world", "test"],
                    "numbers": [1, 42, 100],
                    "floats": [3.14, 2.71, 1.41],
                    "booleans": [True, False],
                    "mixed_array": ["text", 42, True, None]
                },
                "key_features": ["primitive types", "mixed arrays", "null values", "Unicode strings"]
            },
            
            "nested_structures": {
                "description": "Deeply nested objects with complex hierarchies",
                "sample": {
                    "config": {
                        "database": {
                            "hosts": ["db1.example.com", "db2.example.com"],
                            "connection": {
                                "pool_size": 10,
                                "timeout": 30,
                                "retry_policy": {
                                    "max_attempts": 3,
                                    "backoff": "exponential"
                                }
                            }
                        },
                        "services": {
                            "auth": {"enabled": True, "provider": "oauth2"},
                            "cache": {"ttl": 3600, "type": "redis"}
                        }
                    }
                },
                "key_features": ["deep nesting", "configuration objects", "mixed data types"]
            },
            
            "datetime_heavy": {
                "description": "Objects with many datetime fields, timestamps, and UUIDs",
                "sample": {
                    "events": [
                        {
                            "id": "550e8400-e29b-41d4-a716-446655440000",
                            "timestamp": "2024-01-01T12:00:00Z",
                            "event_type": "user_action",
                            "user_id": "123e4567-e89b-12d3-a456-426614174000",
                            "metadata": {
                                "created_at": "2024-01-01T10:00:00Z",
                                "updated_at": "2024-01-01T11:00:00Z",
                                "expires_at": "2024-01-02T12:00:00Z"
                            }
                        },
                        "... 14 more events"
                    ]
                },
                "key_features": ["ISO timestamps", "UUID identifiers", "timezone handling", "date arithmetic"]
            },
            
            "basic_types": {
                "description": "Fundamental data types for core serialization speed testing",
                "sample": {
                    "integer": 42,
                    "float": 3.14159,
                    "string": "Hello, World!",
                    "boolean": True,
                    "null_value": None,
                    "simple_list": [1, 2, 3],
                    "simple_dict": {"key": "value"}
                },
                "key_features": ["core Python types", "minimal overhead", "baseline performance"]
            },
            
            "datetime_types": {
                "description": "Date and time handling with various formats",
                "sample": {
                    "iso_datetime": "2024-01-01T12:00:00Z",
                    "unix_timestamp": 1704110400,
                    "date_only": "2024-01-01",
                    "time_only": "12:00:00",
                    "timezone_aware": "2024-01-01T12:00:00+05:00"
                },
                "key_features": ["ISO 8601", "Unix timestamps", "timezone handling", "date formats"]
            },
            
            "advanced_types": {
                "description": "Complex Python types and newer language features",
                "sample": {
                    "decimal_numbers": "19.99",  # Decimal precision
                    "complex_number": "3+4j",
                    "set_data": [1, 2, 3],  # Set converted to list
                    "bytes_data": "base64_encoded_string",
                    "custom_objects": {"serialized": "representation"}
                },
                "key_features": ["Decimal precision", "complex numbers", "set types", "custom serialization"]
            },
            
            "decimal_precision": {
                "description": "Financial/scientific data requiring precise decimal handling",
                "sample": {
                    "financial_data": {
                        "balance": "1234567.89",  # Decimal
                        "transactions": [
                            {"amount": "99.99", "currency": "USD"},
                            {"amount": "149.50", "currency": "EUR"}
                        ],
                        "exchange_rates": {
                            "USD_EUR": "0.85234567",
                            "EUR_GBP": "0.87654321"
                        }
                    }
                },
                "key_features": ["exact decimal arithmetic", "financial precision", "currency handling"]
            },
            
            "large_dataset": {
                "description": "Large collections testing memory efficiency and streaming",
                "sample": {
                    "users": [
                        {
                            "id": 1,
                            "email": "user1@example.com",
                            "profile": {
                                "name": "User One",
                                "preferences": {"theme": "dark"},
                                "activity": "... large activity log"
                            }
                        },
                        "... 999 more users"
                    ],
                    "metadata": {
                        "total_count": 1000,
                        "generated_at": "2024-01-01T12:00:00Z"
                    }
                },
                "key_features": ["large arrays", "memory pressure", "streaming potential", "bulk data"]
            },
            
            "complex_structure": {
                "description": "Multi-level nested objects with varied data types",
                "sample": {
                    "organization": {
                        "departments": [
                            {
                                "name": "Engineering",
                                "employees": [
                                    {
                                        "id": 1,
                                        "profile": {
                                            "skills": ["Python", "JavaScript"],
                                            "projects": [{"name": "ProjectA", "status": "active"}]
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                },
                "key_features": ["deep nesting", "object hierarchies", "organizational data"]
            }
        }
        
        # Get sample data for this dataset
        if dataset_name not in sample_data_map:
            return ""
        
        sample_info = sample_data_map[dataset_name]
        
        # Create collapsible sample data section
        sample_json = datason_serialize(sample_info["sample"], indent=2)
        features_list = ", ".join(sample_info["key_features"])
        
        html = f"""
        <div style="background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff;">
            <p><strong>📝 Dataset Details:</strong> {sample_info["description"]}</p>
            <p><strong>🔑 Key Features:</strong> {features_list}</p>
            <details style="margin-top: 10px;">
                <summary style="cursor: pointer; font-weight: bold;">📄 View Sample Data Structure</summary>
                <pre style="background: white; padding: 10px; margin: 10px 0; border-radius: 3px; overflow-x: auto; font-size: 12px;"><code>{sample_json}</code></pre>
            </details>
        </div>
        """
        
        return html
    
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
                    x: {datason_serialize(config_names)},
                    y: {datason_serialize(total_times)},
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
                var traces = {datason_serialize(traces)};
                
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


def main():
    """Command-line interface for generating reports."""
    import argparse
    import glob
    import json
    import os
    
    parser = argparse.ArgumentParser(description='Generate benchmark reports')
    parser.add_argument('--input-dir', required=True, help='Directory containing benchmark result JSON files')
    parser.add_argument('--output', required=True, help='Output file path for the report')
    parser.add_argument('--format', choices=['html', 'markdown'], default='html', help='Output format')
    parser.add_argument('--include-charts', action='store_true', help='Include interactive charts (HTML only)')
    
    args = parser.parse_args()
    
    # Find all JSON files in input directory
    json_files = glob.glob(os.path.join(args.input_dir, '*.json'))
    
    if not json_files:
        print(f"No JSON files found in {args.input_dir}")
        return 1
    
    # Combine all results into a single report
    combined_results = {
        "suite_type": "weekly_comprehensive",
        "timestamp": datetime.now().isoformat(),
        "environment": "CI" if os.environ.get('CI') else "local"
    }
    
    # Process each JSON file and categorize results
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                if json_file.endswith('.json'):
                    # Try DataSON first, fallback to json
                    try:
                        content = f.read()
                        data = datason.deserialize(json.loads(content))
                    except:
                        f.seek(0)
                        data = json.load(f)
                else:
                    data = json.load(f)
            
            # The data structure from run_benchmarks.py has the actual results nested
            # Look for the actual benchmark data in the structure
            if isinstance(data, dict):
                # Check for direct competitive results
                if "competitive" in data and isinstance(data["competitive"], dict):
                    # This is the competitive results from the benchmark
                    combined_results["competitive"] = data["competitive"]
                elif "configurations" in data and isinstance(data["configurations"], dict):
                    combined_results["configurations"] = data["configurations"]
                elif "versioning" in data and isinstance(data["versioning"], dict):
                    combined_results["versioning"] = data["versioning"]
                else:
                    # Check if this is a results file with nested structure
                    for key, value in data.items():
                        if isinstance(value, dict):
                            if "competitive" in value:
                                combined_results["competitive"] = value
                            elif "configurations" in value:
                                combined_results["configurations"] = value
                            elif "versioning" in value:
                                combined_results["versioning"] = value
                            # Also check for serialization data which indicates competitive results
                            elif any("serialization" in str(v) for v in value.values() if isinstance(v, dict)):
                                combined_results["competitive"] = value
            
        except Exception as e:
            print(f"Warning: Could not process {json_file}: {e}")
            continue
    
    # Generate report
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    generator = ReportGenerator(output_dir=output_dir if output_dir else ".")
    
    if args.format == 'html':
        report_path = generator.generate_html_report(combined_results)
        # Move to desired output location if different
        if report_path != args.output:
            import shutil
            shutil.move(report_path, args.output)
        print(f"✅ HTML report generated: {args.output}")
    else:
        # Generate markdown report (simplified)
        markdown_content = f"""# Weekly DataSON Benchmark Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

## Summary

This comprehensive report includes:
- Competitive analysis vs other JSON libraries
- Configuration testing results  
- Version comparison analysis

## Results

"""
        
        # Add basic results summary
        if "competitive" in combined_results:
            markdown_content += "### Competitive Analysis\n\n"
            markdown_content += "DataSON performance compared to standard JSON libraries.\n\n"
        
        if "configurations" in combined_results:
            markdown_content += "### Configuration Testing\n\n"
            markdown_content += "Performance across different DataSON configurations.\n\n"
        
        if "versioning" in combined_results:
            markdown_content += "### Version Comparison\n\n"
            markdown_content += "Performance evolution across DataSON versions.\n\n"
        
        markdown_content += f"""
## Raw Data

```json
{json.dumps(combined_results, indent=2)}
```

---
*Generated by DataSON benchmark suite*
"""
        
        with open(args.output, 'w') as f:
            f.write(markdown_content)
        print(f"✅ Markdown report generated: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
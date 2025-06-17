#!/usr/bin/env python3
"""
DataSON Version Comparison Suite
================================

Tests different DataSON versions to track performance evolution and feature changes.
Handles API differences and feature availability across versions.
FOCUSES ON OPTIMIZATION CONFIGURATIONS AND DEEP API ANALYSIS.
"""

import logging
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class DataSONVersionInfo:
    """Information about a specific DataSON version."""
    
    def __init__(self, version: str, pip_spec: Optional[str] = None):
        self.version = version
        self.pip_spec = pip_spec or f"datason=={version}"
        self.api_features: Set[str] = set()
        self.config_methods: Set[str] = set()
        self.optimization_configs: Dict[str, Any] = {}
        self.available = False
        self.module = None
        
    def detect_features(self):
        """Detect available features in this DataSON version."""
        if not self.module:
            return
            
        # Check for configuration methods
        config_methods = [
            'get_performance_config',
            'get_ml_config', 
            'get_strict_config',
            'get_api_config',
            'get_compatibility_config',
            'get_memory_config',
            'get_speed_config'
        ]
        
        for method in config_methods:
            if hasattr(self.module, method):
                self.config_methods.add(method)
                
        # Check for API features
        features = [
            'serialize',
            'deserialize',
            'serialize_to_file',
            'deserialize_from_file'
        ]
        
        for feature in features:
            if hasattr(self.module, feature):
                self.api_features.add(feature)
                
        # Check for advanced features
        if hasattr(self.module, 'deserialize_fast'):
            self.api_features.add('deserialize_fast')
            
        if hasattr(self.module, 'chunked_serialize'):
            self.api_features.add('chunked_serialize')
            
        # Check for optimization-specific features
        optimization_features = [
            'streaming_serialize',
            'batch_serialize',
            'parallel_serialize',
            'compressed_serialize',
            'schema_validate',
            'type_hints_serialize'
        ]
        
        for feature in optimization_features:
            if hasattr(self.module, feature):
                self.api_features.add(feature)
    
    def analyze_optimization_configs(self):
        """Deep analysis of DataSON's optimization configurations."""
        if not self.module:
            return
            
        logger.info(f"  🔍 Analyzing optimization configs for DataSON {self.version}")
        
        # Test all available configuration methods
        for config_method in self.config_methods:
            try:
                config_func = getattr(self.module, config_method)
                config = config_func()
                
                # Deep inspection of configuration
                config_analysis = {
                    'available': True,
                    'type': type(config).__name__,
                    'config_data': {}
                }
                
                if hasattr(config, '__dict__'):
                    # Configuration object - inspect attributes
                    config_analysis['config_data'] = {
                        attr: getattr(config, attr) for attr in dir(config) 
                        if not attr.startswith('_') and not callable(getattr(config, attr))
                    }
                elif isinstance(config, dict):
                    # Dictionary configuration
                    config_analysis['config_data'] = config
                else:
                    # Other type - convert to string
                    config_analysis['config_data'] = str(config)
                
                self.optimization_configs[config_method] = config_analysis
                
            except Exception as e:
                self.optimization_configs[config_method] = {
                    'available': True,
                    'error': str(e)
                }
        
        # Test configuration parameter discovery
        self._discover_config_parameters()
    
    def _discover_config_parameters(self):
        """Discover available configuration parameters."""
        if not self.module:
            return
            
        # Test if serialize function accepts config parameters
        config_params = {}
        
        # Common optimization parameters to test
        test_params = [
            'use_cache', 'enable_compression', 'parallel_mode',
            'strict_mode', 'validate_schema', 'optimize_size',
            'optimize_speed', 'memory_limit', 'chunk_size',
            'encoding', 'precision', 'date_format'
        ]
        
        for param in test_params:
            try:
                # Test with simple data
                test_data = {"test": "value"}
                kwargs = {param: True}  # Try boolean first
                
                # Attempt serialization with parameter
                result = self.module.serialize(test_data, **kwargs)
                config_params[param] = {'supported': True, 'type': 'boolean'}
                
            except TypeError as e:
                if 'unexpected keyword argument' not in str(e):
                    # Parameter exists but wrong type
                    config_params[param] = {'supported': True, 'type': 'unknown', 'error': str(e)}
            except Exception:
                # Parameter might exist but cause other errors
                pass
        
        self.optimization_configs['discovered_parameters'] = config_params


class DataSONVersionManager:
    """Manages different DataSON versions for testing."""
    
    def __init__(self):
        self.versions = self._define_test_versions()
        self.current_version = None
        
    def _define_test_versions(self) -> Dict[str, DataSONVersionInfo]:
        """Define the DataSON versions to test."""
        return {
            # Recent stable versions
            'latest': DataSONVersionInfo('latest', 'datason'),
            '0.11.0': DataSONVersionInfo('0.11.0'),
            '0.10.0': DataSONVersionInfo('0.10.0'),
            '0.9.0': DataSONVersionInfo('0.9.0'),
            
            # Development version
            'dev': DataSONVersionInfo('dev', 'git+https://github.com/datason/datason.git'),
            
            # Major milestone versions (add as needed)
            # '0.8.0': DataSONVersionInfo('0.8.0'),
            # '0.7.0': DataSONVersionInfo('0.7.0'),
        }
    
    @contextmanager
    def version_context(self, version_name: str):
        """Context manager to temporarily install and use a specific DataSON version."""
        version_info = self.versions.get(version_name)
        if not version_info:
            raise ValueError(f"Unknown version: {version_name}")
            
        # Save current state
        original_module = None
        if 'datason' in sys.modules:
            original_module = sys.modules['datason']
            del sys.modules['datason']
            
        try:
            # Install the specific version
            if not self._is_version_installed(version_info):
                self._install_version(version_info)
                
            # Import and set up
            import datason
            version_info.module = datason
            version_info.available = True
            version_info.detect_features()
            version_info.analyze_optimization_configs()
            
            self.current_version = version_info
            yield version_info
            
        except Exception as e:
            logger.error(f"Failed to use DataSON version {version_name}: {e}")
            version_info.available = False
            raise
            
        finally:
            # Restore original state
            if 'datason' in sys.modules:
                del sys.modules['datason']
            if original_module:
                sys.modules['datason'] = original_module
            self.current_version = None
    
    def _is_version_installed(self, version_info: DataSONVersionInfo) -> bool:
        """Check if a specific version is already installed."""
        try:
            result = subprocess.run(
                [sys.executable, '-c', f'import datason; print(datason.__version__)'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                installed_version = result.stdout.strip()
                return installed_version == version_info.version or version_info.version == 'latest'
        except Exception:
            pass
        return False
    
    def _install_version(self, version_info: DataSONVersionInfo):
        """Install a specific DataSON version."""
        logger.info(f"Installing DataSON {version_info.version}...")
        
        try:
            # Uninstall current version first
            subprocess.run(
                [sys.executable, '-m', 'pip', 'uninstall', 'datason', '-y'],
                capture_output=True, timeout=60
            )
            
            # Install specific version
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', version_info.pip_spec],
                capture_output=True, timeout=120, check=True
            )
            
            logger.info(f"✅ DataSON {version_info.version} installed successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install DataSON {version_info.version}: {e}")
            raise
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout installing DataSON {version_info.version}")
            raise


class DataSONVersionBenchmarkSuite:
    """Benchmarks different DataSON versions with focus on optimization configs."""
    
    def __init__(self):
        self.version_manager = DataSONVersionManager()
        
    def create_optimization_test_data(self) -> Dict[str, Any]:
        """Create test data specifically designed to test optimization configs."""
        return {
            "basic_types": {
                "description": "Basic types - tests core serialization speed",
                "data": {
                    "string": "hello world",
                    "integer": 42,
                    "float": 3.14159,
                    "boolean": True,
                    "null": None,
                    "list": [1, 2, 3, "mixed", True],
                    "dict": {"nested": {"key": "value"}}
                }
            },
            
            "datetime_heavy": {
                "description": "Datetime handling - tests date optimization configs",
                "data": {
                    "single_datetime": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                    "datetime_list": [
                        datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                        datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
                        datetime(2024, 1, 3, 12, 0, 0, tzinfo=timezone.utc)
                    ],
                    "nested_dates": {
                        "events": [
                            {"timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc), "event": "start"},
                            {"timestamp": datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc), "event": "middle"},
                            {"timestamp": datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc), "event": "end"}
                        ]
                    }
                }
            },
            
            "decimal_precision": {
                "description": "Decimal handling - tests precision optimization configs",
                "data": {
                    "price": Decimal("19.99"),
                    "prices": [Decimal(f"{i}.99") for i in range(10)],
                    "financial_data": {
                        "transactions": [
                            {"amount": Decimal("100.50"), "fee": Decimal("2.99")},
                            {"amount": Decimal("250.00"), "fee": Decimal("5.00")},
                            {"amount": Decimal("75.25"), "fee": Decimal("1.50")}
                        ],
                        "total": Decimal("425.75")
                    }
                }
            },
            
            "large_dataset": {
                "description": "Large data - tests memory and compression configs",
                "data": {
                    "large_list": list(range(1000)),
                    "large_dict": {f"key_{i}": f"value_{i}" for i in range(500)},
                    "nested_large": {
                        "level1": {
                            "level2": {
                                "data": [{"id": i, "value": f"item_{i}"} for i in range(100)]
                            }
                        }
                    }
                }
            },
            
            "complex_structure": {
                "description": "Complex nested data - tests structural optimization",
                "data": {
                    "users": [
                        {
                            "id": i,
                            "profile": {
                                "name": f"User {i}",
                                "created": datetime(2024, 1, 1, tzinfo=timezone.utc),
                                "balance": Decimal(f"{i * 10}.50"),
                                "preferences": {
                                    "theme": "dark" if i % 2 else "light",
                                    "notifications": True,
                                    "settings": {
                                        "auto_save": True,
                                        "sync_interval": 300
                                    }
                                }
                            }
                        } for i in range(50)
                    ]
                }
            }
        }
    
    def benchmark_optimization_configs(self, version_info: DataSONVersionInfo, 
                                     test_data: Any, dataset_name: str,
                                     iterations: int = 5) -> Dict[str, Any]:
        """Benchmark different optimization configurations for a version."""
        if not version_info.available or not version_info.module:
            return {"error": "Version not available"}
            
        results = {
            "version": version_info.version,
            "dataset": dataset_name,
            "config_results": {},
            "optimization_analysis": {}
        }
        
        # Test each available configuration method
        for config_method in version_info.config_methods:
            logger.info(f"    Testing config: {config_method}")
            
            try:
                config_func = getattr(version_info.module, config_method)
                config = config_func()
                
                # Benchmark with this configuration
                times = []
                errors = []
                
                for _ in range(iterations):
                    try:
                        start = time.perf_counter()
                        
                        # Try to use the configuration
                        if hasattr(config, '__dict__') or isinstance(config, dict):
                            # Configuration object or dict - try to pass as config
                            serialized = version_info.module.serialize(test_data, config=config)
                        else:
                            # Other type - just use default
                            serialized = version_info.module.serialize(test_data)
                        
                        end = time.perf_counter()
                        times.append(end - start)
                    except Exception as e:
                        errors.append(str(e))
                
                if times:
                    results["config_results"][config_method] = {
                        "mean_ms": mean(times) * 1000,
                        "min_ms": min(times) * 1000,
                        "max_ms": max(times) * 1000,
                        "std_ms": stdev(times) * 1000 if len(times) > 1 else 0.0,
                        "successful_runs": len(times),
                        "error_count": len(errors),
                        "config_type": type(config).__name__
                    }
                else:
                    results["config_results"][config_method] = {
                        "error": "All attempts failed",
                        "errors": errors,
                        "config_type": type(config).__name__
                    }
                    
            except Exception as e:
                results["config_results"][config_method] = {
                    "error": f"Config method failed: {str(e)}"
                }
        
        # Test default configuration for comparison
        try:
            times = []
            for _ in range(iterations):
                try:
                    start = time.perf_counter()
                    serialized = version_info.module.serialize(test_data)
                    end = time.perf_counter()
                    times.append(end - start)
                except Exception:
                    pass
            
            if times:
                results["config_results"]["default"] = {
                    "mean_ms": mean(times) * 1000,
                    "min_ms": min(times) * 1000,
                    "max_ms": max(times) * 1000,
                    "std_ms": stdev(times) * 1000 if len(times) > 1 else 0.0,
                    "successful_runs": len(times),
                    "config_type": "default"
                }
        except Exception:
            pass
        
        # Add optimization analysis
        results["optimization_analysis"] = {
            "available_configs": list(version_info.config_methods),
            "optimization_configs": version_info.optimization_configs,
            "fastest_config": self._find_fastest_config(results["config_results"]),
            "performance_variance": self._calculate_config_variance(results["config_results"])
        }
        
        return results
    
    def _find_fastest_config(self, config_results: Dict[str, Any]) -> Dict[str, Any]:
        """Find the fastest configuration from results."""
        fastest_time = float('inf')
        fastest_config = None
        
        for config_name, result in config_results.items():
            if isinstance(result, dict) and "mean_ms" in result:
                if result["mean_ms"] < fastest_time:
                    fastest_time = result["mean_ms"]
                    fastest_config = config_name
        
        return {
            "config": fastest_config,
            "time_ms": fastest_time
        }
    
    def _calculate_config_variance(self, config_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate variance between different configurations."""
        times = []
        
        for result in config_results.values():
            if isinstance(result, dict) and "mean_ms" in result:
                times.append(result["mean_ms"])
        
        if len(times) < 2:
            return {"variance": 0, "range_ms": 0, "analysis": "Insufficient data"}
        
        variance = max(times) / min(times) if min(times) > 0 else 0
        range_ms = max(times) - min(times)
        
        return {
            "variance_ratio": variance,
            "range_ms": range_ms,
            "min_ms": min(times),
            "max_ms": max(times),
            "analysis": "High variance" if variance > 2.0 else "Low variance"
        }
    
    def run_optimization_focused_comparison(self, versions: Optional[List[str]] = None, 
                                          iterations: int = 5) -> Dict[str, Any]:
        """Run optimization-focused version comparison."""
        if versions is None:
            versions = ['latest', '0.11.0', '0.10.0', '0.9.0']
        
        # Filter to available versions
        available_versions = [v for v in versions if v in self.version_manager.versions]
        
        logger.info(f"Testing DataSON optimization configs across versions: {available_versions}")
        
        test_datasets = self.create_optimization_test_data()
        results = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "versions_tested": available_versions,
                "python_version": sys.version,
                "focus": "optimization_configurations"
            },
            "version_results": {},
            "optimization_summary": {}
        }
        
        for version_name in available_versions:
            logger.info(f"Testing DataSON version: {version_name}")
            
            try:
                with self.version_manager.version_context(version_name) as version_info:
                    version_results = {
                        "version": version_info.version,
                        "available_features": list(version_info.api_features),
                        "available_configs": list(version_info.config_methods),
                        "optimization_configs": version_info.optimization_configs,
                        "datasets": {}
                    }
                    
                    # Test each dataset with optimization focus
                    for dataset_name, dataset_info in test_datasets.items():
                        logger.info(f"  Testing optimization dataset: {dataset_name}")
                        
                        dataset_results = self.benchmark_optimization_configs(
                            version_info, dataset_info["data"], dataset_name, iterations
                        )
                        
                        version_results["datasets"][dataset_name] = {
                            "description": dataset_info["description"],
                            "results": dataset_results
                        }
                    
                    results["version_results"][version_name] = version_results
                    
            except Exception as e:
                logger.error(f"Failed to test version {version_name}: {e}")
                results["version_results"][version_name] = {
                    "error": str(e),
                    "available": False
                }
        
        # Generate optimization-focused summary
        results["optimization_summary"] = self._generate_optimization_summary(results["version_results"])
        
        return results
    
    def _generate_optimization_summary(self, version_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization-focused summary."""
        summary = {
            "config_evolution": {},
            "performance_by_config": {},
            "optimization_recommendations": {},
            "api_changes": {}
        }
        
        # Track configuration method evolution
        all_configs = set()
        for version_data in version_results.values():
            if isinstance(version_data, dict) and "available_configs" in version_data:
                all_configs.update(version_data["available_configs"])
        
        for config in all_configs:
            config_availability = {}
            for version_name, version_data in version_results.items():
                if isinstance(version_data, dict) and "available_configs" in version_data:
                    config_availability[version_name] = config in version_data["available_configs"]
            summary["config_evolution"][config] = config_availability
        
        # Track performance by configuration across versions
        for dataset_name in ["basic_types", "datetime_heavy", "decimal_precision", "large_dataset"]:
            dataset_summary = {}
            
            for version_name, version_data in version_results.items():
                if isinstance(version_data, dict) and "datasets" in version_data:
                    dataset_data = version_data["datasets"].get(dataset_name, {})
                    results = dataset_data.get("results", {})
                    config_results = results.get("config_results", {})
                    
                    version_perf = {}
                    for config_name, config_data in config_results.items():
                        if isinstance(config_data, dict) and "mean_ms" in config_data:
                            version_perf[config_name] = config_data["mean_ms"]
                    
                    if version_perf:
                        dataset_summary[version_name] = version_perf
            
            if dataset_summary:
                summary["performance_by_config"][dataset_name] = dataset_summary
        
        # Generate optimization recommendations
        summary["optimization_recommendations"] = self._generate_optimization_recommendations(
            summary["performance_by_config"]
        )
        
        return summary
    
    def _generate_optimization_recommendations(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization recommendations based on performance analysis."""
        recommendations = {
            "fastest_configs_by_dataset": {},
            "most_consistent_configs": {},
            "version_recommendations": {}
        }
        
        # Find fastest config for each dataset
        for dataset, version_data in performance_data.items():
            fastest_overall = float('inf')
            fastest_config = None
            fastest_version = None
            
            for version, config_perfs in version_data.items():
                for config, time_ms in config_perfs.items():
                    if time_ms < fastest_overall:
                        fastest_overall = time_ms
                        fastest_config = config
                        fastest_version = version
            
            if fastest_config:
                recommendations["fastest_configs_by_dataset"][dataset] = {
                    "config": fastest_config,
                    "version": fastest_version,
                    "time_ms": fastest_overall
                }
        
        return recommendations
    
    # Keep the original method for backward compatibility
    def run_version_comparison(self, versions: Optional[List[str]] = None, 
                             iterations: int = 5) -> Dict[str, Any]:
        """Run comprehensive version comparison (calls the optimization-focused version)."""
        return self.run_optimization_focused_comparison(versions, iterations) 
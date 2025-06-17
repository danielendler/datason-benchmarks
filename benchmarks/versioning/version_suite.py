#!/usr/bin/env python3
"""
DataSON Version Comparison Suite
================================

Tests different DataSON versions to track performance evolution and feature changes.
Handles API differences and feature availability across versions.
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
            'get_api_config'
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
    """Benchmarks different DataSON versions."""
    
    def __init__(self):
        self.version_manager = DataSONVersionManager()
        
    def create_version_test_data(self) -> Dict[str, Any]:
        """Create test data suitable for all DataSON versions."""
        return {
            "basic_types": {
                "description": "Basic types supported by all versions",
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
            
            "datetime_types": {
                "description": "Datetime handling (may vary by version)",
                "data": {
                    "single_datetime": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                    "datetime_list": [
                        datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                        datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
                    ]
                }
            },
            
            "advanced_types": {
                "description": "Advanced types (newer versions)",
                "data": {
                    "decimal": Decimal("19.99"),
                    "complex_data": {
                        "items": [
                            {"id": i, "value": Decimal(f"{i}.99")} 
                            for i in range(5)
                        ]
                    }
                }
            }
        }
    
    def benchmark_version_serialization(self, version_info: DataSONVersionInfo, 
                                      test_data: Any, iterations: int = 5) -> Dict[str, Any]:
        """Benchmark serialization for a specific version."""
        if not version_info.available or not version_info.module:
            return {"error": "Version not available"}
            
        results = {}
        
        # Test basic serialization
        if 'serialize' in version_info.api_features:
            times = []
            errors = []
            
            for _ in range(iterations):
                try:
                    start = time.perf_counter()
                    serialized = version_info.module.serialize(test_data)
                    end = time.perf_counter()
                    times.append(end - start)
                except Exception as e:
                    errors.append(str(e))
            
            if times:
                results['serialize'] = {
                    "mean_ms": mean(times) * 1000,
                    "min_ms": min(times) * 1000,
                    "max_ms": max(times) * 1000,
                    "std_ms": stdev(times) * 1000 if len(times) > 1 else 0.0,
                    "successful_runs": len(times),
                    "error_count": len(errors)
                }
            else:
                results['serialize'] = {"error": "All attempts failed", "errors": errors}
        
        # Test fast deserialization if available
        if 'deserialize_fast' in version_info.api_features:
            try:
                serialized = version_info.module.serialize(test_data)
                times = []
                
                for _ in range(iterations):
                    try:
                        start = time.perf_counter()
                        result = version_info.module.deserialize_fast(serialized)
                        end = time.perf_counter()
                        times.append(end - start)
                    except Exception:
                        pass
                
                if times:
                    results['deserialize_fast'] = {
                        "mean_ms": mean(times) * 1000,
                        "successful_runs": len(times)
                    }
            except Exception:
                pass
        
        return results
    
    def test_configuration_compatibility(self, version_info: DataSONVersionInfo) -> Dict[str, Any]:
        """Test which configuration methods are available in this version."""
        config_results = {}
        
        for config_method in ['get_performance_config', 'get_ml_config', 'get_strict_config', 'get_api_config']:
            if config_method in version_info.config_methods:
                try:
                    config_func = getattr(version_info.module, config_method)
                    config = config_func()
                    config_results[config_method] = {
                        "available": True,
                        "config_keys": list(config.keys()) if isinstance(config, dict) else "non-dict"
                    }
                except Exception as e:
                    config_results[config_method] = {
                        "available": True,
                        "error": str(e)
                    }
            else:
                config_results[config_method] = {"available": False}
        
        return config_results
    
    def run_version_comparison(self, versions: Optional[List[str]] = None, 
                             iterations: int = 5) -> Dict[str, Any]:
        """Run comprehensive version comparison."""
        if versions is None:
            versions = ['latest', '0.11.0', '0.10.0', '0.9.0']
        
        # Filter to available versions
        available_versions = [v for v in versions if v in self.version_manager.versions]
        
        logger.info(f"Testing DataSON versions: {available_versions}")
        
        test_datasets = self.create_version_test_data()
        results = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "versions_tested": available_versions,
                "python_version": sys.version
            },
            "version_results": {},
            "summary": {}
        }
        
        for version_name in available_versions:
            logger.info(f"Testing DataSON version: {version_name}")
            
            try:
                with self.version_manager.version_context(version_name) as version_info:
                    version_results = {
                        "version": version_info.version,
                        "available_features": list(version_info.api_features),
                        "available_configs": list(version_info.config_methods),
                        "datasets": {},
                        "configuration_compatibility": self.test_configuration_compatibility(version_info)
                    }
                    
                    # Test each dataset
                    for dataset_name, dataset_info in test_datasets.items():
                        logger.info(f"  Testing dataset: {dataset_name}")
                        
                        dataset_results = self.benchmark_version_serialization(
                            version_info, dataset_info["data"], iterations
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
        
        # Generate summary
        results["summary"] = self._generate_version_summary(results["version_results"])
        
        return results
    
    def _generate_version_summary(self, version_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of version comparison results."""
        summary = {
            "performance_evolution": {},
            "feature_evolution": {},
            "api_compatibility": {}
        }
        
        # Track performance changes
        for dataset_name in ["basic_types", "datetime_types", "advanced_types"]:
            perf_data = {}
            
            for version_name, version_data in version_results.items():
                if isinstance(version_data, dict) and "datasets" in version_data:
                    dataset_results = version_data["datasets"].get(dataset_name, {}).get("results", {})
                    if "serialize" in dataset_results and "mean_ms" in dataset_results["serialize"]:
                        perf_data[version_name] = dataset_results["serialize"]["mean_ms"]
            
            if perf_data:
                summary["performance_evolution"][dataset_name] = perf_data
        
        # Track feature evolution
        all_features = set()
        for version_data in version_results.values():
            if isinstance(version_data, dict) and "available_features" in version_data:
                all_features.update(version_data["available_features"])
        
        for feature in all_features:
            feature_availability = {}
            for version_name, version_data in version_results.items():
                if isinstance(version_data, dict) and "available_features" in version_data:
                    feature_availability[version_name] = feature in version_data["available_features"]
            summary["feature_evolution"][feature] = feature_availability
        
        return summary 
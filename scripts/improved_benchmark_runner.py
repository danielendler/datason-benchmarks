#!/usr/bin/env python3
"""
Improved DataSON Benchmark Runner
=================================

Focus areas:
1. DataSON API Modes - Compare different DataSON methods
2. Competitor Analysis - Fair comparisons with other libraries  
3. Version Evolution - Track DataSON improvements over time

Removes confusing "page 1-4" terminology in favor of clear categories.
"""

import argparse
import json
import logging
import time
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Structured benchmark result"""
    method: str
    library: str
    scenario: str
    mean_time: float
    min_time: float  
    max_time: float
    std_time: float
    successful_runs: int
    error_count: int
    error_message: Optional[str] = None
    output_size: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mean': self.mean_time,
            'min': self.min_time,
            'max': self.max_time,
            'std': self.std_time,
            'successful_runs': self.successful_runs,
            'error_count': self.error_count,
            'mean_ms': self.mean_time * 1000,  # For compatibility
            'error': self.error_message,
            'output_size': self.output_size
        }

class ImprovedBenchmarkRunner:
    """
    Enhanced benchmark runner focusing on:
    - DataSON API method comparison
    - Fair competitor analysis
    - Version evolution tracking
    """
    
    def __init__(self, output_dir: str = "data/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata = {
            "timestamp": datetime.now().isoformat(),
            "python_version": self._get_python_version(),
            "datason_version": self._get_datason_version(),
            "benchmark_framework": "improved_v1",
            "focus": "api_modes_competitors_versions"
        }
        
        # Define DataSON API methods to test
        self.datason_methods = {
            'serialize': {'func': 'serialize', 'type': 'basic_serialization'},
            'dump_secure': {'func': 'dump_secure', 'type': 'secure_serialization'},
            'save_string': {'func': 'save_string', 'type': 'string_serialization'},
            'deserialize': {'func': 'deserialize', 'type': 'basic_deserialization'},
            'load_basic': {'func': 'load_basic', 'type': 'fast_deserialization'},
            'load_smart': {'func': 'load_smart', 'type': 'smart_deserialization'},
            'dump_json': {'func': 'dump_json', 'type': 'json_compat_serialization'},
            'loads_json': {'func': 'loads_json', 'type': 'json_compat_deserialization'}
        }
        
        # Competitor libraries with fair comparison mapping
        self.competitors = {
            'orjson': {
                'serialize': 'orjson.dumps',
                'deserialize': 'orjson.loads',
                'notes': 'High-performance JSON library'
            },
            'ujson': {
                'serialize': 'ujson.dumps', 
                'deserialize': 'ujson.loads',
                'notes': 'Ultra-fast JSON library'
            },
            'json': {
                'serialize': 'json.dumps',
                'deserialize': 'json.loads', 
                'notes': 'Python standard library'
            },
            'pickle': {
                'serialize': 'pickle.dumps',
                'deserialize': 'pickle.loads',
                'notes': 'Python object serialization'
            },
            'msgpack': {
                'serialize': 'msgpack.packb',
                'deserialize': 'msgpack.unpackb',
                'notes': 'Binary serialization format'
            }
        }
        
    def _get_python_version(self) -> str:
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_datason_version(self) -> str:
        try:
            import datason
            return getattr(datason, '__version__', 'unknown')
        except ImportError:
            return 'not_installed'
    
    def _import_competitor(self, competitor: str) -> Optional[Any]:
        """Safely import competitor libraries"""
        try:
            if competitor == 'orjson':
                import orjson
                return orjson
            elif competitor == 'ujson':
                import ujson  
                return ujson
            elif competitor == 'json':
                import json
                return json
            elif competitor == 'pickle':
                import pickle
                return pickle
            elif competitor == 'msgpack':
                import msgpack
                return msgpack
            else:
                return None
        except ImportError as e:
            logger.warning(f"Could not import {competitor}: {e}")
            return None
    
    def _benchmark_method(self, func, data, iterations: int = 5) -> BenchmarkResult:
        """Benchmark a specific function with data"""
        times = []
        errors = 0
        error_message = None
        output_size = None
        
        for i in range(iterations):
            try:
                start_time = time.perf_counter()
                result = func(data)
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                
                # Measure output size on first successful run
                if output_size is None and result is not None:
                    if isinstance(result, (str, bytes)):
                        output_size = len(result)
                    elif hasattr(result, '__len__'):
                        output_size = len(result)
                        
            except Exception as e:
                errors += 1
                if error_message is None:
                    error_message = str(e)
                logger.debug(f"Benchmark iteration {i} failed: {e}")
        
        if not times:
            return BenchmarkResult(
                method="unknown", library="unknown", scenario="unknown",
                mean_time=0, min_time=0, max_time=0, std_time=0,
                successful_runs=0, error_count=errors,
                error_message=error_message or "All attempts failed"
            )
        
        return BenchmarkResult(
            method="unknown", library="unknown", scenario="unknown",
            mean_time=statistics.mean(times),
            min_time=min(times),
            max_time=max(times),
            std_time=statistics.stdev(times) if len(times) > 1 else 0,
            successful_runs=len(times),
            error_count=errors,
            error_message=error_message,
            output_size=output_size
        )
    
    def run_datason_api_comparison(self, test_scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare different DataSON API methods across scenarios
        """
        logger.info("🔍 Running DataSON API Method Comparison")
        
        import datason
        results = {}
        
        for scenario_name, scenario_data in test_scenarios.items():
            logger.info(f"  📊 Testing scenario: {scenario_name}")
            scenario_results = {}
            
            for method_name, method_info in self.datason_methods.items():
                try:
                    func_name = method_info['func']
                    if hasattr(datason, func_name):
                        func = getattr(datason, func_name)
                        
                        # Handle different method signatures
                        if 'deserialize' in method_name or 'load' in method_name:
                            # For deserialization, first serialize the data
                            try:
                                serialized = datason.serialize(scenario_data)
                                benchmark_result = self._benchmark_method(func, serialized)
                            except Exception as e:
                                benchmark_result = BenchmarkResult(
                                    method=method_name, library="datason", scenario=scenario_name,
                                    mean_time=0, min_time=0, max_time=0, std_time=0,
                                    successful_runs=0, error_count=5,
                                    error_message=f"Serialization failed: {e}"
                                )
                        else:
                            # For serialization
                            benchmark_result = self._benchmark_method(func, scenario_data)
                        
                        benchmark_result.method = method_name
                        benchmark_result.library = "datason"
                        benchmark_result.scenario = scenario_name
                        scenario_results[method_name] = benchmark_result.to_dict()
                        
                    else:
                        logger.warning(f"Method {func_name} not found in datason")
                        
                except Exception as e:
                    logger.error(f"Error benchmarking {method_name}: {e}")
                    scenario_results[method_name] = {
                        'error': str(e),
                        'error_count': 5,
                        'successful_runs': 0
                    }
            
            results[scenario_name] = scenario_results
        
        return {
            'type': 'datason_api_comparison',
            'description': 'Performance comparison across DataSON API methods',
            'results': results,
            'methods_tested': list(self.datason_methods.keys()),
            'metadata': self.metadata
        }
    
    def run_competitive_analysis(self, test_scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fair competitive analysis against other libraries
        """
        logger.info("🏁 Running Competitive Analysis")
        
        import datason
        results = {}
        
        for scenario_name, scenario_data in test_scenarios.items():
            logger.info(f"  ⚡ Testing scenario: {scenario_name}")
            scenario_results = {
                'serialization': {},
                'deserialization': {},
                'output_size': {},
                'description': f'Competitive analysis for {scenario_name}'
            }
            
            # Test DataSON baseline
            datason_serialize_time = None
            datason_serialized = None
            
            try:
                datason_result = self._benchmark_method(datason.serialize, scenario_data)
                datason_result.method = "serialize"
                datason_result.library = "datason" 
                datason_result.scenario = scenario_name
                scenario_results['serialization']['datason'] = datason_result.to_dict()
                datason_serialize_time = datason_result.mean_time
                datason_serialized = datason.serialize(scenario_data)
                scenario_results['output_size']['datason'] = {
                    'size': len(datason_serialized) if datason_serialized else 0,
                    'size_type': 'string chars',
                    'supports_binary': True
                }
            except Exception as e:
                scenario_results['serialization']['datason'] = {
                    'error': str(e),
                    'error_count': 5
                }
            
            # Test DataSON deserialization if serialization worked
            if datason_serialized:
                try:
                    deserialize_result = self._benchmark_method(datason.deserialize, datason_serialized)
                    deserialize_result.method = "deserialize"
                    deserialize_result.library = "datason"
                    deserialize_result.scenario = scenario_name
                    scenario_results['deserialization']['datason'] = deserialize_result.to_dict()
                except Exception as e:
                    scenario_results['deserialization']['datason'] = {
                        'error': str(e),
                        'error_count': 5
                    }
            
            # Test competitors
            for competitor_name, competitor_info in self.competitors.items():
                competitor_lib = self._import_competitor(competitor_name)
                if not competitor_lib:
                    continue
                
                # Test serialization
                try:
                    if competitor_name == 'orjson':
                        serialize_func = competitor_lib.dumps
                    elif competitor_name == 'ujson':
                        serialize_func = competitor_lib.dumps
                    elif competitor_name == 'json':
                        serialize_func = competitor_lib.dumps
                    elif competitor_name == 'pickle':
                        serialize_func = competitor_lib.dumps
                    elif competitor_name == 'msgpack':
                        serialize_func = competitor_lib.packb
                    else:
                        continue
                    
                    competitor_result = self._benchmark_method(serialize_func, scenario_data)
                    competitor_result.method = "serialize"
                    competitor_result.library = competitor_name
                    competitor_result.scenario = scenario_name
                    scenario_results['serialization'][competitor_name] = competitor_result.to_dict()
                    
                    # Measure output size
                    try:
                        competitor_output = serialize_func(scenario_data)
                        scenario_results['output_size'][competitor_name] = {
                            'size': len(competitor_output),
                            'size_type': 'bytes' if isinstance(competitor_output, bytes) else 'string chars',
                            'supports_binary': isinstance(competitor_output, bytes)
                        }
                        
                        # Test deserialization
                        if competitor_name == 'orjson':
                            deserialize_func = competitor_lib.loads
                        elif competitor_name == 'ujson':
                            deserialize_func = competitor_lib.loads
                        elif competitor_name == 'json':
                            deserialize_func = competitor_lib.loads
                        elif competitor_name == 'pickle':
                            deserialize_func = competitor_lib.loads
                        elif competitor_name == 'msgpack':
                            deserialize_func = competitor_lib.unpackb
                        else:
                            continue
                        
                        deserialize_result = self._benchmark_method(deserialize_func, competitor_output)
                        deserialize_result.method = "deserialize"
                        deserialize_result.library = competitor_name
                        deserialize_result.scenario = scenario_name
                        scenario_results['deserialization'][competitor_name] = deserialize_result.to_dict()
                        
                    except Exception as e:
                        scenario_results['output_size'][competitor_name] = {
                            'error': str(e)
                        }
                        scenario_results['deserialization'][competitor_name] = {
                            'error': str(e),
                            'error_count': 5
                        }
                        
                except Exception as e:
                    scenario_results['serialization'][competitor_name] = {
                        'error': str(e),
                        'error_count': 5
                    }
            
            # Add competitors tested metadata
            scenario_results['competitors_tested'] = ['datason'] + [
                name for name in self.competitors.keys() 
                if self._import_competitor(name) is not None
            ]
            
            results[scenario_name] = scenario_results
        
        return {
            'type': 'competitive_analysis',
            'description': 'DataSON vs other serialization libraries',
            'results': results,
            'competitors_available': scenario_results.get('competitors_tested', []),
            'metadata': self.metadata
        }
    
    def run_version_evolution_tracking(self, test_scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track DataSON performance evolution (requires baseline data)
        """
        logger.info("📈 Running Version Evolution Analysis")
        
        # This would compare current results against historical baselines
        # For now, record current version performance for future comparison
        import datason
        
        current_performance = {}
        
        for scenario_name, scenario_data in test_scenarios.items():
            logger.info(f"  📊 Baseline scenario: {scenario_name}")
            
            try:
                # Test key DataSON methods for version tracking
                serialize_result = self._benchmark_method(datason.serialize, scenario_data)
                serialize_result.method = "serialize"
                serialize_result.library = "datason"
                serialize_result.scenario = scenario_name
                
                # Test deserialization
                serialized = datason.serialize(scenario_data)
                deserialize_result = self._benchmark_method(datason.deserialize, serialized)
                deserialize_result.method = "deserialize"
                deserialize_result.library = "datason"
                deserialize_result.scenario = scenario_name
                
                current_performance[scenario_name] = {
                    'serialize': serialize_result.to_dict(),
                    'deserialize': deserialize_result.to_dict(),
                    'output_size': len(serialized) if serialized else 0
                }
                
            except Exception as e:
                current_performance[scenario_name] = {
                    'error': str(e)
                }
        
        return {
            'type': 'version_evolution',
            'description': 'DataSON performance tracking across versions',
            'current_version': self._get_datason_version(),
            'baseline_performance': current_performance,
            'metadata': self.metadata,
            'note': 'This establishes baseline for future version comparisons'
        }
    
    def generate_test_scenarios(self) -> Dict[str, Any]:
        """Generate test scenarios focused on real-world use cases"""
        scenarios = {
            'api_response_processing': {
                'users': [
                    {'id': i, 'name': f'user_{i}', 'email': f'user_{i}@example.com', 
                     'created': f'2024-{(i % 12) + 1:02d}-01'}
                    for i in range(10)
                ],
                'metadata': {'total': 10, 'page': 1, 'per_page': 10},
                'timestamp': '2024-08-26T10:30:00Z'
            },
            
            'secure_data_storage': {
                'user_profile': {
                    'username': 'alice_secure',
                    'password': 'secret_password_123',
                    'ssn': '123-45-6789',
                    'credit_card': '4532-1234-5678-9012',
                    'api_keys': {
                        'stripe': 'sk_test_123456789',
                        'aws': 'AKIA123456789EXAMPLE'
                    }
                },
                'session_data': {
                    'session_id': 'sess_abc123def456',
                    'user_id': 12345,
                    'permissions': ['read', 'write', 'admin']
                }
            },
            
            'ml_model_serialization': {
                'model_config': {
                    'architecture': 'transformer',
                    'layers': [
                        {'type': 'embedding', 'dim': 512, 'vocab_size': 50000},
                        {'type': 'attention', 'heads': 8, 'dim': 512},
                        {'type': 'feedforward', 'dim': 2048}
                    ],
                    'weights': [0.1 * i for i in range(100)],  # Simulate model weights
                    'training_metadata': {
                        'epochs': 50,
                        'learning_rate': 0.001,
                        'batch_size': 32
                    }
                }
            },
            
            'mobile_app_sync': {
                'sync_data': {
                    'user_settings': {
                        'theme': 'dark',
                        'notifications': True,
                        'language': 'en'
                    },
                    'offline_actions': [
                        {'action': 'create_note', 'timestamp': 1692345678, 'data': {'title': 'Note 1', 'content': 'Content'}},
                        {'action': 'update_profile', 'timestamp': 1692345679, 'data': {'name': 'Updated Name'}}
                    ],
                    'cached_data': {f'cache_key_{i}': f'cached_value_{i}' for i in range(20)}
                }
            },
            
            'web_service_integration': {
                'request_payload': {
                    'method': 'POST',
                    'endpoint': '/api/v1/process',
                    'headers': {'Content-Type': 'application/json', 'Authorization': 'Bearer token123'},
                    'body': {
                        'transaction_id': 'txn_abc123',
                        'amount': 99.99,
                        'currency': 'USD',
                        'items': [
                            {'id': 'item_1', 'name': 'Product A', 'price': 49.99, 'qty': 1},
                            {'id': 'item_2', 'name': 'Product B', 'price': 49.99, 'qty': 1}
                        ]
                    }
                }
            }
        }
        
        return scenarios
    
    def run_comprehensive_suite(self, suite_type: str = "comprehensive") -> Dict[str, Any]:
        """Run all benchmark categories"""
        logger.info(f"🚀 Starting Comprehensive Benchmark Suite: {suite_type}")
        
        test_scenarios = self.generate_test_scenarios()
        
        results = {
            'suite_type': suite_type,
            'metadata': self.metadata,
            'scenarios': list(test_scenarios.keys()),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # 1. DataSON API Method Comparison
            api_comparison = self.run_datason_api_comparison(test_scenarios)
            results['datason_api_comparison'] = api_comparison
            logger.info("✅ DataSON API comparison completed")
        except Exception as e:
            logger.error(f"❌ DataSON API comparison failed: {e}")
            results['datason_api_comparison'] = {'error': str(e)}
        
        try:
            # 2. Competitive Analysis
            competitive = self.run_competitive_analysis(test_scenarios)
            results['competitive_analysis'] = competitive
            logger.info("✅ Competitive analysis completed")
        except Exception as e:
            logger.error(f"❌ Competitive analysis failed: {e}")
            results['competitive_analysis'] = {'error': str(e)}
        
        try:
            # 3. Version Evolution Tracking
            version_evolution = self.run_version_evolution_tracking(test_scenarios)
            results['version_evolution'] = version_evolution
            logger.info("✅ Version evolution tracking completed")
        except Exception as e:
            logger.error(f"❌ Version evolution tracking failed: {e}")
            results['version_evolution'] = {'error': str(e)}
        
        return results

def main():
    """Main entry point for improved benchmark runner"""
    parser = argparse.ArgumentParser(description='Improved DataSON Benchmark Runner')
    parser.add_argument('--suite-type', choices=['api_modes', 'competitive', 'versions', 'comprehensive'], 
                        default='comprehensive', help='Type of benchmark suite to run')
    parser.add_argument('--output-dir', default='data/results', help='Output directory for results')
    parser.add_argument('--output-file', help='Specific output filename (optional)')
    parser.add_argument('--generate-report', action='store_true', help='Generate HTML report')
    
    args = parser.parse_args()
    
    # Create runner
    runner = ImprovedBenchmarkRunner(output_dir=args.output_dir)
    
    # Run benchmarks
    results = runner.run_comprehensive_suite(suite_type=args.suite_type)
    
    # Save results
    if args.output_file:
        output_file = Path(args.output_dir) / args.output_file
    else:
        timestamp = int(time.time())
        output_file = Path(args.output_dir) / f"improved_{args.suite_type}_{timestamp}.json"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"📄 Results saved to: {output_file}")
    
    # Generate report if requested
    if args.generate_report:
        logger.info("🎨 Generating HTML report...")
        # This would call the improved report generator
        # generate_improved_report(output_file)
    
    return output_file

if __name__ == '__main__':
    main()
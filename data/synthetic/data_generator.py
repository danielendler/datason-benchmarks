"""
Automated synthetic data generation for DataSON benchmarks.
Creates realistic test datasets for various scenarios.
"""
import datason
import random
import string
import datetime
from typing import Dict, List, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from faker import Faker
import warnings
from decimal import Decimal
import uuid
from datetime import timezone

# Suppress DataSON datetime parsing warnings for cleaner logs  
# These warnings are caused by overly aggressive datetime detection in DataSON v0.9.0
# See DATASON_DATETIME_PARSING_ISSUES.md for detailed analysis
warnings.filterwarnings('ignore', message='Failed to parse datetime string', module='datason')

fake = Faker()

@dataclass
class DataGenerationConfig:
    """Configuration for data generation scenarios"""
    name: str
    description: str
    size_range: tuple  # (min_size, max_size) in bytes
    complexity: str    # 'simple', 'realistic', 'complex'
    data_type: str     # 'api_data', 'scientific_data', 'complex_objects', 'edge_cases'
    count: int         # Number of samples to generate

class SyntheticDataGenerator:
    """
    Generates realistic synthetic data for benchmarking.
    Focuses on practical scenarios without complex infrastructure.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize with reproducible seed for consistent benchmarks"""
        random.seed(seed)
        np.random.seed(seed)
        fake.seed_instance(seed)
        
        # Define realistic test scenarios from strategy
        self.scenarios = {
            'api_fast': DataGenerationConfig(
                name='api_fast',
                description='Fast API responses - small objects',
                size_range=(100, 1000),  # 100B - 1KB
                complexity='simple',
                data_type='api_data',
                count=1000
            ),
            'ml_training': DataGenerationConfig(
                name='ml_training',
                description='ML model serialization',
                size_range=(100000, 1000000),  # 100KB - 1MB
                complexity='realistic',
                data_type='scientific_data',
                count=100
            ),
            'secure_storage': DataGenerationConfig(
                name='secure_storage',
                description='Secure data storage - nested objects',
                size_range=(1000, 10000),  # 1KB - 10KB
                complexity='complex',
                data_type='complex_objects',
                count=500
            ),
            'large_data': DataGenerationConfig(
                name='large_data',
                description='Large dataset handling',
                size_range=(1000000, 10000000),  # 1MB - 10MB
                complexity='realistic',
                data_type='scientific_data',
                count=50
            ),
            'edge_cases': DataGenerationConfig(
                name='edge_cases',
                description='Boundary conditions and stress tests',
                size_range=(10, 100000),  # Variable sizes
                complexity='complex',
                data_type='edge_cases',
                count=200
            )
        }
    
    def generate_api_data(self, target_size: int) -> Dict[str, Any]:
        """Generate API-style data (user profiles, product catalogs, etc.)"""
        data_types = ['user_profile', 'product_catalog', 'order_history', 'config_data']
        data_type = random.choice(data_types)
        
        if data_type == 'user_profile':
            return {
                'id': fake.uuid4(),
                'username': fake.user_name(),
                'email': fake.email(),
                'first_name': fake.first_name(),
                'last_name': fake.last_name(),
                'created_at': fake.date_time().isoformat(),
                'profile': {
                    'bio': fake.text(max_nb_chars=random.randint(50, 200)),
                    'location': fake.city(),
                    'website': fake.url(),
                    'verified': fake.boolean(),
                },
                'preferences': {
                    'notifications': fake.boolean(),
                    'theme': random.choice(['light', 'dark', 'auto']),
                    'language': fake.language_code(),
                },
                'stats': {
                    'login_count': fake.random_int(0, 1000),
                    'last_active': fake.date_time().isoformat(),
                }
            }
        
        elif data_type == 'product_catalog':
            return {
                'product_id': fake.uuid4(),
                'name': fake.catch_phrase(),
                'description': fake.text(max_nb_chars=random.randint(100, 500)),
                'price': round(random.uniform(9.99, 999.99), 2),
                'category': fake.word(),
                'tags': [fake.word() for _ in range(random.randint(2, 8))],
                'inventory': {
                    'stock': fake.random_int(0, 1000),
                    'reserved': fake.random_int(0, 50),
                    'available': True,
                },
                'metadata': {
                    'created_at': fake.date_time().isoformat(),
                    'updated_at': fake.date_time().isoformat(),
                    'weight': round(random.uniform(0.1, 50.0), 2),
                    'dimensions': {
                        'length': round(random.uniform(1, 100), 1),
                        'width': round(random.uniform(1, 100), 1),
                        'height': round(random.uniform(1, 100), 1),
                    }
                }
            }
        
        elif data_type == 'order_history':
            return {
                'order_id': fake.uuid4(),
                'customer_id': fake.uuid4(),
                'items': [
                    {
                        'product_id': fake.uuid4(),
                        'quantity': fake.random_int(1, 5),
                        'price': round(random.uniform(10, 200), 2),
                        'discount': round(random.uniform(0, 0.3), 2),
                    }
                    for _ in range(random.randint(1, 10))
                ],
                'totals': {
                    'subtotal': round(random.uniform(50, 500), 2),
                    'tax': round(random.uniform(5, 50), 2),
                    'shipping': round(random.uniform(0, 25), 2),
                },
                'shipping_address': {
                    'street': fake.street_address(),
                    'city': fake.city(),
                    'state': fake.state(),
                    'zip_code': fake.zipcode(),
                    'country': fake.country_code(),
                },
                'status': random.choice(['pending', 'processing', 'shipped', 'delivered', 'cancelled']),
                'timestamps': {
                    'created': fake.date_time().isoformat(),
                    'updated': fake.date_time().isoformat(),
                }
            }
        
        else:  # config_data
            return {
                'app_config': {
                    'database': {
                        'host': fake.ipv4(),
                        'port': fake.random_int(1000, 9999),
                        'name': fake.word(),
                        'ssl': fake.boolean(),
                    },
                    'cache': {
                        'enabled': fake.boolean(),
                        'ttl': fake.random_int(60, 3600),
                        'size_limit': fake.random_int(100, 1000),
                    },
                    'features': {
                        flag: fake.boolean() 
                        for flag in [fake.word() for _ in range(random.randint(5, 15))]
                    },
                    'logging': {
                        'level': random.choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
                        'format': fake.sentence(),
                        'handlers': [fake.word() for _ in range(random.randint(1, 4))],
                    }
                }
            }
    
    def generate_scientific_data(self, target_size: int) -> Dict[str, Any]:
        """Generate NumPy/Pandas data common in ML"""
        data_types = ['time_series', 'feature_matrix', 'model_weights', 'dataset']
        data_type = random.choice(data_types)
        
        # Estimate array size based on target_size
        # Rough estimate: 8 bytes per float64
        array_size = max(10, target_size // 8)
        
        if data_type == 'time_series':
            n_points = min(array_size // 3, 10000)  # 3 columns: timestamp, value, category
            timestamps = pd.date_range(
                start=fake.date_time(),
                periods=n_points,
                freq='1h'
            )
            
            return {
                'metadata': {
                    'type': 'time_series',
                    'source': fake.company(),
                    'created': fake.date_time().isoformat(),
                    'columns': ['timestamp', 'value', 'category'],
                },
                'data': {
                    'timestamps': [ts.isoformat() for ts in timestamps],
                    'values': np.random.normal(100, 15, n_points).tolist(),
                    'categories': [
                        random.choice(['A', 'B', 'C', 'D']) 
                        for _ in range(n_points)
                    ],
                },
                'stats': {
                    'mean': float(np.random.normal(100, 15)),
                    'std': float(np.random.normal(15, 2)),
                    'min': float(np.random.normal(50, 10)),
                    'max': float(np.random.normal(150, 10)),
                }
            }
        
        elif data_type == 'feature_matrix':
            n_samples = min(max(10, array_size // 50), 1000)
            n_features = min(50, array_size // n_samples)
            
            return {
                'metadata': {
                    'type': 'feature_matrix',
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'created': fake.date_time().isoformat(),
                },
                'features': np.random.randn(n_samples, n_features).tolist(),
                'labels': np.random.randint(0, 5, n_samples).tolist(),
                'feature_names': [f'feature_{i}' for i in range(n_features)],
                'preprocessing': {
                    'scaled': fake.boolean(),
                    'normalized': fake.boolean(),
                    'missing_values_filled': fake.boolean(),
                }
            }
        
        elif data_type == 'model_weights':
            # Simulate neural network weights
            layers = random.randint(3, 8)
            weights = {}
            biases = {}
            
            for i in range(layers):
                layer_size = random.randint(10, min(200, array_size // (layers * 2)))
                next_layer_size = random.randint(10, min(200, array_size // (layers * 2)))
                
                weights[f'layer_{i}'] = np.random.randn(layer_size, next_layer_size).tolist()
                biases[f'layer_{i}'] = np.random.randn(next_layer_size).tolist()
            
            return {
                'metadata': {
                    'type': 'model_weights',
                    'architecture': 'feedforward',
                    'layers': layers,
                    'created': fake.date_time().isoformat(),
                    'training_info': {
                        'epochs': fake.random_int(10, 1000),
                        'batch_size': random.choice([16, 32, 64, 128]),
                        'learning_rate': float(np.random.uniform(0.001, 0.1)),
                    }
                },
                'weights': weights,
                'biases': biases,
                'metrics': {
                    'accuracy': float(np.random.uniform(0.7, 0.95)),
                    'loss': float(np.random.uniform(0.1, 2.0)),
                    'val_accuracy': float(np.random.uniform(0.6, 0.9)),
                }
            }
        
        else:  # dataset
            n_rows = min(max(10, array_size // 20), 5000)
            n_cols = min(20, array_size // n_rows)
            
            # Generate mixed data types
            dataset = {}
            for col in range(n_cols):
                col_type = random.choice(['numeric', 'categorical', 'datetime', 'text'])
                
                if col_type == 'numeric':
                    dataset[f'numeric_{col}'] = np.random.randn(n_rows).tolist()
                elif col_type == 'categorical':
                    categories = [fake.word() for _ in range(random.randint(3, 10))]
                    dataset[f'category_{col}'] = [
                        random.choice(categories) for _ in range(n_rows)
                    ]
                elif col_type == 'datetime':
                    start_date = fake.date_time()
                    dataset[f'datetime_{col}'] = [
                        fake.date_time_between(start_date=start_date).isoformat()
                        for _ in range(n_rows)
                    ]
                else:  # text
                    dataset[f'text_{col}'] = [
                        fake.sentence() for _ in range(n_rows)
                    ]
            
            return {
                'metadata': {
                    'type': 'mixed_dataset',
                    'n_rows': n_rows,
                    'n_columns': n_cols,
                    'created': fake.date_time().isoformat(),
                },
                'data': dataset,
                'schema': {
                    col: type(values[0]).__name__ if values else 'unknown'
                    for col, values in dataset.items()
                }
            }
    
    def generate_complex_objects(self, target_size: int) -> Dict[str, Any]:
        """Generate nested Python objects (config files, nested dicts, class instances)"""
        object_types = ['nested_config', 'hierarchical_data', 'graph_structure', 'mixed_types']
        object_type = random.choice(object_types)
        
        if object_type == 'nested_config':
            # Deep configuration structure
            def generate_nested_dict(depth: int, max_depth: int = 5) -> Dict[str, Any]:
                if depth >= max_depth:
                    return {
                        fake.word(): random.choice([
                            fake.word(), fake.random_int(1, 1000), fake.boolean(),
                            fake.url(), fake.email()
                        ])
                        for _ in range(random.randint(2, 8))
                    }
                
                result = {}
                for _ in range(random.randint(2, 6)):
                    key = fake.word()
                    if random.random() < 0.6:  # 60% chance of nesting
                        result[key] = generate_nested_dict(depth + 1, max_depth)
                    else:
                        result[key] = random.choice([
                            fake.sentence(), fake.random_int(1, 1000), fake.boolean(),
                            [fake.word() for _ in range(random.randint(1, 5))],
                            fake.url()
                        ])
                return result
            
            return {
                'config_version': '2.1.0',
                'environment': random.choice(['development', 'staging', 'production']),
                'services': generate_nested_dict(0),
                'metadata': {
                    'created_by': fake.name(),
                    'created_at': fake.date_time().isoformat(),
                    'last_modified': fake.date_time().isoformat(),
                    'checksum': fake.sha256(),
                }
            }
        
        elif object_type == 'hierarchical_data':
            # Organization or file system structure
            def generate_hierarchy(depth: int, max_depth: int = 4) -> Dict[str, Any]:
                if depth >= max_depth:
                    return {
                        'id': fake.uuid4(),
                        'name': fake.word(),
                        'type': 'leaf',
                        'size': fake.random_int(1, 10000),
                        'created': fake.date_time().isoformat(),
                    }
                
                children = []
                for _ in range(random.randint(1, 5)):
                    children.append(generate_hierarchy(depth + 1, max_depth))
                
                return {
                    'id': fake.uuid4(),
                    'name': fake.word(),
                    'type': 'node',
                    'children': children,
                    'metadata': {
                        'owner': fake.name(),
                        'permissions': random.choice(['read', 'write', 'admin']),
                        'created': fake.date_time().isoformat(),
                    }
                }
            
            return {
                'root': generate_hierarchy(0),
                'stats': {
                    'total_nodes': fake.random_int(50, 500),
                    'max_depth': fake.random_int(3, 8),
                    'created': fake.date_time().isoformat(),
                }
            }
        
        elif object_type == 'graph_structure':
            # Network or relationship data
            n_nodes = random.randint(10, 100)
            nodes = [
                {
                    'id': i,
                    'label': fake.word(),
                    'type': random.choice(['person', 'organization', 'location', 'event']),
                    'properties': {
                        'created': fake.date_time().isoformat(),
                        'weight': random.uniform(0, 1),
                        'active': fake.boolean(),
                    }
                }
                for i in range(n_nodes)
            ]
            
            # Generate edges
            n_edges = random.randint(n_nodes, n_nodes * 3)
            edges = []
            for _ in range(n_edges):
                source = random.randint(0, n_nodes - 1)
                target = random.randint(0, n_nodes - 1)
                if source != target:  # No self-loops
                    edges.append({
                        'source': source,
                        'target': target,
                        'type': random.choice(['follows', 'likes', 'works_with', 'located_in']),
                        'weight': random.uniform(0, 1),
                        'created': fake.date_time().isoformat(),
                    })
            
            return {
                'graph': {
                    'nodes': nodes,
                    'edges': edges,
                },
                'metadata': {
                    'type': 'social_network',
                    'created': fake.date_time().isoformat(),
                    'stats': {
                        'node_count': len(nodes),
                        'edge_count': len(edges),
                        'density': len(edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0,
                    }
                }
            }
        
        else:  # mixed_types
            # Mix of different Python types and structures
            return {
                'strings': {
                    'short': fake.word(),
                    'medium': fake.sentence(),
                    'long': fake.text(),
                    'unicode': '🌟 Unicode: ' + fake.sentence(),
                },
                'numbers': {
                    'integer': fake.random_int(-1000000, 1000000),
                    'float': random.uniform(-1000.0, 1000.0),
                    'large_int': fake.random_int(10**10, 10**15),
                    'small_float': random.uniform(-1.0, 1.0),
                },
                'collections': {
                    'list': [fake.word() for _ in range(random.randint(5, 20))],
                    'nested_list': [
                        [fake.random_int(1, 100) for _ in range(random.randint(2, 10))]
                        for _ in range(random.randint(3, 8))
                    ],
                    'mixed_list': [
                        fake.word(), fake.random_int(1, 100), fake.boolean(),
                        {'nested': fake.sentence()}, [1, 2, 3]
                    ],
                },
                'temporal': {
                    'datetime': fake.date_time().isoformat(),
                    'date': fake.date_time().date().isoformat(),
                    'timestamp': fake.unix_time(),
                },
                'identifiers': {
                    'uuid': fake.uuid4(),
                    'email': fake.email(),
                    'url': fake.url(),
                    'ip': fake.ipv4(),
                },
                'boolean_flags': {
                    flag: fake.boolean()
                    for flag in [fake.word() for _ in range(random.randint(5, 15))]
                }
            }
    
    def generate_edge_cases(self, target_size: int) -> Dict[str, Any]:
        """Generate boundary conditions and edge cases"""
        edge_types = ['size_extremes', 'special_values', 'unicode_stress', 'deep_nesting', 'circular_refs']
        edge_type = random.choice(edge_types)
        
        if edge_type == 'size_extremes':
            if target_size < 1000:  # Small edge cases
                return {
                    'empty': {},
                    'single_key': {'key': 'value'},
                    'minimal_list': [1],
                    'tiny_string': 'x',
                    'zero': 0,
                    'none_value': None,
                }
            else:  # Large edge cases
                large_string = 'x' * (target_size // 2)
                large_list = list(range(target_size // 100))
                return {
                    'large_string': large_string,
                    'large_list': large_list,
                    'large_dict': {f'key_{i}': f'value_{i}' for i in range(target_size // 200)},
                    'repeated_pattern': [{'pattern': 'repeat'} for _ in range(target_size // 100)],
                }
        
        elif edge_type == 'special_values':
            return {
                'numeric_extremes': {
                    'max_int': 2**63 - 1,
                    'min_int': -2**63,
                    'max_float': float('inf'),
                    'min_float': float('-inf'),
                    'nan': float('nan'),
                    'zero': 0,
                    'negative_zero': -0.0,
                },
                'string_specials': {
                    'empty': '',
                    'whitespace': '   \t\n  ',
                    'quotes': '"\'`',
                    'backslashes': '\\\\\\',
                    'control_chars': '\x00\x01\x02',
                },
                'container_specials': {
                    'empty_list': [],
                    'empty_dict': {},
                    'none_list': [None, None, None],
                    'mixed_none': [1, None, 'text', None],
                }
            }
        
        elif edge_type == 'unicode_stress':
            return {
                'unicode_variants': {
                    'emoji': '🌟🚀🎉💡🔥⭐🌈🦄',
                    'chinese': '这是一个测试字符串',
                    'arabic': 'هذا نص تجريبي',
                    'russian': 'Это тестовая строка',
                    'japanese': 'これはテスト文字列です',
                    'symbols': '∀∃∈∉∪∩⊂⊃⊆⊇',
                    'mixed': '🌟 Hello 世界 مرحبا Мир こんにちは ∞',
                },
                'encoding_edge_cases': {
                    'high_unicode': '\U0001F600\U0001F601\U0001F602',
                    'combining_chars': 'e\u0301',  # é using combining accent
                    'zero_width': 'test\u200Btext',  # zero-width space
                },
            }
        
        elif edge_type == 'deep_nesting':
            # Create deeply nested structure
            def create_deep_nesting(depth: int) -> Dict[str, Any]:
                if depth <= 0:
                    return {'final': 'value', 'depth': depth}
                return {
                    'level': depth,
                    'data': fake.sentence(),
                    'nested': create_deep_nesting(depth - 1)
                }
            
            # Keep depth well under DataSON's 50 limit to account for metadata wrappers
            max_depth = min(35, target_size // 100)  # Safe depth limit for DataSON
            return {
                'deep_dict': create_deep_nesting(max_depth),
                'deep_list': self._create_deep_list(max_depth),
                'mixed_deep': {
                    'dict_in_list': [create_deep_nesting(max_depth // 2)],
                    'list_in_dict': {'nested': self._create_deep_list(max_depth // 2)},
                }
            }
        
        else:  # circular_refs simulation (can't actually create in JSON)
            # Simulate circular reference patterns with IDs
            return {
                'simulated_circular': {
                    'node_1': {'id': 1, 'refs': [2, 3], 'data': fake.sentence()},
                    'node_2': {'id': 2, 'refs': [1, 3], 'data': fake.sentence()},
                    'node_3': {'id': 3, 'refs': [1, 2], 'data': fake.sentence()},
                },
                'self_reference_sim': {
                    'id': 'root',
                    'parent_id': None,
                    'self_ref_id': 'root',
                    'children_ids': ['child_1', 'child_2'],
                },
                'mutual_reference': {
                    'a': {'id': 'a', 'ref': 'b', 'data': fake.sentence()},
                    'b': {'id': 'b', 'ref': 'a', 'data': fake.sentence()},
                }
            }
    
    def _create_deep_list(self, depth: int) -> List[Any]:
        """Helper to create deeply nested list"""
        if depth <= 0:
            return ['final_value']
        return [f'level_{depth}', self._create_deep_list(depth - 1)]
    
    def generate_scenario_data(self, scenario_name: str, count: int = None) -> List[Dict[str, Any]]:
        """Generate data for a specific scenario"""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        config = self.scenarios[scenario_name]
        count = count or config.count
        
        generated_data = []
        for i in range(count):
            # Target size with some variance
            min_size, max_size = config.size_range
            target_size = random.randint(min_size, max_size)
            
            # Generate data based on type
            if config.data_type == 'api_data':
                data = self.generate_api_data(target_size)
            elif config.data_type == 'scientific_data':
                data = self.generate_scientific_data(target_size)
            elif config.data_type == 'complex_objects':
                data = self.generate_complex_objects(target_size)
            elif config.data_type == 'edge_cases':
                data = self.generate_edge_cases(target_size)
            else:
                raise ValueError(f"Unknown data type: {config.data_type}")
            
            # Add metadata
            data_with_meta = {
                '_metadata': {
                    'scenario': scenario_name,
                    'index': i,
                    'target_size': target_size,
                    'generated_at': datetime.datetime.now().isoformat(),
                    'generator_version': '1.0.0',
                },
                'data': data
            }
            
            generated_data.append(data_with_meta)
        
        return generated_data
    
    def generate_all_scenarios(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate data for all scenarios"""
        all_data = {}
        
        for scenario_name in self.scenarios:
            print(f"Generating data for scenario: {scenario_name}")
            all_data[scenario_name] = self.generate_scenario_data(scenario_name)
        
        return all_data
    
    def save_scenario_data(self, scenario_name: str, output_dir: str = 'data/synthetic'):
        """Save generated data for a scenario to file"""
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        data = self.generate_scenario_data(scenario_name)
        
        # Save as JSON using DataSON for dogfooding
        output_file = os.path.join(output_dir, f'{scenario_name}_data.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            # Use DataSON's JSON-compatible function for true JSON output
            json_string = datason.dumps_json(data)
            f.write(json_string)
        
        print(f"Saved {len(data)} samples for scenario '{scenario_name}' to {output_file}")
        return output_file
    
    def save_all_scenarios(self, output_dir: str = 'data/synthetic'):
        """Save all scenario data to files"""
        import os
        
        generated_files = []
        
        for scenario_name in self.scenarios:
            file_path = self.save_scenario_data(scenario_name, output_dir)
            generated_files.append(file_path)
        
        # Save generation summary
        summary = {
            'generated_at': datetime.datetime.now().isoformat(),
            'scenarios': {
                name: asdict(config) for name, config in self.scenarios.items()
            },
            'files': generated_files,
            'total_scenarios': len(self.scenarios),
        }
        
        summary_file = os.path.join(output_dir, 'generation_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            # Use DataSON's JSON-compatible function for true JSON output
            json_string = datason.dumps_json(summary)
            f.write(json_string)
        
        print(f"Generated {len(generated_files)} scenario files")
        print(f"Summary saved to {summary_file}")
        
        return generated_files, summary_file

    def generate_phase2_security_data(self, count: int = 100) -> List[Dict[str, Any]]:
        """
        Generate data with PII and sensitive information for security testing.
        Tests DataSON's dump_secure() PII redaction capabilities.
        """
        datasets = []
        for i in range(count):
            # Create realistic user data with PII
            user_data = {
                "user_profile": {
                    "user_id": str(uuid.uuid4()),
                    "email": self.fake.email(),
                    "first_name": self.fake.first_name(),
                    "last_name": self.fake.last_name(),
                    "phone": self.fake.phone_number(),
                    "ssn": self.fake.ssn(),
                    "address": {
                        "street": self.fake.street_address(),
                        "city": self.fake.city(),
                        "state": self.fake.state(),
                        "zip_code": self.fake.zipcode(),
                        "country": self.fake.country()
                    },
                    "date_of_birth": self.fake.date_of_birth(minimum_age=18, maximum_age=80),
                    "created_at": datetime.now(timezone.utc)
                },
                "financial_data": {
                    "account_number": self.fake.bban(),
                    "routing_number": self.fake.random_number(digits=9, fix_len=True),
                    "credit_score": self.fake.random_int(min=300, max=850),
                    "annual_income": Decimal(str(self.fake.random_int(min=25000, max=200000))),
                    "bank_balance": Decimal(str(self.fake.random_int(min=100, max=50000))),
                    "transactions": [
                        {
                            "transaction_id": str(uuid.uuid4()),
                            "amount": Decimal(str(self.fake.random_int(min=1, max=1000))),
                            "merchant": self.fake.company(),
                            "timestamp": self.fake.date_time_between(start_date='-30d', end_date='now')
                        } for _ in range(self.fake.random_int(min=1, max=10))
                    ]
                },
                "medical_info": {
                    "patient_id": str(uuid.uuid4()),
                    "conditions": self.fake.words(nb=3),
                    "medications": self.fake.words(nb=2),
                    "doctor_notes": self.fake.text(max_nb_chars=200),
                    "insurance_number": self.fake.random_number(digits=12, fix_len=True)
                },
                "metadata": {
                    "source": "security_test",
                    "generated_at": datetime.now(timezone.utc),
                    "sensitivity_level": "HIGH_PII"
                }
            }
            datasets.append(user_data)
        
        return datasets

    def generate_phase2_accuracy_data(self, count: int = 50) -> List[Dict[str, Any]]:
        """
        Generate complex nested objects for accuracy testing.
        Tests load_smart() vs load_perfect() reconstruction accuracy.
        """
        datasets = []
        for i in range(count):
            # Create deeply nested object with various edge cases
            complex_data = {
                "nested_objects": {
                    "level_1": {
                        "level_2": {
                            "level_3": {
                                "level_4": {
                                    "deep_value": f"nested_value_{i}",
                                    "deep_datetime": datetime.now(timezone.utc),
                                    "deep_decimal": Decimal("123.456789"),
                                    "deep_uuid": uuid.uuid4()
                                }
                            }
                        }
                    }
                },
                "mixed_types_list": [
                    "string_item",
                    42,
                    3.14159,
                    True,
                    None,
                    datetime.now(timezone.utc),
                    uuid.uuid4(),
                    Decimal("999.999"),
                    {"nested": "dict"},
                    [1, 2, 3, {"deep": "list"}]
                ],
                "edge_cases": {
                    "empty_dict": {},
                    "empty_list": [],
                    "null_value": None,
                    "unicode_text": "Hello 世界 🌍 🚀 🎯",
                    "special_chars": "!@#$%^&*()_+-=[]{}|;':\",./<>?",
                    "very_long_string": "x" * 1000,
                    "scientific_notation": Decimal("1.23e-10"),
                    "large_number": Decimal("99999999999999999999.99999999999999")
                },
                "datetime_variants": {
                    "utc_now": datetime.now(timezone.utc),
                    "local_now": datetime.now(),
                    "past_date": datetime(2020, 1, 1, 12, 0, 0),
                    "future_date": datetime(2030, 12, 31, 23, 59, 59),
                    "microseconds": datetime.now(timezone.utc).replace(microsecond=123456)
                },
                "metadata": {
                    "test_type": "accuracy_challenge",
                    "complexity_score": 95,
                    "generated_at": datetime.now(timezone.utc)
                }
            }
            datasets.append(complex_data)
        
        return datasets

    def generate_phase2_ml_framework_data(self, count: int = 25) -> List[Dict[str, Any]]:
        """
        Generate ML framework objects for testing DataSON's ML capabilities.
        Tests integration with numpy, pandas, and optionally torch/tensorflow.
        """
        datasets = []
        for i in range(count):
            ml_data = {
                "experiment_metadata": {
                    "experiment_id": str(uuid.uuid4()),
                    "model_name": f"experiment_model_{i}",
                    "created_at": datetime.now(timezone.utc),
                    "researcher": self.fake.name(),
                    "framework": "mixed"
                },
                "basic_arrays": {
                    "feature_vector": [float(x) for x in range(10)],
                    "label_vector": [i % 2 for i in range(10)],
                    "weights": [Decimal(str(self.fake.random.random())) for _ in range(5)]
                },
                "dataset_info": {
                    "name": f"synthetic_dataset_{i}",
                    "rows": self.fake.random_int(min=1000, max=10000),
                    "features": self.fake.random_int(min=10, max=100),
                    "target_type": self.fake.random_element(["classification", "regression"]),
                    "created": datetime.now(timezone.utc)
                },
                "hyperparameters": {
                    "learning_rate": Decimal(str(self.fake.random.uniform(0.001, 0.1))),
                    "batch_size": self.fake.random_element([16, 32, 64, 128]),
                    "epochs": self.fake.random_int(min=10, max=100),
                    "dropout_rate": Decimal(str(self.fake.random.uniform(0.1, 0.5))),
                    "optimizer": self.fake.random_element(["adam", "sgd", "rmsprop"])
                },
                "performance_metrics": {
                    "train_accuracy": Decimal(str(self.fake.random.uniform(0.7, 0.99))),
                    "val_accuracy": Decimal(str(self.fake.random.uniform(0.65, 0.95))),
                    "train_loss": Decimal(str(self.fake.random.uniform(0.01, 0.5))),
                    "val_loss": Decimal(str(self.fake.random.uniform(0.02, 0.6))),
                    "f1_score": Decimal(str(self.fake.random.uniform(0.7, 0.95))),
                    "precision": Decimal(str(self.fake.random.uniform(0.7, 0.95))),
                    "recall": Decimal(str(self.fake.random.uniform(0.7, 0.95)))
                }
            }
            
            # Try to add numpy arrays if available
            try:
                import numpy as np
                ml_data["numpy_arrays"] = {
                    "feature_matrix": np.random.rand(10, 5).tolist(),  # Convert to list for JSON compatibility
                    "target_array": np.random.randint(0, 2, size=10).tolist(),
                    "weights_matrix": np.random.normal(0, 1, size=(5, 3)).tolist()
                }
            except ImportError:
                ml_data["numpy_arrays"] = "numpy_not_available"
            
            # Try to add pandas structures if available
            try:
                import pandas as pd
                df_data = {
                    "feature_1": [self.fake.random.random() for _ in range(5)],
                    "feature_2": [self.fake.random.random() for _ in range(5)],
                    "target": [self.fake.random_int(0, 1) for _ in range(5)]
                }
                ml_data["pandas_data"] = {
                    "dataframe_dict": df_data,
                    "series_data": list(pd.Series([1, 2, 3, 4, 5])),
                    "index_info": "RangeIndex(start=0, stop=5, step=1)"
                }
            except ImportError:
                ml_data["pandas_data"] = "pandas_not_available"
            
            datasets.append(ml_data)
        
        return datasets

if __name__ == '__main__':
    # Generate all synthetic data
    generator = SyntheticDataGenerator()
    files, summary = generator.save_all_scenarios()
    
    print(f"\nSynthetic data generation complete!")
    print(f"Files generated: {len(files)}")
    print(f"Summary: {summary}") 
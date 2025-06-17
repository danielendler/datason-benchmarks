#!/usr/bin/env python3
"""
Generate synthetic test data for DataSON benchmarks.
This script creates realistic datasets for automated testing.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.synthetic.data_generator import SyntheticDataGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data for benchmarks')
    parser.add_argument(
        '--scenario', 
        choices=['api_fast', 'ml_training', 'secure_storage', 'large_data', 'edge_cases', 'all'],
        default='all',
        help='Scenario to generate data for'
    )
    parser.add_argument(
        '--output-dir',
        default='data/synthetic',
        help='Output directory for generated data'
    )
    parser.add_argument(
        '--count',
        type=int,
        help='Number of samples to generate (overrides scenario default)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible generation'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize generator
    generator = SyntheticDataGenerator(seed=args.seed)
    
    print(f"Generating synthetic data with seed {args.seed}")
    print(f"Output directory: {args.output_dir}")
    
    if args.scenario == 'all':
        # Generate all scenarios
        files, summary = generator.save_all_scenarios(args.output_dir)
        print(f"\n✅ Generated {len(files)} scenario files")
    else:
        # Generate single scenario
        file_path = generator.save_scenario_data(
            args.scenario, 
            args.output_dir
        )
        print(f"\n✅ Generated data file: {file_path}")
    
    print("Synthetic data generation complete!")

if __name__ == '__main__':
    main() 
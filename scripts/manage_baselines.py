#!/usr/bin/env python3
"""
Baseline Management System
==========================

Manages multiple performance baselines for different benchmark types.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class BaselineManager:
    """Manages performance baselines for different benchmark types."""
    
    def __init__(self, results_dir: str = "data/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define baseline mappings for different benchmark types
        self.baseline_mapping = {
            "quick": "latest_competitive.json",
            "quick_enhanced": "latest_competitive.json", 
            "competitive": "latest_competitive.json",
            "enhanced_competitive": "latest_competitive.json",
            "configurations": "latest_configuration.json",
            "versioning": "latest_versioning.json",
            "complete": "latest.json",  # Use tiered baseline for complete benchmarks
            "pr_optimized": "latest.json",
        }
    
    def create_baseline(self, benchmark_type: str, source_file: str, force: bool = False):
        """Create a new baseline from a source file."""
        source_path = Path(source_file)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_file}")
        
        # Determine target baseline file
        baseline_name = self.baseline_mapping.get(benchmark_type)
        if not baseline_name:
            logger.warning(f"Unknown benchmark type '{benchmark_type}', using latest_competitive.json")
            baseline_name = "latest_competitive.json"
        
        baseline_path = self.results_dir / baseline_name
        
        # Check if baseline exists
        if baseline_path.exists() and not force:
            logger.error(f"Baseline {baseline_path} already exists. Use --force to overwrite.")
            return False
        
        # Load and validate source data
        with open(source_path, 'r') as f:
            data = json.load(f)
        
        # Add baseline metadata
        if "metadata" not in data:
            data["metadata"] = {}
        
        data["metadata"]["baseline_info"] = {
            "created_from": str(source_path),
            "benchmark_type": benchmark_type,
            "purpose": f"Baseline for {benchmark_type} benchmarks"
        }
        
        # Save baseline
        with open(baseline_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"✅ Created baseline: {baseline_path}")
        return True
    
    def list_baselines(self):
        """List all available baselines."""
        print("📊 Available Baselines:")
        print("=" * 50)
        
        for benchmark_type, baseline_file in self.baseline_mapping.items():
            baseline_path = self.results_dir / baseline_file
            if baseline_path.exists():
                # Load metadata
                try:
                    with open(baseline_path, 'r') as f:
                        data = json.load(f)
                    
                    suite_type = data.get("suite_type", "unknown")
                    timestamp = data.get("metadata", {}).get("timestamp", "unknown")
                    version = data.get("metadata", {}).get("datason_version", "unknown")
                    
                    print(f"✅ {benchmark_type:15} -> {baseline_file}")
                    print(f"   Suite: {suite_type}, Version: {version}")
                    print(f"   Created: {timestamp}")
                    print()
                except Exception as e:
                    print(f"❌ {benchmark_type:15} -> {baseline_file} (corrupted: {e})")
            else:
                print(f"❌ {benchmark_type:15} -> {baseline_file} (missing)")
    
    def update_baseline(self, benchmark_type: str, source_file: str):
        """Update an existing baseline."""
        return self.create_baseline(benchmark_type, source_file, force=True)
    
    def get_baseline_for_type(self, benchmark_type: str) -> str:
        """Get the appropriate baseline file for a benchmark type."""
        baseline_name = self.baseline_mapping.get(benchmark_type, "latest_competitive.json")
        baseline_path = self.results_dir / baseline_name
        
        if baseline_path.exists():
            return str(baseline_path)
        
        # Fallback logic
        fallbacks = ["latest_competitive.json", "latest_quick.json", "latest.json"]
        for fallback in fallbacks:
            fallback_path = self.results_dir / fallback
            if fallback_path.exists():
                logger.warning(f"Using fallback baseline {fallback} for {benchmark_type}")
                return str(fallback_path)
        
        logger.warning(f"No baseline found for {benchmark_type}")
        return ""
    
    def create_baselines_from_latest(self):
        """Create missing baselines from existing latest_* files."""
        print("🔄 Creating baselines from existing latest_* files...")
        
        # Find all latest_* files
        latest_files = list(self.results_dir.glob("latest_*.json"))
        latest_files = [f for f in latest_files if f.name != "latest.json"]  # Exclude symlink
        
        for latest_file in latest_files:
            # Determine benchmark type from filename
            name_part = latest_file.stem.replace("latest_", "")
            
            # Load file to check its format
            try:
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                
                suite_type = data.get("suite_type", "")
                
                # Determine appropriate benchmark types for this file
                if "competitive" in suite_type or "quick" in suite_type:
                    benchmark_types = ["quick", "competitive", "quick_enhanced", "enhanced_competitive"]
                elif "configuration" in suite_type:
                    benchmark_types = ["configurations"]
                elif "versioning" in suite_type:
                    benchmark_types = ["versioning"]
                else:
                    benchmark_types = [name_part]
                
                for benchmark_type in benchmark_types:
                    expected_baseline = self.baseline_mapping.get(benchmark_type)
                    if expected_baseline and expected_baseline == latest_file.name:
                        logger.info(f"✅ Baseline for {benchmark_type} already exists: {latest_file.name}")
                    elif expected_baseline:
                        baseline_path = self.results_dir / expected_baseline
                        if not baseline_path.exists():
                            self.create_baseline(benchmark_type, str(latest_file))
                
            except Exception as e:
                logger.error(f"Failed to process {latest_file}: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Manage performance baselines')
    
    parser.add_argument('--results-dir', default='data/results', 
                       help='Results directory (default: data/results)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create baseline command
    create_parser = subparsers.add_parser('create', help='Create a new baseline')
    create_parser.add_argument('benchmark_type', help='Benchmark type (e.g., quick, competitive)')
    create_parser.add_argument('source_file', help='Source benchmark file')
    create_parser.add_argument('--force', action='store_true', help='Overwrite existing baseline')
    
    # Update baseline command  
    update_parser = subparsers.add_parser('update', help='Update existing baseline')
    update_parser.add_argument('benchmark_type', help='Benchmark type')
    update_parser.add_argument('source_file', help='Source benchmark file')
    
    # List baselines command
    subparsers.add_parser('list', help='List all baselines')
    
    # Get baseline command
    get_parser = subparsers.add_parser('get', help='Get baseline file for benchmark type')
    get_parser.add_argument('benchmark_type', help='Benchmark type')
    
    # Auto-create command
    subparsers.add_parser('auto-create', help='Auto-create baselines from existing latest_* files')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = BaselineManager(args.results_dir)
    
    if args.command == 'create':
        manager.create_baseline(args.benchmark_type, args.source_file, args.force)
    elif args.command == 'update':
        manager.update_baseline(args.benchmark_type, args.source_file)
    elif args.command == 'list':
        manager.list_baselines()
    elif args.command == 'get':
        baseline_file = manager.get_baseline_for_type(args.benchmark_type)
        if baseline_file:
            print(baseline_file)
        else:
            print(f"No baseline found for {args.benchmark_type}")
    elif args.command == 'auto-create':
        manager.create_baselines_from_latest()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Establish DataSON Performance Baseline
=====================================

Runs the PR-optimized benchmark with the current/stable DataSON version
to establish a performance baseline for regression detection.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
import datason

from pr_optimized_benchmark import OptimizedPRBenchmark

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Establish DataSON performance baseline")
    parser.add_argument("--output", default="data/results/datason_baseline.json",
                       help="Output file for baseline results")
    parser.add_argument("--iterations", type=int, default=10,
                       help="Number of benchmark iterations")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing baseline")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Check if baseline already exists
    if os.path.exists(args.output) and not args.force:
        logger.error(f"Baseline already exists at {args.output}. Use --force to overwrite.")
        return 1
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import datason
        logger.info(f"Establishing baseline with DataSON {datason.__version__}")
    except ImportError:
        logger.error("DataSON not available. Please install DataSON first.")
        return 1
    
    # Run benchmark
    logger.info("Running PR-optimized benchmark to establish baseline...")
    benchmark = OptimizedPRBenchmark()
    results = benchmark.run_pr_benchmark(iterations=args.iterations)
    
    # Add metadata
    results["baseline_metadata"] = {
        "datason_version": datason.__version__,
        "established_at": datetime.now().isoformat(),
        "iterations": args.iterations,
        "purpose": "Performance regression detection baseline"
    }
    
    # Save baseline using DataSON (dogfooding our own product)
    with open(args.output, 'w') as f:
        # Dogfood DataSON for serialization (no indent param in DataSON)
        f.write(datason.dumps(results))
    
    logger.info(f"✅ Baseline established: {args.output}")
    logger.info(f"DataSON {datason.__version__} performance recorded")
    
    return 0


if __name__ == "__main__":
    exit(main()) 
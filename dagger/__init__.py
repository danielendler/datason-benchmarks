"""
DataSON Benchmark Dagger Pipeline Module

This module contains Dagger CI/CD pipelines for DataSON benchmark automation.
Replaces complex GitHub Actions YAML with testable Python code.

Available Pipelines:
- BenchmarkPipeline: Original proof-of-concept implementation
- EnhancedBenchmarkPipeline: Full feature parity with legacy workflows
"""

from .benchmark_pipeline import BenchmarkPipeline
from .enhanced_pipeline import EnhancedBenchmarkPipeline

__all__ = ["BenchmarkPipeline", "EnhancedBenchmarkPipeline"]
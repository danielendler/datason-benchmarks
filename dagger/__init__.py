"""
DataSON Benchmark Dagger Pipeline Module

This module contains Dagger CI/CD pipelines for DataSON benchmark automation.
Replaces complex GitHub Actions YAML with testable Python code.
"""

from .benchmark_pipeline import BenchmarkPipeline

__all__ = ["BenchmarkPipeline"]
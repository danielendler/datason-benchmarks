#!/usr/bin/env python3
"""
Test Setup Script for DataSON Benchmarks
=========================================

Quick verification that the benchmarking environment is working correctly.
"""

import sys
print(f"Python: {sys.version}")
print()

# Test core functionality
try:
    import json
    print("✅ JSON (stdlib) - Available")
except ImportError:
    print("❌ JSON (stdlib) - Missing (this shouldn't happen)")

# Test DataSON
try:
    import datason
    print(f"✅ DataSON - Available (v{datason.__version__})")
    datason_available = True
except ImportError:
    print("❌ DataSON - Not available (install with: pip install datason)")
    datason_available = False

# Test competitive libraries
competitors = {
    'orjson': 'Rust-based JSON library',
    'ujson': 'C-based JSON library', 
    'msgpack': 'Binary serialization format',
    'jsonpickle': 'JSON-based object serialization'
}

print("\nCompetitive Libraries:")
available_competitors = []

for lib, description in competitors.items():
    try:
        module = __import__(lib)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {lib} - Available (v{version}) - {description}")
        available_competitors.append(lib)
    except ImportError:
        print(f"⚠️ {lib} - Not available - {description}")

# Test benchmark imports
print("\nBenchmark Components:")
try:
    from competitors.adapter_registry import CompetitorRegistry
    registry = CompetitorRegistry()
    registry_competitors = registry.list_available_names()
    print(f"✅ Competitor Registry - {len(registry_competitors)} competitors available: {registry_competitors}")
except Exception as e:
    print(f"❌ Competitor Registry - Error: {e}")

try:
    from benchmarks.competitive.competitive_suite import CompetitiveBenchmarkSuite
    print("✅ Competitive Benchmark Suite - Available")
except Exception as e:
    print(f"❌ Competitive Benchmark Suite - Error: {e}")

try:
    from benchmarks.configurations.config_suite import ConfigurationBenchmarkSuite
    print("✅ Configuration Benchmark Suite - Available")
except Exception as e:
    print(f"❌ Configuration Benchmark Suite - Error: {e}")

# Summary
print("\n" + "="*50)
print("SETUP SUMMARY")
print("="*50)

if datason_available:
    print("✅ Core setup complete - DataSON available")
else:
    print("❌ Missing DataSON - install with: pip install datason")

if available_competitors:
    print(f"✅ {len(available_competitors)} competitive libraries available")
else:
    print("⚠️ No competitive libraries - install with: pip install orjson ujson msgpack jsonpickle")

print(f"\nRecommended next step:")
if datason_available and available_competitors:
    print("🚀 Run: python scripts/run_benchmarks.py --quick")
elif datason_available:
    print("📦 Install competitors: pip install orjson ujson msgpack jsonpickle")
else:
    print("📦 Install DataSON: pip install datason")

print("\n🎯 For detailed setup instructions, see: docs/setup.md") 
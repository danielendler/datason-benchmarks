# DataSON API Performance Guide

> **Comprehensive guide to DataSON's different API levels and their performance characteristics**

## Overview

DataSON provides multiple API variants, each optimized for different use cases and performance requirements. This guide helps you choose the right API variant based on your specific needs.

## 🚀 API Variants Performance Matrix

### Performance Categories

Our benchmarking system categorizes APIs into performance tiers:

- **🥇 Fastest**: Top-tier performance, minimal overhead
- **🚀 High Performance**: Excellent speed with some additional features  
- **⚖️ Balanced**: Good balance of features and performance
- **🔧 Feature-Rich**: Advanced features with performance trade-offs

### API Variants Overview

| API Variant | Performance Category | Primary Use Case | Performance Range |
|-------------|---------------------|------------------|-------------------|
| `datason_fast` | 🥇 Fastest | High-throughput systems | 0.3-0.5ms |
| `datason_perfect` | 🥇 Fastest | Critical data integrity | 0.1-0.2ms |
| `datason_ml` | ⚖️ Balanced | Machine learning workflows | 0.4-0.5ms |
| `datason_smart` | ⚖️ Balanced | Type preservation | 0.4-0.5ms |
| `datason_api` | 🔧 Feature-Rich | Web services & APIs | 0.4-0.5ms |
| `datason` | 🔧 Feature-Rich | General purpose | 0.5-0.6ms |
| `datason_secure` | 🔧 Feature-Rich | Security & compliance | 0.6-0.8ms |

*Performance ranges based on typical mixed workloads. Actual performance varies by data type and size.*

## 📊 Detailed API Characteristics

### 🥇 High-Performance APIs

#### `datason_fast`
- **Best for**: High-throughput systems, performance-critical applications, batch processing
- **Performance**: 🥇 Fastest in most scenarios
- **Trade-offs**: Minimal features, optimized for speed
- **Use when**: Performance is the absolute priority
- **Typical performance**: 0.3-0.5ms total serialization+deserialization

```python
import datason

# High-performance serialization
data = {"users": [{"id": i, "name": f"user_{i}"} for i in range(1000)]}
result = datason.dumps_fast(data)  # Optimized for speed
```

#### `datason_perfect`
- **Best for**: Exact round-trips, critical data integrity, schema validation, data archival
- **Performance**: 🥇 Fastest for perfect fidelity use cases
- **Trade-offs**: May have limitations on complex object types
- **Use when**: 100% accurate reconstruction is required
- **Typical performance**: 0.1-0.2ms total serialization+deserialization

```python
import datason

# Perfect fidelity serialization
critical_data = {"timestamp": datetime.now(), "precision": Decimal("99.99")}
result = datason.dumps_perfect(critical_data)  # 100% accurate reconstruction
```

### ⚖️ Balanced APIs

#### `datason_ml`
- **Best for**: Machine learning, data science, NumPy arrays, model serialization
- **Performance**: ⚖️ Balanced - optimized for ML data types
- **Features**: Enhanced NumPy/Pandas support, tensor handling
- **Use when**: Working with ML frameworks and scientific data
- **Typical performance**: 0.4-0.5ms total serialization+deserialization

```python
import datason
import numpy as np

# ML-optimized serialization
ml_data = {
    "model_params": np.array([1.0, 2.0, 3.0]),
    "training_data": pd.DataFrame({"features": [1, 2, 3], "labels": [0, 1, 0]})
}
result = datason.dumps_ml(ml_data)  # Optimized for ML workflows
```

#### `datason_smart`
- **Best for**: Type preservation, complex objects, schema inference, data migration
- **Performance**: ⚖️ Balanced - intelligent type handling
- **Features**: Advanced type inference, automatic schema detection
- **Use when**: Complex type preservation is needed
- **Typical performance**: 0.4-0.5ms total serialization+deserialization

```python
import datason

# Smart type preservation
complex_data = {
    "mixed_types": [1, "string", datetime.now(), {"nested": True}],
    "preserved": Decimal("123.456")
}
result = datason.dumps_smart(complex_data)  # Intelligent type handling
```

### 🔧 Feature-Rich APIs

#### `datason_api`
- **Best for**: REST APIs, web services, API responses, client-server communication
- **Performance**: 🔧 Feature-Rich - optimized for web compatibility
- **Features**: JSON-compatible output, web-safe encoding
- **Use when**: Building web APIs and services
- **Typical performance**: 0.4-0.5ms total serialization+deserialization

```python
import datason

# API-optimized serialization
api_response = {
    "status": "success",
    "data": {"users": [...], "pagination": {"page": 1, "total": 100}},
    "timestamp": datetime.now()
}
result = datason.dumps_api(api_response)  # Web-compatible output
```

#### `datason` (Standard)
- **Best for**: General purpose, balanced workloads, default choice
- **Performance**: 🔧 Feature-Rich - balanced features and performance
- **Features**: Complete feature set, good defaults
- **Use when**: General-purpose serialization needs
- **Typical performance**: 0.5-0.6ms total serialization+deserialization

```python
import datason

# Standard DataSON usage
general_data = {"any": "data", "works": True, "here": [1, 2, 3]}
result = datason.dumps(general_data)  # Standard balanced approach
```

#### `datason_secure`
- **Best for**: Sensitive data, PII handling, compliance requirements, data anonymization
- **Performance**: 🔧 Feature-Rich - security features add overhead
- **Features**: PII redaction, data sanitization, security scanning
- **Use when**: Handling sensitive or regulated data
- **Typical performance**: 0.6-0.8ms total serialization+deserialization

```python
import datason

# Security-enhanced serialization
sensitive_data = {
    "user_email": "user@example.com",  # Will be redacted
    "ssn": "123-45-6789",            # Will be redacted
    "public_data": {"name": "Public Info"}
}
result = datason.dumps_secure(sensitive_data)  # PII protection enabled
```

## 🎯 Performance Selection Guide

### By Use Case

#### High-Volume Processing
**Recommended**: `datason_fast`
- **Scenario**: Processing millions of records
- **Why**: Minimal overhead, maximum throughput
- **Performance gain**: Up to 2-3x faster than standard APIs

#### Machine Learning Pipelines
**Recommended**: `datason_ml`
- **Scenario**: Model training, data preprocessing, tensor operations
- **Why**: Optimized for NumPy/Pandas, scientific data types
- **Performance gain**: Specialized handling of ML data structures

#### Web APIs and Microservices
**Recommended**: `datason_api`
- **Scenario**: REST APIs, GraphQL, microservice communication
- **Why**: JSON-compatible, web-safe encoding
- **Performance gain**: Optimized for web protocols

#### Data Archival and Migration
**Recommended**: `datason_perfect`
- **Scenario**: Long-term storage, data migration, audit trails
- **Why**: Perfect fidelity, exact round-trips
- **Performance gain**: Fastest for perfect reconstruction needs

#### Financial and Healthcare
**Recommended**: `datason_secure`
- **Scenario**: PII data, financial records, healthcare information
- **Why**: Built-in security features, compliance support
- **Performance cost**: 30-50% slower but includes security scanning

### By Performance Requirements

#### Maximum Speed (< 0.3ms)
```
1. datason_perfect  (0.1-0.2ms)
2. datason_fast     (0.3-0.4ms)
```

#### Balanced Performance (0.4-0.5ms)
```
1. datason_ml       (0.4-0.5ms) - ML optimized
2. datason_smart    (0.4-0.5ms) - Type preservation
3. datason_api      (0.4-0.5ms) - Web optimized
```

#### Feature-Rich (0.5-0.8ms)
```
1. datason          (0.5-0.6ms) - General purpose
2. datason_secure   (0.6-0.8ms) - Security features
```

## 📈 Performance Benchmarking

### Methodology
Our performance testing uses:
- **Multi-tier testing**: JSON-safe, object-enhanced, ML-complex data
- **Realistic datasets**: API responses, ML data, financial records
- **Statistical rigor**: Multiple iterations, statistical analysis
- **Fair comparisons**: APIs tested only on appropriate data types

### Benchmark Results Summary

Based on our comprehensive benchmarking across different data types:

| Metric | datason_fast | datason_ml | datason_api | datason | datason_secure |
|--------|-------------|------------|-------------|---------|----------------|
| **Avg Total Time** | 0.40ms | 0.42ms | 0.44ms | 0.51ms | 0.67ms |
| **Performance Category** | 🥇 Fastest | ⚖️ Balanced | 🔧 Feature-Rich | 🔧 Feature-Rich | 🔧 Feature-Rich |
| **Best Use Case** | High-throughput | ML workflows | Web APIs | General | Security |
| **Relative Speed** | 1.0x (baseline) | 1.05x slower | 1.1x slower | 1.3x slower | 1.7x slower |

### Performance Scaling

Performance characteristics by data size:

- **Small data (< 1KB)**: All APIs perform similarly (0.1-0.2ms difference)
- **Medium data (1-100KB)**: Performance differences become apparent
- **Large data (> 100KB)**: `datason_fast` shows significant advantages
- **Complex objects**: `datason_ml` and `datason_smart` excel

## 🛠️ Implementation Recommendations

### Development Guidelines

1. **Start with `datason`** for general development
2. **Profile your specific use case** with realistic data
3. **Switch to specialized APIs** when performance or features are critical
4. **Use capability-based testing** to ensure fair comparisons

### Migration Path

```python
# 1. Start with standard API
result = datason.dumps(data)

# 2. Profile performance
import time
start = time.perf_counter()
result = datason.dumps(data)
standard_time = time.perf_counter() - start

# 3. Test specialized API
start = time.perf_counter()
result = datason.dumps_fast(data)  # or dumps_ml, dumps_api, etc.
specialized_time = time.perf_counter() - start

# 4. Compare and decide
if specialized_time < standard_time * 0.8:  # 20% improvement
    # Switch to specialized API
    pass
```

### Production Deployment

- **Monitor performance** in production environments
- **A/B test** different APIs with real workloads
- **Consider data characteristics** (size, complexity, types)
- **Factor in maintenance overhead** of using multiple APIs

## 📚 Additional Resources

- **[DataSON Main Repository](https://github.com/danielendler/datason)** - Source code and documentation
- **[Benchmark Results](https://danielendler.github.io/datason-benchmarks/)** - Live performance data
- **[API Documentation](https://datason.readthedocs.io/)** - Complete API reference
- **[Performance Testing Guide](README.md)** - How to run your own benchmarks

## 🔄 Keeping Performance Current

This guide is based on DataSON v0.12.0 benchmarks. Performance characteristics may change with new versions. 

**Stay updated**:
- Check [latest benchmark results](https://github.com/danielendler/datason-benchmarks/actions)
- Run your own benchmarks: `python scripts/run_benchmarks.py --datason-showcase`
- Monitor [release notes](https://github.com/danielendler/datason/releases) for performance changes

---

*Last updated: August 2025 | Based on DataSON v0.12.0 comprehensive benchmarks*
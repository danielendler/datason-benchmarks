# DataSON Benchmarks

> **Open source competitive benchmarking for DataSON serialization library**

[![Daily Benchmarks](https://github.com/datason/datason-benchmarks/actions/workflows/daily-benchmarks.yml/badge.svg)](https://github.com/datason/datason-benchmarks/actions/workflows/daily-benchmarks.yml)
[![PR Performance Check](https://github.com/datason/datason-benchmarks/actions/workflows/pr-performance-check.yml/badge.svg)](https://github.com/datason/datason-benchmarks/actions/workflows/pr-performance-check.yml)

## 🎯 Overview

This repository provides **transparent, reproducible benchmarks** for DataSON against major serialization libraries. Using GitHub Actions for zero-cost infrastructure, we deliver accurate competitive analysis that helps users make informed decisions.

### Key Features

- **🏆 Competitive Analysis**: Head-to-head comparison with 6-8 major serialization libraries
- **⚙️ Configuration Testing**: DataSON configuration optimization for different use cases  
- **🤖 Automated Daily Runs**: GitHub Actions automation with historical tracking
- **📊 Open Source Transparency**: Public results, reproducible methodology
- **🚀 Community Friendly**: Easy setup, contribution guidelines, free infrastructure

## 🚀 Quick Start

### Run Benchmarks Locally

```bash
# Clone the repository
git clone https://github.com/datason/datason-benchmarks.git
cd datason-benchmarks

# Install dependencies
pip install -r requirements.txt

# Quick competitive comparison (3-4 libraries)
python scripts/run_benchmarks.py --quick

# Full competitive analysis (all available libraries)
python scripts/run_benchmarks.py --competitive

# DataSON configuration testing
python scripts/run_benchmarks.py --configurations

# Complete benchmark suite
python scripts/run_benchmarks.py --all --generate-report
```

### View Latest Results

- **[Latest Benchmark Results](data/results/)** - JSON files with detailed metrics
- **[GitHub Actions](https://github.com/datason/datason-benchmarks/actions)** - Automated runs and artifacts
- **[Performance Trends](data/results/)** - Historical performance data

## 📊 Current Competitive Landscape

### Tested Libraries

| Library | Type | Why Tested | Latest Status |
|---------|------|------------|---------------|
| **DataSON** | JSON+objects | Our library | ✅ Active |
| **orjson** | JSON (Rust) | Speed benchmark standard | ✅ Available |
| **ujson** | JSON (C) | Popular drop-in replacement | ✅ Available |
| **json** | JSON (stdlib) | Baseline reference | ✅ Available |
| **pickle** | Binary objects | Python default for objects | ✅ Available |
| **jsonpickle** | JSON objects | Direct functional competitor | ✅ Available |
| **msgpack** | Binary compact | Cross-language efficiency | ✅ Available |

### Performance Summary

> **Latest benchmark results from automated daily runs**

*Results updated automatically by GitHub Actions. View [latest artifacts](https://github.com/datason/datason-benchmarks/actions) for detailed data.*

## ⚙️ Configuration Guidance

### Recommended DataSON Configurations

Based on automated testing across realistic scenarios:

| Use Case | Recommended Config | Performance Profile |
|----------|-------------------|-------------------|
| **API Responses** | `get_performance_config()` | Optimized for speed |
| **ML Model Serialization** | `get_ml_config()` | NumPy/Pandas optimized |
| **Secure Storage** | `get_strict_config()` | Maximum type preservation |
| **General Purpose** | Default settings | Balanced performance |

*Configuration recommendations updated automatically based on benchmark results.*

## 🤖 Automated Testing

### GitHub Actions Workflows

- **[PR Performance Check](.github/workflows/pr-performance-check.yml)** - Quick competitive check on pull requests
- **[Daily Benchmarks](.github/workflows/daily-benchmarks.yml)** - Comprehensive daily competitive analysis
- **Manual Triggers** - Run specific benchmark suites on demand

### Benchmark Types

1. **Quick** (~5 minutes) - Core competitors, basic scenarios
2. **Competitive** (~15 minutes) - All available libraries, realistic data
3. **Configuration** (~10 minutes) - DataSON config optimization  
4. **Complete** (~30 minutes) - Full suite with detailed analysis

## 📈 Methodology

### Fair Competition Principles

- **Realistic Data**: API responses, ML datasets, complex objects
- **Multiple Metrics**: Speed, memory usage, output size, success rate
- **Error Handling**: Graceful handling of library limitations
- **Environment Consistency**: Controlled GitHub Actions runners
- **Reproducible**: Anyone can run the same benchmarks

### Test Scenarios

1. **API Response Data** - Typical web API with metadata and arrays
2. **Simple Objects** - Basic data types for baseline comparison
3. **Nested Structures** - Complex hierarchical data
4. **DateTime Heavy** - Real-world timestamp and UUID patterns

### Metrics Collected

- **Serialization Speed** - Operations per second
- **Deserialization Speed** - Parse performance 
- **Output Size** - Bytes for storage/network efficiency
- **Memory Usage** - Peak memory consumption
- **Success Rate** - Reliability across data types

## 🏗️ Architecture

### Repository Structure

```
datason-benchmarks/
├── .github/workflows/          # GitHub Actions automation
├── benchmarks/                 # Core benchmark suites
│   ├── competitive/           # Competitor comparison tests
│   ├── configurations/        # DataSON config testing
│   └── regression/            # Performance regression detection
├── competitors/               # Competitor library adapters
├── data/                     # Test datasets and results
│   ├── results/              # Historical benchmark results  
│   ├── synthetic/            # Generated test data
│   └── configs/              # Test configurations
├── scripts/                  # Automation and utilities
└── docs/                     # Documentation and reports
```

### Core Components

- **CompetitorRegistry** - Unified interface for all serialization libraries
- **CompetitiveBenchmarkSuite** - Head-to-head performance testing
- **ConfigurationBenchmarkSuite** - DataSON optimization testing
- **ReportGenerator** - HTML and markdown report generation

## 🤝 Contributing

### Adding New Competitors

1. Create adapter in `competitors/` directory
2. Implement `CompetitorAdapter` interface
3. Add to `CompetitorRegistry`
4. Test with `python scripts/run_benchmarks.py --quick`

### Adding Test Scenarios

1. Add test data to `create_benchmark_datasets()` in competitive suite
2. Include realistic data patterns from your use case
3. Ensure data is JSON-serializable for fair comparison

### Improving Methodology

1. Review existing methodology in `docs/methodology.md` 
2. Propose changes via GitHub issues
3. Test changes with multiple benchmark runs
4. Document impact on reproducibility

## 📋 Requirements

### System Requirements

- **Python 3.8+**
- **Memory**: 2GB+ recommended for larger benchmarks
- **Time**: 5-30 minutes depending on benchmark scope

### Library Dependencies

Core dependencies automatically installed:

- `datason>=0.4.5` - The library being benchmarked
- `orjson`, `ujson`, `msgpack`, `jsonpickle` - Competitive libraries
- `numpy`, `pandas` - Realistic ML data generation
- `matplotlib`, `plotly` - Report visualization

Optional for extended testing:
- `torch` - PyTorch model serialization
- `scikit-learn` - Scikit-learn model testing

## 🔮 Roadmap

### Current Status: Phase 1 Complete ✅

- [x] Basic competitive comparison (6-8 libraries)
- [x] GitHub Actions automation
- [x] Configuration testing
- [x] Community-friendly setup

### Phase 2: Enhanced Analysis 🚧

- [ ] Memory usage profiling
- [ ] Cross-platform testing (Windows, macOS)  
- [ ] Extended ML library integration
- [ ] Performance regression detection

### Phase 3: Community Growth 📈

- [ ] User-contributed test scenarios
- [ ] Conference presentation materials
- [ ] Integration with CI/CD pipelines
- [ ] Academic research collaboration

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

This benchmarking methodology and infrastructure is open source and freely available for use by the serialization library community.

## 🙏 Acknowledgments

- **Community Contributors** - Test scenarios and improvements
- **Competitive Libraries** - orjson, ujson, msgpack, jsonpickle teams for excellent tools
- **GitHub Actions** - Free infrastructure enabling open source benchmarking
- **DataSON Users** - Feedback and real-world use cases

---

**Latest Update**: Results automatically updated by [Daily Benchmarks workflow](https://github.com/datason/datason-benchmarks/actions/workflows/daily-benchmarks.yml) 
# DataSON Benchmarks

> **Open source competitive benchmarking for DataSON serialization library**

[![Daily Benchmarks](https://github.com/datason/datason-benchmarks/actions/workflows/daily-benchmarks.yml/badge.svg)](https://github.com/datason/datason-benchmarks/actions/workflows/daily-benchmarks.yml)
[![PR Performance Check](https://github.com/datason/datason-benchmarks/actions/workflows/pr-performance-check.yml/badge.svg)](https://github.com/datason/datason-benchmarks/actions/workflows/pr-performance-check.yml)

## 🎯 Overview

This repository provides **transparent, reproducible benchmarks** for DataSON against major serialization libraries. Using GitHub Actions for zero-cost infrastructure, we deliver accurate competitive analysis and deep optimization insights that help users make informed decisions.

### Key Features

- **🏆 Competitive Analysis**: Head-to-head comparison with 6-8 major serialization libraries
- **🔧 Deep Optimization Analysis**: DataSON configuration optimization with API-level insights
- **📊 Version Evolution Tracking**: Performance analysis across DataSON versions
- **🤖 Enhanced CI/CD Integration**: Smart PR performance checks with regression detection
- **📈 Interactive Reports**: Beautiful charts and visualizations with GitHub Pages hosting
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
python scripts/run_benchmarks.py --quick --generate-report

# DataSON version evolution analysis
python scripts/run_benchmarks.py --versioning --generate-report

# Full competitive analysis (all available libraries)
python scripts/run_benchmarks.py --competitive --generate-report

# DataSON configuration optimization testing
python scripts/run_benchmarks.py --configurations --generate-report

# Complete benchmark suite with interactive reports
python scripts/run_benchmarks.py --all --generate-report
```

### View Latest Results

- **[Interactive Reports](https://datason.github.io/datason-benchmarks/)** - Live performance visualizations
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

*Results updated automatically by GitHub Actions with interactive charts. View [latest reports](https://datason.github.io/datason-benchmarks/) for detailed visualizations.*

## 🔧 Optimization Analysis

### DataSON Configuration Deep Dive

Our enhanced benchmarking system now provides **deep API analysis** of DataSON's optimization configurations:

#### Available Optimization Configs
- **`get_performance_config()`** - Speed-optimized settings (`UNIX` dates, `VALUES` orient, no type hints)
- **`get_ml_config()`** - ML-optimized settings (`UNIX_MS` dates, type hints enabled, aggressive coercion)
- **`get_strict_config()`** - Type preservation (`ISO` dates, strict coercion, complex/decimal preservation)
- **`get_api_config()`** - API-compatible settings (`ISO` dates, ASCII encoding, string UUIDs)

#### Key Performance Insights

| Dataset Type | Fastest Configuration | Performance | Version |
|-------------|----------------------|-------------|---------|
| **Basic Types** | Default | 0.009ms | v0.9.0 |
| **DateTime Heavy** | Default | 0.028ms | v0.9.0 |
| **Decimal Precision** | Default | 0.141ms | latest |
| **Large Datasets** | `get_strict_config()` | 0.978ms | latest |

#### ⚠️ Critical Finding: ML Config Performance Regression
- **Latest version**: `get_ml_config()` shows 7,800x slowdown on decimal data (1092ms vs 0.14ms)
- **v0.9.0**: Normal performance across all configs
- **Investigation**: Potential issue with ML config decimal handling in latest version

### Configuration Parameters Analysis

Our system automatically discovers and analyzes optimization parameters:

```python
# Example discovered differences between configs:
{
    "get_performance_config": {
        "date_format": "UNIX",           # vs ISO for strict/api
        "dataframe_orient": "VALUES",    # vs RECORDS for others  
        "include_type_hints": false,     # vs true for ml config
        "type_coercion": "SAFE"         # vs STRICT/AGGRESSIVE
    },
    "get_strict_config": {
        "preserve_complex": true,        # Enhanced preservation
        "preserve_decimals": true,       # Decimal accuracy
        "type_coercion": "STRICT"       # Strictest validation
    }
}
```

## 🤖 Enhanced CI/CD Integration

### Smart PR Performance Checks

Our enhanced PR workflow now provides:

- **⚡ Multi-layer Caching**: Python deps + competitor libraries for 3-5x faster runs
- **🎯 Regression Detection**: Automated performance regression analysis with severity levels
- **📊 Rich Reporting**: Interactive charts with performance analysis
- **💬 Smart Comments**: Updates existing PR comments instead of creating duplicates
- **🔍 Detailed Analysis**: Environment info, test metadata, and performance guidance

#### Performance Severity Levels
- 🚀 **Excellent**: <1.5x slower than fastest competitor
- ✅ **Good**: 1.5-2x slower  
- ⚠️ **Acceptable**: 2-5x slower
- ❌ **Concerning**: >5x slower (triggers investigation)

### GitHub Actions Workflows

- **[PR Performance Check](.github/workflows/pr-performance-check.yml)** - Enhanced competitive check with regression analysis
- **[Daily Benchmarks](.github/workflows/daily-benchmarks.yml)** - Comprehensive competitive + optimization analysis
- **Manual Triggers** - Run specific benchmark suites on demand

### CI vs Local Results

- **🔒 CI-Only Results**: Only CI-generated results are committed (prevents local machine variance)
- **📊 Interactive Reports**: Auto-generated HTML reports with Plotly.js charts
- **🌐 GitHub Pages**: Public hosting of latest benchmark reports
- **♻️ Smart Cleanup**: Automatic 30-day artifact cleanup for storage efficiency

## 📈 Enhanced Methodology

### Fair Competition Principles

- **Realistic Data**: API responses, ML datasets, complex objects, datetime/decimal heavy scenarios
- **Multiple Metrics**: Speed, memory usage, output size, success rate, configuration variance
- **Error Handling**: Graceful handling of library limitations with detailed error tracking
- **Environment Consistency**: Controlled GitHub Actions runners with caching optimization
- **Reproducible**: Anyone can run the same benchmarks with identical results

### Test Scenarios

1. **Basic Types** - Core serialization speed testing
2. **DateTime Heavy** - Real-world timestamp patterns with optimization config testing
3. **Decimal Precision** - Financial/scientific precision handling
4. **Large Datasets** - Memory and compression optimization testing  
5. **Complex Structures** - Nested objects with user profiles and preferences

### Enhanced Metrics

- **Configuration Performance**: Per-config benchmarking across DataSON versions
- **API Evolution**: Feature availability and compatibility tracking
- **Optimization Variance**: Performance difference analysis between configurations
- **Version Regression**: Automated detection of performance changes across versions

## 📊 Interactive Reporting

### New Visualization Features

- **📈 Performance Evolution Charts**: Line charts tracking DataSON performance across versions
- **🔧 Configuration Comparison**: Bar charts comparing optimization configs
- **🏆 Competitive Analysis**: Grouped bar charts with DataSON highlighting
- **📋 API Details**: Expandable sections with deep configuration parameter analysis

### Report Types

- **Competitive Reports**: Head-to-head library comparisons with interactive charts
- **Optimization Reports**: DataSON configuration analysis with recommendations
- **Version Evolution**: Historical performance tracking across DataSON versions
- **Combined Analysis**: Complete benchmarking suite with all insights

## 🏗️ Architecture

### Repository Structure

```
datason-benchmarks/
├── .github/workflows/          # Enhanced GitHub Actions automation
├── benchmarks/                 # Core benchmark suites
│   ├── competitive/           # Competitor comparison tests
│   ├── configurations/        # DataSON config testing
│   ├── versioning/            # Version evolution analysis (NEW)
│   └── regression/            # Performance regression detection
├── competitors/               # Competitor library adapters
├── data/                     # Test datasets and results
│   ├── results/              # CI-only historical results  
│   ├── synthetic/            # Generated test data
│   └── configs/              # Test configurations
├── scripts/                  # Enhanced automation and utilities
│   ├── run_benchmarks.py     # Main benchmark runner
│   └── generate_report.py    # Interactive report generator (ENHANCED)
└── docs/                     # Documentation and live reports
    └── results/              # GitHub Pages hosted reports
```

### Enhanced Core Components

- **DataSONVersionManager** - Version switching and API compatibility testing
- **OptimizationAnalyzer** - Deep configuration parameter analysis
- **EnhancedReportGenerator** - Interactive charts with Plotly.js
- **CIEnvironmentDetector** - Smart CI vs local environment handling

## 🤝 Contributing

### Adding New Optimization Tests

1. Add test scenarios to `create_optimization_test_data()` in version suite
2. Focus on realistic data patterns that benefit from specific configurations
3. Test across multiple DataSON versions for evolution tracking

### Enhancing Analysis

1. Extend configuration parameter discovery in `_discover_config_parameters()`
2. Add new visualization types to report generator
3. Contribute performance optimization insights

### Adding New Competitors

1. Create adapter in `competitors/` directory
2. Implement `CompetitorAdapter` interface
3. Add to `CompetitorRegistry`
4. Test with `python scripts/run_benchmarks.py --quick --generate-report`

## 📋 Requirements

### System Requirements

- **Python 3.8+**
- **Memory**: 4GB+ recommended for optimization analysis
- **Time**: 5-45 minutes depending on benchmark scope

### Library Dependencies

Core dependencies automatically installed:

- `datason>=0.9.0` - The library being benchmarked
- `orjson`, `ujson`, `msgpack`, `jsonpickle` - Competitive libraries
- `numpy`, `pandas` - Realistic ML data generation
- `plotly` - Interactive chart generation

## 🔮 Roadmap

### Current Status: Phase 1 Complete ✅

- [x] **Competitive Benchmarking**: 7 major serialization libraries
- [x] **GitHub Actions Automation**: Daily runs with enhanced caching
- [x] **Optimization Analysis**: Deep DataSON configuration testing
- [x] **Version Evolution**: Performance tracking across DataSON versions
- [x] **Enhanced PR Checks**: Regression detection with smart caching
- [x] **Interactive Reports**: Beautiful visualizations with GitHub Pages
- [x] **CI/Local Separation**: Consistent results with local development support

### Phase 2: Advanced Analysis 🚧

- [ ] **Memory Usage Profiling**: Detailed memory consumption analysis
- [ ] **Cross-platform Testing**: Windows, macOS, Linux consistency verification
- [ ] **Extended ML Integration**: PyTorch, TensorFlow, scikit-learn model benchmarking
- [ ] **Real-world Datasets**: Integration with common ML datasets and API schemas
- [ ] **Performance Alerts**: Automated notifications for regressions
- [ ] **Competitor Version Tracking**: Monitor competitive library updates

### Phase 3: Community Growth 📈

- [ ] **User-contributed Scenarios**: Community test case submissions
- [ ] **Conference Materials**: Presentation templates and research papers
- [ ] **CI/CD Integrations**: Plugins for popular CI systems
- [ ] **Academic Collaboration**: Research partnership opportunities
- [ ] **Benchmarking Standards**: Industry methodology contributions

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

This benchmarking methodology and infrastructure is open source and freely available for use by the serialization library community.

## 🙏 Acknowledgments

- **Community Contributors** - Test scenarios, optimization insights, and improvements
- **Competitive Libraries** - orjson, ujson, msgpack, jsonpickle teams for excellent tools
- **GitHub Actions** - Free infrastructure enabling open source benchmarking with enhanced caching
- **DataSON Users** - Real-world feedback and optimization requirements
- **Open Source Community** - Plotly.js for interactive visualizations

---

**Latest Update**: Results automatically updated by [Daily Benchmarks workflow](https://github.com/datason/datason-benchmarks/actions/workflows/daily-benchmarks.yml) with interactive reports available at [GitHub Pages](https://datason.github.io/datason-benchmarks/) 
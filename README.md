# DataSON Benchmarks

> **Open source competitive benchmarking for [DataSON serialization library](https://github.com/danielendler/datason)**

[![Daily Benchmarks](https://github.com/danielendler/datason-benchmarks/actions/workflows/daily-benchmarks.yml/badge.svg)](https://github.com/danielendler/datason-benchmarks/actions/workflows/daily-benchmarks.yml)
[![PR Performance Check](https://github.com/danielendler/datason-benchmarks/actions/workflows/pr-performance-check.yml/badge.svg)](https://github.com/danielendler/datason-benchmarks/actions/workflows/pr-performance-check.yml)

## 🔗 Important Links

- 📚 **[DataSON Main Repository](https://github.com/danielendler/datason)** - The main DataSON library
- 📖 **[DataSON Documentation](https://datason.readthedocs.io/en/latest/)** - Full API docs and guides  
- 📊 **[Live Benchmark Results](https://danielendler.github.io/datason-benchmarks/)** - Interactive performance reports
- 🧪 **[This Benchmarks Repository](https://github.com/danielendler/datason-benchmarks)** - Benchmarking infrastructure
- 🚀 **[DataSON API Performance Guide](docs/DATASON_API_PERFORMANCE_GUIDE.md)** - **NEW!** Comprehensive guide to API variants and performance characteristics

## 🎯 Overview

This repository provides **transparent, reproducible benchmarks** for [DataSON](https://github.com/danielendler/datason) using a **dual benchmark system** designed to serve different performance analysis needs. Using GitHub Actions for zero-cost infrastructure, we deliver accurate competitive analysis and deep optimization insights.

## 🏗️ Benchmark Architecture

We maintain **two complementary benchmark systems** that serve different purposes:

### 🏆 **System A: Competitive Benchmarks** 
*Daily/Weekly Market Position Analysis*

**Purpose**: Compare DataSON against external serialization libraries  
**Used by**: Daily benchmarks, weekly reports, market analysis  
**Script**: `run_benchmarks.py` → `CompetitiveBenchmarkSuite`  
**Data Format**: `competitive → tiers → datasets`

- **Competitors**: orjson, ujson, json, pickle, jsonpickle, msgpack
- **DataSON Variants**: All API levels tested as separate "competitors"
- **Fairness**: Multi-tier testing (JSON-safe, object-enhanced, ML-complex)
- **Focus**: External market position and competitiveness

### 🔧 **System B: Optimization Benchmarks**
*Internal Performance Optimization & Regression Detection*

**Purpose**: Validate DataSON optimizations and detect performance regressions  
**Used by**: PR performance checks, optimization validation, baseline comparison  
**Script**: `pr_optimized_benchmark.py`  
**Data Format**: `results_by_tier → tiers → datasets`

- **API Tiers**: Basic, API-optimized, Smart, ML-optimized, Compatibility  
- **Profiling Integration**: Detailed optimization validation
- **Focus**: Internal optimization effectiveness and regression prevention
- **Baseline**: Tracks performance improvements over time

> **💡 Why Two Systems?** Each system is optimized for its specific purpose. Competitive benchmarks need fair external comparisons, while optimization benchmarks need detailed internal analysis. This separation provides cleaner insights and more focused reporting.

### Key Features

- **🏆 Competitive Analysis**: Head-to-head comparison with 6-8 major serialization libraries
- **🔧 Deep Optimization Analysis**: DataSON API-level performance insights and regression detection  
- **📊 Version Evolution Tracking**: Performance analysis across DataSON versions
- **🤖 Enhanced CI/CD Integration**: Smart PR performance checks with dual benchmark validation
- **🎨 Phase 4 Enhanced Reports**: **NEW!** Interactive reports with comprehensive performance tables, smart units (μs/ms/s), ML compatibility matrix
- **📈 Interactive Reports**: Beautiful charts and visualizations with GitHub Pages hosting
- **🚀 Community Friendly**: Easy setup, contribution guidelines, free infrastructure

## 🔧 Workflow Management

This repository uses a **Python-to-YAML workflow generation system** for maintainable GitHub Actions workflows:

### How to Change CI Workflows

**✅ Recommended: Edit Python Models**
```bash
# 1. Edit the workflow definitions
vim tools/gen_workflows.py

# 2. Generate updated YAML files  
make workflows

# 3. Commit both Python and generated YAML
git add tools/ .github/workflows/
git commit -m "Update CI workflows"
```

**❌ Don't: Edit YAML Files Directly**
- YAML files in `.github/workflows/` are **generated artifacts**
- Manual edits will be overwritten on next generation
- Always edit the Python models instead

### Workflow Development
```bash
# Generate workflows from Python models
make workflows

# Validate generated workflows  
make validate-workflows

# Run workflow generator tests
make test-workflows

# Set up development environment (includes pre-commit hooks)
make setup
```

### Benefits of This Approach
- **🔒 Type Safety**: Python models with full IDE support
- **🧪 Testable**: Unit tests for workflow logic
- **📝 DRY**: Reusable components and patterns
- **✅ Validated**: Schema validation and actionlint integration
- **🤖 AI-Friendly**: Edit structured code, not whitespace-sensitive YAML

## 🚀 Quick Start

### Setup
```bash
# Clone the repository
git clone https://github.com/danielendler/datason-benchmarks.git
cd datason-benchmarks

# Install dependencies
pip install -r requirements.txt
```

### 🏆 Competitive Benchmarks (System A)
*Compare DataSON against external libraries*

```bash
# Quick competitive comparison (3-4 libraries, fast)
python scripts/run_benchmarks.py --quick --generate-report

# Full competitive analysis (all available libraries)
python scripts/run_benchmarks.py --competitive --generate-report

# Complete competitive suite with reports
python scripts/run_benchmarks.py --complete --generate-report

# DataSON version evolution analysis
python scripts/run_benchmarks.py --versioning --generate-report
```

### 🔧 Optimization Benchmarks (System B)  
*Validate DataSON optimizations and detect regressions*

```bash
# PR optimization validation (fast, 5 datasets × 5 API tiers)
python scripts/pr_optimized_benchmark.py --output results/optimization_check.json

# Establish new performance baseline
python scripts/pr_optimized_benchmark.py --iterations 20 --output data/results/new_baseline.json

# Optimization-specific validation suite
python benchmarks/optimization_validation.py
```

### 🔬 Advanced Analysis

```bash
# Comprehensive API profiling across all DataSON APIs
python scripts/run_benchmarks.py --profile-apis

# DataSON configuration optimization testing
python scripts/run_benchmarks.py --configurations --generate-report

# Detailed profiling analysis (requires DATASON_PROFILE=1)
DATASON_PROFILE=1 python scripts/profile_stages.py --with-rust off --runs 5
```

### Rust Core Benchmarks (experimental)

The `scripts/bench_rust_core.py` helper exercises `datason.save_string` and
`datason.load_basic` with the optional Rust accelerator toggled on or off.
Use it to measure fast-path speedups and fallback overhead.

```bash
# Run save_string with Rust enabled
python scripts/bench_rust_core.py save_string --with-rust on --sizes 10k --shapes flat --repeat 5 --output results_rust_on.json

# Run save_string with Rust disabled
python scripts/bench_rust_core.py save_string --with-rust off --sizes 10k --shapes flat --repeat 5 --output results_rust_off.json
```

Configuration notes:

- `--with-rust` controls the `DATASON_RUST` environment variable (`on`, `off`,
  or `auto` to respect the existing value).
- Ensure your DataSON wheel includes the Rust extension; otherwise the script
  skips `--with-rust on` runs.
- Output files are JSON and can be merged or inspected directly.

### 🎯 **NEW: Dagger-Based CI/CD Pipelines**
*Reliable, Testable, and Maintainable Automation*

**Latest Addition**: Hybrid Dagger + GitHub Actions approach eliminates YAML complexity:

```bash
# Install Dagger CLI and Python SDK
curl -fsSL https://dl.dagger.io/dagger/install.sh | BIN_DIR=$HOME/.local/bin sh
pip install dagger-io

# Test pipelines locally (instant feedback vs 10+ minute CI cycles)
dagger call daily-benchmarks --source=. --focus-area=api_modes
dagger call weekly-benchmarks --source=. --benchmark-type=comprehensive
dagger call validate-system --source=.

# Run comprehensive test suite
dagger call test-pipeline --source=.
```

**Benefits**:
- ✅ **Zero YAML syntax errors** - Complex logic moved to Python
- ⚡ **Local testing** - 30-second iterations vs 10+ minute CI cycles  
- 🔧 **IDE support** - Full autocomplete, debugging, type safety
- 📊 **Same functionality** - All benchmark features preserved
- 🚀 **Better reliability** - Container-based execution

### Phase 4: Enhanced Reporting & Visualization 🎨

**NEW:** Interactive reports with comprehensive performance tables and smart unit formatting:

```bash
# Generate Phase 4 enhanced report from any benchmark result
python scripts/run_benchmarks.py --phase4-report phase2_complete_1750338755.json

# Get intelligent library recommendations by domain  
python scripts/run_benchmarks.py --phase4-decide web      # Web API recommendations
python scripts/run_benchmarks.py --phase4-decide ml       # ML framework recommendations
python scripts/run_benchmarks.py --phase4-decide finance  # Financial services recommendations

# Run trend analysis and regression detection
python scripts/run_benchmarks.py --phase4-trends
```

### Phase 2: Automated Benchmarking 🤖

**NEW:** Full automation with synthetic data generation and regression detection:

```bash
# Generate realistic test data
python scripts/generate_data.py --scenario all

# Run regression analysis
python scripts/regression_detector.py current_results.json --baseline latest_baseline.json

# Analyze performance trends
python scripts/analyze_trends.py --input-dir data/results --lookback-weeks 12
```

### View Latest Results

- **[🎨 Enhanced Phase 4 Reports](https://danielendler.github.io/datason-benchmarks/weekly-reports/latest_phase4_enhanced.html)** - **NEW!** Interactive reports with comprehensive tables and smart units
- **[📊 Interactive Reports](https://danielendler.github.io/datason-benchmarks/results/)** - Live performance visualizations
- **[📈 Latest Benchmark Results](data/results/)** - JSON files with detailed metrics  
- **[🔄 GitHub Actions](https://github.com/danielendler/datason-benchmarks/actions)** - Automated runs and artifacts
- **[📊 Performance Trends](data/results/)** - Historical performance data

## 📊 Current Competitive Landscape

### Tested Libraries

| Library | Type | Why Tested | Latest Status |
|---------|------|------------|---------------|
| **[DataSON](https://github.com/danielendler/datason)** | JSON+objects | Our library | ✅ Active |
| **orjson** | JSON (Rust) | Speed benchmark standard | ✅ Available |
| **ujson** | JSON (C) | Popular drop-in replacement | ✅ Available |
| **json** | JSON (stdlib) | Baseline reference | ✅ Available |
| **pickle** | Binary objects | Python default for objects | ✅ Available |
| **jsonpickle** | JSON objects | Direct functional competitor | ✅ Available |
| **msgpack** | Binary compact | Cross-language efficiency | ✅ Available |

### Performance Summary

> **Latest benchmark results from automated daily runs**

*Results updated automatically by GitHub Actions with interactive charts. View [latest reports](https://danielendler.github.io/datason-benchmarks/results/) for detailed visualizations.*

## 🔧 Optimization Analysis

### DataSON Configuration Deep Dive

Our enhanced benchmarking system now provides **deep API analysis** of [DataSON's](https://github.com/danielendler/datason) optimization configurations:

📋 **[View Complete API Performance Guide →](docs/DATASON_API_PERFORMANCE_GUIDE.md)**

#### Available Optimization Configs
- **`get_performance_config()`** - Speed-optimized settings (`UNIX` dates, `VALUES` orient, no type hints)
- **`get_ml_config()`** - ML-optimized settings (`UNIX_MS` dates, type hints enabled, aggressive coercion)
- **`get_strict_config()`** - Type preservation (`ISO` dates, strict coercion, complex/decimal preservation)
- **`get_api_config()`** - API-compatible settings (`ISO` dates, ASCII encoding, string UUIDs)

#### New DataSON API Methods (Testing Needed)
- **`dump_api()`** - Web API optimized serialization
- **`dump_ml()`** - ML framework optimized serialization  
- **`dump_secure()`** - Security-focused with PII redaction
- **`dump_fast()`** - Performance optimized serialization
- **`load_smart()`** - Intelligent deserialization (80-90% success rate)
- **`load_perfect()`** - 100% accurate reconstruction with templates

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

## 🎨 Phase 4: Enhanced Reporting & Visualization

### Interactive Reports with Comprehensive Analysis

Phase 4 delivers **beautiful, interactive HTML reports** that transform raw benchmark data into actionable insights:

#### 📊 **Enhanced Features**:
- **Performance Summary Tables**: Real benchmark data with method comparison
- **Smart Unit Formatting**: Automatic μs → ms → s conversion based on values
- **ML Framework Compatibility Matrix**: Complete NumPy/Pandas support analysis
- **Security Features Analysis**: PII redaction effectiveness and compliance insights  
- **Interactive Charts**: Chart.js visualizations with real performance data
- **Domain-Specific Recommendations**: Optimized advice for Web API, ML, Finance, Data Engineering

#### 🎯 **Quick Examples**:

```bash
# Generate enhanced report from any benchmark result
python scripts/run_benchmarks.py --phase4-report phase2_complete_1750338755.json

# Get intelligent recommendations for your use case
python scripts/run_benchmarks.py --phase4-decide ml       # → datason.dump_ml() for NumPy/Pandas
python scripts/run_benchmarks.py --phase4-decide finance  # → datason.dump_secure() for PII protection
python scripts/run_benchmarks.py --phase4-decide web      # → datason.dump_api() for JSON compatibility

# Historical trend analysis with regression detection
python scripts/run_benchmarks.py --phase4-trends
```

#### 📈 **Report Highlights**:
- **Performance Table**: Shows `dump_secure()` at 387.31ms vs `serialize()` at 0.32ms with use case guidance
- **ML Compatibility**: Visual matrix showing 100% NumPy/Pandas support for DataSON ML methods
- **Security Analysis**: Quantifies PII redaction effectiveness (90-95%) and performance cost (+930%)
- **Smart Units**: Displays `53.0μs` for fast operations, `387.31ms` for complex ones, `2.5s` for large datasets

#### 🌐 **Automated Integration**:
Phase 4 reports are **automatically generated** by both daily and weekly CI workflows, with enhanced reports available at:
- [Latest Enhanced Report](https://danielendler.github.io/datason-benchmarks/weekly-reports/latest_phase4_enhanced.html)
- [All Enhanced Reports](https://danielendler.github.io/datason-benchmarks/results/)

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

#### 🎯 **Dagger-Based Pipelines** (NEW - Recommended)
- **[Dagger Daily Benchmarks](.github/workflows/dagger-daily-benchmarks.yml)** - Minimal YAML + Python pipeline logic
- **[Dagger Weekly Benchmarks](.github/workflows/dagger-weekly-benchmarks.yml)** - Comprehensive weekly analysis via Dagger
- **[Dagger Validation](.github/workflows/dagger-validation.yml)** - Pipeline testing and validation

#### 📊 **Legacy Workflows** (Maintained for compatibility)
- **[PR Performance Check](.github/workflows/pr-performance-check.yml)** - Enhanced competitive check with regression analysis
- **[Daily Benchmarks](.github/workflows/daily-benchmarks.yml)** - Comprehensive competitive + optimization analysis  
- **[Weekly Benchmarks](.github/workflows/weekly-benchmarks.yml)** - 🆕 **Phase 2:** Complete automation with trend analysis
- **Manual Triggers** - Run specific benchmark suites on demand

**Migration Status**: New Dagger workflows are production-ready and eliminate the YAML complexity issues of legacy workflows.

#### 🆕 Phase 2 Automation Features

- **🔄 Automated Data Generation**: Fresh synthetic test data weekly
- **🔍 Statistical Regression Detection**: Blocks PRs with >25% performance degradation
- **📈 Historical Trend Analysis**: 12-week performance evolution tracking
- **🤖 Self-Sustaining**: Runs without manual intervention
- **📊 Enhanced Reporting**: Comprehensive trend analysis and insights

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

#### 🆕 Phase 2: Realistic Synthetic Data
Automated generation of 5 comprehensive scenarios with **real-world data patterns**:

1. **API Fast** (`api_fast`) - REST API responses, user profiles, product catalogs
   ```json
   {
     "id": "40b2da9f-1c54-4af7-b853-43ee3717a701",
     "username": "jane92", 
     "email": "gwilliams@example.net",
     "profile": {"bio": "Magazine perform foreign air.", "verified": true},
     "preferences": {"notifications": true, "theme": "dark"},
     "stats": {"login_count": 131, "last_active": "1993-01-04T03:19:33.872652"}
   }
   ```

2. **ML Training** (`ml_training`) - ML model serialization, feature matrices, time series
   - NumPy arrays with realistic data distributions
   - Pandas DataFrames with time series patterns
   - Model parameters and training metadata

3. **Secure Storage** (`secure_storage`) - Nested configurations, hierarchical data
   ```json
   {
     "app_config": {
       "database": {"host": "61.225.172.203", "port": 2770, "ssl": true},
       "cache": {"enabled": false, "ttl": 2982, "size_limit": 268},
       "features": {"analytics": true, "debugging": false}
     }
   }
   ```

4. **Large Data** (`large_data`) - Dataset handling, streaming data patterns
5. **Edge Cases** (`edge_cases`) - Boundary conditions, Unicode stress tests

#### 📊 Enhanced Reporting Features
- **Adaptive Unit Formatting** - Automatically chooses best units (ms, μs, ns) for readability
- **Sample Data Visualization** - Shows exactly what data structures are being tested
- **Interactive Charts** - Performance comparison charts with Plotly.js
- **Comprehensive Analysis** - Competitive, configuration, and version comparison in one report

#### Classic Scenarios
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

## 🎉 Phase 2 Complete: Self-Sustaining Automation

**Implementation Date:** January 2025  
**Status:** ✅ Complete  

### 🚀 What's New in Phase 2

- **🔄 Automated Data Generation**: Realistic synthetic data generated weekly
- **🔍 Advanced Regression Detection**: Statistical analysis blocks problematic PRs  
- **📊 Weekly Comprehensive Benchmarks**: Full automation with parallel execution
- **📈 Historical Trend Analysis**: 12-week performance evolution tracking
- **🤖 Self-Sustaining System**: <4 hours/week maintenance as designed

### Core Phase 2 Components

- **`scripts/generate_data.py`** - Synthetic data generation CLI
- **`scripts/regression_detector.py`** - Statistical regression analysis
- **`scripts/analyze_trends.py`** - Historical trend analysis  
- **`.github/workflows/weekly-benchmarks.yml`** - Comprehensive automation
- **Enhanced PR workflows** - Advanced regression detection

### Success Metrics Achieved

- ✅ **95%+ automated execution** with error handling
- ✅ **Zero-cost infrastructure** using GitHub Actions free tier
- ✅ **Part-time maintainable** designed for <4 hours/week
- ✅ **Community transparent** with public results and methodology
- ✅ **Regression prevention** blocks PRs with >25% performance degradation

### Ready for Phase 3

Phase 2 creates the foundation for **Phase 3: Polish** with:
- Documentation improvements 
- Additional competitive libraries
- Enhanced reporting with visualizations
- Community contribution guidelines

**[📋 View Phase 2 Implementation Details →](docs/PHASE_2_IMPLEMENTATION_COMPLETE.md)**

---

## 🏗️ Architecture

### Repository Structure

```
datason-benchmarks/
├── .github/workflows/          # Hybrid GitHub Actions + Dagger automation
│   ├── dagger-*.yml           # NEW: Minimal Dagger-based workflows (recommended)
│   └── *.yml                  # Legacy YAML workflows (maintained)
├── dagger/                    # NEW: Python-based CI/CD pipeline logic
│   ├── benchmark_pipeline.py  # Main pipeline functions (daily/weekly/test)
│   └── __init__.py           # Dagger module exports
├── benchmarks/                # Core benchmark suites
│   ├── competitive/           # Competitor comparison tests
│   ├── configurations/        # DataSON config testing
│   ├── versioning/            # Version evolution analysis (NEW)
│   └── regression/            # Performance regression detection
├── competitors/               # Competitor library adapters
├── data/                      # Test datasets and results
│   ├── results/              # CI-only historical results  
│   ├── synthetic/            # Generated test data
│   └── configs/              # Test configurations
├── scripts/                   # Enhanced automation and utilities
│   ├── run_benchmarks.py     # Main benchmark runner
│   ├── improved_*.py         # Enhanced benchmark & reporting system
│   └── generate_report.py    # Interactive report generator (ENHANCED)
├── docs/                      # Documentation and live reports
│   └── results/              # GitHub Pages hosted reports
├── dagger.json               # Dagger project configuration
└── requirements-dagger.txt   # Dagger-specific dependencies
```

### Enhanced Core Components

#### 🎯 **Dagger Pipeline Components** (NEW)
- **BenchmarkPipeline** - Python-based CI/CD automation with type safety
- **daily_benchmarks()** - Focus area benchmarking (api_modes, competitive, versions)
- **weekly_benchmarks()** - Comprehensive analysis with enhanced test data  
- **validate_system()** - End-to-end validation and testing

#### 📊 **Legacy Core Components** (Maintained)
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

**Latest Update**: Results automatically updated by [Daily Benchmarks workflow](https://github.com/danielendler/datason-benchmarks/actions/workflows/daily-benchmarks.yml) with interactive reports available at [GitHub Pages](https://datason.github.io/datason-benchmarks/) 

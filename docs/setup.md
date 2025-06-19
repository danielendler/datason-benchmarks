# Quick Setup Guide

## DataSON Benchmarks - Getting Started

This guide helps you get the DataSON benchmarking suite running in under 5 minutes.

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/datason/datason-benchmarks.git
cd datason-benchmarks

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Your First Benchmark

```bash
# Quick competitive comparison (fastest way to see results)
python scripts/run_benchmarks.py --quick

# Results will be saved to data/results/latest_quick.json
```

### 3. View Results

```bash
# Check if results were generated
ls data/results/

# View quick summary
python -c "
import json
with open('data/results/latest_quick.json', 'r') as f:
    results = json.load(f)
print('Competitive benchmark completed!')
print(f'Tested libraries: {list(results.get(\"competitive\", {}).get(\"summary\", {}).get(\"competitors_tested\", []))}')
"
```

## 🔧 Troubleshooting

### Missing Libraries

If you see warnings about missing competitive libraries:

```bash
# Install optional competitors
pip install orjson ujson msgpack jsonpickle

# For ML testing (optional)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn
```

### DataSON Not Found

If you get "DataSON not available" errors:

```bash
# Install DataSON
pip install datason

# Or install from source if needed
pip install git+https://github.com/datason/datason.git
```

### Permission Errors

If you get permission errors on macOS/Linux:

```bash
# Make scripts executable
chmod +x scripts/run_benchmarks.py
```

## 📊 Benchmark Types

### Quick Benchmark (~2-5 minutes)
Tests core competitors with basic scenarios:
```bash
python scripts/run_benchmarks.py --quick
```

### Competitive Analysis (~10-15 minutes)
Full comparison with all available libraries:
```bash
python scripts/run_benchmarks.py --competitive
```

### Configuration Testing (~5-10 minutes)
DataSON configuration optimization:
```bash
python scripts/run_benchmarks.py --configurations
```

### Complete Suite (~20-30 minutes)
Everything with detailed reports:
```bash
python scripts/run_benchmarks.py --all --generate-report
```

### 🎨 Phase 4 Enhanced Reports (NEW!)
Generate beautiful interactive reports with comprehensive tables:
```bash
# Generate enhanced report from any benchmark result
python scripts/run_benchmarks.py --phase4-report phase2_complete_1750338755.json

# Get intelligent library recommendations
python scripts/run_benchmarks.py --phase4-decide web      # Web API recommendations
python scripts/run_benchmarks.py --phase4-decide ml       # ML framework recommendations

# Historical trend analysis
python scripts/run_benchmarks.py --phase4-trends
```

## 📁 Understanding Results

### Result Files

- `data/results/latest_quick.json` - Most recent quick benchmark
- `data/results/latest_competitive.json` - Most recent competitive analysis
- `data/results/latest_configuration.json` - Most recent config testing
- `docs/results/*.html` - Generated HTML reports
- `docs/results/phase4_comprehensive_*.html` - **NEW!** Enhanced Phase 4 reports with comprehensive tables and smart units

### Key Metrics

- **mean_ms** - Average time in milliseconds
- **successful_runs** - Number of successful test iterations
- **error_count** - Number of failed attempts
- **size** - Serialized data size in bytes

## 🐛 Common Issues

### 1. Import Errors
**Problem**: `ModuleNotFoundError` for competitive libraries
**Solution**: Install missing libraries or run with `--quick` to use only available ones

### 2. Memory Issues
**Problem**: Out of memory on large datasets
**Solution**: Reduce iteration count or use `--quick` mode

### 3. Slow Performance
**Problem**: Benchmarks taking too long
**Solution**: Start with `--quick`, then try specific suites

### 4. No Results Generated
**Problem**: Script runs but no JSON files created
**Solution**: Check permissions and disk space in `data/results/`

## 🔄 Running in CI/CD

### GitHub Actions
The repository includes ready-to-use GitHub Actions workflows:

- `.github/workflows/pr-performance-check.yml` - Quick PR checks
- `.github/workflows/daily-benchmarks.yml` - Automated daily runs

### Docker (Coming Soon)
```bash
# Will be available in Phase 2
docker run datason/benchmarks --quick
```

## 📞 Getting Help

1. **Check logs**: Look for error messages in terminal output
2. **Review results**: Check if partial results were generated
3. **GitHub Issues**: [Report problems](https://github.com/datason/datason-benchmarks/issues)
4. **Documentation**: See [README.md](../README.md) for detailed info

## ✅ Verification

Run this command to verify your setup:

```bash
python -c "
import sys
print(f'Python: {sys.version}')

# Check core imports
try:
    import datason
    print(f'✅ DataSON: {datason.__version__}')
except ImportError:
    print('❌ DataSON: Not available')

# Check competitors
competitors = ['orjson', 'ujson', 'msgpack', 'jsonpickle']
for lib in competitors:
    try:
        mod = __import__(lib)
        version = getattr(mod, '__version__', 'unknown')
        print(f'✅ {lib}: {version}')
    except ImportError:
        print(f'⚠️ {lib}: Not available (optional)')

print('\n🚀 Ready to run benchmarks!')
"
```

## 🎯 Next Steps

Once basic setup works:

1. **Explore Results**: Look at the JSON output structure
2. **Try Configuration Testing**: See how different DataSON configs perform
3. **Add Custom Data**: Modify test datasets for your use case
4. **Set up Automation**: Use GitHub Actions for regular testing
5. **Contribute**: Add new test scenarios or competitive libraries

Happy benchmarking! 🚀 
# 📊 Performance Baseline Management Guide

## Overview

The datason-benchmarks repository uses a **CI-based baseline system** to ensure consistent and reliable performance comparisons across PRs. All baselines are established in GitHub Actions to guarantee identical hardware and environment conditions.

## 🎯 Why CI-Only Baselines?

Running baselines locally (e.g., on macOS) and comparing against CI results (Ubuntu) leads to:
- **Incomparable metrics** due to different hardware
- **False positives** in regression detection
- **Unreliable performance tracking**

By establishing baselines only in CI, we ensure:
- ✅ **Consistent environment** (ubuntu-latest)
- ✅ **Identical dependencies** and Python version
- ✅ **Reproducible results**
- ✅ **Fair PR comparisons**

## 🚀 How to Establish a Baseline

### Initial Setup (First Time)

1. **Push your changes** to the main branch
2. **Navigate to GitHub Actions** → "📊 Establish Performance Baseline"
3. **Click "Run workflow"** with these options:
   - DataSON version: `latest` (or specific version like `0.12.0`)
   - Update strategy: `create_if_missing`
   - Commit baseline: `true`
4. **Wait for completion** (~2-3 minutes)
5. The baseline files will be automatically committed to the repository

### Updating an Existing Baseline

#### Manual Update
Use when you want to update to a new DataSON version or reset the baseline:

```yaml
Workflow dispatch options:
- DataSON version: 0.13.0  # New version to test
- Update strategy: archive_and_update  # Keeps old baseline in archive
- Commit baseline: true
```

#### Automatic Updates
The baseline updates automatically when:
- **Push to main** changes benchmark scripts
- **Weekly schedule** (Mondays at 2 AM UTC)
- **DataSON version changes** detected in CI

## 📁 Baseline File Structure

```
data/results/
├── datason_baseline.json    # Current active baseline
├── latest.json              # Symlink to datason_baseline.json
└── archived/                # Historical baselines
    ├── baseline_0.11.0_20250816_120000.json
    ├── baseline_0.12.0_20250817_140000.json
    └── ...
```

## 🔄 Update Strategies

### 1. **create_if_missing** (Default)
- Only creates baseline if none exists
- Safest option for initial setup
- No changes if baseline already exists

### 2. **force_update**
- Always creates new baseline
- Overwrites existing without archiving
- Use when resetting after major changes

### 3. **archive_and_update**
- Archives current baseline before updating
- Maintains history (keeps last 10 archives)
- Best for version upgrades

## 📊 How PR Checks Use the Baseline

When a PR is opened in the DataSON repository:

1. **PR workflow triggers** datason-benchmarks via webhook
2. **Benchmarks run** with PR's DataSON wheel
3. **Results compare** against `datason_baseline.json`
4. **Comment posts** to PR with performance analysis:
   ```
   ✅ No regression detected (0.5% faster than baseline)
   ⚠️ Minor regression: 10% slower than baseline
   ❌ Critical regression: 30% slower than baseline
   ```

## 🛠️ Troubleshooting

### "No baseline available for comparison"
**Solution**: Run the establish-baseline workflow to create initial baseline

### "Baseline version mismatch"
**Solution**: Update baseline to match current DataSON version:
```bash
# Run workflow with:
Update strategy: archive_and_update
DataSON version: latest
```

### Local Testing Without Baseline
For local development, you can still run benchmarks:
```bash
python scripts/run_benchmarks.py --quick
```
Just remember: local results are NOT comparable to CI baseline!

## 📋 Best Practices

1. **Never commit local baselines** - They're blocked by .gitignore
2. **Update baseline after major releases** - Keep it current with stable versions
3. **Archive before updating** - Maintain history for trend analysis
4. **Monitor weekly runs** - Catch gradual performance drift
5. **Document baseline changes** - Note why baseline was updated in PR/commit

## 🔍 Viewing Baseline Details

### Check Current Baseline Version
```bash
# From CI artifact or after pulling latest main
cat data/results/datason_baseline.json | jq '.baseline_metadata'
```

### Compare Two Baselines
```python
python scripts/compare_baselines.py \
  data/results/archived/baseline_0.11.0_*.json \
  data/results/datason_baseline.json
```

## 📈 Performance Tracking

The baseline system enables:
- **Regression detection** - Catch performance drops before merge
- **Improvement tracking** - Quantify optimization gains
- **Version comparison** - See performance across DataSON versions
- **Trend analysis** - Identify gradual degradation patterns

## 🚨 Important Notes

- **CI Environment**: All baselines run on `ubuntu-latest` GitHub runners
- **Python Version**: Fixed at 3.11 for consistency
- **Iterations**: 20 iterations for stable measurements
- **Retention**: Baseline artifacts kept for 90 days
- **Archives**: Last 10 baselines preserved in `archived/`

## 📞 Support

For issues or questions about baseline management:
1. Check workflow logs in GitHub Actions
2. Review artifacts from baseline runs
3. Open an issue with the `baseline` label

---

*Remember: Consistent baselines are the foundation of reliable performance testing!*

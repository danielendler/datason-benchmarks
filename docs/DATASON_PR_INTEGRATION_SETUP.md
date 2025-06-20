# DataSON PR Integration Setup Guide

## Overview

The DataSON PR integration allows the main DataSON repository to automatically trigger benchmarks in the datason-benchmarks repository when PRs are created, providing automatic performance analysis.

## Required Setup

### 1. **GitHub Secret Configuration**

The integration requires a GitHub secret with cross-repository permissions:

```yaml
# Required secret in datason-benchmarks repository
BENCHMARK_REPO_TOKEN
```

**This token must have permissions to:**
- Read repository contents
- Access workflow runs and artifacts from external repositories
- Create comments on issues/PRs in the DataSON repository

**Token Setup:**
1. Create a GitHub Personal Access Token with these scopes:
   - `repo` (Full control of private repositories)
   - `workflow` (Update GitHub Action workflows)
2. Add it as a repository secret named `BENCHMARK_REPO_TOKEN`

### 2. **Workflow Files**

Two workflow files are included:

- **`.github/workflows/datason-pr-integration.yml`** - Main active workflow
- **`.github/workflows/datason-pr-integration-example.yml`** - Reference implementation

### 3. **DataSON Repository Integration**

The DataSON repository should trigger this workflow with:

```yaml
- name: Trigger benchmark analysis
  uses: actions/github-script@v7
  with:
    github-token: ${{ secrets.BENCHMARK_REPO_TOKEN }}
    script: |
      await github.rest.actions.createWorkflowDispatch({
        owner: 'danielendler',
        repo: 'datason-benchmarks', 
        workflow_id: 'datason-pr-integration.yml',
        ref: 'main',
        inputs: {
          pr_number: '${{ github.event.number }}',
          commit_sha: '${{ github.event.pull_request.head.sha }}',
          artifact_name: 'datason-wheel',
          datason_repo: '${{ github.repository }}',
          benchmark_type: 'pr_optimized'
        }
      });
```

## Workflow Inputs

| Input | Description | Required | Default |
|-------|-------------|----------|---------|
| `pr_number` | PR number from DataSON repo | Yes | - |
| `commit_sha` | Commit SHA to test | Yes | - |
| `artifact_name` | Name of wheel artifact | Yes | - |
| `datason_repo` | DataSON repo (owner/repo) | Yes | - |
| `benchmark_type` | Type of benchmark to run | No | `pr_optimized` |

## Benchmark Types

- **`pr_optimized`** - Fast 5-dataset suite optimized for PR testing (~2 min)
- **`quick`** - Quick competitive comparison (~3 min)  
- **`competitive`** - Full competitive analysis (~5 min)

## Expected Flow

1. **DataSON PR created** → Builds wheel and uploads as artifact
2. **DataSON triggers** → datason-benchmarks via `workflow_dispatch`
3. **datason-benchmarks** → Downloads wheel, runs benchmarks, analyzes results
4. **Results posted** → Back to DataSON PR as comment with performance analysis

## Baseline Management

To enable regression detection, establish a baseline:

```bash
# In datason-benchmarks repository
python scripts/establish_datason_baseline.py
```

This creates `data/results/datason_baseline.json` for comparison.

## Features

- ✅ **Cross-repository artifact download** using GitHub API
- ✅ **Phase 1-4 optimized test suite** with domain-specific scenarios
- ✅ **Regression detection** with baseline comparison
- ✅ **Enhanced reporting** with interactive analysis
- ✅ **Automatic PR comments** with performance insights
- ✅ **Workflow failure** on significant regressions

## Security Notes

- The `BENCHMARK_REPO_TOKEN` should be scoped to minimum required permissions
- Artifacts are downloaded securely using GitHub's official API
- No external network access required beyond GitHub API

## Troubleshooting

**Common Issues:**

1. **"Could not find artifact"** - Check artifact name matches exactly
2. **"Insufficient permissions"** - Verify `BENCHMARK_REPO_TOKEN` has correct scopes
3. **"No baseline available"** - Run `establish_datason_baseline.py` first
4. **YAML syntax errors** - Both workflows use proper heredoc indentation

**Debug Steps:**
1. Check GitHub Actions logs for detailed error messages
2. Verify artifact exists in DataSON repository workflow runs
3. Confirm token has access to both repositories
4. Test workflow manually using GitHub UI 
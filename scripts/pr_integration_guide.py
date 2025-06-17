#!/usr/bin/env python3
"""
DataSON PR Integration Guide
============================

This script demonstrates how to integrate the datason-benchmarks system
with DataSON repository PRs for testing development versions.

INTEGRATION OPTIONS:
1. GitHub Actions integration in DataSON repo
2. External trigger from DataSON PRs
3. Manual testing with PR URLs
"""

import subprocess
import sys
import tempfile
from pathlib import Path

def create_datason_pr_workflow():
    """
    Generate a GitHub Actions workflow for the DataSON repository
    that triggers benchmarks on PRs.
    """
    
    workflow_content = """
name: Performance Benchmark

on:
  pull_request:
    branches: [ main ]
  workflow_dispatch:
    inputs:
      benchmark_type:
        description: 'Type of benchmark to run'
        required: false
        default: 'quick'
        type: choice
        options:
        - quick
        - competitive
        - full

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout DataSON PR
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install DataSON from source
      run: |
        # Install current PR version of DataSON
        pip install -e .
        
        # Verify installation
        python -c "import datason; print(f'DataSON {datason.__version__} installed from PR')"
    
    - name: Checkout benchmark repository
      uses: actions/checkout@v4
      with:
        repository: 'datason/datason-benchmarks'
        path: 'benchmarks'
    
    - name: Install benchmark dependencies
      run: |
        cd benchmarks
        pip install -r requirements.txt
        pip install orjson ujson msgpack jsonpickle
    
    - name: Run PR benchmarks
      run: |
        cd benchmarks
        
        # Generate test data
        python scripts/generate_data.py --scenario api_fast --count 100
        
        # Run benchmarks with PR version
        python scripts/run_benchmarks.py --quick --output ../pr_results.json
        
        # Run regression detection if baseline exists
        if [ -f data/results/latest.json ]; then
          python scripts/regression_detector.py ../pr_results.json \\
            --baseline data/results/latest.json \\
            --pr-comment ../pr_comment.md \\
            --fail-threshold 0.25
        fi
    
    - name: Comment PR with results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v7
      with:
        script: |
          const fs = require('fs');
          
          if (fs.existsSync('pr_comment.md')) {
            const comment = fs.readFileSync('pr_comment.md', 'utf8');
            
            await github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
          } else {
            await github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '📊 **Performance benchmark completed**\\n\\nResults available in workflow artifacts.'
            });
          }
    
    - name: Upload benchmark results
      uses: actions/upload-artifact@v4
      with:
        name: performance-results
        path: |
          pr_results.json
          pr_comment.md
        retention-days: 30
"""
    
    return workflow_content

def create_external_trigger_system():
    """
    Generate a system that can be triggered from external DataSON PRs
    via webhook or repository dispatch.
    """
    
    workflow_content = """
name: External DataSON PR Benchmark

on:
  repository_dispatch:
    types: [datason-pr-benchmark]
  workflow_dispatch:
    inputs:
      datason_pr_url:
        description: 'DataSON PR URL to test'
        required: true
        type: string
      pr_commit_sha:
        description: 'Commit SHA to test'
        required: false
        type: string

jobs:
  benchmark-external-pr:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout benchmark repository
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Extract PR information
      id: pr-info
      run: |
        # Parse PR URL to get repo and PR number
        PR_URL="${{ github.event.inputs.datason_pr_url || github.event.client_payload.pr_url }}"
        echo "Processing PR: $PR_URL"
        
        # Extract owner/repo/pr_number from URL
        # Example: https://github.com/datason/datason/pull/123
        if [[ $PR_URL =~ github\.com/([^/]+)/([^/]+)/pull/([0-9]+) ]]; then
          echo "repo_owner=${BASH_REMATCH[1]}" >> $GITHUB_OUTPUT
          echo "repo_name=${BASH_REMATCH[2]}" >> $GITHUB_OUTPUT
          echo "pr_number=${BASH_REMATCH[3]}" >> $GITHUB_OUTPUT
        else
          echo "Invalid PR URL format"
          exit 1
        fi
    
    - name: Checkout DataSON PR
      uses: actions/checkout@v4
      with:
        repository: ${{ steps.pr-info.outputs.repo_owner }}/${{ steps.pr-info.outputs.repo_name }}
        ref: refs/pull/${{ steps.pr-info.outputs.pr_number }}/head
        path: datason-pr
    
    - name: Install DataSON from PR
      run: |
        cd datason-pr
        pip install -e .
        echo "Installed DataSON from PR ${{ steps.pr-info.outputs.pr_number }}"
    
    - name: Install benchmark dependencies
      run: |
        pip install -r requirements.txt
        pip install orjson ujson msgpack jsonpickle
    
    - name: Run comprehensive benchmark
      run: |
        # Generate fresh test data
        python scripts/generate_data.py --scenario all
        
        # Run full competitive analysis
        python scripts/run_benchmarks.py --competitive \\
          --output pr_benchmark_results.json
    
    - name: Analyze performance vs baseline
      run: |
        if [ -f data/results/latest.json ]; then
          python scripts/regression_detector.py pr_benchmark_results.json \\
            --baseline data/results/latest.json \\
            --output regression_analysis.json \\
            --pr-comment pr_performance_report.md
        fi
    
    - name: Post results to DataSON PR
      if: success()
      uses: actions/github-script@v7
      with:
        github-token: ${{ secrets.DATASON_REPO_TOKEN }}  # Need PAT with write access
        script: |
          const fs = require('fs');
          
          let comment = '📊 **External Performance Benchmark Results**\\n\\n';
          
          if (fs.existsSync('pr_performance_report.md')) {
            const report = fs.readFileSync('pr_performance_report.md', 'utf8');
            comment += report;
          } else {
            comment += 'Benchmark completed successfully. See artifacts for detailed results.';
          }
          
          comment += '\\n\\n---\\n*Benchmarked by [datason-benchmarks](https://github.com/datason/datason-benchmarks)*';
          
          await github.rest.issues.createComment({
            owner: '${{ steps.pr-info.outputs.repo_owner }}',
            repo: '${{ steps.pr-info.outputs.repo_name }}',
            issue_number: ${{ steps.pr-info.outputs.pr_number }},
            body: comment
          });
"""
    
    return workflow_content

def create_manual_pr_testing_script():
    """
    Generate a script for manually testing DataSON PRs locally.
    """
    
    script_content = """#!/usr/bin/env python3
'''
Manual DataSON PR Testing Script
================================

Test a DataSON PR locally with the benchmark suite.

Usage:
    python test_datason_pr.py https://github.com/datason/datason/pull/123
    python test_datason_pr.py --commit-sha abc123def
'''

import argparse
import subprocess
import sys
import tempfile
import os
from pathlib import Path

def clone_and_install_pr(pr_url_or_sha: str, is_commit: bool = False):
    '''Clone DataSON PR and install it.'''
    
    with tempfile.TemporaryDirectory() as temp_dir:
        datason_dir = Path(temp_dir) / 'datason'
        
        if is_commit:
            # Clone and checkout specific commit
            subprocess.run(['git', 'clone', 'https://github.com/datason/datason.git', str(datason_dir)], check=True)
            subprocess.run(['git', 'checkout', pr_url_or_sha], cwd=datason_dir, check=True)
        else:
            # Parse PR URL and clone PR branch
            if '/pull/' not in pr_url_or_sha:
                raise ValueError('Invalid PR URL format')
            
            # Extract PR number
            pr_number = pr_url_or_sha.split('/pull/')[-1].rstrip('/')
            
            # Clone and fetch PR
            subprocess.run(['git', 'clone', 'https://github.com/datason/datason.git', str(datason_dir)], check=True)
            subprocess.run(['git', 'fetch', 'origin', f'pull/{pr_number}/head:pr-{pr_number}'], cwd=datason_dir, check=True)
            subprocess.run(['git', 'checkout', f'pr-{pr_number}'], cwd=datason_dir, check=True)
        
        # Install DataSON from PR
        print(f'Installing DataSON from {"commit" if is_commit else "PR"}...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], cwd=datason_dir, check=True)
        
        # Verify installation
        result = subprocess.run([sys.executable, '-c', 'import datason; print(f"DataSON {datason.__version__} installed")'], 
                              capture_output=True, text=True)
        print(result.stdout.strip())

def run_benchmark_suite(benchmark_type: str = 'quick'):
    '''Run the benchmark suite with the installed DataSON version.'''
    
    print(f'Running {benchmark_type} benchmark suite...')
    
    # Generate test data
    subprocess.run([sys.executable, 'scripts/generate_data.py', '--scenario', 'api_fast', '--count', '50'], check=True)
    
    # Run benchmarks
    if benchmark_type == 'quick':
        subprocess.run([sys.executable, 'scripts/run_benchmarks.py', '--quick'], check=True)
    elif benchmark_type == 'competitive':
        subprocess.run([sys.executable, 'scripts/run_benchmarks.py', '--competitive'], check=True)
    else:
        subprocess.run([sys.executable, 'scripts/run_benchmarks.py', '--all'], check=True)
    
    print('Benchmark completed! Check data/results/ for results.')

def main():
    parser = argparse.ArgumentParser(description='Test DataSON PR with benchmark suite')
    parser.add_argument('pr_url_or_sha', help='DataSON PR URL or commit SHA')
    parser.add_argument('--commit', action='store_true', help='Treat input as commit SHA instead of PR URL')
    parser.add_argument('--benchmark-type', choices=['quick', 'competitive', 'full'], default='quick',
                       help='Type of benchmark to run')
    
    args = parser.parse_args()
    
    try:
        # Install DataSON from PR
        clone_and_install_pr(args.pr_url_or_sha, args.commit)
        
        # Run benchmarks
        run_benchmark_suite(args.benchmark_type)
        
        print('\\n✅ PR testing completed successfully!')
        
    except Exception as e:
        print(f'❌ Error testing PR: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()
"""
    
    return script_content

def main():
    """Generate integration examples."""
    
    print("🔗 DataSON PR Integration Options")
    print("=" * 50)
    
    print("\n1. 📝 DataSON Repository Workflow")
    print("   Place this in .github/workflows/performance.yml in DataSON repo:")
    workflow1 = create_datason_pr_workflow()
    with open('datason_repo_workflow.yml', 'w') as f:
        f.write(workflow1)
    print("   ✅ Saved to: datason_repo_workflow.yml")
    
    print("\n2. 🌐 External Trigger Workflow")
    print("   Place this in datason-benchmarks .github/workflows/:")
    workflow2 = create_external_trigger_system()
    with open('external_trigger_workflow.yml', 'w') as f:
        f.write(workflow2)
    print("   ✅ Saved to: external_trigger_workflow.yml")
    
    print("\n3. 💻 Manual Testing Script")
    print("   Use this for local PR testing:")
    script = create_manual_pr_testing_script()
    with open('test_datason_pr.py', 'w') as f:
        f.write(script)
    print("   ✅ Saved to: test_datason_pr.py")
    
    print("\n" + "=" * 50)
    print("🚀 Integration Summary:")
    print("✅ All benchmarks work natively in CI")
    print("✅ Can test any DataSON PR via GitHub Actions")
    print("✅ Supports external triggering from DataSON repo")
    print("✅ Manual testing scripts for local development")
    
    print("\n📋 Usage Examples:")
    print("# Manual PR testing:")
    print("python test_datason_pr.py https://github.com/datason/datason/pull/123")
    print()
    print("# Trigger from DataSON PR (with webhook):")
    print("curl -X POST -H 'Authorization: token $TOKEN' \\")
    print("  https://api.github.com/repos/datason/datason-benchmarks/dispatches \\")
    print("  -d '{\"event_type\":\"datason-pr-benchmark\",\"client_payload\":{\"pr_url\":\"...\"}}'")

if __name__ == '__main__':
    main() 
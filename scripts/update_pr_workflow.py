#!/usr/bin/env python3
"""
Update PR workflow to integrate advanced regression detection.
This script enhances the existing PR workflow with Phase 2 improvements.
"""

import os
import re

def enhance_pr_workflow():
    """Add regression detection to PR workflow"""
    workflow_file = '.github/workflows/pr-performance-check.yml'
    
    if not os.path.exists(workflow_file):
        print(f"❌ Workflow file not found: {workflow_file}")
        return
    
    # Read existing workflow
    with open(workflow_file, 'r') as f:
        content = f.read()
    
    # Add regression detection step before the comment step
    regression_step = '''
    - name: 🔍 Advanced Regression Detection
      run: |
        # Install regression detection dependencies if needed
        pip install pandas matplotlib 2>/dev/null || true
        
        # Run comprehensive regression detection
        if [ -f data/results/latest.json ]; then
          echo "📊 Running advanced regression detection..."
          python scripts/regression_detector.py \\
            data/results/latest_competitive_*.json \\
            --baseline data/results/latest.json \\
            --output data/results/pr_regression_report.json \\
            --pr-comment pr_regression_comment.md \\
            --fail-threshold 0.25 \\
            --warn-threshold 0.10
          
          REGRESSION_EXIT_CODE=$?
          echo "REGRESSION_EXIT_CODE=$REGRESSION_EXIT_CODE" >> $GITHUB_ENV
          
          if [ $REGRESSION_EXIT_CODE -ne 0 ]; then
            echo "⚠️ Critical performance regressions detected!"
            echo "PERFORMANCE_REGRESSION=true" >> $GITHUB_ENV
          else
            echo "✅ No critical regressions detected"
            echo "PERFORMANCE_REGRESSION=false" >> $GITHUB_ENV
          fi
          
        else
          echo "📝 No baseline found - this will become the new baseline"
          echo "PERFORMANCE_REGRESSION=false" >> $GITHUB_ENV
          echo "🔄 **Performance Baseline**" > pr_regression_comment.md
          echo "" >> pr_regression_comment.md
          echo "This is the first benchmark run or no previous results were found." >> pr_regression_comment.md
          echo "Future PRs will be compared against this baseline." >> pr_regression_comment.md
        fi
'''
    
    # Find location to insert regression step (before the Comment on PR step)
    comment_step_pattern = r'(\s+- name: 💬 Comment on PR)'
    
    if re.search(comment_step_pattern, content):
        # Insert regression detection step before comment step
        content = re.sub(comment_step_pattern, regression_step + r'\1', content)
    else:
        print("⚠️ Could not find Comment on PR step - adding at end of job")
        # Add before the upload artifacts step
        upload_pattern = r'(\s+- name: 📤 Upload enhanced artifacts)'
        if re.search(upload_pattern, content):
            content = re.sub(upload_pattern, regression_step + r'\1', content)
        else:
            print("❌ Could not find insertion point for regression step")
            return
    
    # Enhance the comment step to use regression analysis
    comment_enhancement = '''
      with:
        script: |
          const fs = require('fs');
          
          try {
            // Priority: Use regression comment if available, fallback to main comment
            let comment = '';
            
            if (fs.existsSync('pr_regression_comment.md')) {
              const regressionComment = fs.readFileSync('pr_regression_comment.md', 'utf8');
              
              if (fs.existsSync('pr_comment.md')) {
                const mainComment = fs.readFileSync('pr_comment.md', 'utf8');
                comment = regressionComment + '\\n\\n---\\n\\n' + mainComment;
              } else {
                comment = regressionComment;
              }
            } else if (fs.existsSync('pr_comment.md')) {
              comment = fs.readFileSync('pr_comment.md', 'utf8');
            }
            
            if (!comment) {
              console.log('No comment content found');
              return;
            }
            
            // Add regression warning to top if needed
            const hasRegression = process.env.PERFORMANCE_REGRESSION === 'true';
            if (hasRegression) {
              comment = '🚨 **PERFORMANCE REGRESSION DETECTED**\\n\\n' + comment;
            }'''
    
    # Replace the existing comment script
    script_pattern = r'(with:\s+script: \|[\s\S]*?)(\s+- name: 📤 Upload enhanced artifacts)'
    if re.search(script_pattern, content):
        content = re.sub(script_pattern, comment_enhancement + r'\2', content)
    
    # Update artifact upload to include regression results
    artifact_enhancement = '''      with:
        name: pr-performance-check-${{ github.run_id }}
        path: |
          data/results/latest_*.json
          data/results/*_benchmark_*.json
          data/results/pr_regression_report.json
          docs/results/*_report.html
          pr_comment.md
          pr_regression_comment.md
        retention-days: 30'''
    
    # Replace artifact upload configuration
    artifact_pattern = r'(- name: 📤 Upload enhanced artifacts\s+uses: actions/upload-artifact@v4\s+if: always\(\)\s+)with:[\s\S]*?retention-days: \d+'
    if re.search(artifact_pattern, content):
        content = re.sub(artifact_pattern, r'\1' + artifact_enhancement, content)
    
    # Write updated workflow
    with open(workflow_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Enhanced {workflow_file} with Phase 2 regression detection")

def update_requirements():
    """Update requirements.txt with regression detection dependencies"""
    requirements_file = 'requirements.txt'
    
    if not os.path.exists(requirements_file):
        print(f"❌ Requirements file not found: {requirements_file}")
        return
    
    with open(requirements_file, 'r') as f:
        content = f.read()
    
    # Add dependencies if not present
    new_deps = [
        'pandas>=1.3.0',
        'matplotlib>=3.5.0',
        'faker>=18.0.0',
        'numpy>=1.21.0'
    ]
    
    for dep in new_deps:
        dep_name = dep.split('>=')[0]
        if dep_name not in content:
            content += f'\n{dep}'
    
    with open(requirements_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated {requirements_file} with regression detection dependencies")

def main():
    """Main enhancement function"""
    print("🚀 Phase 2 Enhancement: Adding Advanced Regression Detection")
    
    try:
        enhance_pr_workflow()
        update_requirements()
        
        print("\n✅ Phase 2 Enhancement Complete!")
        print("\nPhase 2 Features Added:")
        print("- 🔍 Advanced regression detection with statistical analysis")
        print("- ⚠️ Automated PR failure on critical performance regressions")
        print("- 📊 Detailed regression reports with trend analysis")
        print("- 💬 Enhanced PR comments with regression insights")
        print("- 📈 Historical trend tracking integration")
        
        print("\nNext Steps:")
        print("- Test the enhanced PR workflow on a sample PR")
        print("- Generate synthetic data: python scripts/generate_data.py")
        print("- Run weekly benchmarks manually to test: .github/workflows/weekly-benchmarks.yml")
        
    except Exception as e:
        print(f"❌ Enhancement failed: {e}")

if __name__ == '__main__':
    main() 
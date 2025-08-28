#!/usr/bin/env python3
"""
Update Documentation Index
===========================

Generates accurate documentation index based on actual available reports.
"""

import os
import glob
from datetime import datetime

def update_docs_index():
    """Update the docs index with actually available reports."""
    
    # Create docs directories if they don't exist
    os.makedirs('docs/results', exist_ok=True)
    
    # Find all available reports
    daily_reports = glob.glob('docs/results/daily_*_report.html')
    ci_reports = glob.glob('docs/results/ci_*_report.html')
    weekly_reports = glob.glob('docs/results/weekly_*_report.html')
    test_reports = glob.glob('docs/results/*test*_report.html')
    local_reports = glob.glob('docs/results/local_*_report.html')
    
    # Sort by creation time to get latest first
    daily_reports.sort(key=os.path.getctime, reverse=True)
    ci_reports.sort(key=os.path.getctime, reverse=True)
    weekly_reports.sort(key=os.path.getctime, reverse=True)
    test_reports.sort(key=os.path.getctime, reverse=True)
    local_reports.sort(key=os.path.getctime, reverse=True)
    
    print(f"Found {len(daily_reports)} daily reports")
    print(f"Found {len(ci_reports)} CI reports")
    print(f"Found {len(weekly_reports)} weekly reports")
    print(f"Found {len(test_reports)} test reports")
    print(f"Found {len(local_reports)} local reports")
    
    # Build sections based on available content
    daily_section = ""
    if daily_reports:
        daily_section = "### 🚀 Latest Daily Benchmarks\nAutomated benchmark results from our CI system:\n\n"
        for i, report in enumerate(daily_reports[:3]):  # Show latest 3
            filename = os.path.basename(report)
            name = filename.replace('_report.html', '').replace('_', ' ').title()
            daily_section += f"- [📊 {name}](results/{filename}) - Daily benchmark analysis\n"
        daily_section += "\n"
    
    ci_section = ""
    if ci_reports:
        ci_section = "### 🔄 CI Integration Reports\nHistorical CI benchmark results:\n\n"
        for i, report in enumerate(ci_reports[:3]):  # Show latest 3
            filename = os.path.basename(report)
            if 'quick' in filename:
                name = "Quick CI Report"
                desc = "Fast benchmark validation"
            elif 'competitive' in filename:
                name = "Competitive Analysis"
                desc = "Library comparison"
            else:
                name = "Complete Analysis"
                desc = "Full benchmark suite"
            ci_section += f"- [⚡ {name}](results/{filename}) - {desc}\n"
        ci_section += "\n"
    
    weekly_section = ""
    if weekly_reports:
        weekly_section = "### 🗓️ Weekly Reports\nComprehensive weekly benchmark analysis:\n\n"
        for i, report in enumerate(weekly_reports[:2]):  # Show latest 2
            filename = os.path.basename(report)
            name = filename.replace('_report.html', '').replace('_', ' ').title()
            weekly_section += f"- [📈 {name}](results/{filename}) - Weekly analysis\n"
        weekly_section += "\n"
    
    test_section = ""
    if test_reports:
        test_section = "### 🧪 Development Reports\nLocal development and testing results:\n\n"
        for report in test_reports:
            filename = os.path.basename(report)
            if 'improved' in filename:
                name = "Enhanced Test Report"
                desc = "Comprehensive test analysis"
            elif 'final' in filename:
                name = "Final Test Report"
                desc = "Complete test results"
            else:
                name = filename.replace('_report.html', '').replace('_', ' ').title()
                desc = "Development testing"
            test_section += f"- [🎯 {name}](results/{filename}) - {desc}\n"
        test_section += "\n"
    
    # Create comprehensive index
    index_content = f"""# DataSON Benchmarks - Live Results

Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 📊 Available Benchmark Reports

{daily_section}{weekly_section}{ci_section}{test_section}## ⚙️ System Features

✨ **Current Capabilities**:
- 📊 **Interactive HTML Reports** with performance visualizations
- 🏆 **Competitive Analysis** against popular serialization libraries
- 🎯 **Smart Performance Classification** (fast/medium/slow indicators)
- 📈 **Time Series Analysis** for performance trends
- 🔄 **CI Integration** with automated benchmark validation
- 🤖 **Python-Generated Workflows** for maintainable CI/CD

## 🔗 Navigation
- [🏠 Back to Main Repository](https://github.com/danielendler/datason-benchmarks)
- [📚 DataSON Library](https://github.com/danielendler/datason)
- [📖 DataSON Documentation](https://datason.readthedocs.io/en/latest/)
- [🔄 GitHub Actions](https://github.com/danielendler/datason-benchmarks/actions)

---
*Generated automatically by update_docs_index.py*
"""
    
    with open('docs/index.md', 'w') as f:
        f.write(index_content)
    
    print("✅ Documentation index updated with accurate content")

if __name__ == '__main__':
    update_docs_index() 
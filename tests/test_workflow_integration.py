#!/usr/bin/env python3
"""
Comprehensive workflow integration validation tests.

This test suite validates that all GitHub Actions workflows are properly configured
for the datason-benchmarks integration, including:
- Correct triggers (workflow_dispatch, pull_request, etc.)
- Required permissions (contents, pull-requests, etc.) 
- External repository integration capabilities
- Comment management functionality
- Baseline comparison logic
- File handling and artifact management
"""

import os
import sys
import yaml
import json
import unittest
from pathlib import Path
from typing import Dict, Any, List


class TestWorkflowIntegration(unittest.TestCase):
    """Test GitHub Actions workflow integration requirements."""
    
    def setUp(self):
        """Set up test environment."""
        self.repo_root = Path(__file__).parent.parent
        self.workflows_dir = self.repo_root / ".github" / "workflows"
        self.workflows = {}
        
        # Load all workflow files
        for workflow_file in self.workflows_dir.glob("*.yml"):
            try:
                with open(workflow_file, 'r') as f:
                    content = yaml.safe_load(f)
                    self.workflows[workflow_file.stem] = {
                        'file': workflow_file,
                        'content': content
                    }
            except Exception as e:
                self.fail(f"Failed to load workflow {workflow_file}: {e}")
    
    def test_pr_performance_check_workflow_dispatch(self):
        """Test that pr-performance-check has proper workflow_dispatch trigger for external integration."""
        workflow_name = "pr-performance-check"
        self.assertIn(workflow_name, self.workflows, f"Workflow {workflow_name} not found")
        
        workflow = self.workflows[workflow_name]['content']
        
        # Check workflow_dispatch trigger exists (YAML parser converts 'on' to True)
        self.assertIn(True, workflow, "Workflow missing trigger section")
        on_section = workflow[True]
        
        # workflow_dispatch can be a dict with inputs or just True/None
        self.assertIn('workflow_dispatch', on_section, 
                     "pr-performance-check missing workflow_dispatch trigger for external integration")
        
        # Validate expected inputs for external repository integration
        if isinstance(on_section['workflow_dispatch'], dict) and 'inputs' in on_section['workflow_dispatch']:
            inputs = on_section['workflow_dispatch']['inputs']
            required_inputs = ['pr_number', 'commit_sha', 'artifact_name', 'datason_repo', 'benchmark_type']
            
            for input_name in required_inputs:
                self.assertIn(input_name, inputs, 
                            f"Missing required input '{input_name}' for external integration")
                
                # Check input configuration
                input_config = inputs[input_name]
                self.assertIn('description', input_config, f"Input '{input_name}' missing description")
                self.assertIn('type', input_config, f"Input '{input_name}' missing type")
                self.assertEqual(input_config['type'], 'string', f"Input '{input_name}' should be string type")
    
    def test_workflow_permissions(self):
        """Test that workflows have correct permissions for PR integration."""
        critical_workflows = ['pr-performance-check', 'datason-pr-integration-example']
        
        for workflow_name in critical_workflows:
            if workflow_name not in self.workflows:
                continue  # Skip if workflow doesn't exist
                
            workflow = self.workflows[workflow_name]['content']
            
            # Check permissions section exists
            self.assertIn('permissions', workflow, 
                         f"Workflow {workflow_name} missing permissions section")
            
            permissions = workflow['permissions']
            
            # Check required permissions
            required_permissions = {
                'contents': 'read',  # To read repository content
                'pull-requests': 'write',  # To post comments on PRs
            }
            
            for perm, level in required_permissions.items():
                self.assertIn(perm, permissions, 
                            f"Workflow {workflow_name} missing '{perm}' permission")
                self.assertEqual(permissions[perm], level,
                               f"Workflow {workflow_name} has incorrect '{perm}' permission level")
    
    def test_pr_comment_management_steps(self):
        """Test that PR workflows have proper comment management steps."""
        workflow_name = "pr-performance-check"
        if workflow_name not in self.workflows:
            self.skipTest(f"Workflow {workflow_name} not found")
            
        workflow = self.workflows[workflow_name]['content']
        
        # Find the comment management step
        comment_step_found = False
        comment_manager_found = False
        
        for job_name, job in workflow.get('jobs', {}).items():
            for step in job.get('steps', []):
                step_name = step.get('name', '').lower()
                
                # Check for comment update step
                if 'comment' in step_name and ('update' in step_name or 'post' in step_name):
                    comment_step_found = True
                    
                    # Validate step uses proper comment management
                    if 'run' in step:
                        step_script = step['run']
                        # Should use our advanced comment manager
                        if 'manage_pr_comments.py' in step_script:
                            comment_manager_found = True
                            
                            # Check for strategy parameter
                            self.assertIn('--strategy', step_script,
                                        "Comment management step should specify strategy")
                        
        self.assertTrue(comment_step_found, 
                       "PR workflow should have a comment update step")
        self.assertTrue(comment_manager_found,
                       "PR workflow should use the advanced comment manager")
    
    def test_baseline_comparison_logic(self):
        """Test that workflows have proper baseline comparison logic."""
        workflow_name = "pr-performance-check"
        if workflow_name not in self.workflows:
            self.skipTest(f"Workflow {workflow_name} not found")
            
        workflow = self.workflows[workflow_name]['content']
        
        # Look for baseline comparison logic in workflow steps
        baseline_logic_found = False
        
        for job_name, job in workflow.get('jobs', {}).items():
            for step in job.get('steps', []):
                if 'run' in step:
                    step_script = step['run']
                    
                    # Check for baseline file selection logic
                    if ('latest_competitive.json' in step_script and 
                        'baseline' in step_script.lower()):
                        baseline_logic_found = True
                        
                        # Validate smart baseline selection
                        self.assertTrue(
                            'BASELINE_FILE' in step_script or 'REGRESSION_BASELINE' in step_script,
                            "Should use baseline variable for smart selection")
                        self.assertIn('competitive', step_script,
                                    "Should prioritize competitive baseline")
                        
        self.assertTrue(baseline_logic_found,
                       "PR workflow should have intelligent baseline selection logic")
    
    def test_artifact_handling(self):
        """Test that workflows properly handle external artifacts."""
        workflow_name = "pr-performance-check"
        if workflow_name not in self.workflows:
            self.skipTest(f"Workflow {workflow_name} not found")
            
        workflow = self.workflows[workflow_name]['content']
        
        # Check for external artifact download step
        artifact_download_found = False
        artifact_upload_found = False
        
        for job_name, job in workflow.get('jobs', {}).items():
            for step in job.get('steps', []):
                step_name = step.get('name', '').lower()
                
                # Check for external artifact download
                if 'external' in step_name and ('download' in step_name or 'wheel' in step_name):
                    artifact_download_found = True
                    
                    # Should use github-script for API access
                    if step.get('uses') == 'actions/github-script@v7':
                        script = step.get('with', {}).get('script', '')
                        self.assertIn('downloadArtifact', script,
                                    "Should download artifacts via GitHub API")
                        self.assertIn('listWorkflowRunsForRepo', script,
                                    "Should list workflow runs to find artifacts")
                
                # Check for result artifact upload
                if 'upload' in step_name and 'artifact' in step_name:
                    artifact_upload_found = True
                    
                    # Should upload comprehensive results
                    if step.get('uses', '').startswith('actions/upload-artifact'):
                        with_section = step.get('with', {})
                        path = with_section.get('path', '')
                        
                        # Should include key result files
                        self.assertIn('latest_*.json', path,
                                    "Should upload benchmark result files")
                        self.assertIn('pr_comment.md', path,
                                    "Should upload PR comment file")
        
        self.assertTrue(artifact_download_found,
                       "Workflow should handle external artifact download")
        self.assertTrue(artifact_upload_found,
                       "Workflow should upload result artifacts")
    
    def test_environment_variables(self):
        """Test that workflows have proper environment configuration."""
        critical_workflows = ['pr-performance-check']
        
        for workflow_name in critical_workflows:
            if workflow_name not in self.workflows:
                continue
                
            workflow = self.workflows[workflow_name]['content']
            
            # Check global environment variables
            if 'env' in workflow:
                env_vars = workflow['env']
                
                required_env = {
                    'PYTHONUNBUFFERED': '1',
                    'CI': 'true',
                    'PYTHONPATH': '.'
                }
                
                for var, expected_value in required_env.items():
                    if var in env_vars:
                        actual_value = str(env_vars[var]).lower() if isinstance(env_vars[var], bool) else str(env_vars[var])
                        self.assertEqual(actual_value, expected_value,
                                       f"Environment variable {var} should be {expected_value}")
    
    def test_timeout_and_resource_limits(self):
        """Test that workflows have appropriate timeouts and resource limits."""
        critical_workflows = ['pr-performance-check']
        
        for workflow_name in critical_workflows:
            if workflow_name not in self.workflows:
                continue
                
            workflow = self.workflows[workflow_name]['content']
            
            for job_name, job in workflow.get('jobs', {}).items():
                # Check timeout is set
                if 'timeout-minutes' in job:
                    timeout = job['timeout-minutes']
                    self.assertGreater(timeout, 5, f"Job {job_name} timeout too short")
                    self.assertLess(timeout, 60, f"Job {job_name} timeout too long")
                
                # Check runs-on is specified
                self.assertIn('runs-on', job, f"Job {job_name} missing runs-on")
                self.assertIn('ubuntu', job['runs-on'], 
                            f"Job {job_name} should use ubuntu runner")
    
    def test_workflow_trigger_paths(self):
        """Test that workflows trigger on appropriate file changes."""
        workflow_name = "pr-performance-check"
        if workflow_name not in self.workflows:
            self.skipTest(f"Workflow {workflow_name} not found")
            
        workflow = self.workflows[workflow_name]['content']
        
        # Check pull_request trigger has correct paths (YAML parser converts 'on' to True)
        on_section = workflow.get(True, {})
        if 'pull_request' in on_section:
            pr_config = on_section['pull_request']
            
            if isinstance(pr_config, dict) and 'paths' in pr_config:
                paths = pr_config['paths']
                
                # Should trigger on key directories
                expected_paths = ['scripts/**', 'benchmarks/**', 'requirements.txt']
                
                for expected_path in expected_paths:
                    self.assertIn(expected_path, paths,
                                f"PR trigger should include path {expected_path}")
    
    def test_secret_access_configuration(self):
        """Test that workflows properly access required secrets."""
        workflow_name = "pr-performance-check"
        if workflow_name not in self.workflows:
            self.skipTest(f"Workflow {workflow_name} not found")
            
        workflow = self.workflows[workflow_name]['content']
        
        # Look for secret usage in steps
        secret_usage_found = False
        fallback_pattern_found = False
        
        for job_name, job in workflow.get('jobs', {}).items():
            for step in job.get('steps', []):
                # Check 'with' section
                if 'with' in step:
                    with_section = step['with']
                    
                    # Check for GitHub token usage
                    if 'github-token' in with_section:
                        token_config = with_section['github-token']
                        
                        if 'BENCHMARK_REPO_TOKEN' in token_config or 'GITHUB_TOKEN' in token_config:
                            secret_usage_found = True
                            if '||' in token_config:
                                fallback_pattern_found = True
                
                # Check 'run' section for token usage
                if 'run' in step:
                    run_script = step['run']
                    if 'BENCHMARK_REPO_TOKEN' in run_script or 'GITHUB_TOKEN' in run_script:
                        secret_usage_found = True
                        if '||' in run_script and ('BENCHMARK_REPO_TOKEN' in run_script):
                            fallback_pattern_found = True
        
        self.assertTrue(secret_usage_found,
                       "Workflow should properly access GitHub tokens")
        self.assertTrue(fallback_pattern_found,
                       "At least one token usage should have fallback pattern")


class TestScriptIntegration(unittest.TestCase):
    """Test that scripts are properly integrated with workflows."""
    
    def setUp(self):
        """Set up test environment."""
        self.repo_root = Path(__file__).parent.parent
        self.scripts_dir = self.repo_root / "scripts"
        
    def test_comment_manager_exists(self):
        """Test that the comment manager script exists and is executable."""
        comment_manager = self.scripts_dir / "manage_pr_comments.py"
        
        self.assertTrue(comment_manager.exists(),
                       "Comment manager script should exist")
        
        # Check script has proper CLI interface
        if comment_manager.exists():
            with open(comment_manager, 'r') as f:
                content = f.read()
                
                # Should have main function and CLI interface
                self.assertIn('def main()', content,
                            "Comment manager should have main function")
                self.assertIn('argparse', content,
                            "Comment manager should use argparse")
                self.assertIn('--strategy', content,
                            "Comment manager should support strategy parameter")
    
    def test_performance_analysis_scripts(self):
        """Test that performance analysis scripts exist."""
        required_scripts = [
            "generate_pr_comment.py",
            "regression_detector.py",
            "run_benchmarks.py"
        ]
        
        for script_name in required_scripts:
            script_path = self.scripts_dir / script_name
            self.assertTrue(script_path.exists(),
                           f"Required script {script_name} should exist")
            
            if script_path.exists():
                with open(script_path, 'r') as f:
                    content = f.read()
                    
                    # Should be executable Python scripts
                    self.assertTrue(content.startswith('#!/usr/bin/env python') or 
                                  'python' in content[:50],
                                  f"Script {script_name} should be Python script")
    
    def test_baseline_management(self):
        """Test that baseline management functionality exists."""
        # Check for baseline-related scripts
        baseline_scripts = list(self.scripts_dir.glob("*baseline*"))
        
        # Should have at least one baseline-related script
        self.assertGreater(len(baseline_scripts), 0,
                          "Should have baseline management scripts")
        
        # Check data directory structure
        data_dir = self.repo_root / "data"
        if data_dir.exists():
            results_dir = data_dir / "results"
            self.assertTrue(results_dir.exists(),
                           "Should have data/results directory")


class TestWorkflowSyntaxValidation(unittest.TestCase):
    """Test workflow YAML syntax and structure."""
    
    def setUp(self):
        """Set up test environment."""
        self.repo_root = Path(__file__).parent.parent
        self.workflows_dir = self.repo_root / ".github" / "workflows"
    
    def test_yaml_syntax_validity(self):
        """Test that all workflow YAML files have valid syntax."""
        for workflow_file in self.workflows_dir.glob("*.yml"):
            with self.subTest(workflow=workflow_file.name):
                try:
                    with open(workflow_file, 'r') as f:
                        yaml.safe_load(f)
                except yaml.YAMLError as e:
                    self.fail(f"Invalid YAML syntax in {workflow_file.name}: {e}")
    
    def test_workflow_structure(self):
        """Test that workflows have required top-level structure."""
        required_sections = ['name', 'jobs']  # Note: 'on' becomes True in YAML parsing
        
        for workflow_file in self.workflows_dir.glob("*.yml"):
            with self.subTest(workflow=workflow_file.name):
                try:
                    with open(workflow_file, 'r') as f:
                        workflow = yaml.safe_load(f)
                        
                    for section in required_sections:
                        self.assertIn(section, workflow,
                                    f"Workflow {workflow_file.name} missing '{section}' section")
                    
                    # Also check for trigger section (appears as True in parsed YAML)
                    self.assertTrue(True in workflow or 'on' in workflow,
                                  f"Workflow {workflow_file.name} missing trigger section")
                        
                except Exception as e:
                    self.fail(f"Failed to validate {workflow_file.name}: {e}")
    
    def test_job_structure(self):
        """Test that jobs have required structure."""
        required_job_fields = ['runs-on', 'steps']
        
        for workflow_file in self.workflows_dir.glob("*.yml"):
            with self.subTest(workflow=workflow_file.name):
                try:
                    with open(workflow_file, 'r') as f:
                        workflow = yaml.safe_load(f)
                    
                    for job_name, job in workflow.get('jobs', {}).items():
                        for field in required_job_fields:
                            self.assertIn(field, job,
                                        f"Job {job_name} in {workflow_file.name} missing '{field}'")
                            
                        # Steps should be a list
                        self.assertIsInstance(job['steps'], list,
                                            f"Job {job_name} steps should be a list")
                        
                        # Each step should have name or uses
                        for i, step in enumerate(job['steps']):
                            self.assertTrue('name' in step or 'uses' in step,
                                          f"Step {i} in job {job_name} should have 'name' or 'uses'")
                
                except Exception as e:
                    self.fail(f"Failed to validate job structure in {workflow_file.name}: {e}")


def run_workflow_integration_tests():
    """Run all workflow integration tests."""
    print("🧪 Running GitHub Actions workflow integration tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestWorkflowIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestScriptIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestWorkflowSyntaxValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All workflow integration tests passed!")
        print("   GitHub Actions workflows are properly configured.")
    else:
        print("❌ Some workflow integration tests failed!")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
                print(f"  - {test}: {error_msg}")
        
        if result.errors:
            print("\nErrors:")  
            for test, traceback in result.errors:
                error_msg = traceback.split('\n')[-2]
                print(f"  - {test}: {error_msg}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_workflow_integration_tests()
    sys.exit(0 if success else 1)
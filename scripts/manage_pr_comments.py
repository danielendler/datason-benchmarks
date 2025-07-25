#!/usr/bin/env python3
"""
PR Comment Management
====================

Manages DataSON benchmark comments on PRs:
- Updates existing comments instead of creating duplicates
- Marks old comments as outdated
- Keeps PR conversations clean
"""

import os
import sys
import requests
import json
from typing import List, Dict, Any, Optional


class PRCommentManager:
    """Manages PR comments for DataSON benchmarks."""
    
    def __init__(self, token: str, repo_owner: str, repo_name: str):
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json'
        }
        self.base_url = 'https://api.github.com'
        
        # Signature to identify our benchmark comments
        self.comment_signature = "Generated by [datason-benchmarks]"
        self.outdated_marker = "⚠️ **OUTDATED** - See latest analysis below"
    
    def get_pr_comments(self, pr_number: int) -> List[Dict[str, Any]]:
        """Get all comments for a PR."""
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/issues/{pr_number}/comments"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching PR comments: {e}")
            return []
    
    def find_benchmark_comments(self, pr_number: int) -> List[Dict[str, Any]]:
        """Find existing benchmark comments."""
        comments = self.get_pr_comments(pr_number)
        benchmark_comments = []
        
        for comment in comments:
            if self.comment_signature in comment.get('body', ''):
                benchmark_comments.append(comment)
        
        return benchmark_comments
    
    def update_comment(self, comment_id: int, new_body: str) -> bool:
        """Update an existing comment."""
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/issues/comments/{comment_id}"
        data = {'body': new_body}
        
        try:
            response = requests.patch(url, headers=self.headers, json=data)
            response.raise_for_status()
            print(f"✅ Updated comment {comment_id}")
            return True
        except requests.RequestException as e:
            print(f"Error updating comment {comment_id}: {e}")
            return False
    
    def create_comment(self, pr_number: int, body: str) -> bool:
        """Create a new comment."""
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/issues/{pr_number}/comments"
        data = {'body': body}
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            print(f"✅ Created new comment on PR #{pr_number}")
            return True
        except requests.RequestException as e:
            print(f"Error creating comment: {e}")
            return False
    
    def mark_comment_outdated(self, comment_id: int, original_body: str) -> bool:
        """Mark a comment as outdated."""
        # Check if already marked as outdated
        if self.outdated_marker in original_body:
            return True
        
        # Add outdated marker at the top
        outdated_body = f"{self.outdated_marker}\n\n---\n\n{original_body}"
        
        return self.update_comment(comment_id, outdated_body)
    
    def post_or_update_comment(self, pr_number: int, new_comment_body: str, 
                             update_strategy: str = "update") -> bool:
        """
        Post a new comment or update existing one.
        
        Strategies:
        - "update": Update the most recent benchmark comment
        - "mark_outdated": Mark old comments as outdated, create new one  
        - "replace_all": Delete old comments, create new one
        """
        existing_comments = self.find_benchmark_comments(pr_number)
        
        if not existing_comments:
            # No existing comments, create new one
            return self.create_comment(pr_number, new_comment_body)
        
        if update_strategy == "update":
            # Update the most recent comment
            latest_comment = max(existing_comments, key=lambda c: c['updated_at'])
            return self.update_comment(latest_comment['id'], new_comment_body)
        
        elif update_strategy == "mark_outdated":
            # Mark all existing comments as outdated
            for comment in existing_comments:
                if self.outdated_marker not in comment['body']:
                    self.mark_comment_outdated(comment['id'], comment['body'])
            
            # Create new comment
            return self.create_comment(pr_number, new_comment_body)
        
        elif update_strategy == "replace_all":
            # Delete old comments and create new one
            for comment in existing_comments:
                try:
                    delete_url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/issues/comments/{comment['id']}"
                    requests.delete(delete_url, headers=self.headers)
                    print(f"🗑️ Deleted old comment {comment['id']}")
                except requests.RequestException as e:
                    print(f"Warning: Could not delete comment {comment['id']}: {e}")
            
            return self.create_comment(pr_number, new_comment_body)
        
        else:
            print(f"Unknown strategy: {update_strategy}")
            return False


def main():
    """CLI interface for PR comment management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage PR benchmark comments')
    parser.add_argument('--token', required=True, help='GitHub token')
    parser.add_argument('--repo', required=True, help='Repository (owner/name)')
    parser.add_argument('--pr-number', type=int, required=True, help='PR number')
    parser.add_argument('--comment-file', required=True, help='File containing comment body')
    parser.add_argument('--strategy', choices=['update', 'mark_outdated', 'replace_all'], 
                       default='update', help='Comment update strategy')
    
    args = parser.parse_args()
    
    # Parse repository
    try:
        repo_owner, repo_name = args.repo.split('/')
    except ValueError:
        print("Error: Repository must be in format 'owner/name'")
        return 1
    
    # Read comment body
    try:
        with open(args.comment_file, 'r') as f:
            comment_body = f.read()
    except FileNotFoundError:
        print(f"Error: Comment file '{args.comment_file}' not found")
        return 1
    
    # Manage comments
    manager = PRCommentManager(args.token, repo_owner, repo_name)
    success = manager.post_or_update_comment(
        args.pr_number, 
        comment_body, 
        args.strategy
    )
    
    if success:
        print(f"✅ Successfully managed PR #{args.pr_number} comments")
        return 0
    else:
        print(f"❌ Failed to manage PR #{args.pr_number} comments")
        return 1


if __name__ == "__main__":
    exit(main()) 
#!/usr/bin/env python3
"""Script to run security checks on dependencies and code."""

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

def run_safety_check(requirements_file: Path) -> Tuple[int, List[dict]]:
    """Run safety check on Python dependencies.
    
    Args:
        requirements_file: Path to requirements.txt
        
    Returns:
        Tuple of (exit code, list of vulnerabilities)
    """
    try:
        # Run safety check and capture JSON output
        result = subprocess.run(
            [
                sys.executable,
                '-m', 'safety', 'check',
                '-r', str(requirements_file),
                '--json'
            ],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Parse vulnerabilities from JSON output
        vulnerabilities = []
        if result.stdout:
            try:
                data = json.loads(result.stdout)
                vulnerabilities = data.get('vulnerabilities', [])
            except json.JSONDecodeError:
                print("Error parsing safety output", file=sys.stderr)
        
        return result.returncode, vulnerabilities
        
    except Exception as e:
        print(f"Error running safety: {e}", file=sys.stderr)
        return 1, []

def run_bandit(paths: Optional[Sequence[str]] = None) -> int:
    """Run bandit security checks on specified paths."""
    project_root = Path(__file__).parent.parent
    
    if paths is None:
        paths = ['legion']
    
    try:
        # Run bandit with more verbose output
        result = subprocess.run(
            [
                sys.executable,
                '-m', 'bandit',
                '-r',  # Recursive
                '-ll',  # Log level
                '-i',  # Show info msgs
                '-f', 'custom',  # Use custom format for detailed output
                '--msg-template', '{severity}: {msg} [{test_id}] in {relpath}:{line}',
                '-v',  # Verbose output
                *paths
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Always show scan summary
        print("\nScan Summary:")
        print("-" * 40)
        
        # Print any found issues
        if result.stdout:
            lines = result.stdout.split('\n')
            issues_found = False
            for line in lines:
                if line.strip():  # Only print non-empty lines
                    if any(level in line.lower() for level in ['low:', 'medium:', 'high:']):
                        if not issues_found:
                            print("\nIssues Found:")
                            print("-" * 40)
                            issues_found = True
                        print(line)
            
            if not issues_found:
                print("No security issues found.")
        else:
            print("No security issues found.")
        
        # Run a second time to get metrics
        metrics_result = subprocess.run(
            [
                sys.executable,
                '-m', 'bandit',
                '-r',
                *paths
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Print relevant metrics
        if metrics_result.stdout:
            lines = metrics_result.stdout.split('\n')
            print("\nMetrics:")
            print("-" * 40)
            for line in lines:
                if any(x in line.lower() for x in ['total lines of code:', 'total issues', 'files skipped']):
                    print(line.strip())
                
        return result.returncode
        
    except Exception as e:
        print(f"Error running bandit: {e}", file=sys.stderr)
        return 1

def format_vulnerability(vuln: dict) -> str:
    """Format a vulnerability for display.
    
    Args:
        vuln: Vulnerability data from safety
        
    Returns:
        Formatted string describing the vulnerability
    """
    return (
        f"Package: {vuln.get('package_name', 'Unknown')}\n"
        f"Installed: {vuln.get('installed_version', 'Unknown')}\n"
        f"Vulnerable: {vuln.get('vulnerable_spec', 'Unknown')}\n"
        f"CVE: {vuln.get('cve', 'No CVE')}\n"
        f"Advisory: {vuln.get('advisory', 'No details')}\n"
    )

def main() -> int:
    """Run the security scanning process."""
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / 'requirements.txt'
    
    print("Running dependency security scan...")
    safety_code, vulnerabilities = run_safety_check(requirements_file)
    
    if vulnerabilities:
        print("\nFound security vulnerabilities in dependencies:")
        for vuln in vulnerabilities:
            print("\n" + "=" * 40)
            print(format_vulnerability(vuln))
    else:
        print("\nNo known vulnerabilities found in dependencies.")
        
    print("\nRunning code security scan...")
    bandit_code = run_bandit()
    
    # Return non-zero if either check failed
    return max(safety_code, bandit_code)

if __name__ == '__main__':
    sys.exit(main()) 
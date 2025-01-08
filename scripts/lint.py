#!/usr/bin/env python3
"""Script to run code style checks and auto-fix issues."""

import subprocess
import sys
from pathlib import Path
from typing import Optional

def run_ruff(fix: bool = False, unsafe_fixes: bool = False) -> int:
    """Run ruff code style checks and optionally fix issues.
    
    Args:
        fix: If True, attempt to automatically fix issues
        unsafe_fixes: If True, enable additional automated fixes
        
    Returns:
        Exit code from ruff
    """
    project_root = Path(__file__).parent.parent
    
    # Base command with configuration
    cmd = [
        sys.executable,  # Use current Python interpreter
        '-m', 'ruff', 'check',
        '--no-cache',  # Disable cache to avoid any config caching issues
        '--line-length=100',
        '--target-version=py38',
        '--select=E,F,W,I,N,D,Q',
        # Ignore more rules that are too strict or conflict
        '--ignore=UP015,UP009,D100,D101,D102,D103,D107,D203,D212,D400,D415,D213',
        '--exclude=.git,__pycache__,.ruff_cache,build,dist',
        'legion', 'tests', 'examples'
    ]
    
    # Add fix flags
    if fix:
        cmd.append('--fix')
        if unsafe_fixes:
            cmd.append('--unsafe-fixes')
    
    # Run ruff with explicit environment to avoid config file lookup
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        env={"NO_COLOR": "1"}  # Disable color output for cleaner logs
    )
    
    # Print output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
        
    error_count = len(result.stdout.strip().split('\n')) if result.stdout else 0
    action = "Found" if not fix else "Remaining"
    print(f"\n{action} {error_count} style issues")
    
    return result.returncode

def main() -> int:
    """Run the linting process."""
    # First try safe fixes
    print("Attempting safe auto-fixes...")
    run_ruff(fix=True, unsafe_fixes=False)
    
    # Then try unsafe fixes
    print("\nAttempting additional auto-fixes...")
    run_ruff(fix=True, unsafe_fixes=True)
    
    # Final check
    print("\nRunning final style check...")
    return run_ruff(fix=False)

if __name__ == '__main__':
    sys.exit(main())
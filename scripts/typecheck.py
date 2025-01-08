#!/usr/bin/env python3
"""Script to run static type checking with mypy."""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence


def run_mypy(paths: Optional[Sequence[str]] = None) -> int:
    """Run mypy type checker on specified paths.
    
    Args:
        paths: List of paths to check. If None, checks default paths.
        
    Returns:
        Exit code from mypy (0 for success, non-zero for errors)
    """
    project_root = Path(__file__).parent.parent
    
    if paths is None:
        paths = ['legion']
    
    cmd = [
        sys.executable,
        '-m', 'mypy',
        '--config-file', str(project_root / 'mypy.ini'),
        '--show-error-codes',
        '--pretty',
        '--show-column-numbers',
        '--show-error-context',
        '--no-error-summary',
        '--hide-error-codes',
        *paths
    ]
    
    try:
        # Run mypy
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Print output
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
            
        # Count actual errors (ignore notes and other info)
        error_count = len([
            line for line in result.stdout.split('\n') 
            if line.strip() and ': error:' in line
        ])
        
        if error_count > 0:
            print(f"\nFound {error_count} type issues")
        else:
            print("\nNo type issues found!")
            
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"Error running mypy: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Run the type checking process."""
    return run_mypy()


if __name__ == '__main__':
    sys.exit(main()) 
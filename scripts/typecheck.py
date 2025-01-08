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

    print("\n" + "=" * 80)
    print("Running type checking...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80 + "\n")

    try:
        # Run mypy
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False
        )

        # Always print command output for visibility
        if result.stdout:
            print("Output:")
            print("-" * 40)
            print(result.stdout.strip())

        if result.stderr:
            print("\nErrors:")
            print("-" * 40)
            print(result.stderr.strip(), file=sys.stderr)

        # Count actual errors (ignore notes and other info)
        error_count = len([
            line for line in result.stdout.split('\n')
            if line.strip() and ': error:' in line
        ])

        print("\n" + "=" * 40)
        if error_count > 0:
            print(f"Found {error_count} type issues")
        else:
            print("No type issues found!")
        print("=" * 40 + "\n")

        if result.returncode != 0:
            print(f"Type checking failed with exit code: {result.returncode}", file=sys.stderr)

        return result.returncode

    except subprocess.CalledProcessError as e:
        print(f"Error running mypy: {str(e)}", file=sys.stderr)
        if e.stdout:
            print("\nOutput:", file=sys.stderr)
            print(e.stdout.decode().strip(), file=sys.stderr)
        if e.stderr:
            print("\nErrors:", file=sys.stderr)
            print(e.stderr.decode().strip(), file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error during type checking: {str(e)}", file=sys.stderr)
        return 1


def main() -> int:
    """Run the type checking process."""
    try:
        return run_mypy()
    except Exception as e:
        print(f"Fatal error in type checking: {str(e)}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

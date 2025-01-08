import sys
from pathlib import Path

import pytest


def run_tests():
    """Run all tests"""
    test_dir = Path(__file__).parent

    print("\n" + "=" * 80)
    print("Running test suite...")
    print("=" * 80 + "\n")

    # Run tests with pytest
    args = [
        str(test_dir),  # Test directory
        "-v",  # Verbose output
        "--tb=short",  # Shorter traceback format
        "-p", "no:warnings",  # Disable warning capture plugin
        "-p", "asyncio",  # Enable asyncio plugin
        "--no-header",  # Skip pytest header
        "--capture=no",  # Don't capture stdout/stderr
    ]

    try:
        print(f"Test command: pytest {' '.join(args)}\n")
        result = pytest.main(args)

        if result != 0:
            print(f"\nTests failed with exit code: {result}", file=sys.stderr)
            if result == 5:
                print("No tests were collected. Check your test discovery settings.", file=sys.stderr)

        return result

    except Exception as e:
        print(f"\nFatal error running tests: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())

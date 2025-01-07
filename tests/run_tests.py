import pytest
import sys
from pathlib import Path

def run_tests():
    """Run all tests"""
    test_dir = Path(__file__).parent
    
    # Run tests with pytest
    args = [
        str(test_dir),  # Test directory
        "-v",  # Verbose output
        "--tb=short",  # Shorter traceback format
        "-p", "no:warnings",  # Disable warning capture plugin
        "-p", "asyncio"  # Enable asyncio plugin
    ]
    
    # Run pytest with our arguments
    return pytest.main(args)

if __name__ == "__main__":
    sys.exit(run_tests()) 
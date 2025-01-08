import os

import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Run all tests in the directory
    pytest.main([
        os.path.dirname(__file__),  # Run tests in current directory
        "-v",                       # Verbose output
        "--asyncio-mode=auto"       # Enable async test support
    ])

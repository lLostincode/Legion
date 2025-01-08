import os
from pathlib import Path

from dotenv import load_dotenv


def pytest_configure(config):
    """Called before test collection, ensures environment is properly configured"""
    # Get the project root directory (where .env file is located)
    root_dir = Path(__file__).parent.parent

    # Load environment variables from .env file
    env_file = root_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        raise RuntimeError(f"No .env file found at {env_file}")

    # Verify required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Register custom markers
    config.addinivalue_line(
        "markers", "integration: mark test as requiring external API access"
    )

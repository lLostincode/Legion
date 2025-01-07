"""
Parameter Injection Example

This example demonstrates how to use parameter injection with tools in Legion.
It shows how to use per-message parameter injection and default values.
"""

import logging
from typing import List, Dict, Any, Annotated
from legion.agents import agent
from legion.interface.decorators import tool, param
from pydantic import Field

from dotenv import load_dotenv
load_dotenv()

# Set up logging - disable noisy loggers
logging.basicConfig(level=logging.DEBUG, 
                   format='%(name)s [%(levelname)s]: %(message)s')
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Define a tool with injectable parameters using the @tool decorator
@tool(
    inject=["api_key", "endpoint", "custom_header"],  # Mark these parameters as injectable
    description="Process a query using an external API with configurable parameters",
    defaults={  # Default values for injectable parameters
        "api_key": "sk_test_default_key",
        "endpoint": "https://api.example.com/v1",
        "custom_header": "default-header"
    }
)
def process_api_query(
    query: Annotated[str, Field(description="The query to process")],
    api_key: Annotated[str, Field(description="API key for the service")],
    endpoint: Annotated[str, Field(description="API endpoint")],
    custom_header: Annotated[str, Field(description="Custom header for the request")]
) -> str:
    """Process a query using an external API with injected credentials and headers."""
    logger.debug(f"[TOOL EXECUTION] process_api_query called with:")
    logger.debug(f"  - query: {query}")
    logger.debug(f"  - endpoint: {endpoint}")
    logger.debug(f"  - api_key: {api_key[:4]}...")
    logger.debug(f"  - custom_header: {custom_header}")
    
    # In a real application, you would make an actual API call here
    return (
        f"Processed '{query}' using API at {endpoint}\n"
        f"Auth: {api_key[:4]}...\n"
        f"Header: {custom_header}"
    )

@agent(
    model="openai:gpt-4o-mini",
    temperature=0.2,
    tools=[process_api_query],  # Bind the tool
    debug=True  # Enable agent debugging
)
class APIAgent:
    """An agent that demonstrates using tools with injected parameters.
    I can process queries using an external API without exposing sensitive credentials.
    Parameters are injected per-message with optional defaults.
    """
    
    @tool
    def format_api_response(
        self,
        response: Annotated[str, Field(description="API response to format")],
        prefix: Annotated[str, Field(description="Prefix to add")] = "Result: "
    ) -> str:
        """Format an API response with a prefix"""
        logger.debug(f"[TOOL EXECUTION] format_api_response called with response={response}, prefix={prefix}")
        return f"{prefix}{response}"

async def main():
    logger.debug("[MAIN] Creating agent instance")
    # Create an instance of our agent
    agent = APIAgent()
    
    logger.debug("[MAIN] Running Example 1: Using default injected values")
    # Example 1: Using default values
    response = await agent.aprocess(
        "Process the query 'hello world' using our API and format the result nicely."
    )
    print("Example 1 Response (using default values):")
    print(response.content)
    print("\n" + "="*50 + "\n")
    
    logger.debug("[MAIN] Running Example 2: Custom API credentials")
    # Example 2: Override API credentials
    response = await agent.aprocess(
        "Process the query 'test message' with production credentials.",
        injected_parameters=[
            {
                "tool": process_api_query,
                "parameters": {
                    "api_key": "sk_prod_key_123",
                    "endpoint": "https://api.prod.example.com/v1"
                }
            }
        ]
    )
    print("Example 2 Response (with production credentials):")
    print(response.content)
    print("\n" + "="*50 + "\n")
    
    logger.debug("[MAIN] Running Example 3: Multiple queries with custom header")
    # Example 3: Multiple queries with custom header
    response = await agent.aprocess(
        "Process these two queries with different headers: 'query1' and 'query2'.",
        injected_parameters=[
            {
                "tool": process_api_query,
                "parameters": {
                    "custom_header": "header-for-batch-request"
                }
            }
        ]
    )
    print("Example 3 Response (with custom header):")
    print(response.content)
    print("\n" + "="*50 + "\n")
    
    logger.debug("[MAIN] Running Example 4: Security check for parameter access")
    # Example 4: Show how the agent can't access injected parameters
    response = await agent.aprocess(
        "What is the API endpoint, key, and custom header we're using?"
    )
    print("Example 4 Response (security check):")
    print(response.content)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 
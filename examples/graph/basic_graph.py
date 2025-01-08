"""Basic Graph Example

This example demonstrates how to create a simple graph using Legion's decorator syntax.
It shows how to add nodes, connect them with edges, and execute the graph.
This example uses a sequential execution mode.
"""

import asyncio
from typing import Annotated, List  # noqa: F401

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from legion.agents.decorators import agent
from legion.blocks.decorators import block
from legion.graph.channels import LastValue
from legion.graph.decorators import graph
from legion.graph.edges.base import EdgeBase
from legion.graph.nodes.decorators import node  # noqa: F401
from legion.interface.decorators import tool

load_dotenv()


# Define a simple data model for type safety
class TextData(BaseModel):
    text: str = Field(description="Input text to process")


# Define a simple block
@block(
    input_schema=TextData,
    output_schema=TextData,
    tags=["text", "preprocessing"]
)
def normalize_text(input_data: TextData) -> TextData:
    """Normalize text by removing extra whitespace."""
    text = " ".join(input_data.text.split())
    return TextData(text=text)


# Define a simple agent
@agent(model="openai:gpt-4o-mini", temperature=0.2)
class Summarizer:
    """An agent that summarizes text."""

    @tool
    def count_words(
        self,
        text: Annotated[str, Field(description="Text to count words in")]
    ) -> int:
        """Count the number of words in a text."""
        return len(text.split())


# Define a simple edge
class TextEdge(EdgeBase):
    """Edge for connecting text processing nodes."""

    pass


# Define the graph using decorator syntax
@graph(name="basic_text_processing", description="A simple graph for processing text")
class TextProcessingGraph:
    """A graph that first normalizes text and then summarizes it.
    This demonstrates a basic sequential workflow.
    """

    # Define nodes
    normalizer = normalize_text
    summarizer = Summarizer()

    # Define edges - these will be processed by the graph decorator
    edges = [
        {
            "edge_type": TextEdge,
            "source_node": "normalizer",
            "target_node": "summarizer",
            "source_channel": "output",
            "target_channel": "input"
        }
    ]

    # Define input and output channels
    input_channel = LastValue(type_hint=str)
    output_channel = LastValue(type_hint=str)

    async def process(self, input_text: str) -> str:
        """Process text through the graph."""
        # Set input value
        self.input_channel.set(input_text)

        # Execute the graph
        await self.graph.execute()

        # Get output value
        return self.output_channel.get()


async def main():
    # Create an instance of the graph
    text_graph = TextProcessingGraph()

    # Process some text
    input_text = "  This    is   a   test   with    extra   spaces.  "
    output_text = await text_graph.process(input_text)

    print(f"Original Text: '{input_text}'")
    print(f"Processed Text: '{output_text}'")


if __name__ == "__main__":
    asyncio.run(main())

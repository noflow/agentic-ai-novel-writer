"""
Base class for all agent tools.

Every tool needs:
  - name:         unique identifier the LLM uses to call it
  - description:  tells the LLM when/how to use the tool
  - input_schema: JSON Schema describing the expected parameters
  - run():        the actual logic that executes when the tool is called

The Agent automatically converts these into the Anthropic API's
tool format and routes calls back to the right run() method.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class for agent tools."""

    # Subclasses MUST set these
    name: str = ""
    description: str = ""
    input_schema: dict = {}

    def to_api_format(self) -> dict:
        """Convert this tool to Anthropic's API tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    @abstractmethod
    def run(self, **kwargs) -> str:
        """
        Execute the tool with the given arguments.
        Must return a string result that gets fed back to the LLM.
        """
        pass

    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"

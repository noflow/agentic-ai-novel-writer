"""
Tool Registry -- central place to register and look up tools.

When you create a new tool, just import it here and add it to
DEFAULT_TOOLS. The Agent will automatically pick it up.

In Phase 2, different agents will get different tool subsets
(e.g., the Researcher gets web_search, the Writer gets file_ops).
"""

from __future__ import annotations

from tools.base import BaseTool
from tools.calculator import CalculatorTool
from tools.web_search import WebSearchTool
from tools.file_ops import ReadFileTool, WriteFileTool, AppendFileTool, ListFilesTool
from tools.clock import ClockTool
from tools.docx_tool import CreateDocxTool, TxtToDocxTool
from tools.text_stats import CountFileCharsTool, ValidateChapterTool, ValidateChunkTool


class ToolRegistry:
    """Manages a collection of tools for an agent."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Add a tool to the registry."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def to_api_format(self) -> list[dict]:
        """Convert all tools to Anthropic API format."""
        return [tool.to_api_format() for tool in self._tools.values()]

    def __repr__(self) -> str:
        names = ", ".join(self._tools.keys())
        return f"<ToolRegistry: [{names}]>"


def create_default_registry() -> ToolRegistry:
    """Create a registry with all the built-in tools."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool())
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(AppendFileTool())
    registry.register(ListFilesTool())
    registry.register(ClockTool())
    registry.register(CreateDocxTool())
    registry.register(TxtToDocxTool())
    registry.register(CountFileCharsTool())
    registry.register(ValidateChapterTool())
    registry.register(ValidateChunkTool())
    return registry

"""
Base LLM Provider interface.

All providers (Anthropic, OpenAI, Gemini, Grok) implement this interface.
The Agent class talks to providers through this abstraction, so swapping
providers is just changing one config line.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    text: str                          # The text content (may be empty if tool call)
    stop_reason: str                   # "end_turn", "tool_use", "max_tokens"
    tool_calls: list[ToolCall]         # List of tool calls (empty if none)
    raw: object                        # The raw response from the provider


@dataclass
class ToolCall:
    """A single tool/function call from the LLM."""
    id: str           # Unique ID for this call
    name: str         # Tool/function name
    arguments: dict   # The arguments as a dict


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    # Subclasses set these
    provider_name: str = ""
    default_model: str = ""

    @abstractmethod
    def call(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict],
        max_tokens: int,
        model: str,
    ) -> LLMResponse:
        """
        Make a completion call to the LLM.

        Args:
            messages:      Conversation history in a universal format
            system_prompt: System instructions
            tools:         Tool definitions (Anthropic format -- provider converts)
            max_tokens:    Max response tokens
            model:         Model ID to use

        Returns:
            Standardized LLMResponse
        """
        pass

    @abstractmethod
    def format_tool_result(self, tool_call_id: str, result: str) -> dict:
        """Format a tool result message for this provider's API."""
        pass

    @abstractmethod
    def format_assistant_message(self, response: LLMResponse) -> dict:
        """Format the assistant's response as a message for conversation history."""
        pass

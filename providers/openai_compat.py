"""
OpenAI-compatible provider.

Works with:
  - OpenAI (ChatGPT): api_key=OPENAI_API_KEY, base_url=default
  - xAI (Grok):       api_key=XAI_API_KEY,    base_url=https://api.x.ai/v1
  - Google (Gemini):   api_key=GEMINI_API_KEY,  base_url=https://generativelanguage.googleapis.com/v1beta/openai/

All three use the same OpenAI chat completions format for tool calling,
so one provider class handles them all.

Required install: pip install openai
"""

from __future__ import annotations
import json
import os
from providers import BaseLLMProvider, LLMResponse, ToolCall


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    Provider for any OpenAI-compatible API.
    Handles ChatGPT, Grok, and Gemini through different base URLs.
    """

    def __init__(self, provider_name: str, api_key: str, base_url: str | None = None,
                 default_model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI SDK required. Install with: pip install openai"
            )

        self.provider_name = provider_name
        self.default_model = default_model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def _convert_tools(self, anthropic_tools: list[dict]) -> list[dict]:
        """Convert Anthropic tool format to OpenAI function format."""
        openai_tools = []
        for tool in anthropic_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        return openai_tools

    def _convert_messages(self, messages: list[dict], system_prompt: str) -> list[dict]:
        """Convert Anthropic message format to OpenAI format."""
        openai_messages = [{"role": "system", "content": system_prompt}]

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                openai_messages.append({"role": role, "content": content})

            elif isinstance(content, list):
                # Handle Anthropic's content blocks
                if role == "assistant":
                    # Extract text and tool calls
                    text_parts = []
                    tool_calls = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "tool_use":
                                tool_calls.append({
                                    "id": block["id"],
                                    "type": "function",
                                    "function": {
                                        "name": block["name"],
                                        "arguments": json.dumps(block.get("input", {})),
                                    },
                                })

                    msg_dict = {"role": "assistant", "content": "\n".join(text_parts) or None}
                    if tool_calls:
                        msg_dict["tool_calls"] = tool_calls
                    openai_messages.append(msg_dict)

                elif role == "user":
                    # Could be tool results or text
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "tool_result":
                                openai_messages.append({
                                    "role": "tool",
                                    "tool_call_id": block.get("tool_use_id", ""),
                                    "content": str(block.get("content", "")),
                                })
                            elif block.get("type") == "text":
                                openai_messages.append({
                                    "role": "user",
                                    "content": block.get("text", ""),
                                })
                        elif isinstance(block, str):
                            openai_messages.append({"role": "user", "content": block})

        return openai_messages

    def call(self, messages, system_prompt, tools, max_tokens, model) -> LLMResponse:
        openai_messages = self._convert_messages(messages, system_prompt)
        openai_tools = self._convert_tools(tools) if tools else None

        kwargs = {
            "model": model,
            "messages": openai_messages,
        }

        # Grok 4+ reasoning models use max_completion_tokens instead of max_tokens
        if "grok-4" in model and "non-reasoning" not in model:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        if openai_tools:
            kwargs["tools"] = openai_tools

        response = self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        message = choice.message

        # Extract text
        text = message.content or ""

        # Extract tool calls
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        # Map finish reason
        finish = choice.finish_reason
        if finish == "stop":
            stop_reason = "end_turn"
        elif finish == "tool_calls":
            stop_reason = "tool_use"
        elif finish == "length":
            stop_reason = "max_tokens"
        else:
            # If there are tool calls, it's a tool_use even if finish says "stop"
            stop_reason = "tool_use" if tool_calls else "end_turn"

        return LLMResponse(
            text=text,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            raw=response,
        )

    def format_tool_result(self, tool_call_id: str, result: str) -> dict:
        # OpenAI uses "tool" role messages, handled in _convert_messages
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": result,
        }

    def format_assistant_message(self, response: LLMResponse) -> dict:
        # Store in Anthropic format (our internal standard) -- converted on the fly
        content = []
        if response.text:
            content.append({"type": "text", "text": response.text})
        for tc in response.tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })
        return {"role": "assistant", "content": content}


# --- Convenience factory functions ---

def create_openai_provider() -> OpenAICompatibleProvider:
    """Create an OpenAI (ChatGPT) provider."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Add it to your .env file:\n"
            "  OPENAI_API_KEY=sk-your-key-here"
        )
    return OpenAICompatibleProvider(
        provider_name="openai",
        api_key=api_key,
        default_model="gpt-4o-mini",
    )


def create_grok_provider() -> OpenAICompatibleProvider:
    """Create an xAI (Grok) provider."""
    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "XAI_API_KEY not set. Add it to your .env file:\n"
            "  XAI_API_KEY=xai-your-key-here"
        )
    return OpenAICompatibleProvider(
        provider_name="grok",
        api_key=api_key,
        base_url="https://api.x.ai/v1",
        default_model="grok-4-1-fast-non-reasoning",
    )


def create_gemini_provider() -> OpenAICompatibleProvider:
    """Create a Google (Gemini) provider."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set. Add it to your .env file:\n"
            "  GEMINI_API_KEY=your-key-here"
        )
    return OpenAICompatibleProvider(
        provider_name="gemini",
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-2.0-flash",
    )

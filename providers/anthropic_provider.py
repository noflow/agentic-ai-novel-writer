"""Anthropic (Claude) provider."""

from __future__ import annotations
from anthropic import Anthropic
from providers import BaseLLMProvider, LLMResponse, ToolCall


class AnthropicProvider(BaseLLMProvider):
    provider_name = "anthropic"
    default_model = "claude-haiku-4-5-20251001"

    def __init__(self):
        self.client = Anthropic()

    def call(self, messages, system_prompt, tools, max_tokens, model) -> LLMResponse:
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        # Extract text and tool calls
        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        # Map stop reason
        stop = response.stop_reason
        if stop == "end_turn":
            stop_reason = "end_turn"
        elif stop == "tool_use":
            stop_reason = "tool_use"
        elif stop == "max_tokens":
            stop_reason = "max_tokens"
        else:
            stop_reason = stop or "end_turn"

        return LLMResponse(
            text="\n".join(text_parts),
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            raw=response,
        )

    def format_tool_result(self, tool_call_id: str, result: str) -> dict:
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": result,
        }

    def format_assistant_message(self, response: LLMResponse) -> dict:
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

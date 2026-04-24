"""
Agent -- the core agentic loop with multi-provider support.

Handles tool_use/tool_result pairing carefully to avoid API errors.
"""

from __future__ import annotations
import json
from config import MAX_TOKENS, MAX_AGENT_TURNS, MAX_CONTINUATIONS, DEFAULT_SYSTEM_PROMPT, MAX_CONTEXT_CHARS
from tools import ToolRegistry, create_default_registry
from providers import BaseLLMProvider, LLMResponse


def get_default_provider() -> BaseLLMProvider:
    from providers.anthropic_provider import AnthropicProvider
    from config import MODEL
    provider = AnthropicProvider()
    provider.default_model = MODEL
    return provider


class Agent:
    def __init__(
        self,
        name: str = "Agent",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tool_registry: ToolRegistry | None = None,
        provider: BaseLLMProvider | None = None,
        model: str | None = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.provider = provider or get_default_provider()
        self.model = model or self.provider.default_model
        self.tool_registry = tool_registry or create_default_registry()
        self.conversation_history: list[dict] = []

    def run(self, user_message: str) -> str:
        self._check_context_size()

        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        for turn in range(MAX_AGENT_TURNS):
            print(f"  [{self.name}] Turn {turn + 1}...", end="", flush=True)

            response = self._call_api()

            # --- Handle based on what the response contains ---

            # If response has tool calls, ALWAYS handle them first
            # (regardless of stop_reason -- some providers say "stop"
            #  even when there are tool calls)
            if response.tool_calls:
                print(" -> Tool call")
                self._handle_tool_use(response)
                continue

            # No tool calls -- it's a text response
            if response.stop_reason == "max_tokens":
                print(" -> Hit token limit, continuing...")
                partial = self._save_text_response(response)
                continuation = self._auto_continue()
                return partial + continuation

            else:
                # end_turn or anything else -- we're done
                print(" -> Final answer")
                return self._save_text_response(response)

        print(f"  [{self.name}] Hit max turns ({MAX_AGENT_TURNS}), wrapping up.")
        return self._force_final_answer()

    def _call_api(self) -> LLMResponse:
        """Call the LLM with automatic retry on rate limits."""
        import time as _time

        max_retries = 5
        for attempt in range(max_retries):
            try:
                return self.provider.call(
                    messages=self.conversation_history,
                    system_prompt=self.system_prompt,
                    tools=self.tool_registry.to_api_format(),
                    max_tokens=MAX_TOKENS,
                    model=self.model,
                )
            except Exception as e:
                error_str = str(e).lower()
                # Retry on rate limits and overload
                if ("rate_limit" in error_str or "rate limit" in error_str
                        or "429" in error_str or "overloaded" in error_str
                        or "529" in error_str):
                    wait = min(15 * (attempt + 1), 60)  # 15s, 30s, 45s, 60s, 60s
                    print(f"\n  [{self.name}] Rate limited. Waiting {wait}s "
                          f"(attempt {attempt + 1}/{max_retries})...",
                          end="", flush=True)
                    _time.sleep(wait)
                    print(" retrying", end="", flush=True)
                    continue
                else:
                    raise  # Non-rate-limit errors bubble up immediately

        # All retries exhausted
        raise Exception(
            f"Rate limit exceeded after {max_retries} retries. "
            "Try again in a minute, or upgrade your API plan for higher limits."
        )

    # -------------------------------------------------------------------
    # Response handling -- text only (no tool calls)
    # -------------------------------------------------------------------
    def _save_text_response(self, response: LLMResponse) -> str:
        """Save a text-only response. Strips any accidental tool_use blocks."""
        # Only save the text part -- never save dangling tool_use blocks
        text = response.text or ""
        self.conversation_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": text}] if text else [{"type": "text", "text": ""}],
        })
        return text

    # -------------------------------------------------------------------
    # Tool use handling -- always pairs tool_use with tool_result
    # -------------------------------------------------------------------
    def _handle_tool_use(self, response: LLMResponse) -> None:
        """
        Process tool calls. This ALWAYS adds:
          1. An assistant message with the tool_use blocks
          2. A user message with matching tool_result blocks
        So they're always properly paired.
        """
        # Build assistant message with text + tool_use blocks
        assistant_msg = self.provider.format_assistant_message(response)
        self.conversation_history.append(assistant_msg)

        # Execute each tool and collect results
        tool_results = []
        for tc in response.tool_calls:
            print(f"    -> Calling {tc.name}({json.dumps(tc.arguments, indent=2)[:200]})")

            tool = self.tool_registry.get(tc.name)
            if tool is None:
                result = f"Error: Unknown tool '{tc.name}'"
            else:
                try:
                    result = tool.run(**tc.arguments)
                except Exception as e:
                    result = f"Error executing {tc.name}: {e}"

            if len(result) > 5000:
                result = result[:5000] + "\n... [truncated]"

            print(f"    <- Result: {result[:150]}...")

            tool_results.append(
                self.provider.format_tool_result(tc.id, result)
            )

        # Add ALL tool results as a user message -- must match every tool_use
        self.conversation_history.append({
            "role": "user",
            "content": tool_results,
        })

    # -------------------------------------------------------------------
    # Auto-continuation for truncated responses
    # -------------------------------------------------------------------
    def _auto_continue(self) -> str:
        full_continuation = ""

        for i in range(MAX_CONTINUATIONS):
            print(f"  [{self.name}] Continuation {i + 1}/{MAX_CONTINUATIONS}...",
                  end="", flush=True)

            self.conversation_history.append({
                "role": "user",
                "content": (
                    "Your response was cut off. Continue EXACTLY where you "
                    "left off. Do not repeat what you already wrote. "
                    "Do not add any preamble -- just pick up mid-sentence if needed."
                ),
            })

            response = self._call_api()

            # If continuation triggers tool calls, handle them and loop
            if response.tool_calls:
                print(" -> Tool call in continuation")
                self._handle_tool_use(response)
                continue

            chunk = self._save_text_response(response)
            full_continuation += chunk

            if response.stop_reason != "max_tokens":
                print(" -> Complete")
                break
            else:
                print(" -> Still going...")

        return full_continuation

    # -------------------------------------------------------------------
    # Context management
    # -------------------------------------------------------------------
    def _check_context_size(self):
        total_chars = sum(self._estimate_message_chars(m)
                         for m in self.conversation_history)

        if total_chars > MAX_CONTEXT_CHARS:
            print(f"  [{self.name}] Context too large ({total_chars} chars), trimming...")

            if len(self.conversation_history) > 8:
                # Keep first message and recent messages
                # But make sure we don't split a tool_use/tool_result pair
                first = self.conversation_history[0]
                recent = self.conversation_history[-6:]

                # Check if the first message in 'recent' is a tool_result
                # without its preceding tool_use -- if so, include one more
                if recent and isinstance(recent[0].get("content"), list):
                    first_block = recent[0]["content"][0] if recent[0]["content"] else {}
                    if isinstance(first_block, dict) and first_block.get("type") == "tool_result":
                        recent = self.conversation_history[-7:]

                summary = {
                    "role": "user",
                    "content": (
                        "[Note: Earlier messages were trimmed to save space. "
                        "The original task and recent exchanges are preserved.]"
                    ),
                }
                self.conversation_history = [first, summary] + recent
                new_size = sum(self._estimate_message_chars(m)
                              for m in self.conversation_history)
                print(f"  [{self.name}] Trimmed: {total_chars} -> {new_size} chars")

    def _estimate_message_chars(self, message: dict) -> int:
        content = message.get("content", "")
        if isinstance(content, str):
            return len(content)
        elif isinstance(content, list):
            total = 0
            for block in content:
                if isinstance(block, dict):
                    total += len(str(block.get("text", "")))
                    total += len(str(block.get("content", "")))
                    total += len(json.dumps(block.get("input", {})))
            return total
        return 0

    def _force_final_answer(self) -> str:
        self.conversation_history.append({
            "role": "user",
            "content": (
                "You've used many tool calls. Please provide your best "
                "final answer now based on what you've gathered so far."
            ),
        })
        response = self._call_api()
        # Even here, handle tool calls if they appear
        if response.tool_calls:
            self._handle_tool_use(response)
            response = self._call_api()
        return self._save_text_response(response)

    def reset(self) -> None:
        self.conversation_history = []

    def __repr__(self) -> str:
        return f"<Agent '{self.name}' | {self.provider.provider_name}:{self.model} | {self.tool_registry}>"

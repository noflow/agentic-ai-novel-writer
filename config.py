"""
Configuration for the Agentic AI framework.

Supports multiple LLM providers:
  - Anthropic (Claude)  -- default
  - OpenAI (ChatGPT)
  - xAI (Grok)
  - Google (Gemini)

Set your API keys in the .env file.
"""

import os
import sys
from pathlib import Path


# --- Load .env file ---
def _load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    os.environ.setdefault(key, value)

_load_env()


# --- Validate at least one API key ---
def check_api_key():
    keys = {
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "grok": os.environ.get("XAI_API_KEY", ""),
        "gemini": os.environ.get("GEMINI_API_KEY", ""),
    }

    has_any = any(
        v and v not in ("sk-ant-your-key-here", "sk-your-key-here",
                        "xai-your-key-here", "your-key-here")
        for v in keys.values()
    )

    if not has_any:
        print()
        print("=" * 56)
        print("  !!  No API keys configured!")
        print("=" * 56)
        print()
        print("  Edit the .env file and add at least one API key:")
        print()
        print("  Anthropic: ANTHROPIC_API_KEY=sk-ant-...")
        print("  OpenAI:    OPENAI_API_KEY=sk-...")
        print("  Grok:      XAI_API_KEY=xai-...")
        print("  Gemini:    GEMINI_API_KEY=...")
        print()
        print("  Get keys at:")
        print("    https://console.anthropic.com")
        print("    https://platform.openai.com/api-keys")
        print("    https://console.x.ai")
        print("    https://aistudio.google.com/apikey")
        print()
        sys.exit(1)


# --- Provider Settings ---
PROVIDER = os.environ.get("PROVIDER", "anthropic")

# Model IDs per provider (overridable via .env)
MODEL = os.environ.get("MODEL", "claude-haiku-4-5-20251001")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
GROK_MODEL = os.environ.get("GROK_MODEL", "grok-4-1-fast-non-reasoning")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# --- Token Settings ---
MAX_TOKENS = 8192
MAX_AGENT_TURNS = 12
MAX_CONTINUATIONS = 5
TEMPERATURE = 1.0
CHARS_PER_TOKEN_ESTIMATE = 4
MAX_CONTEXT_CHARS = 150_000

# --- Chapter Writing Settings ---
CHUNK_MIN_CHARS = 10_000
CHUNK_MAX_CHARS = 12_000
CHAPTER_MIN_CHARS = 36_000
CHAPTER_MAX_CHARS = 45_000
TARGET_CHUNKS_PER_CHAPTER = 4


# --- Chapter Validation Helpers ---
def count_chars(text: str) -> int:
    """Count characters in text (excluding headers)."""
    return len(text)


def validate_chunk(text: str, chunk_num: int) -> tuple[bool, str]:
    """Validate a chunk is within character count bounds."""
    count = count_chars(text)
    if count < CHUNK_MIN_CHARS:
        return False, f"Chunk {chunk_num} is too short: {count}. Needs {CHUNK_MIN_CHARS - count} more chars."
    if count > CHUNK_MAX_CHARS:
        return False, f"Chunk {chunk_num} is too long: {count}. Trim {count - CHUNK_MAX_CHARS} chars."
    return True, f"Chunk {chunk_num} passed: {count} chars."


def validate_chapter(text: str) -> tuple[bool, str]:
    """Validate a chapter is within character count bounds."""
    count = count_chars(text)
    if count < CHAPTER_MIN_CHARS:
        return False, f"Chapter is too short: {count}. Needs {CHAPTER_MIN_CHARS - count} more chars."
    if count > CHAPTER_MAX_CHARS:
        return False, f"Chapter is too long: {count}. Trim {count - CHAPTER_MAX_CHARS} chars."
    return True, f"Chapter passed: {count} chars (target: {CHAPTER_MIN_CHARS}-{CHAPTER_MAX_CHARS})."

# --- Agent Defaults ---
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

When given a task:
1. Think step-by-step about what you need to do.
2. Use the available tools to gather information or perform actions.
3. If a tool result isn't what you expected, try a different approach.
4. When you have enough information, provide a clear final answer.

IMPORTANT - Managing long content:
- If you need to write something long (essays, reports, articles), ALWAYS
  use the write_file tool to save the content to a file.
- Write in sections/chunks rather than trying to produce everything at once.
- For documents longer than ~1000 words, write the first half to a file,
  then read it back and append the second half.
- Never try to output more than ~800 words in a single response.

Be concise but thorough. If you're unsure, say so rather than guessing."""


# --- Provider Factory ---
def create_provider(name: str | None = None):
    """Create an LLM provider by name."""
    name = (name or PROVIDER).lower()

    if name == "anthropic":
        from providers.anthropic_provider import AnthropicProvider
        p = AnthropicProvider()
        p.default_model = MODEL
        return p

    elif name == "openai":
        from providers.openai_compat import create_openai_provider
        p = create_openai_provider()
        p.default_model = OPENAI_MODEL
        return p

    elif name == "grok":
        from providers.openai_compat import create_grok_provider
        p = create_grok_provider()
        p.default_model = GROK_MODEL
        return p

    elif name == "gemini":
        from providers.openai_compat import create_gemini_provider
        p = create_gemini_provider()
        p.default_model = GEMINI_MODEL
        return p

    else:
        raise ValueError(
            f"Unknown provider '{name}'. "
            "Available: anthropic, openai, grok, gemini"
        )


def list_available_providers() -> list[str]:
    """Return list of providers that have API keys configured."""
    available = []
    if os.environ.get("ANTHROPIC_API_KEY", "") not in ("", "sk-ant-your-key-here"):
        available.append("anthropic")
    if os.environ.get("OPENAI_API_KEY", "") not in ("", "sk-your-key-here"):
        available.append("openai")
    if os.environ.get("XAI_API_KEY", "") not in ("", "xai-your-key-here"):
        available.append("grok")
    if os.environ.get("GEMINI_API_KEY", "") not in ("", "your-key-here"):
        available.append("gemini")
    return available

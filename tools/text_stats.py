"""
Text statistics and validation tools for long-form novel chapters.

These tools give the agents deterministic character counts instead of asking
an LLM to estimate. Character counts include letters, spaces, punctuation, and
line breaks.
"""

from __future__ import annotations

from pathlib import Path
from tools.base import BaseTool
from tools.file_ops import _resolve_path


CHUNK_MIN_CHARS = 10_000
CHUNK_MAX_CHARS = 12_000
CHAPTER_MIN_CHARS = 36_000
CHAPTER_MAX_CHARS = 45_000


def _read_file(filepath: str) -> str:
    resolved = Path(_resolve_path(filepath))
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if resolved.is_dir():
        raise IsADirectoryError(f"'{filepath}' is a directory, not a file")
    return resolved.read_text(encoding="utf-8")


def _count_chars(text: str) -> int:
    return len(text or "")


class CountFileCharsTool(BaseTool):
    name = "count_file_chars"
    description = (
        "Count all characters in a text file, including spaces, punctuation, "
        "and line breaks."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "filepath": {
                "type": "string",
                "description": "Filename in output folder or full path to count",
            }
        },
        "required": ["filepath"],
    }

    def run(self, filepath: str) -> str:
        try:
            text = _read_file(filepath)
            return f"Character count: {_count_chars(text)}"
        except Exception as e:
            return f"Error counting file characters: {e}"


class ValidateChunkTool(BaseTool):
    name = "validate_chunk"
    description = "Validate that a chapter chunk is between 10,000 and 12,000 characters."
    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Chunk text to validate",
            },
            "chunk_number": {
                "type": "integer",
                "description": "Chunk number being validated",
            },
        },
        "required": ["text", "chunk_number"],
    }

    def run(self, text: str, chunk_number: int = 1) -> str:
        count = _count_chars(text)
        if count < CHUNK_MIN_CHARS:
            return (
                f"FAIL: Chunk {chunk_number} is too short at {count} characters. "
                f"Add at least {CHUNK_MIN_CHARS - count} characters."
            )
        if count > CHUNK_MAX_CHARS:
            return (
                f"FAIL: Chunk {chunk_number} is too long at {count} characters. "
                f"Trim at least {count - CHUNK_MAX_CHARS} characters."
            )
        return f"PASS: Chunk {chunk_number} is {count} characters."


class ValidateChapterTool(BaseTool):
    name = "validate_chapter"
    description = "Validate that a completed chapter file is between 36,000 and 45,000 characters."
    input_schema = {
        "type": "object",
        "properties": {
            "filepath": {
                "type": "string",
                "description": "Chapter filename in output folder or full path to validate",
            }
        },
        "required": ["filepath"],
    }

    def run(self, filepath: str) -> str:
        try:
            text = _read_file(filepath)
            count = _count_chars(text)
            if count < CHAPTER_MIN_CHARS:
                return (
                    f"FAIL: Chapter is too short at {count} characters. "
                    f"Add at least {CHAPTER_MIN_CHARS - count} characters."
                )
            if count > CHAPTER_MAX_CHARS:
                return (
                    f"FAIL: Chapter is too long at {count} characters. "
                    f"Trim at least {count - CHAPTER_MAX_CHARS} characters."
                )
            return f"PASS: Chapter is {count} characters."
        except Exception as e:
            return f"Error validating chapter length: {e}"

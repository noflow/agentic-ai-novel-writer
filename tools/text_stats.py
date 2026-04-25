from tools.base import BaseTool
from tools.file_ops import _resolve_path

CHUNK_MIN = 10_000
CHUNK_MAX = 12_000
CHAPTER_MIN = 36_000
CHAPTER_MAX = 45_000


def count_chars(text: str) -> int:
    return len(text or "")


class CountFileCharsTool(BaseTool):
    name = "count_file_chars"
    description = "Count characters in a file"

    input_schema = {
        "type": "object",
        "properties": {
            "filepath": {"type": "string"}
        },
        "required": ["filepath"],
    }

    def run(self, filepath: str) -> str:
        path = _resolve_path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        count = count_chars(text)
        return f"{count} characters"


class ValidateChapterTool(BaseTool):
    name = "validate_chapter"
    description = "Validate chapter character count"

    input_schema = {
        "type": "object",
        "properties": {
            "filepath": {"type": "string"}
        },
        "required": ["filepath"],
    }

    def run(self, filepath: str) -> str:
        path = _resolve_path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        count = count_chars(text)

        if count < CHAPTER_MIN:
            return f"FAIL: {count} chars (needs {CHAPTER_MIN - count} more)"
        elif count > CHAPTER_MAX:
            return f"FAIL: {count} chars (trim {count - CHAPTER_MAX})"
        else:
            return f"PASS: {count} chars"
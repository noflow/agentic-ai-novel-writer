"""
File Operations tools -- read and write text files.

All files are saved to an 'output' subfolder by default so they're
easy to find. The output folder is created automatically.
"""

import os
from pathlib import Path
from tools.base import BaseTool


def _get_output_dir() -> Path:
    """Get (and create) the output directory next to the script."""
    out = Path(__file__).parent.parent / "output"
    out.mkdir(exist_ok=True)
    return out


def _resolve_path(filepath: str) -> str:
    """
    Resolve a filepath intelligently:
    - If it's just a filename like 'report.txt', put it in the output folder.
    - If it's an absolute path, use it as-is.
    - Strip quotes and whitespace.
    """
    filepath = filepath.strip().strip("\"'")

    p = Path(filepath)

    # If it's just a filename (no directory parts), save to output/
    if p.parent == Path(".") or str(p.parent) == ".":
        return str(_get_output_dir() / p.name)

    # Otherwise use the path as given
    return str(p.resolve())


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "Read the contents of a text file. Provide just the filename "
        "(e.g. 'report.txt') to read from the output folder, or a full "
        "path to read from elsewhere."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "filepath": {
                "type": "string",
                "description": "Filename (e.g. 'report.txt') or full path to read",
            }
        },
        "required": ["filepath"],
    }

    def run(self, filepath: str) -> str:
        try:
            resolved = _resolve_path(filepath)

            # Check if it's a directory instead of a file
            if os.path.isdir(resolved):
                files = os.listdir(resolved)
                file_list = ", ".join(files[:20]) if files else "(empty)"
                return (
                    f"Error: '{filepath}' is a directory, not a file. "
                    f"Files in this directory: {file_list}"
                )

            if not os.path.exists(resolved):
                # Try to help -- list files in the output folder
                out_dir = _get_output_dir()
                available = [f.name for f in out_dir.iterdir() if f.is_file()]
                if available:
                    avail_str = ", ".join(available[:10])
                    return (
                        f"Error: File not found: {filepath}\n"
                        f"Available files in output folder: {avail_str}"
                    )
                return f"Error: File not found: {filepath}"

            with open(resolved, "r", encoding="utf-8") as f:
                content = f.read()

            if len(content) > 10_000:
                content = content[:10_000] + "\n\n... [truncated -- file is very large]"

            return content

        except PermissionError:
            return (
                f"Error: Permission denied reading '{filepath}'. "
                "This might be a directory or a protected file."
            )
        except Exception as e:
            return f"Error reading file: {e}"


class WriteFileTool(BaseTool):
    name = "write_file"
    description = (
        "Write content to a text file. Provide a filename like 'essay.txt' "
        "and the content string. IMPORTANT: If the content is very long, "
        "write it in smaller pieces -- first use write_file for the beginning, "
        "then use append_file for each additional section."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "filepath": {
                "type": "string",
                "description": "Filename (e.g. 'essay.txt') or full path to write",
            },
            "content": {
                "type": "string",
                "description": "The text content to write to the file. Must not be empty.",
            },
        },
        "required": ["filepath", "content"],
    }

    def run(self, filepath: str = "", content: str = "", **kwargs) -> str:
        # Handle cases where the model sends args wrong
        if not content:
            return (
                "Error: No content provided. You must include both 'filepath' "
                "AND 'content' arguments. Example:\n"
                '  write_file(filepath="story.txt", content="Once upon a time...")\n'
                "If your content is very long, write just the first section here, "
                "then use append_file to add more sections."
            )
        if not filepath:
            return "Error: No filepath provided. Example: write_file(filepath='story.txt', content='...')"

        try:
            resolved = _resolve_path(filepath)

            if os.path.isdir(resolved):
                return (
                    f"Error: '{filepath}' is a directory. "
                    "Please provide a filename like 'report.txt'."
                )

            parent = os.path.dirname(resolved)
            if parent:
                os.makedirs(parent, exist_ok=True)

            with open(resolved, "w", encoding="utf-8") as f:
                f.write(content)

            return (
                f"Successfully wrote {len(content)} characters to: {resolved}\n"
                f"You can find the file at: {resolved}"
            )

        except PermissionError:
            return (
                f"Error: Permission denied writing to '{filepath}'. "
                "Try using just a filename like 'report.txt'."
            )
        except Exception as e:
            return f"Error writing file: {e}"


class AppendFileTool(BaseTool):
    name = "append_file"
    description = (
        "Append content to the END of an existing file. Use this to build "
        "up long documents section by section. First use write_file to create "
        "the file with the beginning, then use append_file to add more sections."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "filepath": {
                "type": "string",
                "description": "Filename (e.g. 'essay.txt') or full path",
            },
            "content": {
                "type": "string",
                "description": "Content to append to the end of the file",
            },
        },
        "required": ["filepath", "content"],
    }

    def run(self, filepath: str = "", content: str = "", **kwargs) -> str:
        if not content:
            return (
                "Error: No content provided. You must include both 'filepath' "
                "AND 'content' arguments."
            )
        if not filepath:
            return "Error: No filepath provided."

        try:
            resolved = _resolve_path(filepath)

            if os.path.isdir(resolved):
                return f"Error: '{filepath}' is a directory, not a file."

            if not os.path.exists(resolved):
                return (
                    f"Error: File '{filepath}' does not exist yet. "
                    "Use write_file first to create it, then append_file to add more."
                )

            with open(resolved, "a", encoding="utf-8") as f:
                f.write(content)

            # Report total file size
            total_size = os.path.getsize(resolved)
            return (
                f"Appended {len(content)} characters to: {resolved}\n"
                f"Total file size: {total_size} characters"
            )

        except PermissionError:
            return f"Error: Permission denied appending to '{filepath}'."
        except Exception as e:
            return f"Error appending to file: {e}"


class ListFilesTool(BaseTool):
    name = "list_files"
    description = (
        "List all files in the output folder. Use this to see what "
        "chapters, summaries, and documents have been created so far. "
        "Helps you know what already exists before writing new content."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "filter": {
                "type": "string",
                "description": "Optional: filter files containing this text (e.g. 'ch' for chapters, 'summary' for summaries, 'outline' for novel outlines)",
            },
        },
    }

    def run(self, filter: str = "", **kwargs) -> str:
        try:
            out_dir = _get_output_dir()
            files = sorted([f.name for f in out_dir.iterdir() if f.is_file()])

            if not files:
                return "Output folder is empty. No files have been created yet."

            if filter:
                filter_lower = filter.lower()
                files = [f for f in files if filter_lower in f.lower()]
                if not files:
                    return f"No files matching '{filter}' found in output folder."

            file_list = "\n".join(f"  - {f}" for f in files)
            return f"Files in output folder ({len(files)}):\n{file_list}"

        except Exception as e:
            return f"Error listing files: {e}"

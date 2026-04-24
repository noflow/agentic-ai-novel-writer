"""
Clock tool -- returns the current date and time.

Simple but important: LLMs don't inherently know what time it is.
This lets the agent answer time-sensitive questions and timestamp things.
"""

from datetime import datetime
from tools.base import BaseTool


class ClockTool(BaseTool):
    name = "get_current_time"
    description = (
        "Get the current date and time. Use this when you need to know "
        "the current time, today's date, or the day of the week."
    )
    input_schema = {
        "type": "object",
        "properties": {},  # No inputs needed
    }

    def run(self) -> str:
        now = datetime.now()
        return now.strftime("%A, %B %d, %Y at %I:%M %p")

"""
Web Search tool -- REAL web search via Anthropic's built-in search.

Uses your existing ANTHROPIC_API_KEY -- no extra keys needed.
Makes a separate API call with the web_search tool enabled,
then returns the search results to the main agent.
"""

from anthropic import Anthropic
from tools.base import BaseTool


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web for current, real-time information. Use this when you need "
        "up-to-date facts, news, prices, or any information you don't already know. "
        "Returns real web search results with sources."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            }
        },
        "required": ["query"],
    }

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = Anthropic()
        return self._client

    def run(self, query: str) -> str:
        try:
            client = self._get_client()

            # Use Claude with web search tool to find real results
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 3,
                }],
                messages=[{
                    "role": "user",
                    "content": (
                        f"Search the web for: {query}\n\n"
                        "Return the key facts and information you find. "
                        "Include specific numbers, prices, and details. "
                        "Cite your sources."
                    ),
                }],
            )

            # Extract text from the response
            results = []
            for block in response.content:
                if block.type == "text":
                    results.append(block.text)

            if results:
                return "\n".join(results)
            else:
                return f"No results found for: {query}"

        except Exception as e:
            error_msg = str(e)
            # Check if web search isn't enabled on their account
            if "web_search" in error_msg.lower() or "tool" in error_msg.lower():
                return (
                    f"Web search error: {error_msg}\n\n"
                    "Note: Web search may need to be enabled in your Anthropic console.\n"
                    "Go to https://console.anthropic.com/settings and check that "
                    "web search is enabled for your API key."
                )
            return f"Search error: {error_msg}"

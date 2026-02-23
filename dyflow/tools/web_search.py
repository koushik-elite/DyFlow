"""
dyflow/tools/web_search.py
──────────────────────────
WebSearchTool — external web retrieval via the Serper API.

Falls back gracefully to a MockWebSearchTool (returns canned results)
when no API key is configured, so experiments can run without a live key.

Params accepted by execute()
-----------------------------
  query  : str  — the search string
  top_k  : int  — max results to return (default 5)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .base import BaseTool, ToolResult, ToolStatus


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_results(items: List[Dict]) -> str:
    """Convert a list of {title, link, snippet} dicts to a readable block."""
    lines = []
    for i, item in enumerate(items, 1):
        title   = item.get("title",   "(no title)")
        link    = item.get("link",    "")
        snippet = item.get("snippet", "")
        lines.append(f"[{i}] {title}\n    URL: {link}\n    {snippet}")
    return "\n\n".join(lines)


# ── Live tool (Serper) ────────────────────────────────────────────────────────

class WebSearchTool(BaseTool):
    """
    Issues a query to the Serper.dev Google Search API and returns
    the top-k organic results.

    Parameters
    ----------
    api_key : str, optional
        Serper API key. Falls back to the SERPER_API_KEY env var.
    top_k   : int
        Default number of results to retrieve (overrideable per call).
    """

    tool_name = "WEB_SEARCH"

    def __init__(self, api_key: Optional[str] = None, top_k: int = 5) -> None:
        self.api_key   = api_key or os.getenv("SERPER_API_KEY", "")
        self.default_k = top_k

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        query = params.get("query", "").strip()
        top_k = int(params.get("top_k", self.default_k))

        if not query:
            return ToolResult(
                status=ToolStatus.ERROR,
                raw_output="",
                tool_name=self.tool_name,
                query=query,
                error_message="Empty query string.",
            )

        if not self.api_key:
            # ── Fallback: no API key configured ──────────────────────────────
            return ToolResult(
                status=ToolStatus.ERROR,
                raw_output="",
                tool_name=self.tool_name,
                query=query,
                error_message=(
                    "SERPER_API_KEY is not set. "
                    "Use MockWebSearchTool for offline testing."
                ),
            )

        try:
            import requests  # soft import — only needed at call time
        except ImportError:
            return ToolResult(
                status=ToolStatus.ERROR,
                raw_output="",
                tool_name=self.tool_name,
                query=query,
                error_message="'requests' library not installed. Run: pip install requests",
            )

        payload = json.dumps({"q": query, "num": top_k})
        headers = {
            "X-API-KEY":     self.api_key,
            "Content-Type":  "application/json",
        }

        try:
            response = requests.post(
                "https://google.serper.dev/search",
                headers=headers,
                data=payload,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            return ToolResult(
                status=ToolStatus.ERROR,
                raw_output="",
                tool_name=self.tool_name,
                query=query,
                error_message=f"Serper API error: {exc}",
            )

        organic = data.get("organic", [])[:top_k]
        if not organic:
            return ToolResult(
                status=ToolStatus.EMPTY,
                raw_output="No results found.",
                tool_name=self.tool_name,
                query=query,
            )

        raw = _format_results(organic)
        return ToolResult(
            status=ToolStatus.SUCCESS,
            raw_output=raw,
            tool_name=self.tool_name,
            query=query,
            structured={"results": organic},
            metadata={"answer_box": data.get("answerBox"), "knowledge_graph": data.get("knowledgeGraph")},
        )


# ── Offline stub (for unit tests / no-key environments) ──────────────────────

class MockWebSearchTool(BaseTool):
    """
    Returns a deterministic fake result.
    Useful for unit tests and local development without a live API key.
    """

    tool_name = "WEB_SEARCH"

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        query = params.get("query", "(empty)")
        mock_results = [
            {
                "title":   f"Mock Result 1 for: {query}",
                "link":    "https://example.com/1",
                "snippet": f"This is a mock search result for the query '{query}'. "
                           "It contains placeholder information for offline testing.",
            },
            {
                "title":   f"Mock Result 2 for: {query}",
                "link":    "https://example.com/2",
                "snippet": f"Another mock result providing additional context about '{query}'.",
            },
        ]
        return ToolResult(
            status=ToolStatus.SUCCESS,
            raw_output=_format_results(mock_results),
            tool_name=self.tool_name,
            query=query,
            structured={"results": mock_results},
            metadata={"mock": True},
        )

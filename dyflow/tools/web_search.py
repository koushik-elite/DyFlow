"""
dyflow/tools/web_search.py
──────────────────────────
WebSearchTool — external web retrieval via SerpAPI (Google Search engine).

Uses the official `serpapi` Python package.

Install:
    pip install google-search-results

Params accepted by execute()
-----------------------------
  query  : str  — the search string
  top_k  : int  — max organic results to return (default 5)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .base import BaseTool, ToolResult, ToolStatus


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_results(items: List[Dict]) -> str:
    """Convert a list of organic result dicts to a readable block."""
    lines = []
    for i, item in enumerate(items, 1):
        title   = item.get("title",   "(no title)")
        link    = item.get("link",    "")
        snippet = item.get("snippet", "")
        lines.append(f"[{i}] {title}\n    URL: {link}\n    {snippet}")
    return "\n\n".join(lines)


def _extract_organic(data: Dict, top_k: int) -> List[Dict]:
    """
    Extract organic results from a SerpAPI response dict.
    SerpAPI returns the list under the key 'organic_results'.
    """
    raw = data.get("organic_results", [])
    results = []
    for item in raw[:top_k]:
        results.append({
            "position": item.get("position"),
            "title":    item.get("title", ""),
            "link":     item.get("link", ""),
            "snippet":  item.get("snippet", ""),
            "date":     item.get("date", ""),
        })
    return results


# ── Live tool (SerpAPI) ───────────────────────────────────────────────────────

class WebSearchTool(BaseTool):
    """
    Issues a Google search via SerpAPI and returns the top-k organic results.

    Parameters
    ----------
    api_key : str, optional
        SerpAPI key. Falls back to the SERPAPI_API_KEY env var.
    top_k   : int
        Default number of organic results to retrieve per call.

    Requirements
    ------------
        pip install google-search-results
    """

    tool_name = "WEB_SEARCH"

    def __init__(self, api_key: Optional[str] = None, top_k: int = 5) -> None:
        # Support both old (SERPER_API_KEY) and new (SERPAPI_API_KEY) env var names
        self.api_key = (
            api_key
            or os.getenv("SERPAPI_API_KEY", "")
            or os.getenv("SERPER_API_KEY", "")   # backward compat
        )
        self.default_k = top_k

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        query = params.get("query", "").strip()
        # Strip backticks the LLM sometimes wraps around the query
        query = query.strip("`").strip()
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
            return ToolResult(
                status=ToolStatus.ERROR,
                raw_output="",
                tool_name=self.tool_name,
                query=query,
                error_message=(
                    "SERPAPI_API_KEY is not set. "
                    "Use MockWebSearchTool for offline testing."
                ),
            )

        try:
            from serpapi import GoogleSearch
        except ImportError:
            return ToolResult(
                status=ToolStatus.ERROR,
                raw_output="",
                tool_name=self.tool_name,
                query=query,
                error_message=(
                    "'serpapi' package not installed. "
                    "Run: pip install google-search-results"
                ),
            )

        try:
            search = GoogleSearch({
                "engine":  "google",
                "q":       query,
                "num":     top_k,
                "api_key": self.api_key,
            })
            data = search.get_dict()
        except Exception as exc:
            return ToolResult(
                status=ToolStatus.ERROR,
                raw_output="",
                tool_name=self.tool_name,
                query=query,
                error_message=f"SerpAPI error: {exc}",
            )

        # Debug: show top-level keys so mismatches are visible in logs
        print(f"  [SerpAPI] Response keys: {list(data.keys())}")

        # Check for API-level errors
        if "error" in data:
            return ToolResult(
                status=ToolStatus.ERROR,
                raw_output="",
                tool_name=self.tool_name,
                query=query,
                error_message=f"SerpAPI returned error: {data['error']}",
            )

        organic = _extract_organic(data, top_k)
        print(f"  [SerpAPI] organic_results count: {len(organic)}")

        if not organic:
            # Print full keys to help diagnose missing organic_results
            print(f"  [SerpAPI] WARNING: no organic_results. Available keys: {list(data.keys())}")
            # Try alternate key names some SerpAPI versions use
            fallback = data.get("results", data.get("web_results", []))
            if fallback:
                print(f"  [SerpAPI] Using fallback key with {len(fallback)} results")
                organic = [
                    {
                        "position": i + 1,
                        "title":    r.get("title", ""),
                        "link":     r.get("link", r.get("url", "")),
                        "snippet":  r.get("snippet", r.get("description", "")),
                        "date":     r.get("date", ""),
                    }
                    for i, r in enumerate(fallback[:top_k])
                ]

        if not organic:
            return ToolResult(
                status=ToolStatus.EMPTY,
                raw_output="No organic results found.",
                tool_name=self.tool_name,
                query=query,
            )

        raw = _format_results(organic)

        # Include useful extras from the response
        metadata = {}
        if "answer_box" in data:
            ab = data["answer_box"]
            metadata["answer_box"] = ab.get("answer") or ab.get("snippet", "")
        if "knowledge_graph" in data:
            kg = data["knowledge_graph"]
            metadata["knowledge_graph"] = kg.get("description", "")

        return ToolResult(
            status=ToolStatus.SUCCESS,
            raw_output=raw,
            tool_name=self.tool_name,
            query=query,
            structured={
                "results": organic,
                "total_results": data.get("search_information", {}).get("total_results"),
            },
            metadata=metadata,
        )


# ── Offline stub ──────────────────────────────────────────────────────────────

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
                "position": 1,
                "title":    f"Mock Result 1 for: {query}",
                "link":     "https://example.com/1",
                "snippet":  f"This is a mock search result for '{query}'. "
                            "It contains placeholder information for offline testing.",
                "date":     "",
            },
            {
                "position": 2,
                "title":    f"Mock Result 2 for: {query}",
                "link":     "https://example.com/2",
                "snippet":  f"Another mock result with additional context about '{query}'.",
                "date":     "",
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

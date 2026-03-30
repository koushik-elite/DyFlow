"""
dyflow/tools/web_search.py
──────────────────────────
WebSearchTool — live web retrieval via tavily-python SDK.

Install:
    pip install tavily-python

Environment variable:
    TAVILY_API_KEY=tvly-...

Usage:
    from dyflow.tools.web_search import WebSearchTool
    tool = WebSearchTool()
    result = tool.execute({"query": "Who won the 2025 US Open?"})
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .base import BaseTool, ToolResult, ToolStatus


class WebSearchTool(BaseTool):
    """
    Live web search via tavily-python SDK.

    Parameters
    ----------
    api_key : str, optional — overrides TAVILY_API_KEY env var
    top_k   : int           — max results (default 5)
    """

    tool_name = "WEB_SEARCH"

    def __init__(self, api_key: Optional[str] = None, top_k: int = 5) -> None:
        self.api_key   = api_key or os.getenv("TAVILY_API_KEY", "")
        self.default_k = top_k

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        query = params.get("query", "").strip().strip("`").strip()
        top_k = int(params.get("top_k", self.default_k))

        if not query:
            return ToolResult(
                status=ToolStatus.ERROR, raw_output="",
                tool_name=self.tool_name, query=query,
                error_message="Empty query string.",
            )

        if not self.api_key:
            return ToolResult(
                status=ToolStatus.ERROR, raw_output="",
                tool_name=self.tool_name, query=query,
                error_message=(
                    "TAVILY_API_KEY not set.\n"
                    "  pip install tavily-python\n"
                    "  Add TAVILY_API_KEY=tvly-... to your .env"
                ),
            )

        try:
            from tavily import TavilyClient
        except ImportError:
            return ToolResult(
                status=ToolStatus.ERROR, raw_output="",
                tool_name=self.tool_name, query=query,
                error_message="tavily-python not installed. Run: pip install tavily-python",
            )

        try:
            client   = TavilyClient(self.api_key)
            response = client.search(
                query=query,
                max_results=top_k,
                include_answer="basic",
                search_depth="advanced",
            )
        except Exception as exc:
            return ToolResult(
                status=ToolStatus.ERROR, raw_output="",
                tool_name=self.tool_name, query=query,
                error_message=f"Tavily error: {exc}",
            )

        # ── Parse response ─────────────────────────────────────────────────────
        ai_answer  = response.get("answer", "")
        results    = response.get("results", [])

        print(f"  [Tavily] query='{query[:60]}' | results={len(results)}")

        if not ai_answer and not results:
            return ToolResult(
                status=ToolStatus.EMPTY,
                raw_output="Tavily returned no content.",
                tool_name=self.tool_name, query=query,
            )

        # ── Format output ──────────────────────────────────────────────────────
        references = [
            {
                "position": i + 1,
                "title":    r.get("title",   ""),
                "link":     r.get("url",     ""),
                "snippet":  r.get("content", "")[:300],
                "score":    r.get("score",   0),
            }
            for i, r in enumerate(results[:top_k])
        ]

        parts = []
        if ai_answer:
            parts.append("=== Tavily Answer ===")
            parts.append(ai_answer)
        if references:
            parts.append("\n=== Sources ===")
            for ref in references:
                parts.append(
                    f"[{ref['position']}] {ref['title']}\n"
                    f"    URL: {ref['link']}\n"
                    f"    {ref['snippet']}"
                )
        raw_output = "\n".join(parts)

        return ToolResult(
            status=ToolStatus.SUCCESS,
            raw_output=raw_output,
            tool_name=self.tool_name,
            query=query,
            structured={"ai_answer": ai_answer, "references": references},
            metadata={"engine": "tavily"},
        )


# ── Offline stub ───────────────────────────────────────────────────────────────

class MockWebSearchTool(BaseTool):
    """Fake results for offline testing — no API key needed."""

    tool_name = "WEB_SEARCH"

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        query      = params.get("query", "(empty)")
        ai_answer  = f"Mock answer for: {query} [MOCK — set TAVILY_API_KEY for real results]"
        references = [{
            "position": 1, "title": f"Mock result for: {query}",
            "link": "https://example.com", "snippet": ai_answer, "score": 0.0,
        }]
        return ToolResult(
            status=ToolStatus.SUCCESS,
            raw_output=f"=== Tavily Answer ===\n{ai_answer}",
            tool_name=self.tool_name, query=query,
            structured={"ai_answer": ai_answer, "references": references},
            metadata={"mock": True, "engine": "mock"},
        )

"""
dyflow/tools/web_search.py
──────────────────────────
WebSearchTool — live web retrieval via Tavily MCP or Tavily REST API.

Primary:  Tavily MCP  → https://mcp.tavily.com/mcp/?tavilyApiKey=<key>
Fallback: Tavily REST → https://api.tavily.com/search

Tavily MCP is an MCP (Model Context Protocol) server that exposes a
`tavily-search` tool via JSON-RPC over HTTP. It returns AI-synthesised
answers grounded in real-time web results — ideal for news questions
and post-training knowledge retrieval.

MCP endpoint: https://mcp.tavily.com/mcp/?tavilyApiKey=<key>
REST endpoint: https://api.tavily.com/search

Install:
    pip install requests

Environment variables (set one):
    TAVILY_API_KEY   = tvly-...   (recommended)
    SERPAPI_API_KEY  = ...        (legacy fallback — SerpAPI google_ai_mode)
    SERPER_API_KEY   = ...        (legacy compat)

Usage in .env:
    TAVILY_API_KEY=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
"""

from __future__ import annotations

import os
import json
import requests
from typing import Any, Dict, List, Optional

from .base import BaseTool, ToolResult, ToolStatus


# ── Constants ──────────────────────────────────────────────────────────────────
TAVILY_MCP_BASE  = "https://mcp.tavily.com/mcp/"
TAVILY_REST_URL  = "https://api.tavily.com/search"
SERPAPI_URL      = "https://serpapi.com/search"


# ── Response formatters ────────────────────────────────────────────────────────

def _format_tavily_results(data: Dict, top_k: int) -> tuple:
    """
    Parse Tavily response (MCP tool_result or REST response).
    Returns (ai_answer: str, references: list, raw_output: str)
    """
    # MCP tool_result wraps content inside content[].text as JSON string
    # REST returns directly as JSON dict
    results = data.get("results", [])
    ai_answer = data.get("answer", "")

    references = []
    for i, r in enumerate(results[:top_k]):
        references.append({
            "position": i + 1,
            "title":    r.get("title",   ""),
            "link":     r.get("url",     r.get("link", "")),
            "snippet":  r.get("content", r.get("snippet", "")),
            "score":    r.get("score",   0),
        })

    parts = []
    if ai_answer:
        parts.append("=== Tavily AI Answer ===")
        parts.append(ai_answer)
    if references:
        parts.append("\n=== Sources ===")
        for ref in references:
            title   = ref["title"]   or "(no title)"
            link    = ref["link"]    or ""
            snippet = ref["snippet"] or ""
            parts.append(f"[{ref['position']}] {title}\n    URL: {link}\n    {snippet[:200]}")

    return ai_answer, references, "\n".join(parts)


def _format_serpapi_results(data: Dict, top_k: int) -> tuple:
    """Parse SerpAPI google_ai_mode response."""
    lines = []
    for block in data.get("text_blocks", []):
        btype   = block.get("type", "paragraph")
        snippet = block.get("snippet", "")
        title   = block.get("title", "")
        if btype == "heading":
            lines.append(f"\n### {snippet or title}")
        elif btype == "list":
            if title: lines.append(f"\n{title}:")
            for item in block.get("list", []):
                s = item.get("snippet", item) if isinstance(item, dict) else str(item)
                lines.append(f"  - {s}")
        else:
            if snippet: lines.append(snippet)
    ai_answer = "\n".join(lines).strip()

    refs = data.get("references", [])
    references = [
        {"position": i+1, "title": r.get("title",""), "link": r.get("link",""),
         "snippet": r.get("snippet",""), "source": r.get("source","")}
        for i, r in enumerate(refs[:top_k])
    ]

    parts = []
    if ai_answer:
        parts.append("=== Google AI Mode Answer ===")
        parts.append(ai_answer)
    if references:
        parts.append("\n=== Sources ===")
        for ref in references:
            label = f"{ref['title']} ({ref.get('source','')})" if ref.get('source') else ref['title']
            parts.append(f"[{ref['position']}] {label}\n    URL: {ref['link']}\n    {ref['snippet'][:200]}")

    return ai_answer, references, "\n".join(parts)


# ── Live WebSearchTool ─────────────────────────────────────────────────────────

class WebSearchTool(BaseTool):
    """
    Live web search tool supporting:
      1. Tavily MCP   (primary)   — MCP JSON-RPC over HTTP
      2. Tavily REST  (fallback)  — direct REST API
      3. SerpAPI      (legacy)    — google_ai_mode engine

    Priority: TAVILY_API_KEY → SERPAPI_API_KEY / SERPER_API_KEY

    Parameters
    ----------
    api_key : str, optional — override env var lookup
    top_k   : int           — max results to return (default 5)
    """

    tool_name   = "WEB_SEARCH"

    def __init__(
        self,
        api_key: Optional[str] = None,
        top_k: int = 5,
    ) -> None:
        # Tavily key (primary)
        self.tavily_key = (
            api_key
            or os.getenv("TAVILY_API_KEY", "")
        )
        # SerpAPI key (legacy fallback)
        self.serpapi_key = (
            os.getenv("SERPAPI_API_KEY", "")
            or os.getenv("SERPER_API_KEY", "")
        )
        self.default_k = top_k

    # ── Public ────────────────────────────────────────────────────────────────

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        query = params.get("query", "").strip().strip("`").strip()
        top_k = int(params.get("top_k", self.default_k))

        if not query:
            return ToolResult(
                status=ToolStatus.ERROR, raw_output="",
                tool_name=self.tool_name, query=query,
                error_message="Empty query string.",
            )

        if self.tavily_key:
            # Try MCP first, then REST
            result = self._tavily_mcp(query, top_k)
            if result.status == ToolStatus.ERROR:
                print(f"  [Tavily MCP] failed, trying REST: {result.error_message}")
                result = self._tavily_rest(query, top_k)
            return result

        if self.serpapi_key:
            return self._serpapi(query, top_k)

        return ToolResult(
            status=ToolStatus.ERROR, raw_output="",
            tool_name=self.tool_name, query=query,
            error_message=(
                "No API key found. Set TAVILY_API_KEY in your .env file.\n"
                "  Get a free key at: https://tavily.com (1000 searches/month)\n"
                "  echo \"TAVILY_API_KEY=tvly-...\" >> .env"
            ),
        )

    # ── Tavily MCP ─────────────────────────────────────────────────────────────

    def _tavily_mcp(self, query: str, top_k: int) -> ToolResult:
        """
        Call Tavily MCP server via JSON-RPC 2.0.

        MCP endpoint: POST https://mcp.tavily.com/mcp/?tavilyApiKey=<key>
        Method: tools/call
        Tool:   tavily-search
        """
        url = f"{TAVILY_MCP_BASE}?tavilyApiKey={self.tavily_key}"
        payload = {
            "jsonrpc": "2.0",
            "id":      1,
            "method":  "tools/call",
            "params":  {
                "name":      "tavily-search",
                "arguments": {
                    "query":              query,
                    "max_results":        top_k,
                    "include_answer":     True,
                    "include_raw_content": False,
                    "search_depth":       "advanced",
                },
            },
        }
        try:
            resp = requests.post(
                url, json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            rjson = resp.json()
        except requests.exceptions.Timeout:
            return ToolResult(status=ToolStatus.ERROR, raw_output="",
                              tool_name=self.tool_name, query=query,
                              error_message="Tavily MCP timeout (30s).")
        except Exception as exc:
            return ToolResult(status=ToolStatus.ERROR, raw_output="",
                              tool_name=self.tool_name, query=query,
                              error_message=f"Tavily MCP error: {exc}")

        # MCP JSON-RPC error
        if "error" in rjson:
            return ToolResult(status=ToolStatus.ERROR, raw_output="",
                              tool_name=self.tool_name, query=query,
                              error_message=f"MCP error: {rjson['error']}")

        # Extract result — MCP returns content as list of {type, text}
        result_obj = rjson.get("result", {})
        content    = result_obj.get("content", [])
        raw_text   = ""
        for block in content:
            if block.get("type") == "text":
                raw_text = block.get("text", "")
                break

        # Parse the JSON string inside the text block
        try:
            data = json.loads(raw_text) if raw_text else {}
        except (json.JSONDecodeError, TypeError):
            # Sometimes raw_text IS the answer directly
            data = {"answer": raw_text, "results": []}

        print(f"  [Tavily MCP] query='{query[:50]}' | results={len(data.get('results', []))}")

        ai_answer, references, raw_output = _format_tavily_results(data, top_k)

        if not ai_answer and not references:
            return ToolResult(status=ToolStatus.EMPTY,
                              raw_output="Tavily MCP returned no content.",
                              tool_name=self.tool_name, query=query)

        return ToolResult(
            status=ToolStatus.SUCCESS,
            raw_output=raw_output,
            tool_name=self.tool_name,
            query=query,
            structured={"ai_answer": ai_answer, "references": references},
            metadata={"engine": "tavily_mcp"},
        )

    # ── Tavily REST ────────────────────────────────────────────────────────────

    def _tavily_rest(self, query: str, top_k: int) -> ToolResult:
        """
        Tavily REST API fallback.
        POST https://api.tavily.com/search
        """
        payload = {
            "api_key":        self.tavily_key,
            "query":          query,
            "max_results":    top_k,
            "include_answer": True,
            "search_depth":   "advanced",
        }
        try:
            resp = requests.post(TAVILY_REST_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            return ToolResult(status=ToolStatus.ERROR, raw_output="",
                              tool_name=self.tool_name, query=query,
                              error_message="Tavily REST timeout (30s).")
        except Exception as exc:
            return ToolResult(status=ToolStatus.ERROR, raw_output="",
                              tool_name=self.tool_name, query=query,
                              error_message=f"Tavily REST error: {exc}")

        if "error" in data:
            return ToolResult(status=ToolStatus.ERROR, raw_output="",
                              tool_name=self.tool_name, query=query,
                              error_message=f"Tavily error: {data['error']}")

        print(f"  [Tavily REST] query='{query[:50]}' | results={len(data.get('results', []))}")

        ai_answer, references, raw_output = _format_tavily_results(data, top_k)

        if not ai_answer and not references:
            return ToolResult(status=ToolStatus.EMPTY,
                              raw_output="Tavily returned no content.",
                              tool_name=self.tool_name, query=query)

        return ToolResult(
            status=ToolStatus.SUCCESS,
            raw_output=raw_output,
            tool_name=self.tool_name,
            query=query,
            structured={"ai_answer": ai_answer, "references": references},
            metadata={"engine": "tavily_rest"},
        )

    # ── SerpAPI legacy ─────────────────────────────────────────────────────────

    def _serpapi(self, query: str, top_k: int) -> ToolResult:
        """SerpAPI google_ai_mode — legacy fallback if no Tavily key."""
        req_params = {
            "engine": "google_ai_mode", "q": query,
            "api_key": self.serpapi_key, "output": "json",
        }
        try:
            resp = requests.get(SERPAPI_URL, params=req_params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return ToolResult(status=ToolStatus.ERROR, raw_output="",
                              tool_name=self.tool_name, query=query,
                              error_message=f"SerpAPI error: {exc}")

        if "error" in data:
            return ToolResult(status=ToolStatus.ERROR, raw_output="",
                              tool_name=self.tool_name, query=query,
                              error_message=f"SerpAPI: {data['error']}")

        ai_answer, references, raw_output = _format_serpapi_results(data, top_k)
        print(f"  [SerpAPI] text_blocks={len(data.get('text_blocks', []))} | refs={len(references)}")

        if not ai_answer and not references:
            return ToolResult(status=ToolStatus.EMPTY,
                              raw_output="SerpAPI returned no content.",
                              tool_name=self.tool_name, query=query)

        return ToolResult(
            status=ToolStatus.SUCCESS,
            raw_output=raw_output,
            tool_name=self.tool_name,
            query=query,
            structured={"ai_answer": ai_answer, "references": references},
            metadata={"engine": "serpapi_google_ai_mode"},
        )


# ── Offline stub ──────────────────────────────────────────────────────────────

class MockWebSearchTool(BaseTool):
    """Deterministic fake results for offline testing / CI."""

    tool_name = "WEB_SEARCH"

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        query = params.get("query", "(empty)")
        refs  = [
            {"position": 1, "title": f"Mock Source for: {query}",
             "link": "https://example.com/1", "snippet": f"Mock result for '{query}'.", "score": 0.9},
        ]
        ai_answer   = f"Based on current sources, here is information about: {query}. [MOCK]"
        raw_output  = f"=== Tavily AI Answer ===\n{ai_answer}\n\n=== Sources ===\n[1] {refs[0]['title']}"
        return ToolResult(
            status=ToolStatus.SUCCESS,
            raw_output=raw_output,
            tool_name=self.tool_name,
            query=query,
            structured={"ai_answer": ai_answer, "references": refs},
            metadata={"mock": True, "engine": "mock"},
        )

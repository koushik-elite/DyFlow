"""
dyflow/tools/web_search.py
──────────────────────────
WebSearchTool — live web retrieval via SerpAPI Google AI Mode engine.

Uses engine=google_ai_mode which returns a Gemini-2.5-powered AI answer
grounded in real-time Google search results — ideal for news questions,
current facts, and post-training knowledge retrieval.

API docs:   https://serpapi.com/google-ai-mode-api
Engine:     google_ai_mode
Response:   text_blocks[] + references[] (no organic_results)

Install:
    pip install google-search-results

Params accepted by execute()
-----------------------------
  query                   : str  — the search string
  top_k                   : int  — max reference sources to include (default 5)
  subsequent_request_token: str  — optional token for multi-turn follow-ups
"""

from __future__ import annotations

import os
import requests
from typing import Any, Dict, List, Optional

from .base import BaseTool, ToolResult, ToolStatus


# ── Response parsing ───────────────────────────────────────────────────────────

def _extract_text_blocks(data: Dict) -> str:
    """
    Flatten google_ai_mode text_blocks into a readable string.

    text_blocks is a list of dicts with fields:
      type    : "paragraph" | "heading" | "list" | "code" | ...
      snippet : the text content
      title   : optional heading for list/expandable blocks
    """
    lines = []
    for block in data.get("text_blocks", []):
        btype   = block.get("type", "paragraph")
        snippet = block.get("snippet", "")
        title   = block.get("title", "")

        if btype == "heading":
            lines.append(f"\n### {snippet or title}")
        elif btype == "list":
            if title:
                lines.append(f"\n{title}:")
            for item in block.get("list", []):
                s = item.get("snippet", item) if isinstance(item, dict) else str(item)
                lines.append(f"  - {s}")
        elif btype in ("paragraph", "expandable"):
            if title:
                lines.append(f"\n{title}")
            if snippet:
                lines.append(snippet)
        elif btype == "code":
            lines.append(f"```\n{snippet}\n```")
        else:
            if snippet:
                lines.append(snippet)

    return "\n".join(lines).strip()


def _extract_references(data: Dict, top_k: int) -> List[Dict]:
    """
    Extract reference sources from the google_ai_mode response.

    References are listed under the 'references' key as:
      [{ title, link, snippet, source }, ...]
    """
    refs = data.get("references", [])
    results = []
    for i, ref in enumerate(refs[:top_k]):
        results.append({
            "position": i + 1,
            "title":    ref.get("title",   ""),
            "link":     ref.get("link",    ref.get("url", "")),
            "snippet":  ref.get("snippet", ref.get("description", "")),
            "source":   ref.get("source",  ""),
        })
    return results


def _format_references(refs: List[Dict]) -> str:
    lines = []
    for r in refs:
        title   = r.get("title",   "(no title)")
        link    = r.get("link",    "")
        snippet = r.get("snippet", "")
        source  = r.get("source",  "")
        label   = f"{title} ({source})" if source else title
        lines.append(f"[{r['position']}] {label}\n    URL: {link}\n    {snippet}")
    return "\n\n".join(lines)


# ── Live tool — google_ai_mode ────────────────────────────────────────────────

class WebSearchTool(BaseTool):
    """
    Retrieves answers from SerpAPI's Google AI Mode engine.

    Google AI Mode (powered by Gemini 2.5) returns a structured AI answer
    grounded in real-time Google search — longer, more detailed, and more
    accurate for current-knowledge questions than standard organic results.

    The tool output combines:
      1. AI-generated answer text (from text_blocks)
      2. Reference sources (title, URL, snippet) used to ground the answer

    Parameters
    ----------
    api_key   : SerpAPI key — falls back to SERPAPI_API_KEY env var
    top_k     : max reference sources to include in output (default 5)
    engine    : SerpAPI engine to use (default: google_ai_mode)

    Requirements
    ------------
        pip install requests
    """

    tool_name  = "WEB_SEARCH"
    SERPAPI_URL = "https://serpapi.com/search"

    def __init__(
        self,
        api_key: Optional[str] = None,
        top_k: int = 5,
        engine: str = "google_ai_mode",
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("SERPAPI_API_KEY", "")
            or os.getenv("SERPER_API_KEY", "")   # backward compat
        )
        self.default_k = top_k
        self.engine    = engine

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        query = params.get("query", "").strip().strip("`").strip()
        top_k = int(params.get("top_k", self.default_k))
        subsequent_token = params.get("subsequent_request_token", "")

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

        # ── Build request params ───────────────────────────────────────────────
        req_params: Dict[str, Any] = {
            "engine":  self.engine,
            "q":       query,
            "api_key": self.api_key,
            "output":  "json",
        }
        if subsequent_token:
            req_params["subsequent_request_token"] = subsequent_token

        # ── Call SerpAPI ───────────────────────────────────────────────────────
        try:
            resp = requests.get(self.SERPAPI_URL, params=req_params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            return ToolResult(
                status=ToolStatus.ERROR,
                raw_output="",
                tool_name=self.tool_name,
                query=query,
                error_message="SerpAPI request timed out (30s).",
            )
        except requests.exceptions.RequestException as exc:
            return ToolResult(
                status=ToolStatus.ERROR,
                raw_output="",
                tool_name=self.tool_name,
                query=query,
                error_message=f"SerpAPI HTTP error: {exc}",
            )
        except ValueError as exc:
            return ToolResult(
                status=ToolStatus.ERROR,
                raw_output="",
                tool_name=self.tool_name,
                query=query,
                error_message=f"SerpAPI JSON parse error: {exc}",
            )

        print(f"  [SerpAPI/{self.engine}] Response keys: {list(data.keys())}")

        # ── API-level error ────────────────────────────────────────────────────
        if "error" in data:
            return ToolResult(
                status=ToolStatus.ERROR,
                raw_output="",
                tool_name=self.tool_name,
                query=query,
                error_message=f"SerpAPI error: {data['error']}",
            )

        # ── Extract AI answer text ─────────────────────────────────────────────
        ai_text    = _extract_text_blocks(data)
        references = _extract_references(data, top_k)

        print(f"  [SerpAPI/{self.engine}] text_blocks: {len(data.get('text_blocks', []))}  |  references: {len(references)}")

        if not ai_text and not references:
            print(f"  [SerpAPI/{self.engine}] WARNING: empty response. Keys: {list(data.keys())}")
            return ToolResult(
                status=ToolStatus.EMPTY,
                raw_output="Google AI Mode returned no content for this query.",
                tool_name=self.tool_name,
                query=query,
            )

        # ── Compose final output ───────────────────────────────────────────────
        parts = []
        if ai_text:
            parts.append("=== Google AI Mode Answer ===")
            parts.append(ai_text)
        if references:
            parts.append("\n=== Sources ===")
            parts.append(_format_references(references))

        raw_output = "\n".join(parts)

        # subsequent_request_token enables multi-turn follow-up queries
        next_token = data.get("subsequent_request_token", "")
        if next_token:
            print(f"  [SerpAPI/{self.engine}] subsequent_request_token available for follow-up")

        return ToolResult(
            status=ToolStatus.SUCCESS,
            raw_output=raw_output,
            tool_name=self.tool_name,
            query=query,
            structured={
                "ai_answer":   ai_text,
                "references":  references,
                "text_blocks": data.get("text_blocks", []),
            },
            metadata={
                "engine":                   self.engine,
                "subsequent_request_token": next_token,
                "search_metadata":          data.get("search_metadata", {}),
            },
        )


# ── Offline stub ──────────────────────────────────────────────────────────────

class MockWebSearchTool(BaseTool):
    """
    Returns deterministic fake AI Mode results.
    Useful for unit tests and local development without a live API key.
    """

    tool_name = "WEB_SEARCH"

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        query = params.get("query", "(empty)")
        ai_text = (
            f"=== Google AI Mode Answer ===\n"
            f"Based on current information, here is a synthesised answer for: '{query}'.\n"
            f"This is a mock response generated for offline testing. "
            f"In production, the Google AI Mode engine returns a real "
            f"Gemini-2.5-powered answer grounded in live Google search results."
        )
        refs = [
            {
                "position": 1,
                "title":    f"Mock Source 1 for: {query}",
                "link":     "https://example.com/1",
                "snippet":  f"Mock snippet describing context for '{query}'.",
                "source":   "example.com",
            },
            {
                "position": 2,
                "title":    f"Mock Source 2 for: {query}",
                "link":     "https://example.com/2",
                "snippet":  f"Another mock source with additional context.",
                "source":   "example.com",
            },
        ]
        ref_text = "\n\n=== Sources ===\n" + _format_references(refs)
        return ToolResult(
            status=ToolStatus.SUCCESS,
            raw_output=ai_text + ref_text,
            tool_name=self.tool_name,
            query=query,
            structured={"ai_answer": ai_text, "references": refs, "text_blocks": []},
            metadata={"mock": True, "engine": "google_ai_mode"},
        )

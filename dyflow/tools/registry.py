"""
dyflow/tools/registry.py
────────────────────────
ToolRegistry maps operator instruction_type strings (e.g. "WEB_SEARCH",
"SQL_QUERY") to concrete BaseTool instances.

The WorkflowExecutor checks this registry to decide whether to route
an operator to the LLM executor or to a real tool.

Usage
-----
    registry = ToolRegistry()
    registry.register("WEB_SEARCH", WebSearchTool(api_key="..."))
    registry.register("SQL_QUERY",  SQLQueryTool(db_url="sqlite:///data.db"))

    tool = registry.get("WEB_SEARCH")
    result = tool._timed_execute({"query": "...", "top_k": 5})
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import BaseTool


class ToolRegistry:
    """
    Singleton-style registry that maps operator instruction_type names
    to initialised BaseTool instances.
    """

    # Built-in tool operator names — all others are routed to the LLM executor
    TOOL_OPERATOR_TYPES = {
        "WEB_SEARCH",
        "SQL_QUERY",
    }

    # Operators that wrap / verify tool outputs — still LLM-executed but
    # need tool result injected into their context
    TOOL_AWARE_OPERATOR_TYPES = {
        "TOOL_REVIEW",
        "TOOL_REFINE",
        "RESULT_EXTRACT",
        "SEARCH_QUERY_FORMULATE",
    }

    def __init__(self) -> None:
        self._registry: Dict[str, BaseTool] = {}

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, instruction_type: str, tool: BaseTool) -> None:
        """
        Register a tool under the given operator instruction_type key.

        Parameters
        ----------
        instruction_type : str
            Must match the instruction_type used in the designer's stage JSON
            (e.g. "WEB_SEARCH").
        tool : BaseTool
            An initialised tool instance ready to call .execute().
        """
        key = instruction_type.upper()
        if key not in self.TOOL_OPERATOR_TYPES:
            raise ValueError(
                f"'{key}' is not a recognised tool operator type. "
                f"Valid types: {sorted(self.TOOL_OPERATOR_TYPES)}"
            )
        self._registry[key] = tool

    def get(self, instruction_type: str) -> Optional[BaseTool]:
        """Return the registered tool or None if not found."""
        return self._registry.get(instruction_type.upper())

    def is_tool_operator(self, instruction_type: str) -> bool:
        """True if the instruction_type should be dispatched to a real tool."""
        return instruction_type.upper() in self.TOOL_OPERATOR_TYPES

    def is_tool_aware(self, instruction_type: str) -> bool:
        """
        True if the operator is LLM-based but needs tool context injected
        (TOOL_REVIEW, TOOL_REFINE, RESULT_EXTRACT, SEARCH_QUERY_FORMULATE).
        """
        return instruction_type.upper() in self.TOOL_AWARE_OPERATOR_TYPES

    def registered_tools(self) -> List[str]:
        """List all registered tool instruction_types."""
        return list(self._registry.keys())

    def __repr__(self) -> str:
        return f"ToolRegistry(registered={self.registered_tools()})"

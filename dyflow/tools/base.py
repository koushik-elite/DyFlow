"""
dyflow/tools/base.py
────────────────────
Abstract base class for all tool integrations in DyFlow-T.

Every concrete tool (WebSearchTool, SQLQueryTool, …) must implement:
  - execute(params) → ToolResult
  - tool_name: str  (used as the routing key in ToolOperator)

ToolResult is a standardised container so TOOL_REVIEW and TOOL_REFINE
can inspect tool outputs uniformly regardless of tool type.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ToolStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"          # returned results, but incomplete
    ERROR = "error"              # tool raised an exception
    EMPTY = "empty"              # executed fine but returned nothing useful
    RATE_LIMITED = "rate_limited"


@dataclass
class ToolResult:
    """
    Standardised return value for every tool call.

    Fields
    ------
    status        : ToolStatus enum value
    raw_output    : raw string output from the tool (search snippets, SQL rows, etc.)
    structured    : optional dict for machine-readable parsed output
    tool_name     : name of the tool that produced this result
    query         : the exact query / parameters submitted to the tool
    elapsed_sec   : wall-clock execution time in seconds
    error_message : populated only when status == ERROR
    metadata      : anything extra the tool wants to surface (e.g., source URLs)
    """
    status:        ToolStatus
    raw_output:    str
    tool_name:     str
    query:         str
    structured:    Optional[Dict[str, Any]] = None
    elapsed_sec:   float = 0.0
    error_message: Optional[str] = None
    metadata:      Dict[str, Any] = field(default_factory=dict)

    def to_prompt_string(self) -> str:
        """
        Render the result as a human-readable block that can be
        injected directly into an operator prompt template.
        """
        lines = [
            f"Tool: {self.tool_name}",
            f"Status: {self.status.value}",
            f"Query: {self.query}",
            "---",
            self.raw_output or "(no output)",
        ]
        if self.error_message:
            lines += ["---", f"Error: {self.error_message}"]
        return "\n".join(lines)


class BaseTool(ABC):
    """
    Abstract base class for all DyFlow-T tools.

    Subclasses must define:
      - tool_name  (class attribute)
      - execute(params)

    The execute method receives a flat dict of parameters whose keys
    match the {placeholder} names in the operator prompt template.
    """

    tool_name: str = "base_tool"

    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> ToolResult:
        """
        Run the tool with the given parameters.

        Parameters
        ----------
        params : dict
            Tool-specific parameters (e.g. {"query": "...", "top_k": 5}).

        Returns
        -------
        ToolResult
        """
        raise NotImplementedError

    def _timed_execute(self, params: Dict[str, Any]) -> ToolResult:
        """Wraps execute() to record elapsed time automatically."""
        t0 = time.perf_counter()
        try:
            result = self.execute(params)
        except Exception as exc:
            result = ToolResult(
                status=ToolStatus.ERROR,
                raw_output="",
                tool_name=self.tool_name,
                query=str(params),
                error_message=str(exc),
            )
        result.elapsed_sec = time.perf_counter() - t0
        return result

    def __repr__(self) -> str:
        return f"<Tool: {self.tool_name}>"

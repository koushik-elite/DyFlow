"""
dyflow/core/tool_workflow.py
────────────────────────────
ToolAwareWorkflowExecutor — extends DyFlow's WorkflowExecutor to support
external tool operators (WEB_SEARCH, SQL_QUERY) via the ToolRegistry.

Key changes vs. original WorkflowExecutor
──────────────────────────────────────────
1. Accepts a ToolRegistry at init time.
2. In _execute_stage(), checks whether each operator's instruction_type
   is a tool operator or a tool-aware LLM operator, and dispatches
   accordingly — everything else routes to the original LLM executor path.
3. Adds TOOL_REVIEW verdict-driven branching:
   after a TOOL_REVIEW operator, if verdict == 'retry_with_refinement',
   the executor inserts a TOOL_REFINE stage before continuing.
4. Adds updated DESIGN_STAGE_PROMPT that includes the new operator types.
5. Is a drop-in replacement — existing DyFlow tasks run unchanged.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .tool_operator import (
    ToolExecutorOperator,
    ToolAwareLLMOperator,
    TOOL_PROMPT_TEMPLATES,
    parse_tool_review_verdict,
)
from ..tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# UPDATED DESIGN STAGE PROMPT  (includes tool operators)
# ══════════════════════════════════════════════════════════════════════════════

TOOL_DESIGN_STAGE_PROMPT = """
You are the workflow stage designer for DyFlow-T, an extended DyFlow agent
with access to external tools.

Original Problem:
{problem_description}

Current Execution Summary:
{state_summary}

# Available Operators and When to Use

## Reasoning Operators (LLM-executed)
1. DECOMPOSE_PROBLEM   — Break down a complex problem into 2–4 sub-goals.
2. GENERATE_PLAN       — Create a step-by-step plan for the current subgoal.
3. GENERATE_ANSWER     — Generate a complete solution with step-by-step reasoning.
4. REFINE_ANSWER       — Improve an existing solution based on review feedback.
5. REVIEW_SOLUTION     — Critically evaluate a solution for correctness.
6. GENERATE_CODE       — Write Python code to solve the current subgoal.
7. REFINE_CODE         — Improve or debug previously generated code.
8. ENSEMBLE            — Generate multiple solutions and select best by majority vote.
9. ORGANIZE_SOLUTION   — Format the final answer and TERMINATE the workflow.
10. DEFAULT            — General-purpose fallback operator.

## Tool Operators (External tool execution)
11. SEARCH_QUERY_FORMULATE — Construct the optimal web search query before retrieval.
    Use when: the subgoal requires external factual knowledge and the query needs refinement.
    Output: formulated_query_{id}

12. WEB_SEARCH         — Retrieve and summarise information from the web.
    Use when: the task requires up-to-date factual knowledge beyond the model's parametric memory.
    Always pair with: SEARCH_QUERY_FORMULATE → WEB_SEARCH → TOOL_REVIEW
    Output: search_result_{id}

13. SQL_QUERY          — Generate and execute a SQL query against a database.
    Use when: the task involves structured data retrieval, aggregation, or filtering.
    Always pair with: SQL_QUERY → TOOL_REVIEW → RESULT_EXTRACT
    Output: sql_result_{id}

14. TOOL_REVIEW        — Audit tool output quality and relevance.
    Use when: immediately after WEB_SEARCH or SQL_QUERY.
    Output verdict: accept / retry_with_refinement / reject
    Output: tool_review_{id}

15. TOOL_REFINE        — Diagnose and correct a failed tool call.
    Use when: TOOL_REVIEW verdict is 'retry_with_refinement'.
    Output: tool_refined_result_{id}

16. RESULT_EXTRACT     — Distil raw tool output into clean memory buffer entries.
    Use when: after TOOL_REVIEW verdict is 'accept', before GENERATE_ANSWER.
    Output: extracted_result_{id}

# Design Rules

1. Stage Structure:
   - Typical stage: 3–4 operators working together to achieve an intermediate goal
   - Tool stage pattern: [SEARCH_QUERY_FORMULATE →] WEB_SEARCH → TOOL_REVIEW → RESULT_EXTRACT
   - SQL stage pattern: SQL_QUERY → TOOL_REVIEW → RESULT_EXTRACT
   - Termination stage: ONLY ORGANIZE_SOLUTION — never mix with other operators
   - Maximum 7 stages total

2. Naming Conventions:
   - Stage IDs: Increase monotonically (e.g., "stage_1", "stage_2", "stage_3")
   - Operator IDs: Format "op_<stageNum>_<index>" (e.g., "op_3_1", "op_3_2")
   - Output keys: Use exact keys from the summary; never invent new ones

3. Input Keys:
   - Always reference existing output_key names from the summary
   - For tool operators: reference the formulated_query or sql_query from prior operators
   - For any reasoning operator needing the problem: include "original_problem" in input_keys

4. Tool Selection Guidelines:
   - If a question requires recent/external facts → use WEB_SEARCH chain
   - If a question involves structured data/tables → use SQL_QUERY chain
   - If a question is self-contained and logical → use pure reasoning operators
   - Never use WEB_SEARCH and SQL_QUERY in the same stage

Output Format:
Return valid JSON:
{{
  "stage_id": "stage_N",
  "stage_description": "One-sentence description of what this stage accomplishes",
  "operators": [
    {{
      "operator_id": "op_N_1",
      "operator_description": "What this specific operator does",
      "params": {{
        "instruction_type": "OPERATOR_NAME",
        "input_keys": ["key1", "key2"],
        "output_key": "act_X",
        "input_usage": "How to use the input_keys data"
      }}
    }}
  ]
}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# TOOL-AWARE WORKFLOW EXECUTOR
# ══════════════════════════════════════════════════════════════════════════════

class ToolAwareWorkflowExecutor:
    """
    Drop-in replacement for DyFlow's WorkflowExecutor that adds external
    tool support via ToolRegistry.

    Parameters
    ----------
    problem_description : str
        The task / question to solve.
    designer_service    : ModelService
        LLM service used by the designer (DyPlanner or GPT-4.1).
    executor_service    : ModelService
        LLM service used by reasoning operators.
    tool_registry       : ToolRegistry
        Pre-configured registry with WebSearchTool / SQLQueryTool.
    save_design_history : bool
        If True, appends each designed stage to state.design_history.
    max_tool_retries    : int
        Max TOOL_REFINE iterations before escalating.
    """

    def __init__(
        self,
        problem_description: str,
        designer_service,
        executor_service,
        tool_registry:        ToolRegistry,
        save_design_history:  bool = False,
        max_tool_retries:     int = 2,
    ) -> None:
        self.problem_description = problem_description
        self.designer_service    = designer_service
        self.executor_service    = executor_service
        self.tool_registry       = tool_registry
        self.save_design_history = save_design_history
        self.max_tool_retries    = max_tool_retries

        # Lazy import to avoid circular dependency with original codebase
        from .state import State
        self.state = State(original_problem=problem_description)

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, max_steps: int = 7) -> Tuple[str, Any]:
        """
        Execute the tool-augmented DyFlow loop.

        Returns
        -------
        (final_answer: str, trajectory: list)
        """
        logger.info(f"[ToolAwareWorkflowExecutor] Starting | steps={max_steps}")
        print(f"\n{'='*60}")
        print(f"DyFlow-T  |  Problem: {self.problem_description[:80]}...")
        print(f"{'='*60}\n")

        for step in range(max_steps):
            print(f"\n>>> Stage {step + 1} / {max_steps}")

            # 1. Designer produces next stage JSON
            stage_json = self._design_stage()
            if stage_json is None:
                logger.warning("Designer returned invalid JSON — stopping.")
                break

            stage_id   = stage_json.get("stage_id", f"stage_{step+1}")
            operators  = stage_json.get("operators", [])
            terminated = False

            # 2. Execute each operator in the stage
            for op_def in operators:
                op_id   = op_def.get("operator_id", "op_unknown")
                op_desc = op_def.get("operator_description", "")
                params  = op_def.get("params", {})
                instr   = params.get("instruction_type", "").upper()

                print(f"  [{op_id}] {instr}: {op_desc[:60]}")

                signal = self._dispatch_operator(op_id, op_desc, instr, params)

                # Handle TOOL_REVIEW verdict-driven branching
                if instr == "TOOL_REVIEW":
                    signal = self._handle_tool_review_branch(op_id, params, operators)

                if signal == "terminate" or instr == "ORGANIZE_SOLUTION":
                    terminated = True
                    break

            if self.save_design_history:
                self.state.design_history = getattr(self.state, "design_history", [])
                self.state.design_history.append(stage_json)

            if terminated:
                print("\n[DyFlow-T] Workflow complete.")
                break

        return self._extract_final_answer(), getattr(self.state, "workflow_log", [])

    # ── Designer call ─────────────────────────────────────────────────────────

    def _design_stage(self) -> Optional[Dict]:
        summary = self._summarise_state()
        prompt  = TOOL_DESIGN_STAGE_PROMPT.format(
            problem_description=self.problem_description,
            state_summary=summary,
        )
        try:
            raw = self.designer_service.chat(prompt)
            # Extract JSON block
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(raw)
        except Exception as exc:
            logger.error(f"Designer JSON parse error: {exc}\nRaw: {raw[:300]}")
            return None

    # ── Operator dispatch ─────────────────────────────────────────────────────

    def _dispatch_operator(
        self,
        op_id:   str,
        op_desc: str,
        instr:   str,
        params:  Dict[str, Any],
    ) -> str:
        """
        Route the operator to:
          - ToolExecutorOperator   if instruction_type is in TOOL_OPERATOR_TYPES
          - ToolAwareLLMOperator   if instruction_type is in TOOL_AWARE_OPERATOR_TYPES
          - InstructExecutorOperator (original) for all other operators
        """
        params["instruction_type"] = instr

        if self.tool_registry.is_tool_operator(instr):
            op = ToolExecutorOperator(op_id, op_desc, self.tool_registry)
            return op.execute(self.state, params)

        elif self.tool_registry.is_tool_aware(instr):
            llm_client = self._make_llm_client()
            op = ToolAwareLLMOperator(op_id, op_desc, llm_client)
            return op.execute(self.state, params)

        else:
            # Original DyFlow path
            from .operator import InstructExecutorOperator
            op = InstructExecutorOperator(op_id, op_desc, self.executor_service)
            return op.execute(self.state, params)

    # ── TOOL_REVIEW branching ─────────────────────────────────────────────────

    def _handle_tool_review_branch(
        self,
        review_op_id: str,
        review_params: Dict,
        stage_operators: List[Dict],
    ) -> str:
        """
        After TOOL_REVIEW runs, check its verdict.
        If 'retry_with_refinement', inject a TOOL_REFINE operator inline.
        """
        review_key = review_params.get("output_key", review_op_id)
        path       = f"actions.{review_key}.content"
        content    = self.state.get_data_by_path(path) or ""
        verdict    = parse_tool_review_verdict(content)

        print(f"    TOOL_REVIEW verdict: {verdict}")

        if verdict == "retry_with_refinement":
            self._run_tool_refine_inline(review_op_id, review_params)

        elif verdict == "reject":
            logger.warning(f"[TOOL_REVIEW] verdict=reject | escalating to designer replan")

        return "next"

    def _run_tool_refine_inline(self, review_op_id: str, review_params: Dict) -> None:
        """Inline TOOL_REFINE execution (up to max_tool_retries)."""
        retries = getattr(self.state, "_tool_refine_count", 0)
        if retries >= self.max_tool_retries:
            logger.warning("Max tool retries reached — skipping TOOL_REFINE.")
            return

        self.state._tool_refine_count = retries + 1
        refine_op_id = f"tool_refine_{retries + 1}"
        refine_params = {
            "instruction_type": "TOOL_REFINE",
            "input_keys":       review_params.get("input_keys", []) + [review_params.get("output_key", "")],
            "output_key":       f"tool_refined_result_{retries + 1}",
            "input_usage":      "Use the review verdict and original tool output to reformulate the query.",
        }
        print(f"    [Inline TOOL_REFINE] attempt {retries + 1}")
        self._dispatch_operator(refine_op_id, "Refine failed tool call", "TOOL_REFINE", refine_params)

        # Re-run the original tool with refined query
        self._rerun_tool_with_refined_query(review_params, retries)

    def _rerun_tool_with_refined_query(self, review_params: Dict, attempt: int) -> None:
        """Re-execute the tool using the refined query from TOOL_REFINE output."""
        refine_key  = f"tool_refined_result_{attempt + 1}"
        path        = f"actions.{refine_key}.content"
        refine_text = self.state.get_data_by_path(path) or ""

        refined_query_match = re.search(r"Refined Query\s*:\s*(.+?)(?:\n|$)", refine_text, re.IGNORECASE)
        if not refined_query_match:
            logger.warning("Could not parse refined query from TOOL_REFINE output.")
            return

        refined_query = refined_query_match.group(1).strip()
        logger.info(f"[TOOL_REFINE] Refined query: {refined_query}")

        # Find which tool to re-run from the original tool output keys
        for op_type in ["WEB_SEARCH", "SQL_QUERY"]:
            if self.tool_registry.get(op_type):
                rerun_op_id    = f"tool_rerun_{op_type.lower()}_{attempt + 1}"
                rerun_params   = {
                    "instruction_type": op_type,
                    "input_keys":       [],
                    "output_key":       f"search_result_refined_{attempt + 1}",
                    "input_usage":      f"Re-execute with refined query: {refined_query}",
                    "guidance":         refined_query,
                    "query":            refined_query,
                }
                print(f"    [Re-run {op_type}] query='{refined_query[:60]}'")
                self._dispatch_operator(rerun_op_id, f"Re-run {op_type}", op_type, rerun_params)
                break

    # ── State utilities ───────────────────────────────────────────────────────

    def _summarise_state(self) -> str:
        """Build a compact summary of state for the designer prompt."""
        lines = [f"Problem: {self.state.original_problem[:200]}"]
        for key, action in self.state.actions.items():
            content = action.get("content", "")
            preview = content[:300].replace("\n", " ") if content else "(empty)"
            lines.append(f"- [{key}]: {preview}")
        return "\n".join(lines)

    def _extract_final_answer(self) -> str:
        """Find the ORGANIZE_SOLUTION output or last meaningful action."""
        # Prefer organized solution
        for key, action in self.state.actions.items():
            if "organiz" in key.lower() or "final" in key.lower():
                return action.get("content", "")
        # Fallback: last action with content
        for action in reversed(list(self.state.actions.values())):
            content = action.get("content", "")
            if content and len(content) > 20:
                return content
        return "(no answer found)"

    def _make_llm_client(self):
        """Create a thin wrapper around executor_service for ToolAwareLLMOperator."""
        service = self.executor_service

        class _LLMClientWrapper:
            def chat(self, prompt: str) -> str:
                return service.generate(prompt)

        return _LLMClientWrapper()

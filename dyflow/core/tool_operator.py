"""
dyflow/core/tool_operator.py
────────────────────────────
Extends DyFlow's operator system with two new classes:

  ToolExecutorOperator  — routes an operator to a real tool (WebSearch / SQL)
                          instead of an LLM, stores ToolResult in state memory.

  ToolAwareLLMOperator  — LLM-executed operator that is "tool-aware":
                          it has access to a prior ToolResult injected into its
                          prompt context (used by TOOL_REVIEW, TOOL_REFINE,
                          RESULT_EXTRACT, SEARCH_QUERY_FORMULATE).

Both classes extend the original InstructExecutorOperator interface so the
WorkflowExecutor can dispatch them without modification to its core loop.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .operator import Operator, ExecuteSignal
from .state import State
from ..tools.base import ToolResult, ToolStatus
from ..tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES  —  New tool operators (Appendix G extension)
# ══════════════════════════════════════════════════════════════════════════════

TOOL_PROMPT_TEMPLATES: Dict[str, str] = {

    # ── SEARCH_QUERY_FORMULATE ────────────────────────────────────────────────
    "SEARCH_QUERY_FORMULATE": """You are an expert at constructing precise and effective web search queries.
Given the current task context and subgoal, formulate the optimal search query
that will retrieve the most relevant and useful information.

Context:
{context}

Guidance:
{guidance}

Instructions:
- Identify the core information need from the current subgoal.
- Use specific, unambiguous keywords — avoid generic terms.
- Include domain qualifiers if the topic is specialised.
- If the information need has multiple facets, generate up to 3 ranked query variants.
- Select the single best query as the primary output.
- Keep the primary query under 10 tokens.

Output Format:
Primary Query: <best single query string>
Alternative Queries:
  1. <variant 1>
  2. <variant 2>
Rationale: <one sentence explaining why this query best targets the subgoal>""",

    # ── WEB_SEARCH ────────────────────────────────────────────────────────────
    # Note: this template is used to FORMAT the result for memory storage
    # (actual retrieval is done by WebSearchTool.execute, not the LLM)
    "WEB_SEARCH": """You are an expert at synthesising web search results into actionable findings.
Analyse the retrieved results and extract the most relevant information for the current subgoal.

Context:
{context}

Guidance:
{guidance}

Retrieved Results:
{tool_output}

Instructions:
- Identify the most relevant results for the current subgoal.
- Extract key facts, figures, or statements directly from the retrieved snippets.
- Do NOT hallucinate or infer beyond what the results explicitly state.
- If results are insufficient or contradictory, flag this in the output.
- Summarise findings in 3–5 concise bullet points.

Output Format:
Query Issued: <the exact query string submitted>
Relevant Sources: <list of titles and URLs used>
Key Findings:
  - <finding 1>
  - <finding 2>
Confidence: <high / medium / low>
Gaps: <any missing information or ambiguity noted>""",

    # ── SQL_QUERY ─────────────────────────────────────────────────────────────
    # Template used to FORMAT the SQL result for memory storage
    "SQL_QUERY": """You are an expert SQL engineer capable of generating precise SQL queries
and interpreting database results.

Context:
{context}

Guidance:
{guidance}

Database Schema:
{schema}

SQL Query Executed:
{sql_query}

Execution Result:
{tool_output}

Instructions:
- Interpret the result set in plain language relevant to the current subgoal.
- Highlight key rows, aggregates, or patterns in the returned data.
- If the query returned an error, describe the likely cause clearly.
- If 0 rows were returned, assess whether this is expected or signals a problem.

Output Format:
SQL Generated:
  <sql query here>
Execution Status: <success / error / empty>
Result Summary: <natural language interpretation of the returned rows>
Row Count: <number of rows returned>
Error Details: <if applicable>""",

    # ── TOOL_REVIEW ───────────────────────────────────────────────────────────
    "TOOL_REVIEW": """You are a rigorous auditor of externally retrieved information. Your role is to
critically assess whether the tool output is accurate, relevant, and sufficient
to support the current subgoal before it is used in downstream reasoning.

Context:
{context}

Guidance:
{guidance}

Tool Operator Executed: {tool_operator_name}
Tool Input (Query/Parameters): {tool_input}
Tool Output:
{tool_output}

Instructions:
- Verify that the tool output directly addresses the information need of the subgoal.
- Check for factual inconsistencies, outdated information, or domain mismatches.
- Assess whether the retrieved data is complete or only partially answers the need.
- Identify any hallucinated, interpolated, or unverifiable content in the output.
- Flag low-confidence or contradictory results that require a retry or reformulation.
- Do NOT attempt to re-execute the tool; only audit what was returned.

Output Format:
Relevance Check: <does the output address the subgoal? yes / partial / no>
Accuracy Assessment: <are the facts verifiable and internally consistent?>
Completeness: <is the retrieved information sufficient for the subgoal?>
Identified Issues:
  - <issue 1, if any>
Overall Verdict: <accept / retry_with_refinement / reject>
Recommended Action: <proceed / invoke TOOL_REFINE / escalate to designer>""",

    # ── TOOL_REFINE ───────────────────────────────────────────────────────────
    "TOOL_REFINE": """You are an expert at diagnosing and correcting failed or low-quality tool calls.
Analyse the failure and produce a corrected reformulation of the tool invocation.

Context:
{context}

Guidance:
{guidance}

Original Tool Operator: {tool_operator_name}
Original Input (Query/Parameters): {original_tool_input}
Original Tool Output:
{original_tool_output}

Review Verdict: {tool_review_verdict}
Identified Issues: {tool_review_issues}

Instructions:
- Diagnose the root cause of the failure or insufficiency from the review verdict.
- Reformulate the query or parameters to address the identified issue specifically.
- For WEB_SEARCH: use alternative keywords, add domain qualifiers, or narrow scope.
- For SQL_QUERY: correct syntax errors, fix JOIN conditions, or adjust column refs.
- Provide the corrected query as output; the executor will re-run the tool.
- Limit refinement to 2 iterations; if still failing, escalate to designer.

Output Format:
Failure Diagnosis: <root cause identified from the review verdict>
Refinement Strategy: <what was changed and why>
Refined Query: <the corrected query string or SQL>
Resolution Status: <resolved / partially_resolved / unresolved>
Next Action: <proceed / escalate to designer>""",

    # ── RESULT_EXTRACT ────────────────────────────────────────────────────────
    "RESULT_EXTRACT": """You are an expert at distilling raw tool outputs into clean, structured information
that can be directly used by downstream reasoning steps.

Context:
{context}

Guidance:
{guidance}

Source Tool: {tool_operator_name}
Raw Tool Output:
{tool_output}

Target Subgoal: {subgoal_description}

Instructions:
- Extract only information that is directly relevant to the stated subgoal.
- Remove duplicates, advertisements, navigation text, and irrelevant metadata.
- Resolve minor contradictions across sources by noting the most reliable reference.
- Structure extracted information as typed key-value entries.
- Preserve source attribution for each extracted fact (URL or table name).
- If no relevant information can be extracted, state this explicitly.

Output Format:
Extracted Facts:
  <key_1>: <value_1>  [source: <ref>]
  <key_2>: <value_2>  [source: <ref>]
Summary: <2–3 sentence synthesis of the extracted content>
Extraction Confidence: <high / medium / low>
Unresolved Conflicts: <any contradictory facts noted>""",
}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL VERDICT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_tool_review_verdict(review_text: str) -> str:
    """
    Extracts the 'Overall Verdict' from a TOOL_REVIEW output string.

    Returns one of: 'accept', 'retry_with_refinement', 'reject', 'unknown'.
    """
    pattern = r"Overall Verdict\s*:\s*(accept|retry_with_refinement|reject)"
    match = re.search(pattern, review_text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "unknown"


def parse_refined_query(refine_text: str) -> Optional[str]:
    """
    Extracts the 'Refined Query' from a TOOL_REFINE output string.
    Returns None if not found.
    """
    pattern = r"Refined Query\s*:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, refine_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# ══════════════════════════════════════════════════════════════════════════════
# TOOL EXECUTOR OPERATOR
# ══════════════════════════════════════════════════════════════════════════════

class ToolExecutorOperator(Operator):
    """
    Dispatches an operator to a real external tool (WebSearch or SQLQuery)
    instead of an LLM.

    The operator reads its input parameters from state memory (M) using
    the ψ (input_keys) convention, calls the registered tool, and writes
    a ToolResult-derived string back to M under the output_key.

    Behaviour mirrors InstructExecutorOperator's memory read/write pattern
    so the WorkflowExecutor loop requires no structural changes.
    """

    def __init__(
        self,
        operator_id:          str,
        operator_description: str,
        tool_registry:        ToolRegistry,
    ) -> None:
        super().__init__(operator_id, operator_description)
        self.tool_registry = tool_registry

    def execute(self, state: State, params: Dict[str, Any]) -> ExecuteSignal:
        instruction_type = params.get("instruction_type", "").upper()
        output_key       = params.get("output_key", self.operator_id)
        input_keys       = params.get("input_keys", [])
        input_usage      = params.get("input_usage", "")

        # ── Resolve inputs from memory ────────────────────────────────────────
        resolved_inputs: Dict[str, str] = {}
        for key in input_keys:
            path = f"actions.{key}.content" if not key.startswith("actions.") and "." not in key else key
            val  = state.get_data_by_path(path) or state.get_data_by_path(key) or ""
            resolved_inputs[key] = str(val)

        # ── Get tool ──────────────────────────────────────────────────────────
        tool = self.tool_registry.get(instruction_type)
        if tool is None:
            err = f"No tool registered for instruction_type='{instruction_type}'"
            logger.error(err)
            self._store_output(state, output_key, f"ERROR: {err}")
            self._log_execution(state, params, "error", error_message=err)
            return "next"

        # ── Build tool params from resolved memory inputs ──────────────────
        # Convention: the 'guidance' field of the operator contains the query
        # or, if a SEARCH_QUERY_FORMULATE result is available, use that.
        tool_params = self._build_tool_params(
            instruction_type, resolved_inputs, params, state
        )

        # ── Execute tool ──────────────────────────────────────────────────────
        logger.info(f"[ToolExecutorOperator] {instruction_type} | params={tool_params}")
        print(f"\n--- Tool Operator: {instruction_type} ---")
        print(f"  Params: {tool_params}")

        result: ToolResult = tool._timed_execute(tool_params)

        print(f"  Status: {result.status.value} | elapsed: {result.elapsed_sec:.2f}s")

        # ── Store raw ToolResult in memory ────────────────────────────────────
        output_str = result.to_prompt_string()
        self._store_output(state, output_key, output_str)

        # Also store structured data separately for downstream operators
        structured_key = f"{output_key}_structured"
        if result.structured:
            self._store_output(
                state, structured_key, json.dumps(result.structured, default=str)
            )

        # Attach tool metadata to state for TOOL_REVIEW injection
        if not hasattr(state, "tool_results"):
            state.tool_results = {}
        state.tool_results[output_key] = result

        self._log_execution(
            state, params, "next",
            tool_status=result.status.value,
            tool_elapsed=result.elapsed_sec,
        )
        print(f"--- Tool Operator {self.operator_id} Finished ---")
        return "next"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_tool_params(
        self,
        instruction_type: str,
        resolved_inputs:  Dict[str, str],
        params:           Dict[str, Any],
        state:            State,
    ) -> Dict[str, Any]:
        """
        Build the concrete dict to pass to tool.execute().

        Priority for query:
          1. Explicit 'query' key in input_keys
          2. Refined query from TOOL_REFINE output in memory
          3. Formulated query from SEARCH_QUERY_FORMULATE in memory
          4. Operator guidance field (free-text instruction from designer)
        """
        guidance = params.get("guidance", params.get("input_usage", ""))

        if instruction_type == "WEB_SEARCH":
            # Prefer a formulated query from memory
            query = (
                resolved_inputs.get("query")
                or self._find_formulated_query(state)
                or guidance.strip()
            )
            return {"query": query, "top_k": params.get("top_k", 5)}

        elif instruction_type == "SQL_QUERY":
            query = (
                resolved_inputs.get("sql_query")
                or resolved_inputs.get("query")
                or self._find_refined_query(state)
                or guidance.strip()
            )
            return {
                "query": query,
                "limit": params.get("limit", 50),
            }

        return {"query": guidance.strip()}

    def _find_formulated_query(self, state: State) -> Optional[str]:
        """Scan state memory for the output of a SEARCH_QUERY_FORMULATE operator."""
        for key, val in state.actions.items():
            if "formulated_query" in key or "search_query" in key:
                content = val.get("content", "")
                match   = re.search(r"Primary Query\s*:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
                if match:
                    return match.group(1).strip().strip("`")
        return None

    def _find_refined_query(self, state: State) -> Optional[str]:
        """Scan state memory for a TOOL_REFINE refined query."""
        for key, val in state.actions.items():
            if "refined" in key:
                content = val.get("content", "")
                parsed  = parse_refined_query(content)
                if parsed:
                    return parsed
        return None

    def _store_output(self, state: State, output_key: str, content: str) -> None:
        full_key = (
            f"actions.{output_key}.content"
            if not output_key.startswith("actions.") and "." not in output_key
            else output_key
        )
        state.set_data_by_path(full_key, content)
        action_id = output_key.split(".")[-3] if output_key.startswith("actions.") else output_key
        if action_id not in state.actions:
            state.actions[action_id] = {}
        state.actions[action_id]["content"] = content


# ══════════════════════════════════════════════════════════════════════════════
# TOOL-AWARE LLM OPERATOR
# ══════════════════════════════════════════════════════════════════════════════

class ToolAwareLLMOperator(Operator):
    """
    An LLM-executed operator that enriches its context with ToolResult data
    from state memory before calling the LLM.

    Used for: TOOL_REVIEW, TOOL_REFINE, RESULT_EXTRACT, SEARCH_QUERY_FORMULATE.

    The operator automatically injects {tool_output}, {tool_operator_name},
    and {tool_input} into the prompt if a preceding ToolResult is found in
    state.tool_results.
    """

    def __init__(
        self,
        operator_id:          str,
        operator_description: str,
        llm_client,           # ExecutorLLMClient instance
    ) -> None:
        super().__init__(operator_id, operator_description)
        self.llm_client = llm_client

    def execute(self, state: State, params: Dict[str, Any]) -> ExecuteSignal:
        instruction_type = params.get("instruction_type", "").upper()
        output_key       = params.get("output_key", self.operator_id)
        input_keys       = params.get("input_keys", [])

        # ── Resolve standard inputs ───────────────────────────────────────────
        resolved = {}
        for key in input_keys:
            path = f"actions.{key}.content" if "." not in key else key
            resolved[key] = state.get_data_by_path(path) or state.get_data_by_path(key) or ""

        # ── Build prompt context ──────────────────────────────────────────────
        context  = self._build_context(state, resolved, input_keys)
        guidance = params.get("input_usage", params.get("guidance", ""))

        # ── Inject tool-specific fields ───────────────────────────────────────
        tool_fields = self._extract_tool_fields(state, input_keys)

        # ── Select and fill template ──────────────────────────────────────────
        template = TOOL_PROMPT_TEMPLATES.get(instruction_type, TOOL_PROMPT_TEMPLATES.get("RESULT_EXTRACT", ""))
        if not template:
            logger.warning(f"No template for instruction_type='{instruction_type}'")
            template = "Context:\n{context}\n\nGuidance:\n{guidance}\n\nOutput:\n"

        prompt = self._safe_format(template, {
            "context":             context,
            "guidance":            guidance,
            "tool_output":         tool_fields.get("tool_output", "(not available)"),
            "tool_operator_name":  tool_fields.get("tool_operator_name", "unknown"),
            "tool_input":          tool_fields.get("tool_input", "unknown"),
            "original_tool_input": tool_fields.get("original_tool_input", "unknown"),
            "original_tool_output":tool_fields.get("tool_output", "(not available)"),
            "tool_review_verdict": tool_fields.get("tool_review_verdict", "unknown"),
            "tool_review_issues":  tool_fields.get("tool_review_issues", "none"),
            "subgoal_description": guidance,
            "schema":              tool_fields.get("schema", "(not provided)"),
            "sql_query":           tool_fields.get("sql_query", ""),
            **resolved,
        })

        # ── Call LLM ──────────────────────────────────────────────────────────
        print(f"\n--- Tool-Aware LLM Operator: {instruction_type} ---")
        try:
            llm_output = self.llm_client.chat(prompt)
        except Exception as exc:
            llm_output = f"ERROR calling LLM: {exc}"
            logger.error(llm_output)

        # ── Store output ──────────────────────────────────────────────────────
        full_key = (
            f"actions.{output_key}.content"
            if "." not in output_key
            else output_key
        )
        state.set_data_by_path(full_key, llm_output)
        if output_key not in state.actions:
            state.actions[output_key] = {}
        state.actions[output_key]["content"] = llm_output

        self._log_execution(state, params, "next", llm_output_preview=llm_output[:120])
        print(f"--- Tool-Aware LLM Operator {self.operator_id} Finished ---")
        return "next"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_context(
        self, state: State, resolved: Dict[str, str], input_keys: List[str]
    ) -> str:
        parts = [f"Problem: {state.original_problem}"]
        for key in input_keys:
            if key in resolved and resolved[key]:
                parts.append(f"[{key}]:\n{resolved[key]}")
        return "\n\n".join(parts)

    def _extract_tool_fields(
        self, state: State, input_keys: List[str]
    ) -> Dict[str, str]:
        """
        Find the most recent ToolResult in state.tool_results that matches
        one of the input_keys, and extract fields for prompt injection.
        """
        fields: Dict[str, str] = {}
        tool_results = getattr(state, "tool_results", {})

        for key in input_keys:
            base_key = key.replace("actions.", "").split(".")[0]
            if base_key in tool_results:
                tr: ToolResult = tool_results[base_key]
                fields["tool_output"]        = tr.raw_output
                fields["tool_operator_name"] = tr.tool_name
                fields["tool_input"]         = tr.query
                fields["original_tool_input"]= tr.query
                break

        # Parse TOOL_REVIEW verdict from memory for TOOL_REFINE
        for key, val in state.actions.items():
            if "review" in key.lower():
                content = val.get("content", "")
                verdict = parse_tool_review_verdict(content)
                if verdict != "unknown":
                    fields["tool_review_verdict"] = verdict
                    # Extract issues section
                    issues_match = re.search(
                        r"Identified Issues\s*:(.*?)(?:Overall Verdict|$)",
                        content, re.IGNORECASE | re.DOTALL
                    )
                    fields["tool_review_issues"] = (
                        issues_match.group(1).strip() if issues_match else "none"
                    )
                break

        return fields

    @staticmethod
    def _safe_format(template: str, kwargs: Dict[str, Any]) -> str:
        """Format a template, replacing missing keys with a placeholder."""
        for key, val in kwargs.items():
            template = template.replace(f"{{{key}}}", str(val) if val is not None else "")
        return template

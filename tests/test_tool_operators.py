"""
tests/test_tool_operators.py
─────────────────────────────
Unit tests for DyFlow-T tool extension components.

Run with:  python -m pytest tests/test_tool_operators.py -v
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from unittest.mock import MagicMock, patch


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: ToolResult
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolResult(unittest.TestCase):

    def setUp(self):
        from dyflow.tools.base import ToolResult, ToolStatus
        self.ToolResult = ToolResult
        self.ToolStatus = ToolStatus

    def test_to_prompt_string_success(self):
        result = self.ToolResult(
            status=self.ToolStatus.SUCCESS,
            raw_output="Result line 1\nResult line 2",
            tool_name="WEB_SEARCH",
            query="quantum computing basics",
        )
        s = result.to_prompt_string()
        self.assertIn("WEB_SEARCH", s)
        self.assertIn("quantum computing basics", s)
        self.assertIn("Result line 1", s)
        self.assertIn("success", s)

    def test_to_prompt_string_error(self):
        result = self.ToolResult(
            status=self.ToolStatus.ERROR,
            raw_output="",
            tool_name="SQL_QUERY",
            query="SELECT * FROM users",
            error_message="Table 'users' does not exist",
        )
        s = result.to_prompt_string()
        self.assertIn("error", s)
        self.assertIn("Table 'users' does not exist", s)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: ToolRegistry
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolRegistry(unittest.TestCase):

    def setUp(self):
        from dyflow.tools.registry import ToolRegistry
        from dyflow.tools.web_search import MockWebSearchTool
        from dyflow.tools.sql_query import MockSQLQueryTool
        self.registry = ToolRegistry()
        self.MockWeb  = MockWebSearchTool
        self.MockSQL  = MockSQLQueryTool

    def test_register_and_get(self):
        self.registry.register("WEB_SEARCH", self.MockWeb())
        tool = self.registry.get("WEB_SEARCH")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.tool_name, "WEB_SEARCH")

    def test_register_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            self.registry.register("GENERATE_ANSWER", self.MockWeb())

    def test_is_tool_operator(self):
        self.assertTrue(self.registry.is_tool_operator("WEB_SEARCH"))
        self.assertTrue(self.registry.is_tool_operator("SQL_QUERY"))
        self.assertFalse(self.registry.is_tool_operator("GENERATE_ANSWER"))
        self.assertFalse(self.registry.is_tool_operator("REVIEW_SOLUTION"))

    def test_is_tool_aware(self):
        self.assertTrue(self.registry.is_tool_aware("TOOL_REVIEW"))
        self.assertTrue(self.registry.is_tool_aware("TOOL_REFINE"))
        self.assertTrue(self.registry.is_tool_aware("RESULT_EXTRACT"))
        self.assertTrue(self.registry.is_tool_aware("SEARCH_QUERY_FORMULATE"))
        self.assertFalse(self.registry.is_tool_aware("WEB_SEARCH"))

    def test_registered_tools_list(self):
        self.registry.register("WEB_SEARCH", self.MockWeb())
        self.registry.register("SQL_QUERY",  self.MockSQL())
        self.assertIn("WEB_SEARCH", self.registry.registered_tools())
        self.assertIn("SQL_QUERY",  self.registry.registered_tools())

    def test_get_unregistered_returns_none(self):
        self.assertIsNone(self.registry.get("SQL_QUERY"))


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: MockWebSearchTool
# ═══════════════════════════════════════════════════════════════════════════════

class TestMockWebSearchTool(unittest.TestCase):

    def setUp(self):
        from dyflow.tools.web_search import MockWebSearchTool
        from dyflow.tools.base import ToolStatus
        self.tool   = MockWebSearchTool()
        self.Status = ToolStatus

    def test_returns_success(self):
        result = self.tool.execute({"query": "climate change effects"})
        self.assertEqual(result.status, self.Status.SUCCESS)
        self.assertIn("climate change effects", result.raw_output)

    def test_structured_output_present(self):
        result = self.tool.execute({"query": "test"})
        self.assertIsNotNone(result.structured)
        self.assertIn("results", result.structured)
        self.assertGreater(len(result.structured["results"]), 0)

    def test_empty_query_still_runs(self):
        result = self.tool.execute({"query": ""})
        self.assertEqual(result.status, self.Status.SUCCESS)

    def test_timed_execute_sets_elapsed(self):
        result = self.tool._timed_execute({"query": "test timing"})
        self.assertGreaterEqual(result.elapsed_sec, 0.0)

    def test_tool_name(self):
        self.assertEqual(self.tool.tool_name, "WEB_SEARCH")


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: MockSQLQueryTool
# ═══════════════════════════════════════════════════════════════════════════════

class TestMockSQLQueryTool(unittest.TestCase):

    def setUp(self):
        from dyflow.tools.sql_query import MockSQLQueryTool
        from dyflow.tools.base import ToolStatus
        self.tool   = MockSQLQueryTool()
        self.Status = ToolStatus

    def test_returns_success(self):
        result = self.tool.execute({"query": "SELECT id, name FROM users LIMIT 10"})
        self.assertEqual(result.status, self.Status.SUCCESS)
        self.assertIn("id", result.raw_output)

    def test_structured_contains_rows(self):
        result = self.tool.execute({"query": "SELECT *"})
        self.assertIsNotNone(result.structured)
        self.assertIn("rows", result.structured)
        self.assertGreater(result.structured["row_count"], 0)

    def test_timed_execute(self):
        result = self.tool._timed_execute({"query": "SELECT 1"})
        self.assertGreaterEqual(result.elapsed_sec, 0.0)

    def test_tool_name(self):
        self.assertEqual(self.tool.tool_name, "SQL_QUERY")


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Real SQLQueryTool (read-only guard)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSQLQueryToolReadOnlyGuard(unittest.TestCase):

    def setUp(self):
        from dyflow.tools.sql_query import SQLQueryTool
        from dyflow.tools.base import ToolStatus
        # Use in-memory SQLite — no file needed
        self.tool   = SQLQueryTool(db_url="sqlite:///:memory:", read_only=True)
        self.Status = ToolStatus

    def test_insert_rejected_when_read_only(self):
        result = self.tool.execute({"query": "INSERT INTO foo VALUES (1)"})
        self.assertEqual(result.status, self.Status.ERROR)
        self.assertIn("Read-only", result.error_message)

    def test_delete_rejected_when_read_only(self):
        result = self.tool.execute({"query": "DELETE FROM bar WHERE id=1"})
        self.assertEqual(result.status, self.Status.ERROR)

    def test_empty_query_rejected(self):
        result = self.tool.execute({"query": ""})
        self.assertEqual(result.status, self.Status.ERROR)
        self.assertIn("Empty SQL query", result.error_message)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Verdict / Query parsers
# ═══════════════════════════════════════════════════════════════════════════════

class TestParsers(unittest.TestCase):

    def setUp(self):
        from dyflow.core.tool_operator import parse_tool_review_verdict, parse_refined_query
        self.parse_verdict = parse_tool_review_verdict
        self.parse_query   = parse_refined_query

    def test_parse_accept_verdict(self):
        text = "Relevance Check: yes\nOverall Verdict: accept\nRecommended Action: proceed"
        self.assertEqual(self.parse_verdict(text), "accept")

    def test_parse_retry_verdict(self):
        text = "Overall Verdict: retry_with_refinement"
        self.assertEqual(self.parse_verdict(text), "retry_with_refinement")

    def test_parse_reject_verdict(self):
        text = "Overall Verdict: reject\nRecommended Action: escalate"
        self.assertEqual(self.parse_verdict(text), "reject")

    def test_parse_unknown_verdict(self):
        text = "The output seems okay but uncertain."
        self.assertEqual(self.parse_verdict(text), "unknown")

    def test_parse_refined_query(self):
        text = "Refinement Strategy: narrowed keywords\nRefined Query: climate change Arctic 2024\nNext Action: proceed"
        result = self.parse_query(text)
        self.assertEqual(result, "climate change Arctic 2024")

    def test_parse_refined_query_missing(self):
        text = "No refined query found here."
        self.assertIsNone(self.parse_query(text))


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: ToolExecutorOperator (with mock State)
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolExecutorOperator(unittest.TestCase):

    def _make_mock_state(self):
        """Create a real State instance (not a MagicMock) so dict checks work."""
        from dyflow.core.state import State
        state = State(problem_description="Test problem")
        state.stages = {"stage_1": {}}
        return state

    def test_web_search_dispatch(self):
        from dyflow.tools.registry import ToolRegistry
        from dyflow.tools.web_search import MockWebSearchTool
        from dyflow.core.tool_operator import ToolExecutorOperator

        registry = ToolRegistry()
        registry.register("WEB_SEARCH", MockWebSearchTool())

        op = ToolExecutorOperator("op_1_1", "Web search for facts", registry)
        state = self._make_mock_state()

        params = {
            "instruction_type": "WEB_SEARCH",
            "input_keys":       [],
            "output_key":       "search_result_1",
            "input_usage":      "Search for climate change effects",
            "guidance":         "climate change effects on Arctic ice",
        }

        signal = op.execute(state, params)
        self.assertEqual(signal, "next")
        # ToolResult should be stored
        self.assertIn("search_result_1", state.tool_results)

    def test_sql_query_dispatch(self):
        from dyflow.tools.registry import ToolRegistry
        from dyflow.tools.sql_query import MockSQLQueryTool
        from dyflow.core.tool_operator import ToolExecutorOperator

        registry = ToolRegistry()
        registry.register("SQL_QUERY", MockSQLQueryTool())

        op = ToolExecutorOperator("op_2_1", "SQL data retrieval", registry)
        state = self._make_mock_state()

        params = {
            "instruction_type": "SQL_QUERY",
            "input_keys":       [],
            "output_key":       "sql_result_1",
            "input_usage":      "Get all users from database",
            "guidance":         "SELECT * FROM users LIMIT 10",
        }

        signal = op.execute(state, params)
        self.assertEqual(signal, "next")
        self.assertIn("sql_result_1", state.tool_results)

    def test_unregistered_tool_returns_next(self):
        from dyflow.tools.registry import ToolRegistry
        from dyflow.core.tool_operator import ToolExecutorOperator

        registry = ToolRegistry()  # empty — no tools registered
        op = ToolExecutorOperator("op_1_1", "Unregistered tool test", registry)
        state = self._make_mock_state()

        params = {
            "instruction_type": "WEB_SEARCH",
            "input_keys":       [],
            "output_key":       "search_result_1",
            "guidance":         "test query",
        }

        signal = op.execute(state, params)
        # Should return 'next' with an error stored, not raise
        self.assertEqual(signal, "next")


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: TOOL_PROMPT_TEMPLATES completeness
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolPromptTemplates(unittest.TestCase):

    def setUp(self):
        from dyflow.core.tool_operator import TOOL_PROMPT_TEMPLATES
        self.templates = TOOL_PROMPT_TEMPLATES

    def test_all_six_templates_present(self):
        expected = {
            "SEARCH_QUERY_FORMULATE",
            "WEB_SEARCH",
            "SQL_QUERY",
            "TOOL_REVIEW",
            "TOOL_REFINE",
            "RESULT_EXTRACT",
        }
        self.assertEqual(expected, set(self.templates.keys()))

    def test_each_template_has_context_placeholder(self):
        for name, template in self.templates.items():
            self.assertIn("{context}",  template, f"{name} missing {{context}}")
            self.assertIn("{guidance}", template, f"{name} missing {{guidance}}")

    def test_tool_review_has_verdict_format(self):
        t = self.templates["TOOL_REVIEW"]
        self.assertIn("Overall Verdict", t)
        self.assertIn("accept", t)
        self.assertIn("retry_with_refinement", t)

    def test_tool_refine_references_verdict(self):
        t = self.templates["TOOL_REFINE"]
        self.assertIn("{tool_review_verdict}", t)
        self.assertIn("Refined Query", t)

    def test_result_extract_has_output_format(self):
        t = self.templates["RESULT_EXTRACT"]
        self.assertIn("Extracted Facts", t)
        self.assertIn("Summary", t)


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)

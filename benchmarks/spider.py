"""
benchmarks/spider.py
─────────────────────
Spider / BIRD Text-to-SQL benchmark for DyFlow-T.

Metric: Execution Accuracy (EX)
  - Model generates a SQL query
  - Both predicted and gold queries are executed against the DB
  - Results are compared as sets of tuples (order-insensitive)

Dataset format: benchmarks/data/Spider/spider_dev.json
Each record:
    {
        "db_id":     str,          # database name
        "question":  str,          # natural language question
        "query":     str,          # gold SQL query
        "hardness":  str           # easy | medium | hard | extra
    }

Download: python scripts/install_spider.py
"""

import os
import re
import sys
import json
import sqlite3
import datetime
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .framework import BaseBenchmark, get_relative_path
from dyflow.model_service import ModelService
from tqdm import tqdm


# ── SQL execution helpers ──────────────────────────────────────────────────────

def _clean_sql(sql: str) -> str:
    """Strip markdown fences and normalise whitespace."""
    sql = re.sub(r"```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"```\s*", "", sql)
    # Take only the first statement
    sql = sql.split(";")[0].strip()
    return sql


def _extract_sql_from_text(text: str) -> str:
    """Try to extract a SQL SELECT from a verbose model answer."""
    # Look for a markdown code block first
    match = re.search(r"```(?:sql)?\s*(SELECT.*?)```", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # Find the last SELECT … ; block
    matches = list(re.finditer(r"(SELECT\s+.+?)(?:;|$)", text, re.IGNORECASE | re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()

    return _clean_sql(text)


def _execute_sql(db_path: str, sql: str, timeout: int = 10) -> Tuple[bool, Any]:
    """
    Execute sql against db_path.
    Returns (success: bool, result: list of tuples | error_str).
    """
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.execute("PRAGMA query_only = ON;")
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return True, rows
    except Exception as exc:
        return False, str(exc)


def _results_match(pred_rows: list, gold_rows: list) -> bool:
    """Order-insensitive set comparison of result tuples (EX metric)."""
    def normalise(rows):
        return sorted(
            [tuple(str(v).strip().lower() if v is not None else "" for v in row) for row in rows]
        )
    return normalise(pred_rows) == normalise(gold_rows)


# ── Benchmark class ────────────────────────────────────────────────────────────

class SpiderBenchmark(BaseBenchmark):
    """
    Evaluates DyFlow-T on Spider (or BIRD) using Execution Accuracy.

    EX = fraction of questions where predicted SQL returns the same
         result set as the gold SQL when run on the target database.
    """

    def __init__(
        self,
        execution_model: str = "gemini-2.5-flash",
        baseline: str = "DyFlow-T",
        mode: str = "dev",               # 'dev' | 'train' | 'test'
        dataset: str = "spider",         # 'spider' | 'bird'
        db_root: Optional[str] = None,   # path to folder of SQLite databases
    ):
        super().__init__(execution_model, baseline, mode)

        self.dataset   = dataset.lower()
        self.db_root   = db_root or get_relative_path(f"data/{dataset.capitalize()}/databases")

        self.dataset_path = get_relative_path(
            f"data/{dataset.capitalize()}/{dataset}_{mode}.json"
        )
        self.output_path = get_relative_path(
            f"results/{dataset.capitalize()}/{mode}/{baseline}_{execution_model}.json"
        )
        self.cost_path = get_relative_path(
            f"results/{dataset.capitalize()}/{mode}/{baseline}_{execution_model}_cost.json"
        )
        self.metrics_path = get_relative_path(
            f"results/{dataset.capitalize()}/{mode}/{baseline}_{execution_model}_metrics.json"
        )

        self.generate_service: Optional[ModelService] = None
        self.executor_service: Optional[ModelService] = None

    # ── DB helpers ─────────────────────────────────────────────────────────────

    def _db_path(self, db_id: str) -> str:
        return os.path.join(self.db_root, db_id, f"{db_id}.db")

    def _get_schema(self, db_id: str) -> str:
        """Return CREATE TABLE statements for all tables in db_id."""
        db_path = self._db_path(db_id)
        if not os.path.exists(db_path):
            return f"(database {db_id} not found at {db_path})"
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL;"
            )
            schemas = [row[0] for row in cursor.fetchall()]
            conn.close()
            return "\n\n".join(schemas) if schemas else "(no tables found)"
        except Exception as exc:
            return f"(schema error: {exc})"

    def _build_question_prompt(self, question: str, db_id: str) -> str:
        schema = self._get_schema(db_id)
        return (
            f"You are an expert SQL assistant.\n\n"
            f"Database: {db_id}\n\n"
            f"Schema:\n{schema}\n\n"
            f"Question: {question}\n\n"
            "Write a single valid SQLite SELECT query to answer the question.\n"
            "Return ONLY the SQL query inside a ```sql ... ``` code block."
        )

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _score_prediction(
        self, db_id: str, pred_sql: str, gold_sql: str
    ) -> Tuple[bool, str, str]:
        """
        Returns (correct, pred_sql_clean, execution_error).
        correct = True if EX passes.
        """
        db_path = self._db_path(db_id)
        if not os.path.exists(db_path):
            return False, pred_sql, f"DB not found: {db_path}"

        pred_clean = _extract_sql_from_text(pred_sql)
        gold_clean = _clean_sql(gold_sql)

        ok_gold, gold_rows = _execute_sql(db_path, gold_clean)
        if not ok_gold:
            return False, pred_clean, f"Gold SQL failed: {gold_rows}"

        ok_pred, pred_rows = _execute_sql(db_path, pred_clean)
        if not ok_pred:
            return False, pred_clean, f"Pred SQL failed: {pred_rows}"

        correct = _results_match(pred_rows, gold_rows)
        return correct, pred_clean, ""

    # ── Core evaluation ────────────────────────────────────────────────────────

    def evaluate_problem(
        self,
        problem: dict,
        function: Callable,
        judge_service: Optional[ModelService] = None,
    ) -> dict:
        db_id     = problem.get("db_id", "")
        question  = problem.get("question", "")
        gold_sql  = problem.get("query", "")
        hardness  = problem.get("hardness", "unknown")

        # Enrich question with schema so the workflow has full context
        enriched_q = self._build_question_prompt(question, db_id)

        # Generate predicted SQL
        try:
            result = function(enriched_q)
            if isinstance(result, tuple):
                pred_output, design_history = result
            else:
                pred_output, design_history = result, None
        except Exception as exc:
            problem.update({
                "predicted_sql": "",
                "judge_result":  False,
                "correct":       False,
                "error":         str(exc),
                "hardness":      hardness,
            })
            return problem

        pred_output = pred_output or ""
        correct, pred_clean, exec_err = self._score_prediction(db_id, pred_output, gold_sql)

        problem["predicted_sql"]    = pred_clean
        problem["raw_output"]       = pred_output
        problem["gold_sql"]         = gold_sql
        problem["judge_result"]     = correct
        problem["correct"]          = correct
        problem["execution_error"]  = exec_err
        problem["design_history"]   = design_history
        problem["hardness"]         = hardness
        return problem

    def evaluate_all_problems(
        self,
        function: Callable,
        judge_service: Optional[ModelService] = None,
        generate_service: Optional[ModelService] = None,
        max_workers: int = 10,
        size: Optional[int] = None,
    ) -> List[dict]:
        problems = self.load_json(self.dataset_path)
        if size is not None:
            problems = problems[:size]

        # Resume support
        temp_dir  = os.path.dirname(self.output_path)
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"temp_{self.mode}_{self.dataset}_results.json")
        results, done_ids = [], set()

        if os.path.exists(temp_file):
            try:
                with open(temp_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                done_ids = {
                    r.get("db_id", "") + r.get("question", "")[:40]
                    for r in results
                }
                print(f"Resumed: {len(results)} already done.")
                problems = [
                    p for p in problems
                    if p.get("db_id", "") + p.get("question", "")[:40] not in done_ids
                ]
                print(f"Remaining: {len(problems)}")
            except Exception as e:
                print(f"Could not resume: {e}")
                results, done_ids = [], set()

        def _save_temp():
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        def _save_cost():
            stats: Dict[str, Any] = {
                "mode": self.mode, "dataset": self.dataset,
                "completed": len(results),
                "timestamp": str(datetime.datetime.now()),
            }
            if generate_service:
                stats["generate_cost"] = generate_service.get_usage_stats()
            if self.executor_service:
                stats["executor_cost"] = self.executor_service.get_usage_stats()
            with open(self.cost_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=4)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {
                ex.submit(self.evaluate_problem, prob, function, judge_service): prob
                for prob in problems
            }
            for fut in tqdm(
                concurrent.futures.as_completed(future_map),
                total=len(problems),
                initial=len(done_ids),
                desc=f"{self.dataset.upper()} {self.mode}",
                unit="q",
            ):
                prob = future_map[fut]
                try:
                    result = fut.result(timeout=300)
                    results.append(result)
                except concurrent.futures.TimeoutError:
                    prob.update({"judge_result": False, "correct": False, "error": "timeout"})
                    results.append(prob)
                except Exception as exc:
                    prob.update({"judge_result": False, "correct": False, "error": str(exc)})
                    results.append(prob)
                finally:
                    _save_temp()
                    if len(results) % 10 == 0:
                        _save_cost()

        _save_cost()
        return results

    # ── Metrics ────────────────────────────────────────────────────────────────

    def calculate_metrics(self, results: Optional[List[dict]] = None) -> Dict[str, Any]:
        if results is None:
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
            except FileNotFoundError:
                print(f"Results file not found: {self.output_path}")
                return {"execution_accuracy": 0.0, "error": "results file not found"}

        total, correct, errors = 0, 0, 0
        hardness_stats: Dict[str, Dict[str, int]] = {}

        for r in results:
            if r.get("error") and not r.get("predicted_sql"):
                errors += 1
                continue
            total += 1
            h = r.get("hardness", "unknown")
            if h not in hardness_stats:
                hardness_stats[h] = {"total": 0, "correct": 0}
            hardness_stats[h]["total"] += 1
            if r.get("judge_result", False):
                correct += 1
                hardness_stats[h]["correct"] += 1

        overall_ex = correct / total if total > 0 else 0.0
        hardness_acc = {
            h: (s["correct"] / s["total"] if s["total"] > 0 else 0.0)
            for h, s in hardness_stats.items()
        }

        metrics = {
            "benchmark":          self.dataset.upper(),
            "mode":               self.mode,
            "baseline":           self.baseline,
            "model":              self.execution_model,
            "total":              total,
            "correct":            correct,
            "errors":             errors,
            "execution_accuracy": round(overall_ex, 4),
            "hardness_accuracy":  {h: round(a, 4) for h, a in hardness_acc.items()},
            "hardness_counts":    hardness_stats,
            "timestamp":          str(datetime.datetime.now()),
        }

        # Print report
        print("\n" + "=" * 60)
        print(f"{self.dataset.upper()} BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Baseline   : {self.baseline}")
        print(f"Model      : {self.execution_model}")
        print(f"Mode       : {self.mode}")
        print(f"Total      : {total}  (errors: {errors})")
        print(f"Correct    : {correct}")
        print(f"EX Accuracy: {overall_ex:.2%}")
        print("-" * 40)
        for h in ["easy", "medium", "hard", "extra", "unknown"]:
            if h in hardness_acc:
                s = hardness_stats[h]
                print(f"  {h:<8}: {hardness_acc[h]:.2%}  ({s['correct']}/{s['total']})")
        print("=" * 60)

        return metrics

    def record_cost(
        self,
        generate_service: Optional[ModelService],
        judge_service: Optional[ModelService] = None,
    ) -> None:
        stats: Dict[str, Any] = {
            "mode":      self.mode,
            "dataset":   self.dataset,
            "timestamp": str(datetime.datetime.now()),
            "generate_cost": generate_service.get_usage_stats() if generate_service else {},
        }
        if self.executor_service:
            stats["executor_cost"] = self.executor_service.get_usage_stats()
        with open(self.cost_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)

    # ── Entry point ────────────────────────────────────────────────────────────

    def run(
        self,
        generate_service: ModelService,
        judge_service: Optional[ModelService] = None,
        function: Optional[Callable] = None,
        size: Optional[int] = None,
        max_workers: int = 10,
        pass_k: Optional[int] = None,
    ) -> List[dict]:
        if function is None:
            raise ValueError("function parameter is required for SpiderBenchmark.run()")

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.generate_service = generate_service

        results = self.evaluate_all_problems(
            function=function,
            generate_service=generate_service,
            max_workers=max_workers,
            size=size,
        )

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\nResults saved → {self.output_path}")

        self.record_cost(generate_service)

        # Clean temp file
        temp_file = os.path.join(
            os.path.dirname(self.output_path),
            f"temp_{self.mode}_{self.dataset}_results.json",
        )
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass

        metrics = self.calculate_metrics(results)

        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved → {self.metrics_path}")

        return results

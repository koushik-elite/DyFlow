"""
GAIA Benchmark evaluation for DyFlow-T (tool-augmented).

GAIA (General AI Assistants) tests real-world question answering that requires
web search, multi-step reasoning, and tool use across three difficulty levels.

Dataset format expected at: benchmarks/data/GAIA/GAIA_{mode}.json
Each record:
    {
        "task_id":      str,
        "question":     str,
        "final_answer": str,
        "level":        int,        # 1 | 2 | 3
        "file_name":    str | null, # optional attachment
        "annotator_metadata": {...} # optional
    }

Download from: https://huggingface.co/datasets/gaia-benchmark/GAIA
"""

import os
import re
import sys
import json
import string
import datetime
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .framework import BaseBenchmark, get_relative_path
from dyflow.model_service import ModelService
from tqdm import tqdm


# ── Exact-match helpers ────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lower-case, strip punctuation/whitespace for loose exact match."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Primary GAIA metric: normalised exact match."""
    return _normalise(prediction) == _normalise(ground_truth)


def contains_match(prediction: str, ground_truth: str) -> bool:
    """Secondary check: ground truth appears inside the prediction."""
    return _normalise(ground_truth) in _normalise(prediction)


# ── Benchmark class ────────────────────────────────────────────────────────────

class GAIABenchmark(BaseBenchmark):
    """
    Evaluates DyFlow-T on the GAIA benchmark.

    Scoring follows the official GAIA protocol:
      - Primary:   normalised exact match
      - Secondary: contains-match (used for partial credit reporting only)
      - Reports accuracy per level (1 / 2 / 3) and overall
    """

    def __init__(
        self,
        execution_model: str = "gemini-2.5-flash",
        baseline: str = "DyFlow-T",
        mode: str = "validation",       # 'validation' | 'test'
        samples_per_task: int = 1,
        use_llm_judge: bool = True,     # use Gemini to extract final answer
    ):
        super().__init__(execution_model, baseline, mode)

        self.use_llm_judge = use_llm_judge
        self.samples_per_task = max(1, samples_per_task)
        self.generate_service: Optional[ModelService] = None
        self.executor_service: Optional[ModelService] = None

        self.dataset_path = get_relative_path(f"data/GAIA/GAIA_{mode}.json")
        self.output_path  = get_relative_path(
            f"results/GAIA/{mode}/{baseline}_{execution_model}.json"
        )
        self.cost_path = get_relative_path(
            f"results/GAIA/{mode}/{baseline}_{execution_model}_cost.json"
        )
        self.metrics_path = get_relative_path(
            f"results/GAIA/{mode}/{baseline}_{execution_model}_metrics.json"
        )

    # ── Judge helpers ──────────────────────────────────────────────────────────

    def judge_prompt(self, question: str, model_output: str, ground_truth: str) -> str:
        return (
            "You are an expert evaluator for the GAIA benchmark.\n"
            "Your task: extract the FINAL answer from the model output and decide "
            "if it matches the ground truth.\n\n"
            f"Question:\n{question}\n\n"
            f"Model Output:\n{model_output}\n\n"
            f"Ground Truth Answer:\n{ground_truth}\n\n"
            "Instructions:\n"
            "1. Extract the model's final answer (ignore reasoning steps).\n"
            "2. Compare it to the ground truth using normalised exact match "
            "(case-insensitive, ignore punctuation).\n"
            "3. Output the extracted answer on a line starting with 'Extracted: '\n"
            "4. Then respond with [[True]] if correct or [[False]] if incorrect."
        )

    def _extract_answer_with_llm(
        self, question: str, model_output: str, ground_truth: str, judge_service: ModelService
    ) -> Tuple[str, bool]:
        """Use LLM judge to extract final answer and score it."""
        prompt = self.judge_prompt(question, model_output, ground_truth)
        try:
            response = judge_service.generate(prompt=prompt)
            output = response.get("response", "") if isinstance(response, dict) else str(response)

            # Extract the answer the model gave
            extracted = ""
            for line in output.splitlines():
                if line.strip().lower().startswith("extracted:"):
                    extracted = line.split(":", 1)[1].strip()
                    break

            # Get judge verdict
            is_correct = self.extract_judge_result(output)
            return extracted, is_correct
        except Exception as e:
            print(f"  Judge error: {e}")
            return "", False

    def _score(
        self,
        question: str,
        model_output: str,
        ground_truth: str,
        judge_service: Optional[ModelService],
    ) -> Tuple[str, bool, bool]:
        """
        Returns (extracted_answer, exact_match_bool, contains_match_bool).
        """
        # 1. Try to parse explicit 'Final Answer:' line from the workflow output
        fa_match = re.search(r"Final Answer\s*:\s*(.+?)(?:\n|$)", model_output, re.IGNORECASE)
        clean_output = fa_match.group(1).strip() if fa_match else model_output

        # 2. Fast-path exact/contains match on cleaned output
        em = exact_match(clean_output, ground_truth)
        cm = contains_match(clean_output, ground_truth)
        if em:
            return clean_output, True, True

        # 3. LLM judge for ambiguous cases
        if self.use_llm_judge and judge_service is not None:
            extracted, judge_correct = self._extract_answer_with_llm(
                question, model_output, ground_truth, judge_service
            )
            if extracted:
                em = em or exact_match(extracted, ground_truth) or judge_correct
                cm = cm or contains_match(extracted, ground_truth)
                return extracted, em, cm

        return clean_output, em, cm

    # ── Core evaluation ────────────────────────────────────────────────────────

    def evaluate_problem(
        self,
        problem: dict,
        function: Callable,
        judge_service: Optional[ModelService] = None,
    ) -> dict:
        question     = problem.get("question", "")
        ground_truth = str(problem.get("final_answer", "")).strip()
        level        = problem.get("level", 1)

        # Generate answer(s)
        answers, design_histories = [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.samples_per_task) as ex:
            futures = [ex.submit(function, question) for _ in range(self.samples_per_task)]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    result = fut.result()
                    if isinstance(result, tuple):
                        ans, hist = result
                    else:
                        ans, hist = result, None
                    answers.append(ans or "")
                    design_histories.append(hist)
                except Exception as e:
                    print(f"  Generation error: {e}")
                    answers.append("")
                    design_histories.append(None)

        # Score each answer
        em_results, cm_results, extracted_answers = [], [], []
        for ans in answers:
            extracted, em, cm = self._score(question, ans, ground_truth, judge_service)
            em_results.append(em)
            cm_results.append(cm)
            extracted_answers.append(extracted)

        problem["generated_solutions"]  = answers
        problem["extracted_answers"]    = extracted_answers
        problem["design_histories"]     = design_histories
        problem["em_results"]           = em_results
        problem["cm_results"]           = cm_results
        problem["judge_result"]         = em_results[0] if em_results else False
        problem["generated_solution"]   = answers[0] if answers else ""
        problem["extracted_answer"]     = extracted_answers[0] if extracted_answers else ""
        problem["level"]                = level
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
        temp_file = os.path.join(temp_dir, f"temp_{self.mode}_gaia_results.json")
        results, completed_ids = [], set()

        if os.path.exists(temp_file):
            try:
                with open(temp_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                completed_ids = {r.get("task_id", r.get("question", "")[:40]) for r in results}
                print(f"Resumed: {len(results)} problems already done.")
                problems = [
                    p for p in problems
                    if p.get("task_id", p.get("question", "")[:40]) not in completed_ids
                ]
                print(f"Remaining: {len(problems)}")
            except Exception as e:
                print(f"Could not resume: {e}")
                results, completed_ids = [], set()

        def _save_temp():
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        def _save_cost():
            stats: Dict[str, Any] = {
                "mode": self.mode,
                "completed": len(results),
                "timestamp": str(datetime.datetime.now()),
            }
            if generate_service:
                stats["generate_cost"] = generate_service.get_usage_stats()
            if judge_service:
                stats["judge_cost"] = judge_service.get_usage_stats()
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
                initial=len(completed_ids),
                desc=f"GAIA {self.mode}",
                unit="q",
            ):
                prob = future_map[fut]
                try:
                    result = fut.result(timeout=300)
                    results.append(result)
                except concurrent.futures.TimeoutError:
                    prob.update({
                        "generated_solution": None,
                        "judge_result": False,
                        "error": "timeout",
                    })
                    results.append(prob)
                except Exception as exc:
                    prob.update({
                        "generated_solution": None,
                        "judge_result": False,
                        "error": str(exc),
                    })
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
                return {"accuracy": 0.0, "error": "results file not found"}

        total, correct = 0, 0
        level_stats: Dict[int, Dict[str, int]] = {1: {"total": 0, "correct": 0},
                                                   2: {"total": 0, "correct": 0},
                                                   3: {"total": 0, "correct": 0}}
        errors = 0

        for r in results:
            if r.get("error"):
                errors += 1
                continue
            if r.get("generated_solution") is None:
                continue
            total += 1
            lv = int(r.get("level", 1))
            if lv not in level_stats:
                level_stats[lv] = {"total": 0, "correct": 0}
            level_stats[lv]["total"] += 1
            if r.get("judge_result", False):
                correct += 1
                level_stats[lv]["correct"] += 1

        overall_acc = correct / total if total > 0 else 0.0
        level_acc   = {
            lv: (s["correct"] / s["total"] if s["total"] > 0 else 0.0)
            for lv, s in level_stats.items()
        }

        metrics = {
            "benchmark":      "GAIA",
            "mode":           self.mode,
            "baseline":       self.baseline,
            "model":          self.execution_model,
            "total":          total,
            "correct":        correct,
            "errors":         errors,
            "accuracy":       round(overall_acc, 4),
            "level_accuracy": {f"level_{lv}": round(acc, 4) for lv, acc in level_acc.items()},
            "level_counts":   level_stats,
            "timestamp":      str(datetime.datetime.now()),
        }

        # Print report
        print("\n" + "=" * 60)
        print("GAIA BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Baseline   : {self.baseline}")
        print(f"Model      : {self.execution_model}")
        print(f"Mode       : {self.mode}")
        print(f"Total      : {total}  (errors: {errors})")
        print(f"Correct    : {correct}")
        print(f"Accuracy   : {overall_acc:.2%}")
        print("-" * 40)
        for lv in sorted(level_acc):
            s = level_stats[lv]
            print(f"  Level {lv}  : {level_acc[lv]:.2%}  ({s['correct']}/{s['total']})")
        print("=" * 60)

        return metrics

    def record_cost(
        self,
        generate_service: ModelService,
        judge_service: ModelService,
    ) -> None:
        stats: Dict[str, Any] = {
            "mode":      self.mode,
            "timestamp": str(datetime.datetime.now()),
            "generate_cost": generate_service.get_usage_stats() if generate_service else {},
            "judge_cost":    judge_service.get_usage_stats()    if judge_service    else {},
        }
        if self.executor_service:
            stats["executor_cost"] = self.executor_service.get_usage_stats()
        with open(self.cost_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)

    # ── Entry point ────────────────────────────────────────────────────────────

    def run(
        self,
        generate_service: ModelService,
        judge_service: ModelService,
        function: Callable,
        size: Optional[int] = None,
        max_workers: int = 10,
        pass_k: Optional[int] = None,   # kept for API compatibility
    ) -> List[dict]:
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.generate_service = generate_service

        results = self.evaluate_all_problems(
            function=function,
            judge_service=judge_service,
            generate_service=generate_service,
            max_workers=max_workers,
            size=size,
        )

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\nResults saved → {self.output_path}")

        self.record_cost(generate_service, judge_service)

        # Clean up temp file
        temp_file = os.path.join(
            os.path.dirname(self.output_path),
            f"temp_{self.mode}_gaia_results.json",
        )
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass

        metrics = self.calculate_metrics(results)

        # Persist metrics
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved → {self.metrics_path}")

        return results

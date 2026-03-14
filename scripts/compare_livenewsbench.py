#!/usr/bin/env python3
"""
compare_livenewsbench.py — DyFlow vs DyFlow-T comparison on LiveNewsBench.

LiveNewsBench evaluates LLM web search capabilities using freshly curated
news questions that REQUIRE live retrieval to answer correctly. Every
question is about a specific post-training news event — the LLM cannot
know the answer from parametric memory.

Paper:  https://arxiv.org/abs/2602.13543
Data:   https://huggingface.co/datasets/YunfanZhang42/LiveNewsBench

Why this proves the web tool's value:
  DyFlow (no tool):  must recall from parametric memory — post-training
                     events are unknown, producing hallucinated answers
  DyFlow-T (web):    queries SerpAPI for the specific news event →
                     grounds the answer in real retrieved content

Example:
  Q: "In the 2025 Emmy Awards, how many Emmys did Adolescence win?"
  DyFlow:   "Adolescence won 3 Emmys" ← hallucinated; event post-training
  DyFlow-T: searches → "Adolescence won 6 Emmys" ← real retrieved answer

Usage
─────
    # Instant test with built-in 20-question sample
    python scripts/install_livenewsbench.py --sample-only
    python scripts/compare_livenewsbench.py --sample

    # Full validation set (download first)
    python scripts/install_livenewsbench.py --subset sep_2025 --split val
    python scripts/compare_livenewsbench.py --subset sep_2025 --split val

    # Specific options
    python scripts/compare_livenewsbench.py --sample --size 10 --workers 2
    python scripts/compare_livenewsbench.py --report-only
    python scripts/compare_livenewsbench.py --sample --force   # clear cache
"""

import sys
import os
import json
import re
import string
import datetime
import argparse
import time
import concurrent.futures

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

from dyflow.model_service import ModelService
from dyflow.core.workflow import WorkflowExecutor
from dyflow.core.tool_workflow import ToolAwareWorkflowExecutor
from dyflow.tools.registry import ToolRegistry
from dyflow.tools.web_search import WebSearchTool, MockWebSearchTool
from dyflow.tools.sql_query import MockSQLQueryTool

REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR    = os.path.join(REPO_ROOT, "benchmarks", "data", "LiveNewsBench")
REPORT_DIR  = os.path.join(REPO_ROOT, "benchmarks", "results", "LiveNewsBench", "comparison")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample", "livenewsbench_sample.json")


# ── Scoring ────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(pred: str, gold: str) -> bool:
    return _normalise(pred) == _normalise(gold)


def contains_match(pred: str, gold: str) -> bool:
    return _normalise(gold) in _normalise(pred)


def score_answer(prediction: str, ground_truth: str) -> tuple:
    """Returns (extracted, exact_match, contains_match)."""
    # Try to pull out Final Answer line first
    fa = re.search(r"Final Answer\s*:\s*(.+?)(?:\n|$)", prediction, re.IGNORECASE)
    clean = fa.group(1).strip() if fa else prediction.strip()
    em = exact_match(clean, ground_truth)
    cm = contains_match(clean, ground_truth)
    return clean, em, cm


# ── LLM judge for ambiguous cases ─────────────────────────────────────────────

JUDGE_PROMPT = """\
You are an expert evaluator for the LiveNewsBench benchmark.
Your task: extract the FINAL answer from the model output and decide if it matches the ground truth.

Question:
{question}

Model Output:
{output}

Ground Truth Answer:
{gold}

Instructions:
1. Extract the model's final answer (ignore reasoning steps).
2. Compare it to the ground truth using normalised exact match (case-insensitive, ignore punctuation).
3. Output the extracted answer on a line starting with 'Extracted: '
4. Then respond with [[True]] if correct or [[False]] if incorrect."""


def llm_judge(question, output, gold, judge_service):
    prompt = JUDGE_PROMPT.format(question=question, output=output, gold=gold)
    try:
        resp = judge_service.generate(prompt=prompt)
        text = resp.get("response", "") if isinstance(resp, dict) else str(resp)
        extracted = ""
        for line in text.splitlines():
            if line.strip().lower().startswith("extracted:"):
                extracted = line.split(":", 1)[1].strip()
                break
        is_correct = "[[true]]" in text.lower()
        return extracted, is_correct
    except Exception as e:
        print(f"  Judge error: {e}")
        return "", False


# ── Workflow factories ─────────────────────────────────────────────────────────

def make_dyflow_fn(designer: ModelService, executor: ModelService):
    """
    DyFlow baseline — no tools.
    For post-training news questions, the LLM has no parametric memory of the
    events and must either hallucinate an answer or refuse.
    """
    def run(question: str):
        wf = WorkflowExecutor(
            problem_description=question,
            designer_service=designer,
            executor_service=executor,
            save_design_history=True,
        )
        try:
            answer  = wf.execute()
            history = getattr(wf.state, "design_history", [])
            return str(answer) if answer else "", history
        except Exception as e:
            return f"ERROR: {e}", []
    return run


def make_dyflow_t_fn(designer: ModelService, executor: ModelService):
    """
    DyFlow-T — WebSearchTool via SerpAPI.
    Searches for the specific news event at inference time, then grounds
    the answer in real retrieved content.
    """
    serpapi_key = os.getenv("SERPAPI_API_KEY", "") or os.getenv("SERPER_API_KEY", "")

    def run(question: str):
        registry = ToolRegistry()
        if serpapi_key:
            registry.register("WEB_SEARCH", WebSearchTool(api_key=serpapi_key))
        else:
            registry.register("WEB_SEARCH", MockWebSearchTool())
        registry.register("SQL_QUERY", MockSQLQueryTool())

        wf = ToolAwareWorkflowExecutor(
            problem_description=question,
            designer_service=designer,
            executor_service=executor,
            tool_registry=registry,
            save_design_history=True,
            max_tool_retries=2,
        )
        try:
            answer, _ = wf.run(max_steps=8)
            history   = getattr(wf.state, "design_history", [])
            return answer or "", history
        except Exception as e:
            return f"ERROR: {e}", []
    return run


# ── Single problem evaluation ──────────────────────────────────────────────────

def evaluate_one(item, run_fn, judge_service=None):
    question     = item.get("question", "")
    ground_truth = str(item.get("answer", "")).strip()
    qid          = item.get("id", question[:60])
    category     = item.get("category", "unknown")
    date         = item.get("date", "")

    try:
        prediction, history = run_fn(question)
    except Exception as e:
        prediction, history = f"ERROR: {e}", []

    extracted, em, cm = score_answer(prediction, ground_truth)

    # LLM judge if not already correct
    if not em and judge_service:
        ext2, judge_ok = llm_judge(question, prediction, ground_truth, judge_service)
        if ext2:
            extracted = ext2
            em = em or exact_match(ext2, ground_truth) or judge_ok
            cm = cm or contains_match(ext2, ground_truth)

    return {
        "id":               qid,
        "question":         question,
        "ground_truth":     ground_truth,
        "date":             date,
        "category":         category,
        "prediction":       prediction,
        "extracted_answer": extracted,
        "exact_match":      em,
        "contains_match":   cm,
        "judge_result":     em,
        "design_history":   history,
    }


# ── Parallel evaluation ────────────────────────────────────────────────────────

def evaluate_all(items, run_fn, judge_service=None, max_workers=2,
                 size=None, temp_path=None, label=""):
    if size:
        items = items[:size]

    # Resume support
    done, done_ids = [], set()
    if temp_path and os.path.exists(temp_path):
        try:
            done = json.load(open(temp_path, encoding="utf-8"))
            done_ids = {r["id"] for r in done}
            print(f"  [{label}] Resumed: {len(done)} done, {len(items)-len(done_ids)} remaining")
            items = [it for it in items if it.get("id", it["question"][:60]) not in done_ids]
        except Exception:
            done, done_ids = [], set()

    def _save():
        if temp_path:
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(done, f, indent=2, ensure_ascii=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(evaluate_one, it, run_fn, judge_service): it for it in items}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            try:
                result = fut.result()
                done.append(result)
                status = "✓" if result["judge_result"] else "✗"
                print(f"  [{label}] {i+len(done_ids)}/{len(items)+len(done_ids)}  {status}  {result['id'][:50]}")
                if i % 5 == 0:
                    _save()
            except Exception as e:
                print(f"  [{label}] ERROR: {e}")
    _save()
    return done


# ── Summary stats ──────────────────────────────────────────────────────────────

def summarise(results: list) -> dict:
    total = correct = 0
    categories: dict = {}
    for r in results:
        if r.get("prediction", "").startswith("ERROR") and not r.get("extracted_answer"):
            continue
        total += 1
        cat = r.get("category", "unknown")
        categories.setdefault(cat, {"total": 0, "correct": 0})
        categories[cat]["total"] += 1
        if r.get("judge_result", False):
            correct += 1
            categories[cat]["correct"] += 1
    return {
        "total":   total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "category_accuracy": {
            cat: round(s["correct"] / s["total"], 4) if s["total"] else 0.0
            for cat, s in categories.items()
        },
        "category_counts": categories,
    }


# ── Report builder ─────────────────────────────────────────────────────────────

def build_report(df_res, dft_res, args, t_df, t_dft) -> dict:
    df_sum  = summarise(df_res)
    dft_sum = summarise(dft_res)

    qmap = {}
    for r in df_res:
        qmap[r["id"]] = {"dyflow": r}
    for r in dft_res:
        qmap.setdefault(r["id"], {})["dyflow_t"] = r

    both_ok = dft_only = df_only = both_wrong = 0
    per_q   = []
    for qid, pair in qmap.items():
        df  = pair.get("dyflow",   {})
        dft = pair.get("dyflow_t", {})
        df_ok  = df.get("judge_result",  False)
        dft_ok = dft.get("judge_result", False)

        if   df_ok and dft_ok:        both_ok   += 1
        elif dft_ok and not df_ok:    dft_only  += 1
        elif df_ok  and not dft_ok:   df_only   += 1
        else:                         both_wrong += 1

        per_q.append({
            "id":               qid,
            "question":         df.get("question",     dft.get("question", "")),
            "category":         df.get("category",     dft.get("category", "")),
            "date":             df.get("date",         dft.get("date", "")),
            "ground_truth":     df.get("ground_truth", dft.get("ground_truth", "")),
            "dyflow_correct":   df_ok,
            "dyflow_t_correct": dft_ok,
            "dyflow_answer":    df.get("extracted_answer",  ""),
            "dyflow_t_answer":  dft.get("extracted_answer", ""),
            "outcome": (
                "both_correct"  if df_ok  and dft_ok  else
                "dyflow_t_only" if dft_ok and not df_ok else
                "dyflow_only"   if df_ok  and not dft_ok else
                "both_wrong"
            ),
        })

    delta = dft_sum["accuracy"] - df_sum["accuracy"]
    serpapi_key = os.getenv("SERPAPI_API_KEY", "") or os.getenv("SERPER_API_KEY", "")

    return {
        "meta": {
            "timestamp":       str(datetime.datetime.now()),
            "dataset":         "LiveNewsBench",
            "subset":          getattr(args, "subset", "sample"),
            "split":           getattr(args, "split", "sample"),
            "size":            args.size or "all",
            "model":           "gemini-2.5-flash",
            "web_search_live": bool(serpapi_key),
        },
        "summary": {
            "dyflow": {
                **df_sum,
                "elapsed_sec": round(t_df, 1),
                "label": "DyFlow (no tools — parametric memory only)",
            },
            "dyflow_t": {
                **dft_sum,
                "elapsed_sec": round(t_dft, 1),
                "label": "DyFlow-T (WebSearchTool — live news retrieval)",
            },
            "delta_accuracy":  round(delta, 4),
            "improvement_pct": round(delta * 100, 2),
        },
        "outcome_breakdown": {
            "total_questions": max(len(df_res), len(dft_res)),
            "both_correct":    both_ok,
            "dyflow_t_only":   dft_only,
            "dyflow_only":     df_only,
            "both_wrong":      both_wrong,
        },
        "key_insight": (
            "LiveNewsBench questions are about post-training news events — "
            "DyFlow has no parametric knowledge of them and must hallucinate. "
            "DyFlow-T searches for the specific event at inference time and "
            "grounds the answer in real retrieved news content."
        ),
        "per_question": per_q,
    }


def print_report(report: dict) -> None:
    s      = report["summary"]
    df     = s["dyflow"]
    dft    = s["dyflow_t"]
    ob     = report["outcome_breakdown"]
    delta  = s["delta_accuracy"]
    sign   = "+" if delta >= 0 else ""
    live   = report["meta"].get("web_search_live", False)
    subset = report["meta"].get("subset", "—")
    split  = report["meta"].get("split", "—")

    print("\n" + "=" * 70)
    print("LiveNewsBench COMPARISON: DyFlow vs DyFlow-T")
    print("=" * 70)
    print(f"  Subset / Split : {subset} / {split}")
    print(f"  Web search     : {'LIVE (SerpAPI)' if live else '⚠  MOCK — set SERPAPI_API_KEY'}")
    print(f"  Why this bench : Every question requires post-training news retrieval")
    print()
    print(f"{'Metric':<34} {'DyFlow':>14} {'DyFlow-T':>14}")
    print("-" * 70)
    print(f"{'Accuracy':<34} {df['accuracy']:>13.2%} {dft['accuracy']:>13.2%}")
    print(f"{'Correct':<34} {df['correct']:>14} {dft['correct']:>14}")
    print(f"{'Total evaluated':<34} {df['total']:>14} {dft['total']:>14}")
    print(f"{'Elapsed (sec)':<34} {df['elapsed_sec']:>14} {dft['elapsed_sec']:>14}")
    print("-" * 70)

    # Per-category
    all_cats = sorted(set(
        list(df["category_accuracy"].keys()) +
        list(dft["category_accuracy"].keys())
    ))
    for cat in all_cats:
        da  = df["category_accuracy"].get(cat, 0.0)
        dta = dft["category_accuracy"].get(cat, 0.0)
        n   = df["category_counts"].get(cat, {}).get("total", 0)
        label = cat[:30]
        print(f"  {label:<32} {da:>13.2%} {dta:>13.2%}  (n={n})")

    print("-" * 70)
    print(f"  Tool improvement : {sign}{delta:.2%}  ({sign}{s['improvement_pct']}pp)")
    print("-" * 70)
    print(f"  Both correct     : {ob['both_correct']}")
    print(f"  DyFlow-T only    : {ob['dyflow_t_only']}  ← web search retrieved correct answer")
    print(f"  DyFlow only      : {ob['dyflow_only']}   ← lucky parametric guess")
    print(f"  Both wrong       : {ob['both_wrong']}")
    print("-" * 70)
    words, line = report["key_insight"].split(), ""
    for w in words:
        if len(line) + len(w) + 1 > 66:
            print(f"  {line}")
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        print(f"  {line}")
    print("=" * 70)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_comparison(args):
    # Determine dataset path
    if args.sample:
        dataset_path = SAMPLE_PATH
        if not os.path.exists(dataset_path):
            print("[LiveNewsBench] Sample not found — generating it now...")
            from scripts.install_livenewsbench import save_sample
            save_sample()
    else:
        dataset_path = os.path.join(
            DATA_DIR, args.subset,
            f"livenewsbench_{args.subset}_{args.split}.json"
        )
        if not os.path.exists(dataset_path):
            print(f"[ERROR] Dataset not found: {dataset_path}")
            print(f"  Run: python scripts/install_livenewsbench.py --subset {args.subset} --split {args.split}")
            sys.exit(1)

    os.makedirs(REPORT_DIR, exist_ok=True)
    tag = "sample" if args.sample else f"{args.subset}_{args.split}"

    if args.force:
        import glob
        for f in glob.glob(os.path.join(REPORT_DIR, f"temp_{tag}_*.json")):
            os.remove(f); print(f"[force] Removed cache: {f}")

    with open(dataset_path, encoding="utf-8") as f:
        items = json.load(f)

    serpapi_key = os.getenv("SERPAPI_API_KEY", "") or os.getenv("SERPER_API_KEY", "")

    print("=" * 70)
    print("LiveNewsBench: DyFlow vs DyFlow-T")
    print("=" * 70)
    print(f"  Dataset : {dataset_path}")
    print(f"  Items   : {len(items)}  |  Size: {args.size or 'all'}")
    print(f"  Workers : {args.workers}")
    print(f"  Search  : {'live (SerpAPI)' if serpapi_key else '⚠  mock — set SERPAPI_API_KEY'}")
    print("=" * 70)

    designer = ModelService(model="gemini-2.5-flash")
    executor = ModelService(model="gemini-2.5-flash")
    judge    = ModelService(model="gemini-2.5-flash")

    # ── Run DyFlow (no tool) ───────────────────────────────────────────────────
    print("\n[1/2] DyFlow (no tools — parametric memory)...")
    t0 = time.time()
    df_results = evaluate_all(
        items,
        make_dyflow_fn(designer, executor),
        judge_service=judge,
        max_workers=args.workers,
        size=args.size,
        temp_path=os.path.join(REPORT_DIR, f"temp_{tag}_dyflow.json"),
        label="DyFlow",
    )
    t_df = time.time() - t0

    # ── Run DyFlow-T (web search) ──────────────────────────────────────────────
    print("\n[2/2] DyFlow-T (WebSearchTool — live news retrieval)...")
    t0 = time.time()
    dft_results = evaluate_all(
        items,
        make_dyflow_t_fn(designer, executor),
        judge_service=judge,
        max_workers=args.workers,
        size=args.size,
        temp_path=os.path.join(REPORT_DIR, f"temp_{tag}_dyflow_t.json"),
        label="DyFlow-T",
    )
    t_dft = time.time() - t0

    # ── Report ─────────────────────────────────────────────────────────────────
    report = build_report(df_results, dft_results, args, t_df, t_dft)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"comparison_{tag}_{ts}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_report(report)
    print(f"\nReport → {report_path}")


def show_latest(args):
    if not os.path.isdir(REPORT_DIR):
        print("[ERROR] No reports found. Run the comparison first.")
        sys.exit(1)
    reports = sorted(
        [f for f in os.listdir(REPORT_DIR) if f.startswith("comparison_")],
        reverse=True,
    )
    if not reports:
        print(f"[ERROR] No comparison reports in {REPORT_DIR}")
        sys.exit(1)
    path = os.path.join(REPORT_DIR, reports[0])
    print(f"Loading: {path}")
    print_report(json.load(open(path)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compare DyFlow vs DyFlow-T on LiveNewsBench"
    )
    p.add_argument("--sample", action="store_true",
                   help="Use built-in 20-question sample (no download needed)")
    p.add_argument("--subset", default="sep_2025",
                   choices=["sep_2025", "jan_2026"],
                   help="Dataset subset (default: sep_2025)")
    p.add_argument("--split", default="val",
                   choices=["train", "val", "test", "human_verified_test"],
                   help="Dataset split (default: val)")
    p.add_argument("--size", type=int, default=None,
                   help="Limit number of questions (default: all)")
    p.add_argument("--workers", type=int, default=2,
                   help="Parallel workers per system (default: 2)")
    p.add_argument("--force", action="store_true",
                   help="Clear cached temp files and re-run from scratch")
    p.add_argument("--report-only", action="store_true",
                   help="Print metrics from last saved report")
    args = p.parse_args()

    if args.report_only:
        show_latest(args)
    else:
        run_comparison(args)

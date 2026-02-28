#!/usr/bin/env python3
"""
install_bird.py — Download BIRD (Big Bench for Large-scale Database Grounded
Text-to-SQL Evaluation) dataset and prepare SQLite databases for DyFlow-T.

Why BIRD over Spider for tool comparison?
──────────────────────────────────────────
- Real-world databases with up to 33,000 rows (vs Spider's ~5 rows avg)
- Dirty/noisy data — LLM cannot guess aggregates by pattern matching
- Domain-specific values (finance, sports, medical) unknown to parametric LLM
- Queries require actual execution to get correct numeric answers
- Official leaderboard metric: Execution Accuracy (EX) + Valid Efficiency Score (VES)

This makes BIRD ideal for demonstrating the tool's value:
the LLM genuinely cannot hallucinate correct COUNT/SUM/AVG answers
over thousands of realistic rows.

Dataset: https://bird-bench.github.io/
HuggingFace: https://huggingface.co/datasets/birdbench/bird

Usage
─────
    # Download via HuggingFace (recommended)
    python scripts/install_bird.py

    # Download via direct URL (alternative)
    python scripts/install_bird.py --direct

    # Force re-download
    python scripts/install_bird.py --force

Requirements
────────────
    pip install datasets huggingface_hub requests tqdm
    HF_TOKEN in .env (accept terms at https://huggingface.co/datasets/birdbench/bird)
"""

import sys
import os
import json
import sqlite3
import argparse
import zipfile
import shutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR  = os.path.join(REPO_ROOT, "benchmarks", "data", "BIRD")
DB_ROOT   = os.path.join(DATA_DIR, "databases")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
except ImportError:
    pass


# ── BIRD format normalisation ─────────────────────────────────────────────────

def normalise_bird_record(row: dict) -> dict:
    """
    Map BIRD JSON fields → DyFlow-T SpiderBenchmark format.

    BIRD dev fields:
        question, SQL, db_id, difficulty, evidence (extra domain hint)
    """
    return {
        "db_id":    row.get("db_id", ""),
        "question": row.get("question", ""),
        "query":    row.get("SQL", row.get("query", row.get("sql", ""))),
        "hardness": row.get("difficulty", row.get("hardness", "unknown")),
        "evidence": row.get("evidence", ""),   # domain hint unique to BIRD
    }


# ── Download via HuggingFace datasets ────────────────────────────────────────

def download_via_huggingface(force: bool = False) -> bool:
    """Download BIRD via HuggingFace datasets library."""
    try:
        from datasets import load_dataset
        from huggingface_hub import login
    except ImportError:
        print("[ERROR] Missing packages: pip install datasets huggingface_hub")
        return False

    hf_token = os.getenv("HF_TOKEN", "")
    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=False)
            print("[HF] Logged in with HF_TOKEN")
        except Exception as e:
            print(f"[HF] Login warning: {e}")

    dev_path   = os.path.join(DATA_DIR, "bird_dev.json")
    train_path = os.path.join(DATA_DIR, "bird_train.json")

    if os.path.exists(dev_path) and not force:
        print(f"[BIRD] Already downloaded: {dev_path}")
        return True

    print("[BIRD] Downloading via HuggingFace...")
    try:
        dataset = load_dataset("birdbench/bird", trust_remote_code=True)
    except Exception as e:
        print(f"[ERROR] HuggingFace download failed: {e}")
        print("  → Try: python scripts/install_bird.py --direct")
        return False

    os.makedirs(DATA_DIR, exist_ok=True)

    for split_name, out_path in [("validation", dev_path), ("train", train_path)]:
        if split_name not in dataset:
            print(f"  [skip] Split '{split_name}' not found")
            continue
        records = [normalise_bird_record(dict(r)) for r in dataset[split_name]]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        _print_split_stats(out_path, records)

    return True


def _print_split_stats(path: str, records: list):
    counts = {}
    for r in records:
        h = r.get("hardness", "unknown")
        counts[h] = counts.get(h, 0) + 1
    print(f"  ✅  {os.path.basename(path)}: {len(records)} records")
    for h in ["simple", "moderate", "challenging", "unknown"]:
        if h in counts:
            print(f"       {h}: {counts[h]}")


# ── Download via direct URL ───────────────────────────────────────────────────

BIRD_DIRECT_URL = (
    "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
)

def download_via_direct(force: bool = False) -> bool:
    """Download BIRD dev set via direct URL (no HF token needed)."""
    try:
        import requests
        from tqdm import tqdm as _tqdm
    except ImportError:
        print("[ERROR] Missing: pip install requests tqdm")
        return False

    extract_dir = os.path.join(DATA_DIR, "_bird_extracted")
    zip_path    = os.path.join(DATA_DIR, "bird_dev.zip")

    if os.path.exists(extract_dir) and not force:
        print(f"[BIRD] Already extracted: {extract_dir}")
        return _process_direct_download(extract_dir)

    print(f"[BIRD] Downloading dev set from {BIRD_DIRECT_URL}")
    print("       (~1.5 GB — includes databases with real data rows)\n")

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        response = requests.get(BIRD_DIRECT_URL, stream=True, timeout=300)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(zip_path, "wb") as f, _tqdm(
            total=total, unit="B", unit_scale=True, desc="Downloading"
        ) as bar:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))

        print("\n[BIRD] Extracting...")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        os.remove(zip_path)
        print(f"[BIRD] Extracted to {extract_dir}")

    except Exception as e:
        print(f"[ERROR] Direct download failed: {e}")
        print("\nManual download instructions:")
        print("  1. Visit https://bird-bench.github.io/")
        print("  2. Download dev.zip")
        print(f"  3. Extract to: {extract_dir}")
        return False

    return _process_direct_download(extract_dir)


def _process_direct_download(extract_dir: str) -> bool:
    """Process extracted BIRD zip — copy DBs and normalise JSON."""
    # Find the JSON file
    dev_json = None
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f in ("dev.json", "dev_20240627.json"):
                dev_json = os.path.join(root, f)
                break
        if dev_json:
            break

    if not dev_json:
        print("[ERROR] Could not find dev.json in extracted files")
        print(f"  Contents: {os.listdir(extract_dir)}")
        return False

    with open(dev_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = [normalise_bird_record(r) for r in raw]
    out_path = os.path.join(DATA_DIR, "bird_dev.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    _print_split_stats(out_path, records)

    # Copy databases
    _copy_bird_databases(extract_dir)
    return True


# ── Copy/link BIRD databases ──────────────────────────────────────────────────

def _copy_bird_databases(extract_dir: str):
    """
    BIRD ships pre-built SQLite .sqlite files — just copy them into
    our databases/<db_id>/<db_id>.db structure.
    """
    os.makedirs(DB_ROOT, exist_ok=True)

    # Find all .sqlite / .db files in the extracted dir
    db_files = []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.endswith((".sqlite", ".db", ".sqlite3")):
                db_files.append(os.path.join(root, f))

    if not db_files:
        print("[BIRD] ⚠️  No SQLite database files found in extracted archive.")
        print("        The databases may be in a separate download.")
        print("        See https://bird-bench.github.io/ for the database pack.")
        return

    print(f"\n[BIRD] Copying {len(db_files)} databases...")
    copied = 0
    for src in db_files:
        db_id = os.path.splitext(os.path.basename(src))[0]
        out_dir = os.path.join(DB_ROOT, db_id)
        out_path = os.path.join(out_dir, f"{db_id}.db")
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.exists(out_path):
            shutil.copy2(src, out_path)
            copied += 1

    print(f"  ✅  Copied {copied} databases to {DB_ROOT}")


# ── Verify ────────────────────────────────────────────────────────────────────

def verify_bird():
    print("\n[Verify] BIRD setup...")
    dev_path = os.path.join(DATA_DIR, "bird_dev.json")
    all_ok = True

    if not os.path.exists(dev_path):
        print(f"  [FAIL] dev JSON not found: {dev_path}")
        all_ok = False
    else:
        with open(dev_path) as f:
            records = json.load(f)
        print(f"  [OK]   bird_dev.json: {len(records)} records")

    db_count = 0
    if os.path.isdir(DB_ROOT):
        for db_id in os.listdir(DB_ROOT):
            db_path = os.path.join(DB_ROOT, db_id, f"{db_id}.db")
            if os.path.exists(db_path):
                db_count += 1
    print(f"  [{'OK' if db_count > 0 else 'WARN'}]   Databases found: {db_count}")

    if db_count == 0:
        print("  ⚠️   No databases found. BIRD databases may need separate download.")
        print("       See: https://bird-bench.github.io/")

    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print("BIRD Dataset Installer for DyFlow-T")
    print("=" * 60)
    print("\nWhy BIRD for tool comparison:")
    print("  • Real databases with up to 33,000 rows")
    print("  • LLM cannot guess COUNT/SUM/AVG over real noisy data")
    print("  • Proves tool execution value over LLM hallucination")
    print()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DB_ROOT,  exist_ok=True)

    if args.direct:
        success = download_via_direct(force=args.force)
    else:
        success = download_via_huggingface(force=args.force)
        if not success:
            print("\n[Fallback] Trying direct URL download...")
            success = download_via_direct(force=args.force)

    verify_bird()

    print("\n" + "=" * 60)
    if success:
        print("✅  BIRD dataset ready!")
        print(f"    Dev JSON:  {os.path.join(DATA_DIR, 'bird_dev.json')}")
        print(f"    Databases: {DB_ROOT}")
        print("\nRun comparison:")
        print("    python scripts/compare_spider.py --dataset bird --mode dev --size 50 --workers 3")
        print("\nRun evaluation only:")
        print("    python scripts/run_spider.py --dataset bird --mode dev --size 50")
    else:
        print("⚠️   Download incomplete — check errors above.")
        print("     Manual download: https://bird-bench.github.io/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download BIRD dataset for DyFlow-T SQL tool comparison"
    )
    parser.add_argument(
        "--direct", action="store_true",
        help="Download via direct URL instead of HuggingFace (no token needed)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if files already exist",
    )
    args = parser.parse_args()
    main(args)

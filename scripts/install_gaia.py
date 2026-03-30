#!/usr/bin/env python3
"""
install_gaia.py — Download and prepare the GAIA benchmark dataset.

Usage
─────
    python scripts/install_gaia.py

    # Skip HuggingFace login prompt (if you already have access)
    python scripts/install_gaia.py --no-login

    # Only download validation split
    python scripts/install_gaia.py --splits validation

Requirements
────────────
    pip install datasets huggingface_hub

Access
──────
The GAIA dataset requires accepting the terms on HuggingFace:
    https://huggingface.co/datasets/gaia-benchmark/GAIA
"""

import sys
import os
import json
import argparse

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR  = os.path.join(REPO_ROOT, "benchmarks", "data", "GAIA")
DATASET_ID  = "gaia-benchmark/GAIA"
DATASET_CFG = "2023_all"

SPLIT_MAP = {
    "validation": "GAIA_validation.json",
    "test":       "GAIA_test.json",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def check_deps():
    missing = []
    try:
        import datasets       # noqa: F401
    except ImportError:
        missing.append("datasets")
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        missing.append("huggingface_hub")

    if missing:
        print(f"[ERROR] Missing dependencies: {', '.join(missing)}")
        print(f"        Run:  pip install {' '.join(missing)}")
        sys.exit(1)


def hf_login():
    from huggingface_hub import login, whoami
    try:
        info = whoami()
        print(f"[Auth]  Already logged in as: {info['name']}")
        return
    except Exception:
        pass

    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        print("[Auth]  Using HF_TOKEN from environment.")
        login(token=token)
    else:
        print("[Auth]  No HF_TOKEN found. Starting interactive login...")
        print("        (You can also set HF_TOKEN=<your_token> in .env to skip this)\n")
        login()


def normalise_record(row: dict) -> dict:
    """Map HuggingFace GAIA columns → DyFlow-T format."""
    return {
        "task_id":      row.get("task_id", ""),
        "question":     row.get("Question", row.get("question", "")),
        "final_answer": row.get("Final answer", row.get("final_answer", "")),
        "level":        int(row.get("Level", row.get("level", 1))),
        "file_name":    row.get("file_name", ""),
        "annotator_metadata": row.get("Annotator Metadata", {}),
    }


def download_split(ds, split: str, output_path: str):
    print(f"\n[Download] Split: {split}")

    if split not in ds:
        print(f"  [SKIP] Split '{split}' not found in dataset. Available: {list(ds.keys())}")
        return 0

    records = [normalise_record(dict(row)) for row in ds[split]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Level breakdown
    levels = {}
    for r in records:
        lv = r["level"]
        levels[lv] = levels.get(lv, 0) + 1

    print(f"  Saved {len(records)} records → {output_path}")
    for lv in sorted(levels):
        print(f"    Level {lv}: {levels[lv]} questions")

    return len(records)


def verify(splits: list):
    print("\n[Verify] Checking saved files...")
    all_ok = True
    for split in splits:
        path = os.path.join(OUTPUT_DIR, SPLIT_MAP[split])
        if not os.path.exists(path):
            print(f"  [FAIL] {path} — not found")
            all_ok = False
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        required = {"task_id", "question", "final_answer", "level"}
        sample = data[0] if data else {}
        missing_fields = required - set(sample.keys())
        if missing_fields:
            print(f"  [FAIL] {path} — missing fields: {missing_fields}")
            all_ok = False
        else:
            print(f"  [OK]   {path} — {len(data)} records, fields OK")

    return all_ok


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print("GAIA Dataset Installer for DyFlow-T")
    print("=" * 60)

    # 1. Check dependencies
    check_deps()
    from datasets import load_dataset

    # 2. Authenticate
    if not args.no_login:
        hf_login()

    # 3. Load dataset
    print(f"\n[Load]  Downloading {DATASET_ID} ({DATASET_CFG})...")
    print("        This may take a moment on first run.\n")
    try:
        ds = load_dataset(DATASET_ID, DATASET_CFG, trust_remote_code=True)
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        print(
            "\nMake sure you have accepted the terms at:\n"
            "  https://huggingface.co/datasets/gaia-benchmark/GAIA\n"
            "and that you are logged in with an account that has access."
        )
        sys.exit(1)

    print(f"[Load]  Available splits: {list(ds.keys())}")

    # 4. Download each split
    total = 0
    for split in args.splits:
        output_path = os.path.join(OUTPUT_DIR, SPLIT_MAP[split])
        if os.path.exists(output_path) and not args.force:
            print(f"\n[Skip]  {output_path} already exists. Use --force to re-download.")
            continue
        total += download_split(ds, split, output_path)

    # 5. Verify
    ok = verify(args.splits)

    # 6. Summary
    print("\n" + "=" * 60)
    if ok:
        print("✅  GAIA dataset ready!")
        print(f"    Location: {OUTPUT_DIR}")
        print("\nRun evaluation:")
        print("    python scripts/run_gaia.py --mode validation --size 20 --workers 5")
    else:
        print("❌  Some files failed verification. Check errors above.")
    print("=" * 60)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare the GAIA benchmark dataset for DyFlow-T"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "test"],
        choices=["validation", "test"],
        help="Which splits to download (default: validation test)",
    )
    parser.add_argument(
        "--no-login",
        action="store_true",
        help="Skip HuggingFace login (use if already authenticated)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    args = parser.parse_args()
    main(args)

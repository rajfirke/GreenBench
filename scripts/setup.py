#!/usr/bin/env python3
"""Setup: download datasets, verify Ollama, and check model availability."""

import sys
import json
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from greenbench.config import (
    MODELS, DATA_DIR, RESULTS_DIR,
    MMLU_SUBJECTS, MMLU_SAMPLE_SIZE, SUMMARIZATION_SAMPLE_SIZE,
)
from greenbench.client import OllamaClient


def download_humaneval(force=False):
    dest = DATA_DIR / "humaneval" / "humaneval.json"
    if dest.exists() and not force:
        print(f"  [OK] HumanEval already exists")
        return True
    print("  Downloading HumanEval...")
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
        records = [dict(row) for row in ds]
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  [OK] Saved {len(records)} problems")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def download_mmlu_subset(force=False):
    dest = DATA_DIR / "mmlu_subset" / "mmlu_subset.json"
    if dest.exists() and not force:
        print(f"  [OK] MMLU subset already exists")
        return True
    print(f"  Downloading MMLU ({MMLU_SAMPLE_SIZE} questions, {len(MMLU_SUBJECTS)} subjects)...")
    try:
        from datasets import load_dataset
        random.seed(42)
        ds = load_dataset("cais/mmlu", "all", split="test")
        filtered = [row for row in ds if row["subject"] in MMLU_SUBJECTS]
        per_subject = max(1, MMLU_SAMPLE_SIZE // len(MMLU_SUBJECTS))
        by_subject = {}
        for row in filtered:
            by_subject.setdefault(row["subject"], []).append(row)
        sampled = []
        for subj in sorted(MMLU_SUBJECTS):
            pool = by_subject.get(subj, [])
            sampled.extend(random.sample(pool, min(per_subject, len(pool))))
        sampled = sampled[:MMLU_SAMPLE_SIZE]
        records = [dict(row) for row in sampled]
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  [OK] Saved {len(records)} questions")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def download_summarization(force=False):
    dest = DATA_DIR / "summarization" / "cnn_dailymail_sample.json"
    if dest.exists() and not force:
        print(f"  [OK] Summarization already exists")
        return True
    print(f"  Downloading CNN/DailyMail ({SUMMARIZATION_SAMPLE_SIZE} articles)...")
    try:
        from datasets import load_dataset
        random.seed(42)
        ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
        indices = random.sample(range(len(ds)), SUMMARIZATION_SAMPLE_SIZE)
        sampled = ds.select(indices)
        records = [dict(row) for row in sampled]
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  [OK] Saved {len(records)} articles")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def check_ollama():
    client = OllamaClient()
    alive = client.is_alive()
    print(f"\n  Ollama server: {'[OK]' if alive else '[NOT RUNNING]'}")
    if not alive:
        print("  Start with: ollama serve")
        return False

    installed = set(client.list_models())
    required = {m["name"] for m in MODELS}
    missing = required - installed

    print(f"  Models required: {len(required)}")
    for m in sorted(required):
        status = "[OK]" if m in installed else "[MISSING]"
        print(f"    {status} {m}")

    if missing:
        print(f"\n  Pull missing models:")
        for m in sorted(missing):
            print(f"    ollama pull {m}")
    return len(missing) == 0


def main():
    parser = argparse.ArgumentParser(description="GreenBench setup")
    parser.add_argument("--force", action="store_true", help="Re-download datasets")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("GreenBench Setup")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "energy").mkdir(parents=True, exist_ok=True)

    print("\n[1/2] Checking Ollama...")
    ollama_ok = check_ollama()

    if not args.skip_download:
        print("\n[2/2] Downloading datasets...")
        d1 = download_humaneval(args.force)
        d2 = download_mmlu_subset(args.force)
        d3 = download_summarization(args.force)
        datasets_ok = all([d1, d2, d3])
    else:
        print("\n[2/2] Skipping downloads (--skip-download)")
        datasets_ok = True

    print("\n" + "=" * 60)
    if ollama_ok and datasets_ok:
        print("READY. Run the benchmark with:")
        print("  python scripts/run.py")
    else:
        print("ISSUES FOUND. Resolve them above, then re-run setup.")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the GreenBench benchmark suite.

Usage:
    python scripts/run.py                    # Full benchmark (all models, all tasks)
    python scripts/run.py --models llama3.2:3b qwen2.5:7b   # Subset of models
    python scripts/run.py --dry-run          # Print plan without running
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from greenbench.config import MODELS
from greenbench.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run GreenBench")
    parser.add_argument("--models", nargs="+", help="Specific model tags to run")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = parser.parse_args()

    if args.models:
        selected = [m for m in MODELS if m["name"] in args.models]
        if not selected:
            print(f"ERROR: No matching models found. Available: {[m['name'] for m in MODELS]}")
            sys.exit(1)
    else:
        selected = MODELS

    print(f"GreenBench: {len(selected)} model(s) × 3 tasks")
    for m in selected:
        print(f"  - {m['name']} ({m['params']})")

    if args.dry_run:
        print("\n[DRY RUN] Exiting.")
        return

    run_benchmark(models=selected)


if __name__ == "__main__":
    main()

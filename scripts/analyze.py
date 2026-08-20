#!/usr/bin/env python3
"""Analyze benchmark results and produce summary tables.

Reads JSONL output files from results/energy/ and computes:
- Per-model throughput (tok/s)
- Per-token energy at package and system level
- CO2 estimates per 1,000 tokens
- MMLU accuracy
"""

import sys
import json
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from greenbench.config import RESULTS_DIR, PACKAGE_POWER_W, SYSTEM_POWER_W, GRID_CARBON_INTENSITY


def load_results():
    energy_dir = RESULTS_DIR / "energy"
    if not energy_dir.exists():
        print("No results found. Run the benchmark first: python scripts/run.py")
        sys.exit(1)

    records = []
    for f in sorted(energy_dir.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return pd.DataFrame(records)


def main():
    df = load_results()
    if df.empty:
        print("No results found.")
        return

    print("=" * 70)
    print("GreenBench Results Summary")
    print("=" * 70)

    # Throughput by model and task
    print("\n[Throughput (tokens/second)]")
    throughput = df.groupby(["model", "task"])["tokens_per_sec"].mean().unstack(fill_value=0)
    print(throughput.round(1).to_string())

    # Average throughput across tasks
    print("\n\n[Overall Metrics by Model]")
    summary = df[df["tokens_per_sec"] > 0].groupby("model").agg(
        avg_tok_s=("tokens_per_sec", "mean"),
        avg_ms_per_tok=("time_per_token_ms", "mean"),
    ).round(2)

    summary["mJ_per_tok_pkg"] = (PACKAGE_POWER_W * summary["avg_ms_per_tok"]).round(2)
    summary["J_per_tok_sys"] = (SYSTEM_POWER_W * summary["avg_ms_per_tok"] / 1000).round(4)
    summary["co2_mg_per_1k"] = (
        1000 * SYSTEM_POWER_W * summary["avg_ms_per_tok"] / 1000 * GRID_CARBON_INTENSITY / 3.6e6
    ).round(3)

    print(summary.to_string())

    # MMLU accuracy
    mmlu_df = df[df["task"] == "mmlu"]
    if "is_correct" in mmlu_df.columns:
        print("\n\n[MMLU Accuracy]")
        accuracy = mmlu_df.groupby("model")["is_correct"].mean() * 100
        print(accuracy.round(1).to_string())

    # Save to CSV
    out_path = RESULTS_DIR / "summary.csv"
    summary.to_csv(out_path)
    print(f"\n\nSummary saved to: {out_path}")


if __name__ == "__main__":
    main()

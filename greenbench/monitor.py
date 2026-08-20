"""Energy monitor: records per-inference timing and computes energy metrics.

On Apple Silicon, CPU+GPU package power is roughly constant (~0.47W measured)
during memory-bandwidth-bound LLM decoding. Per-token energy is therefore
proportional to per-token latency:

    E_token_pkg = P_package * (time_per_token)
    E_token_sys = P_system  * (time_per_token)

For absolute power readings, run powermetrics separately:
    sudo powermetrics --samplers cpu_power,gpu_power -i 2000
"""

import json
import time
from pathlib import Path

from greenbench.config import RESULTS_DIR, PACKAGE_POWER_W, SYSTEM_POWER_W, GRID_CARBON_INTENSITY


class EnergyMonitor:
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir or RESULTS_DIR / "energy")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records = []

    def record(self, model, task, question_id, ollama_result, extra=None):
        time_per_token_s = ollama_result.get("time_per_token_ms", 0) / 1000.0

        rec = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": model,
            "task": task,
            "question_id": question_id,
            "eval_count": ollama_result.get("eval_count", 0),
            "eval_duration_ns": ollama_result.get("eval_duration_ns", 0),
            "tokens_per_sec": ollama_result.get("tokens_per_sec", 0),
            "time_per_token_ms": ollama_result.get("time_per_token_ms", 0),
            "wall_time_s": ollama_result.get("wall_time_s", 0),
            "energy_per_token_mj_pkg": PACKAGE_POWER_W * time_per_token_s * 1000,
            "energy_per_token_j_sys": SYSTEM_POWER_W * time_per_token_s,
            "co2_mg_per_1k_tokens": (
                1000 * SYSTEM_POWER_W * time_per_token_s * GRID_CARBON_INTENSITY / 3.6e6
            ),
        }
        if extra:
            rec.update(extra)

        self.records.append(rec)

        log_file = self.output_dir / f"{model.replace(':', '_')}_{task}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def get_summary(self, model=None, task=None):
        filtered = self.records
        if model:
            filtered = [r for r in filtered if r["model"] == model]
        if task:
            filtered = [r for r in filtered if r["task"] == task]

        if not filtered:
            return {}

        eval_counts = [r["eval_count"] for r in filtered if r["eval_count"] > 0]
        tps = [r["tokens_per_sec"] for r in filtered if r["tokens_per_sec"] > 0]
        energy_pkg = [r["energy_per_token_mj_pkg"] for r in filtered if r["energy_per_token_mj_pkg"] > 0]
        energy_sys = [r["energy_per_token_j_sys"] for r in filtered if r["energy_per_token_j_sys"] > 0]

        return {
            "model": model,
            "task": task,
            "n_calls": len(filtered),
            "total_tokens": sum(eval_counts),
            "avg_tokens_per_sec": sum(tps) / len(tps) if tps else 0,
            "avg_energy_per_token_mj_pkg": sum(energy_pkg) / len(energy_pkg) if energy_pkg else 0,
            "avg_energy_per_token_j_sys": sum(energy_sys) / len(energy_sys) if energy_sys else 0,
        }

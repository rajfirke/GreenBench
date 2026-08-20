"""Core benchmark runner: executes inference across models and tasks."""

import json
import time
from pathlib import Path

from greenbench.config import (
    MODELS, DATA_DIR, RESULTS_DIR,
    TEMPERATURE, MAX_TOKENS_MCQ, MAX_TOKENS_SUMMARY, MAX_TOKENS_CODE,
)
from greenbench.client import OllamaClient
from greenbench.monitor import EnergyMonitor


def load_mmlu():
    with open(DATA_DIR / "mmlu_subset" / "mmlu_subset.json") as f:
        return json.load(f)


def load_summarization():
    with open(DATA_DIR / "summarization" / "cnn_dailymail_sample.json") as f:
        return json.load(f)


def load_humaneval():
    with open(DATA_DIR / "humaneval" / "humaneval.json") as f:
        return json.load(f)


def format_mmlu_prompt(item):
    choices = item["choices"]
    labels = ["A", "B", "C", "D"]
    options = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    return (
        f"Answer the following multiple choice question. "
        f"Reply with only the letter (A, B, C, or D).\n\n"
        f"Question: {item['question']}\n{options}\n\nAnswer:"
    )


def format_summary_prompt(item):
    article = item["article"][:2000]
    return (
        f"Summarize the following article in 2-3 sentences.\n\n"
        f"Article: {article}\n\nSummary:"
    )


def format_code_prompt(item):
    return item["prompt"]


def parse_mmlu_answer(response, correct_idx):
    labels = ["A", "B", "C", "D"]
    text = response.strip().upper()
    for label in labels:
        if label in text:
            return label == labels[correct_idx]
    return False


def get_done_ids(output_file):
    done = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec.get("question_id", ""))
                except json.JSONDecodeError:
                    pass
    return done


def run_task(client, monitor, model_name, task_name, items, format_fn,
             max_tokens, output_file):
    """Run a single task for a single model, with resume support."""
    done_ids = get_done_ids(output_file)
    remaining = [(i, item) for i, item in enumerate(items)
                 if f"{task_name}_{i}" not in done_ids]

    if not remaining:
        print(f"    {task_name}: all {len(items)} done, skipping")
        return

    print(f"    {task_name}: {len(remaining)}/{len(items)} remaining")

    for idx, item in remaining:
        qid = f"{task_name}_{idx}"
        prompt = format_fn(item)

        result = client.generate(
            model=model_name,
            prompt=prompt,
            temperature=TEMPERATURE,
            num_predict=max_tokens,
        )

        extra = {"question_id": qid}
        if task_name == "mmlu":
            extra["is_correct"] = parse_mmlu_answer(
                result["response"], item["answer"]
            )

        monitor.record(model_name, task_name, qid, result, extra)

        record = {
            "question_id": qid,
            "model": model_name,
            "task": task_name,
            "response": result["response"][:500],
            "eval_count": result["eval_count"],
            "tokens_per_sec": result["tokens_per_sec"],
            "time_per_token_ms": result["time_per_token_ms"],
            "wall_time_s": result["wall_time_s"],
            **extra,
        }
        with open(output_file, "a") as f:
            f.write(json.dumps(record) + "\n")


def run_benchmark(models=None):
    """Run the full GreenBench suite across all models and tasks."""
    models = models or MODELS
    client = OllamaClient()
    monitor = EnergyMonitor()

    if not client.is_alive():
        print("ERROR: Ollama is not running. Start it with: ollama serve")
        return

    print("Loading datasets...")
    mmlu = load_mmlu()
    summaries = load_summarization()
    humaneval = load_humaneval()
    print(f"  MMLU: {len(mmlu)} questions")
    print(f"  Summarization: {len(summaries)} articles")
    print(f"  HumanEval: {len(humaneval)} problems")

    total_start = time.time()
    for model_info in models:
        model_name = model_info["name"]
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_info['params']})")
        print(f"{'='*60}")

        print("  Warming up...")
        client.warmup(model_name)

        out_dir = RESULTS_DIR / "energy"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = model_name.replace(":", "_").replace("/", "_")

        run_task(client, monitor, model_name, "mmlu", mmlu,
                 format_mmlu_prompt, MAX_TOKENS_MCQ,
                 out_dir / f"{safe_name}_mmlu.jsonl")

        run_task(client, monitor, model_name, "summarization", summaries,
                 format_summary_prompt, MAX_TOKENS_SUMMARY,
                 out_dir / f"{safe_name}_summarization.jsonl")

        run_task(client, monitor, model_name, "humaneval", humaneval,
                 format_code_prompt, MAX_TOKENS_CODE,
                 out_dir / f"{safe_name}_humaneval.jsonl")

        summary = monitor.get_summary(model=model_name)
        print(f"  Total tokens: {summary.get('total_tokens', 0)}")
        print(f"  Avg tok/s: {summary.get('avg_tokens_per_sec', 0):.1f}")
        print(f"  Avg mJ/tok (pkg): {summary.get('avg_energy_per_token_mj_pkg', 0):.2f}")
        print(f"  Avg J/tok (sys): {summary.get('avg_energy_per_token_j_sys', 0):.4f}")

    elapsed = time.time() - total_start
    print(f"\nBenchmark complete in {elapsed/3600:.1f}h")

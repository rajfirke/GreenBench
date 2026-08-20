# GreenBench

Benchmarking energy efficiency and carbon footprint of open-source LLM inference on Apple Silicon.

GreenBench measures **throughput**, **per-token energy**, and **CO₂ emissions** for quantized open-source LLMs running locally via [Ollama](https://ollama.com) on Apple Silicon hardware.

---

## Key Metrics

| Metric | What it measures |
|--------|-----------------|
| Throughput (tok/s) | Tokens generated per second during decoding |
| Energy/token (mJ, package) | CPU+GPU package energy per output token |
| Energy/token (J, system) | Full-system energy per output token |
| CO₂/1k tokens (mg) | Carbon footprint per 1,000 tokens (grid-specific) |

---

## Hardware Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4 series) with ≥16 GB unified memory
- macOS 13+ (Ventura or later)
- [Ollama](https://ollama.com) installed and running

> The benchmark is designed for Apple Silicon's unified memory architecture. It can run on other platforms with Ollama support, but power measurements assume Apple Silicon characteristics.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/rajfirke/GreenBench.git
cd GreenBench

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure (optional — defaults work for M4 Pro)
cp .env.example .env
# Edit .env if your hardware differs

# 5. Pull models and download datasets
ollama pull llama3.2:3b
ollama pull phi3:3.8b
ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama pull gemma2:9b

python scripts/setup.py

# 6. Run benchmark
python scripts/run.py

# 7. Analyze results
python scripts/analyze.py
```

---

## Project Structure

```
GreenBench/
├── greenbench/             # Core library
│   ├── __init__.py
│   ├── config.py           # Models, parameters, energy constants
│   ├── client.py           # Ollama API client with timing extraction
│   ├── monitor.py          # Energy computation from timing data
│   └── benchmark.py        # Benchmark runner (tasks × models)
├── scripts/
│   ├── setup.py            # Download datasets + verify Ollama
│   ├── run.py              # Execute the benchmark
│   ├── analyze.py          # Produce summary tables from results
│   └── measure_power.sh    # Direct power measurement via powermetrics
├── data/                   # Datasets (downloaded by setup, gitignored)
├── results/                # Benchmark outputs (gitignored)
├── .env.example            # Environment variable template
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Models

Five open-source model families spanning 3–9B parameters, all at **Q4_K_M** quantization (Ollama's default):

| Model | Developer | Parameters | Disk Size | Architecture |
|-------|-----------|-----------|-----------|--------------|
| Llama 3.2 | Meta | 3B | 2.0 GB | Grouped Query Attention |
| Phi-3 | Microsoft | 3.8B | 2.2 GB | Dense Transformer |
| Qwen 2.5 | Alibaba | 7B | 4.7 GB | GQA + SwiGLU |
| Mistral | Mistral AI | 7B | 4.4 GB | Sliding Window Attention |
| Gemma 2 | Google | 9B | 5.4 GB | Multi-Query Attention |

---

## Benchmark Tasks

| Task | Dataset | Samples | Output Length | Purpose |
|------|---------|---------|-------------|---------|
| MMLU | Hendrycks et al. (100 questions, 10 subjects) | 100 | ~15 tokens | Short-answer QA |
| Summarization | CNN/DailyMail | 50 | ~100–200 tokens | Medium-length generation |
| Code Generation | HumanEval | 164 | ~150–300 tokens | Long structured output |

---

## Energy Measurement

GreenBench uses a **dual-source measurement pipeline**:

1. **`powermetrics`** (macOS) — Direct hardware power readings for CPU+GPU package power
2. **Ollama API** — Nanosecond-precision per-request timing (`eval_duration`, `eval_count`)

These are combined to compute per-token energy:

```
E_token_pkg = P_package × (eval_duration / eval_count)
E_token_sys = P_system  × (eval_duration / eval_count)
CO₂         = (N × E_token_sys × I_grid) / 3.6×10⁶
```

### Measuring Package Power

To capture direct power readings (requires `sudo`):

```bash
sudo bash scripts/measure_power.sh qwen2.5:7b
```

This reports idle baseline, inference load, and cooldown power for CPU, GPU, and ANE.

---

## Configuration

All configuration lives in `greenbench/config.py` or can be overridden via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server endpoint |
| `PACKAGE_POWER_W` | `0.47` | Measured CPU+GPU package power (W) |
| `SYSTEM_POWER_W` | `10` | Estimated total system power (W) |
| `GRID_CARBON_INTENSITY` | `708` | Grid carbon intensity (g CO₂/kWh) |

**Grid carbon intensity reference values:**
- India: 708 g/kWh (coal-dominated)
- United States: 390 g/kWh (mixed)
- EU average: 295 g/kWh
- France: 55 g/kWh (nuclear-heavy)

---

## Running a Subset

```bash
# Run only specific models
python scripts/run.py --models llama3.2:3b qwen2.5:7b

# Dry run (print plan without executing)
python scripts/run.py --dry-run
```

---

## Output Format

Results are stored as JSONL files in `results/energy/`, one file per model-task combination:

```json
{
  "question_id": "mmlu_42",
  "model": "qwen2.5:7b",
  "task": "mmlu",
  "eval_count": 3,
  "tokens_per_sec": 96.9,
  "time_per_token_ms": 10.32,
  "energy_per_token_mj_pkg": 4.85,
  "energy_per_token_j_sys": 0.103,
  "co2_mg_per_1k_tokens": 0.203
}
```

---

## Reproducibility Notes

- Temperature is set to **0** for all experiments (deterministic outputs)
- Only one model is kept in memory at a time (previous model unloaded before next)
- Each model begins with a **warm-up inference** to ensure weights are resident
- The benchmark supports **resume** — re-running skips completed items
- Random seeds are fixed (`seed=42`) for dataset sampling

---

## Extending

**Add a new model:**

Edit `greenbench/config.py` and append to the `MODELS` list:

```python
{"name": "your-model:tag", "family": "family", "params": "XB", "size_gb": Y.Z},
```

Then pull it: `ollama pull your-model:tag`

**Adjust for different hardware:**

Update `.env` with your measured package power and system power estimates. The energy calculations will automatically use the new values.

---

## License

MIT

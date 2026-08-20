"""Benchmark configuration: models, tasks, and parameters."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

PACKAGE_POWER_W = float(os.environ.get("PACKAGE_POWER_W", "0.47"))
SYSTEM_POWER_W = float(os.environ.get("SYSTEM_POWER_W", "10"))
GRID_CARBON_INTENSITY = float(os.environ.get("GRID_CARBON_INTENSITY", "708"))

MODELS = [
    {"name": "llama3.2:3b",  "family": "llama",   "params": "3B",   "size_gb": 2.0},
    {"name": "phi3:3.8b",    "family": "phi",     "params": "3.8B", "size_gb": 2.2},
    {"name": "qwen2.5:7b",   "family": "qwen",    "params": "7B",   "size_gb": 4.7},
    {"name": "mistral:7b",   "family": "mistral", "params": "7B",   "size_gb": 4.4},
    {"name": "gemma2:9b",    "family": "gemma",   "params": "9B",   "size_gb": 5.4},
]

TEMPERATURE = 0.0
MAX_TOKENS_MCQ = 20
MAX_TOKENS_SUMMARY = 300
MAX_TOKENS_CODE = 512

MMLU_SAMPLE_SIZE = 100
SUMMARIZATION_SAMPLE_SIZE = 50

MMLU_SUBJECTS = [
    "abstract_algebra", "college_chemistry", "college_physics",
    "computer_security", "conceptual_physics", "electrical_engineering",
    "high_school_biology", "high_school_chemistry", "high_school_physics",
    "machine_learning",
]

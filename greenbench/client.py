"""Ollama API client with nanosecond-precision timing extraction."""

import json
import time
import requests
from pathlib import Path

from greenbench.config import OLLAMA_URL


class OllamaClient:
    def __init__(self, base_url=None):
        self.base_url = base_url or OLLAMA_URL
        self.session = requests.Session()

    def generate(self, model, prompt, temperature=0.0, num_predict=100,
                 retries=3, timeout=300):
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        for attempt in range(retries):
            try:
                wall_start = time.time()
                resp = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=timeout,
                )
                wall_end = time.time()
                resp.raise_for_status()
                data = resp.json()

                eval_count = data.get("eval_count", 0)
                eval_duration = data.get("eval_duration", 0)

                return {
                    "response": data.get("response", ""),
                    "model": model,
                    "eval_count": eval_count,
                    "eval_duration_ns": eval_duration,
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "prompt_eval_duration_ns": data.get("prompt_eval_duration", 0),
                    "total_duration_ns": data.get("total_duration", 0),
                    "load_duration_ns": data.get("load_duration", 0),
                    "wall_time_s": wall_end - wall_start,
                    "tokens_per_sec": (
                        eval_count / (eval_duration / 1e9)
                        if eval_duration > 0 else 0
                    ),
                    "time_per_token_ms": (
                        (eval_duration / 1e6) / eval_count
                        if eval_count > 0 else 0
                    ),
                }

            except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    return {
                        "response": "",
                        "model": model,
                        "error": str(e),
                        "eval_count": 0,
                        "eval_duration_ns": 0,
                        "tokens_per_sec": 0,
                        "time_per_token_ms": 0,
                        "wall_time_s": 0,
                    }

    def warmup(self, model):
        """Single warm-up call to ensure weights are resident in memory."""
        self.generate(model, "Hello", temperature=0, num_predict=1, timeout=120)

    def is_alive(self):
        try:
            resp = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self):
        try:
            resp = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            return [m["name"] for m in resp.json().get("models", [])]
        except (requests.RequestException, json.JSONDecodeError):
            return []

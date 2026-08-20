#!/bin/bash
# Measure CPU+GPU package power during idle and LLM inference.
# Requires macOS with Apple Silicon and sudo access.
#
# Usage: sudo bash scripts/measure_power.sh [model_name]
#
# Output: prints idle, inference, and cooldown power readings.

MODEL=${1:-"qwen2.5:7b"}
SAMPLES=5
INTERVAL_MS=2000

echo "=============================================="
echo "GreenBench Power Measurement"
echo "Model: $MODEL"
echo "=============================================="

echo ""
echo "[Phase 1] Idle baseline (4 seconds)..."
powermetrics --samplers cpu_power,gpu_power -n 2 -i $INTERVAL_MS 2>/dev/null \
    | grep -E "^(CPU|GPU|ANE|Combined) Power:" \
    | while read line; do echo "  [IDLE] $line"; done

echo ""
echo "[Phase 2] LLM inference burst..."
# Fire 20 inference requests in background
(ollama run $MODEL "Write a detailed paragraph about the history of computing." --nowordwrap 2>/dev/null &
 sleep 2
 for i in $(seq 1 19); do
     ollama run $MODEL "Explain quantum computing in simple terms." --nowordwrap 2>/dev/null &
 done
 wait) &
INFERENCE_PID=$!

sleep 3

echo "  Measuring during inference..."
powermetrics --samplers cpu_power,gpu_power -n $SAMPLES -i $INTERVAL_MS 2>/dev/null \
    | grep -E "^(CPU|GPU|ANE|Combined) Power:" \
    | while read line; do echo "  [LOAD] $line"; done

wait $INFERENCE_PID 2>/dev/null

echo ""
echo "[Phase 3] Cooldown (4 seconds)..."
sleep 2
powermetrics --samplers cpu_power,gpu_power -n 2 -i $INTERVAL_MS 2>/dev/null \
    | grep -E "^(CPU|GPU|ANE|Combined) Power:" \
    | while read line; do echo "  [COOL] $line"; done

echo ""
echo "=============================================="
echo "Measurement complete."
echo "=============================================="

#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# Runs the timing/token-usage benchmark for all three generator backbones,
# at a matched row count, across the six free-text datasets.
# Gfl and gpt-4o-mini run sequentially within each track (gpt-4o-mini
# deliberately so, to avoid tripping OpenAI's per-account rate limit --
# see source/metrics/run_timing_benchmark.py for details). Qwen-14B runs
# its six datasets in parallel, one per GPU, since local generation has
# no shared rate limit to worry about.

echo "=== Gfl (gemini-3.1-flash-lite-preview) ==="
python -m source.metrics.run_timing_benchmark gfl

echo "=== gpt-4o-mini ==="
python -m source.metrics.run_timing_benchmark gpt

echo "=== Qwen-14B (local, parallel across GPUs) ==="
python -m source.metrics.run_timing_benchmark qwen

echo "All timing benchmark tracks done. Results: reports/timing_benchmark_results.json"

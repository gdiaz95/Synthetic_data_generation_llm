"""
Timing + token-usage benchmark: free-text datasets x generator backbones, at a
matched row count, for a fair per-generator speed/cost comparison.

Three tracks, run as separate invocations of this script:
  - gfl:  gemini-3.1-flash-lite-preview, sequential across datasets
  - gpt:  gpt-4o-mini, sequential across datasets (kept sequential deliberately --
    running gpt-4o-mini combos in parallel against each other hits OpenAI's
    per-account rate limit hard; confirmed empirically before this script existed)
  - qwen: Qwen/Qwen2.5-14B-Instruct (local), parallel -- one dataset per GPU,
    since each gets dedicated hardware and there's no shared API rate limit

Invokes run_full.py as a subprocess per (dataset, generator) combo, then reads
back the run-log JSON it writes under logs/ (which includes "timing" and
"llm_usage" fields) using the EXACT expected log filename -- not a glob, since
a glob that matches multiple generators' logs for the same dataset can pick
the wrong one depending on sort order (this bit us once during development).

Usage:
    python -m source.metrics.run_timing_benchmark gfl
    python -m source.metrics.run_timing_benchmark gpt
    python -m source.metrics.run_timing_benchmark qwen
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

BENCH_INPUT_DIR = REPO_ROOT / "data" / "processed" / "timing_bench_inputs"
BENCH_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "timing_bench_outputs"
RESULTS_PATH = REPO_ROOT / "reports" / "timing_benchmark_results.json"
GPT_LLM_CONFIG = REPO_ROOT / "configs" / "llm_generation_gpt4omini_bench.yml"

DATASETS = ["adult_with_three", "Airbnb", "consumer_complaint", "fake_job_postings", "Kickstarter", "netflix_titles"]
N_ROWS = 500


def _safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"


def prepare_bench_inputs() -> None:
    """Sample N_ROWS rows (fixed seed) from each dataset's raw CSV, once."""
    import pandas as pd

    BENCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        out_path = BENCH_INPUT_DIR / f"{ds}_bench{N_ROWS}.csv"
        if out_path.exists():
            continue
        df = pd.read_csv(REPO_ROOT / "data" / "raw" / f"{ds}.csv", low_memory=False)
        df.sample(n=N_ROWS, random_state=42).reset_index(drop=True).to_csv(out_path, index=False)
        print(f"Prepared bench input: {out_path}", flush=True)


def _read_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {}


def _write_results(update: dict) -> None:
    current = _read_results()
    current.update(update)
    tmp = RESULTS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, indent=2, sort_keys=True))
    tmp.replace(RESULTS_PATH)


def run_one(ds: str, label: str, model: str, provider: str,
            gpu_index: int | None = None, llm_config: Path | None = None) -> dict:
    key = f"{ds}|{label}"
    existing = _read_results()
    if key in existing and "error" not in existing[key]:
        print(f"[{key}] already done, skipping", flush=True)
        return existing[key]

    input_csv = BENCH_INPUT_DIR / f"{ds}_bench{N_ROWS}.csv"
    output_csv = BENCH_OUTPUT_DIR / f"{ds}_bench{N_ROWS}.csv"
    BENCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, "-m", "source.pipelines.run_full",
        "--dataset-name", ds,
        "--input-csv", str(input_csv),
        "--output-csv", str(output_csv),
        "--model", model,
        "--provider", provider,
        "--yes",
    ]
    if gpu_index is not None:
        cmd += ["--gpu-index", str(gpu_index)]
    if llm_config is not None:
        cmd += ["--llm-config", str(llm_config)]

    print(f"[{key}] starting: {' '.join(cmd)}", flush=True)
    import time
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"[{key}] FAILED (exit {proc.returncode}) after {elapsed:.1f}s", flush=True)
        print(proc.stderr[-2000:], flush=True)
        result = {"error": f"exit_code={proc.returncode}", "stderr_tail": proc.stderr[-2000:]}
    else:
        safe_model = _safe_model_name(model)
        exact_log = REPO_ROOT / "logs" / f"{output_csv.stem}_{safe_model}.log"
        if not exact_log.exists():
            print(f"[{key}] WARNING: expected run-log not found: {exact_log}", flush=True)
            result = {"error": "no_run_log_found", "expected_path": str(exact_log)}
        else:
            log_data = json.loads(exact_log.read_text())
            result = {
                "timing": log_data.get("timing"),
                "llm_usage": log_data.get("llm_usage"),
                "wrapper_elapsed_seconds": round(elapsed, 3),
            }
            print(f"[{key}] OK: {result}", flush=True)

    _write_results({key: result})
    return result


def track_sequential(label: str, model: str, provider: str, llm_config: Path | None = None) -> None:
    for ds in DATASETS:
        run_one(ds, label, model, provider, llm_config=llm_config)
    print(f"=== track {label} ALL DONE ===", flush=True)


def track_qwen_parallel() -> None:
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(DATASETS)) as ex:
        futures = [
            ex.submit(run_one, ds, "Qwen-14B", "Qwen/Qwen2.5-14B-Instruct", "local", gpu_index=i)
            for i, ds in enumerate(DATASETS)
        ]
        concurrent.futures.wait(futures)
    print("=== track Qwen-14B ALL DONE ===", flush=True)


def _ensure_gpt_config() -> Path:
    """gpt-4o-mini needs a lower api_parallel_shards than the project default (64) --
    empirically, 64-way concurrency against OpenAI's per-account rate limit causes
    a storm of HTTP 429s and a very slow retry queue. 8 is safe."""
    if GPT_LLM_CONFIG.exists():
        return GPT_LLM_CONFIG
    base = (REPO_ROOT / "configs" / "llm_generation.yml").read_text()
    patched = re.sub(r"^api_parallel_shards:\s*\d+", "api_parallel_shards: 8", base, flags=re.MULTILINE)
    GPT_LLM_CONFIG.write_text(patched)
    return GPT_LLM_CONFIG


if __name__ == "__main__":
    track = sys.argv[1] if len(sys.argv) > 1 else None
    prepare_bench_inputs()
    if track == "gfl":
        track_sequential("Gfl", "gemini-3.1-flash-lite-preview", "gemini")
    elif track == "gpt":
        track_sequential("gpt-4o-mini", "gpt-4o-mini", "openai", llm_config=_ensure_gpt_config())
    elif track == "qwen":
        track_qwen_parallel()
    else:
        print("usage: python -m source.metrics.run_timing_benchmark [gfl|gpt|qwen]")
        sys.exit(1)

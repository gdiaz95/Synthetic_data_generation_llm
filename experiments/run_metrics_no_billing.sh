#!/usr/bin/env bash
# Every experiment in the paper that costs no API money: local generation,
# local metrics, the local (Qwen-14B) judge, and the significance tests.
# run_metrics_billing.sh covers everything that calls a paid API.
#
# Each step is skipped when its output already exists AND is complete, so
# re-running only does what is missing or broken. A step whose input data is
# produced by the other script is recorded as WAIT, not FAIL.
#
# From scratch: run this script, then run_metrics_billing.sh, then this script
# again (it picks up the metrics on the API-generated data and the significance
# tests, which need every judge's results).
#
# Usage: nohup bash experiments/run_metrics_no_billing.sh > run_metrics_no_billing.out 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

RUN_LOG_DIR="logs/redo_run"
mkdir -p "$RUN_LOG_DIR" reports/judges
MANIFEST="$RUN_LOG_DIR/manifest.csv"
[ -f "$MANIFEST" ] || echo "timestamp,phase,dataset,generator,status,note" > "$MANIFEST"

FREE_TEXT_DATASETS=(netflix_titles fake_job_postings adult_with_three Kickstarter Airbnb consumer_complaint)
STRUCTURED_ONLY_DATASETS=(english_teaching Spotify student_satisfaction)
ALL_DATASETS=("${FREE_TEXT_DATASETS[@]}" "${STRUCTURED_ONLY_DATASETS[@]}")
GENERATORS=(gemini-3.1-flash-lite gpt-6-luna "Qwen/Qwen2.5-14B-Instruct")
GFL=gemini-3.1-flash-lite
GFL_K0=gemini-3.1-flash-lite_k0
QWEN14=Qwen/Qwen2.5-14B-Instruct
QSEVEN=Qwen/Qwen2.5-7B-Instruct_great
QA_REPEATS=3

# Synthetic rows per dataset: min(real rows, 20,000), as stated in the paper.
declare -A EXPECTED_ROWS=(
  [netflix_titles]=8807 [fake_job_postings]=17880 [adult_with_three]=20000
  [Kickstarter]=20000 [Airbnb]=20000 [consumer_complaint]=20000
  [english_teaching]=3580 [Spotify]=20000 [student_satisfaction]=580
)

FAILED=()
WAITING=()

record() { echo "$(date -Is),$1,$2,$3,$4,\"$5\"" >> "$MANIFEST"; }
safe() { echo "$1" | sed -E 's/[^A-Za-z0-9._-]+/-/g; s/^-+//; s/-+$//'; }
data_path() { echo "data/processed/$1_$(safe "$2").csv"; }

# --- completeness checks (exit 0 = complete) ---
data_complete() {  # <dataset> <generator>
  python3 - "$(data_path "$1" "$2")" "${EXPECTED_ROWS[$1]}" <<'EOF'
import csv, sys
path, expected = sys.argv[1], int(sys.argv[2])
csv.field_size_limit(10**9)
try:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = sum(1 for r in csv.reader(f) if r) - 1
except FileNotFoundError:
    sys.exit(1)
sys.exit(0 if rows == expected else 1)
EOF
}

judge_complete() {  # <report> <whole_row_only: 0|1>
  python3 - "$1" "$2" <<'EOF'
import json, sys
path, whole_row_only = sys.argv[1], sys.argv[2] == "1"
try:
    d = json.load(open(path))
except (FileNotFoundError, json.JSONDecodeError):
    sys.exit(1)
lj = d["llm_judge"]
wr = lj.get("whole_row") or {}
ok = d.get("whole_row_only") == whole_row_only and "error" not in wr and wr.get("n_examples", 0) > 0
if ok and not whole_row_only:
    cols = lj.get("free_text_columns") or {}
    meta = json.load(open(f".metadata/{d['dataset_name']}/metadata.json"))
    ok = set(cols) == set(meta.get("free_text_columns") or []) and all(
        "error" not in e and e.get("n_examples", 0) > 0 for e in cols.values())
sys.exit(0 if ok else 1)
EOF
}

structured_complete() {  # <report>
  python3 - "$1" "$QA_REPEATS" <<'EOF'
import json, sys
try:
    qa = json.load(open(sys.argv[1]))["metrics_qa"]
except (FileNotFoundError, json.JSONDecodeError, KeyError):
    sys.exit(1)
ok = qa.get("trivariate_accuracy") is not None and qa.get("n_repeats") == int(sys.argv[2])
sys.exit(0 if ok else 1)
EOF
}

timing_complete() {  # <track label in timing_benchmark_results.json>
  python3 - "$1" "${FREE_TEXT_DATASETS[@]}" <<'EOF'
import json, sys
try:
    d = json.load(open("reports/timing_benchmark_results.json"))
except (FileNotFoundError, json.JSONDecodeError):
    sys.exit(1)
sys.exit(0 if all(f"{ds}|{sys.argv[1]}" in d for ds in sys.argv[2:]) else 1)
EOF
}

# run_step <phase> <dataset> <label> <log_file> <check> -- <command...>
# Skips if <check> passes; otherwise runs <command> (retrying once) and
# counts it as PASS only if <check> passes afterwards.
run_step() {
  local phase="$1" ds="$2" label="$3" log_file="$4" check="$5"; shift 6
  if eval "$check"; then
    echo "=== SKIP (complete): $phase | $ds | $label ==="
    record "$phase" "$ds" "$label" "SKIP" "already complete"
    return 0
  fi
  for attempt in 1 2; do
    echo "=== $phase (attempt $attempt): $ds | $label ($(date)) ==="
    "$@" > "$log_file" 2>&1
    if eval "$check"; then
      record "$phase" "$ds" "$label" "PASS" "attempt $attempt"
      return 0
    fi
    sleep 10
  done
  record "$phase" "$ds" "$label" "FAIL" "see $log_file"
  FAILED+=("$phase | $ds | $label")
}

wait_for() {  # <phase> <dataset> <label> <missing input>
  echo "=== WAIT (input not generated yet): $1 | $2 | $3 ==="
  record "$1" "$2" "$3" "WAIT" "input missing: $4"
  WAITING+=("$1 | $2 | $3")
}

# =========================== Generation ===================================

echo "### Generation: NIFTSy k=5 with Qwen-14B (local), free-text datasets ###"
for ds in "${FREE_TEXT_DATASETS[@]}"; do
  extra=()
  # Long fake_job_postings prompts exceed 4,096 tokens; this one run uses 8,192.
  [ "$ds" = "fake_job_postings" ] && extra=(--max-model-len 8192)
  run_step generate "$ds" "$QWEN14" "$RUN_LOG_DIR/${ds}_$(safe "$QWEN14").generate.log" \
    "data_complete $ds $QWEN14" -- \
    poetry run python -m source.pipelines.run_full --dataset-name "$ds" --model "$QWEN14" \
      --provider local "${extra[@]}" --yes
done

echo "### Generation: NIFTSy k=5, structured-only datasets (NPGC only, no LLM call) ###"
for ds in "${STRUCTURED_ONLY_DATASETS[@]}"; do
  for gen in "${GENERATORS[@]}"; do
    provider=auto; [ "$gen" = "$QWEN14" ] && provider=local
    run_step generate "$ds" "$gen" "$RUN_LOG_DIR/${ds}_$(safe "$gen").generate.log" \
      "data_complete $ds $gen" -- \
      poetry run python -m source.pipelines.run_full --dataset-name "$ds" --model "$gen" \
        --provider "$provider" --yes
  done
done

echo "### Generation: GReaT baseline (Qwen-7B fine-tune, local), all datasets ###"
for ds in "${ALL_DATASETS[@]}"; do
  run_step generate "$ds" "$QSEVEN" "$RUN_LOG_DIR/${ds}_$(safe "$QSEVEN").generate.log" \
    "data_complete $ds $QSEVEN" -- \
    poetry run python -m source.pipelines.run_great --dataset-name "$ds" --gpu-index 0
done

# ============================ Metrics =====================================

structured_metrics() {  # <dataset> <generator>
  local ds="$1" gen="$2"
  if ! data_complete "$ds" "$gen"; then wait_for structured_metrics "$ds" "$gen" "$(data_path "$ds" "$gen")"; return; fi
  local report="reports/${ds}_$(safe "$gen")_structured_metrics.json"
  run_step structured_metrics "$ds" "$gen" "$RUN_LOG_DIR/${ds}_$(safe "$gen").structured_metrics.log" \
    "structured_complete $report" -- \
    poetry run python -m source.metrics.run_structured_metrics --dataset-name "$ds" --generator-model "$gen"
}

echo "### Structured metrics: NIFTSy (all datasets x 3 LLMs) and GReaT (all datasets) ###"
for ds in "${ALL_DATASETS[@]}"; do
  for gen in "${GENERATORS[@]}" "$QSEVEN"; do structured_metrics "$ds" "$gen"; done
done

echo "### Privacy (identical match, DCR share, MIA): NPGC fit on the training split, 3 runs per dataset ###"
for ds in "${ALL_DATASETS[@]}"; do
  for run in 1 2 3; do
    run_step privacy "$ds" "run$run" "$RUN_LOG_DIR/${ds}_run${run}.privacy.log" \
      "[ -f reports/privacy/${ds}_run${run}_privacy.json ]" -- \
      poetry run python -m source.metrics.run_privacy --dataset-name "$ds" --run "$run"
  done
done

echo "### BERTScore / MAUVE: NIFTSy free-text datasets x 3 LLMs ###"
for ds in "${FREE_TEXT_DATASETS[@]}"; do
  for gen in "${GENERATORS[@]}"; do
    if ! data_complete "$ds" "$gen"; then wait_for llm_text_metrics "$ds" "$gen" "$(data_path "$ds" "$gen")"; continue; fi
    run_step llm_text_metrics "$ds" "$gen" "$RUN_LOG_DIR/${ds}_$(safe "$gen").llm_text_metrics.log" \
      "[ -f reports/llm_metrics/${ds}_$(safe "$gen")_llm_text_metrics.json ]" -- \
      poetry run python source/metrics/run_llm_text_metrics.py --dataset-name "$ds" --generator-model "$gen" \
        --mauve-max-samples 500 --bertscore-model distilbert-base-uncased --nndr-threshold 0.5
  done
done

# judge <phase> <dataset> <generator> <whole_row_only: 0|1>
judge() {
  local phase="$1" ds="$2" gen="$3" wro="$4"
  if ! data_complete "$ds" "$gen"; then wait_for "$phase" "$ds" "${gen}|judge=Qwen-14B" "$(data_path "$ds" "$gen")"; return; fi
  local report="reports/judges/${ds}_$(safe "$gen")_$(safe "$QWEN14")_judge.json"
  local flag=(); [ "$wro" = 1 ] && flag=(--whole-row-only)
  run_step "$phase" "$ds" "${gen}|judge=Qwen-14B" "$RUN_LOG_DIR/${ds}_$(safe "$gen")_$(safe "$QWEN14").judge.log" \
    "judge_complete $report $wro" -- \
    poetry run python -m source.metrics.run_judge --dataset-name "$ds" --generator-model "$gen" \
      --judge-model "$QWEN14" --judge-provider local "${flag[@]}" --yes
}

echo "### LLM judge (Qwen-14B, local) ###"
for ds in "${FREE_TEXT_DATASETS[@]}"; do
  for gen in "${GENERATORS[@]}"; do judge judge_k5 "$ds" "$gen" 0; done   # column-level + whole-row
done
for ds in "${FREE_TEXT_DATASETS[@]}"; do judge judge_k0 "$ds" "$GFL_K0" 1; done
for ds in "${STRUCTURED_ONLY_DATASETS[@]}"; do judge judge_k5 "$ds" "$GFL" 1; done
for ds in "${ALL_DATASETS[@]}"; do judge judge_great "$ds" "$QSEVEN" 1; done

echo "### Linguistic quality (word count, MATTR, Distinct-2) ###"
run_step linguistic_quality ALL ALL "$RUN_LOG_DIR/linguistic_quality.log" \
  "[ -f reports/linguistic_quality/results.json ]" -- \
  poetry run python -m source.metrics.run_linguistic_quality

echo "### Runtime benchmark, n=500 (Qwen-14B, local) ###"
run_step timing_benchmark ALL qwen "$RUN_LOG_DIR/timing_benchmark_qwen.log" \
  "timing_complete Qwen-14B" -- \
  flock reports/.timing_benchmark.lock poetry run python -m source.metrics.run_timing_benchmark qwen

echo "### Significance tests (need every judge's results) ###"
log_file="$RUN_LOG_DIR/significance_tests.log"
poetry run python -m source.metrics.run_significance_tests > "$log_file" 2>&1
case $? in
  0) record significance_tests ALL ALL PASS "see $log_file" ;;
  3) record significance_tests ALL ALL WAIT "inputs incomplete, see $log_file"
     WAITING+=("significance_tests (inputs listed in $log_file)") ;;
  *) record significance_tests ALL ALL FAIL "see $log_file"; FAILED+=("significance_tests") ;;
esac

echo
echo "### No-billing script done: $(date) ###"
[ "${#WAITING[@]}" -gt 0 ] && { echo "WAITING on the other script's outputs (re-run after it):"; printf '  - %s\n' "${WAITING[@]}"; }
[ "${#FAILED[@]}" -gt 0 ] && { echo "FAILED (after 2 attempts):"; printf '  - %s\n' "${FAILED[@]}"; }
[ "${#WAITING[@]}" -eq 0 ] && [ "${#FAILED[@]}" -eq 0 ] && echo "Everything complete."
echo "Full manifest: $MANIFEST"

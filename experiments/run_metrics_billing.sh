#!/usr/bin/env bash
# Every experiment in the paper that calls a paid API: NIFTSy generation with
# the hosted LLMs, the three hosted judges, and the hosted runtime benchmarks.
# run_metrics_no_billing.sh covers everything local (and the significance tests).
#
# Each step is skipped when its output already exists AND is complete, so
# re-running only does what is missing or broken. A step whose input data is
# produced by the other script is recorded as WAIT, not FAIL.
#
# Usage: nohup bash experiments/run_metrics_billing.sh > run_metrics_billing.out 2>&1 &

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
HOSTED_GENERATORS=(gemini-3.1-flash-lite gpt-6-luna)
GFL=gemini-3.1-flash-lite
GFL_K0=gemini-3.1-flash-lite_k0
QSEVEN=Qwen/Qwen2.5-7B-Instruct_great
declare -A JUDGE_PROVIDER=([gemini-3.1-flash-lite]=gemini [gemini-4-argon]=gemini [gpt-6-luna]=openai)

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

echo "### Generation: NIFTSy k=5 with hosted LLMs, free-text datasets ###"
for ds in "${FREE_TEXT_DATASETS[@]}"; do
  for gen in "${HOSTED_GENERATORS[@]}"; do
    run_step generate "$ds" "$gen" "$RUN_LOG_DIR/${ds}_$(safe "$gen").generate.log" \
      "data_complete $ds $gen" -- \
      poetry run python -m source.pipelines.run_full --dataset-name "$ds" --model "$gen" --provider auto --yes
  done
done

echo "### Generation: NIFTSy k=0 (retrieval ablation), Gfl, free-text datasets ###"
for ds in "${FREE_TEXT_DATASETS[@]}"; do
  run_step generate "$ds" "$GFL_K0" "$RUN_LOG_DIR/${ds}_$(safe "$GFL_K0").generate.log" \
    "data_complete $ds $GFL_K0" -- \
    poetry run python -m source.pipelines.run_full --dataset-name "$ds" --model "$GFL" \
      --provider auto --k-neighbors 0 --yes
done

# ============================ Judges ======================================

# judge <phase> <dataset> <generator> <judge> <whole_row_only: 0|1>
judge() {
  local phase="$1" ds="$2" gen="$3" jm="$4" wro="$5"
  if ! data_complete "$ds" "$gen"; then wait_for "$phase" "$ds" "${gen}|judge=${jm}" "$(data_path "$ds" "$gen")"; return; fi
  local report="reports/judges/${ds}_$(safe "$gen")_$(safe "$jm")_judge.json"
  local flag=(); [ "$wro" = 1 ] && flag=(--whole-row-only)
  run_step "$phase" "$ds" "${gen}|judge=${jm}" "$RUN_LOG_DIR/${ds}_$(safe "$gen")_$(safe "$jm").judge.log" \
    "judge_complete $report $wro" -- \
    poetry run python -m source.metrics.run_judge --dataset-name "$ds" --generator-model "$gen" \
      --judge-model "$jm" --judge-provider "${JUDGE_PROVIDER[$jm]}" "${flag[@]}" --yes
}

for jm in gemini-3.1-flash-lite gemini-4-argon gpt-6-luna; do
  echo "### LLM judge: $jm ###"
  for ds in "${FREE_TEXT_DATASETS[@]}"; do
    for gen in "${GENERATORS[@]}"; do judge judge_k5 "$ds" "$gen" "$jm" 0; done   # column-level + whole-row
  done
  for ds in "${FREE_TEXT_DATASETS[@]}"; do judge judge_k0 "$ds" "$GFL_K0" "$jm" 1; done
  for ds in "${STRUCTURED_ONLY_DATASETS[@]}"; do judge judge_k5 "$ds" "$GFL" "$jm" 1; done
  for ds in "${ALL_DATASETS[@]}"; do judge judge_great "$ds" "$QSEVEN" "$jm" 1; done
done

# ========================= Runtime benchmark ==============================

echo "### Runtime benchmark, n=500 (hosted) ###"
run_step timing_benchmark ALL gfl "$RUN_LOG_DIR/timing_benchmark_gfl.log" \
  "timing_complete Gfl" -- flock reports/.timing_benchmark.lock poetry run python -m source.metrics.run_timing_benchmark gfl
run_step timing_benchmark ALL gpt "$RUN_LOG_DIR/timing_benchmark_gpt.log" \
  "timing_complete gpt-6-luna" -- flock reports/.timing_benchmark.lock poetry run python -m source.metrics.run_timing_benchmark gpt

echo
echo "### Billing script done: $(date) ###"
[ "${#WAITING[@]}" -gt 0 ] && { echo "WAITING on the other script's outputs (re-run after it):"; printf '  - %s\n' "${WAITING[@]}"; }
[ "${#FAILED[@]}" -gt 0 ] && { echo "FAILED (after 2 attempts):"; printf '  - %s\n' "${FAILED[@]}"; }
[ "${#WAITING[@]}" -eq 0 ] && [ "${#FAILED[@]}" -eq 0 ] && echo "Everything complete."
echo "Next: run experiments/run_metrics_no_billing.sh to compute the significance tests."
echo "Full manifest: $MANIFEST"

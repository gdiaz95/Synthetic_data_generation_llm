#!/usr/bin/env bash
# Computes BERTScore + MAUVE + privacy NNDR for all datasets.
# adult_with_three only (no other adult variants).
# All other datasets run with both models.
# Output: reports/llm_metrics/{dataset}_{model}_llm_text_metrics.json
#
# Usage:
#   bash run_text_metrics_all.sh
#   bash run_text_metrics_all.sh 2>&1 | tee logs/text_metrics_all.log

set -uo pipefail  # no -e: errors are caught per-call so one failure never stops the loop

# Always run from the project root regardless of where this script lives
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=5,6,7   # device 0 inside script → physical GPU 5

SCRIPT="source/pipelines/run_llm_text_metrics.py"
MODELS=("gpt-4o-mini" "gemini-3.1-flash-lite-preview")
OTHER_DATASETS=(
    Airbnb
    consumer_complaint
    english_teaching
    fake_job_postings
    harvard_reviews
    Kickstarter
    netflix_titles
    Spotify
    student_satisfaction
)

run_one() {
    local dataset=$1
    local model=$2
    echo ""
    echo "========================================================"
    echo "  dataset: $dataset   model: $model"
    echo "========================================================"
    python "$SCRIPT" \
        --dataset-name "$dataset" \
        --generator-model "$model" \
        --device-id 0 \
        --mauve-max-samples 500 \
        --bertscore-model distilbert-base-uncased \
        --nndr-threshold 0.5
}

SKIPPED=()

echo "### adult_with_three ###"
for model in "${MODELS[@]}"; do
    run_one "adult_with_three" "$model" \
        || { echo "  => SKIPPED: adult_with_three / $model"; SKIPPED+=("adult_with_three / $model"); }
done

echo ""
echo "### other datasets (two models each) ###"
for dataset in "${OTHER_DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        run_one "$dataset" "$model" \
            || { echo "  => SKIPPED: $dataset / $model"; SKIPPED+=("$dataset / $model"); }
    done
done

echo ""
echo "All done. Reports written to reports/llm_metrics/"
if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo ""
    echo "Skipped ${#SKIPPED[@]} run(s) due to errors:"
    for entry in "${SKIPPED[@]}"; do
        echo "  - $entry"
    done
fi

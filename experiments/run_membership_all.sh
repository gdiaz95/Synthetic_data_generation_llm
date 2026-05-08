#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

DATASETS=(
    adult_with_both_small
    adult_with_three
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

for ds in "${DATASETS[@]}"; do
    echo "=== $ds ==="
    python -m source.pipelines.run_membership --dataset-name "$ds"
done

echo "All membership tests done."

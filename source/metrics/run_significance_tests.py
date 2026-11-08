"""
Significance tests exactly as described in the paper's statistical testing protocol.

All comparisons are two-sided exact binomial sign tests on paired results; ties
are excluded and the number of pairs favoring one side is tested against p=0.5.

  (i)   Judge comparison: for each pair of judges, pair their accuracies over the
        51 k=5 evaluations (11 column-level + 6 whole-row, for each of the three
        NIFTSy generators). Bonferroni correction over the six judge pairs.
  (ii)  Retrieval ablation: whole-row accuracy of NIFTSy Gfl at k=5 vs k=0, paired
        per (dataset, judge) over the six free-text datasets. Lower is better.
  (iii) GReaT baseline: whole-row accuracy of NIFTSy Gfl (k=5) vs Qseven, paired per
        (dataset, judge) over all nine datasets. Lower is better.
  (iv)  Structured fidelity: overall structured fidelity of NIFTSy Gfl (k=5) vs
        Qseven, paired per dataset over all nine datasets. Higher is better.

Quality control (reported, not excluded): an evaluation is degenerate when at
least 95% of its predictions fall in one class, and has parsing failures when
more than 5% of responses could not be interpreted.

Inputs:  reports/judges/*_judge.json, reports/*_structured_metrics.json
Output:  reports/significance_tests.json
Exits with code 3 if any expected input is missing or contains an error entry, so
an incomplete run can never be reported as a pass.
"""

from __future__ import annotations

import itertools
import json
import pathlib
import re
import sys
from datetime import datetime, timezone

from scipy.stats import binomtest

REPORTS_DIR = pathlib.Path("reports")
JUDGES_DIR = REPORTS_DIR / "judges"
METADATA_DIR = pathlib.Path(".metadata")
OUTPUT_JSON = REPORTS_DIR / "significance_tests.json"

FREE_TEXT_DATASETS = [
    "adult_with_three", "Airbnb", "consumer_complaint",
    "fake_job_postings", "Kickstarter", "netflix_titles",
]
STRUCTURED_ONLY_DATASETS = ["english_teaching", "Spotify", "student_satisfaction"]
ALL_DATASETS = FREE_TEXT_DATASETS + STRUCTURED_ONLY_DATASETS

GENERATORS = ["gemini-3.1-flash-lite", "gpt-6-luna", "Qwen/Qwen2.5-14B-Instruct"]
JUDGES = ["gemini-3.1-flash-lite", "gemini-4-argon", "gpt-6-luna", "Qwen/Qwen2.5-14B-Instruct"]
NIFTSY_GFL = "gemini-3.1-flash-lite"
NIFTSY_GFL_K0 = "gemini-3.1-flash-lite_k0"
QSEVEN = "Qwen/Qwen2.5-7B-Instruct_great"

ALPHA = 0.05
DEGENERATE_SHARE = 0.95
PARSE_FAILURE_SHARE = 0.05

missing_inputs: list[str] = []
qc_seen: dict[str, dict] = {}


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def _text_columns(dataset: str) -> list[str]:
    meta = json.loads((METADATA_DIR / dataset / "metadata.json").read_text())
    return [str(c) for c in (meta.get("free_text_columns") or [])]


def _judge_block(dataset: str, generator: str, judge: str) -> dict | None:
    path = JUDGES_DIR / f"{dataset}_{_safe(generator)}_{_safe(judge)}_judge.json"
    if not path.exists():
        missing_inputs.append(f"missing file: {path}")
        return None
    return json.loads(path.read_text())["llm_judge"]


def _entry_accuracy(block: dict | None, key: str | None, label: str) -> float | None:
    """key=None selects the whole-row entry, otherwise a free-text column."""
    if block is None:
        return None
    entry = block.get("whole_row") if key is None else (block.get("free_text_columns") or {}).get(key)
    if not isinstance(entry, dict) or "error" in entry or not entry.get("n_examples"):
        missing_inputs.append(f"missing or errored evaluation: {label}")
        return None
    n = entry["n_examples"]
    predicted = (entry.get("predicted_real", 0), entry.get("predicted_synthetic", 0))
    qc_seen[label] = {
        "evaluation": label,
        "accuracy": entry["correct"] / n,
        "majority_class_share": max(predicted) / n,
        "parse_failure_share": entry.get("failed_to_parse", 0) / n,
    }
    return entry["correct"] / n


def _whole_row(dataset: str, generator: str, judge: str) -> float | None:
    label = f"{dataset} | {generator} | judge={judge} | whole_row"
    return _entry_accuracy(_judge_block(dataset, generator, judge), None, label)


def _fidelity(dataset: str, generator: str) -> float | None:
    path = REPORTS_DIR / f"{dataset}_{_safe(generator)}_structured_metrics.json"
    if not path.exists():
        missing_inputs.append(f"missing file: {path}")
        return None
    value = json.loads(path.read_text()).get("metrics_qa", {}).get("overall_accuracy")
    if value is None:
        missing_inputs.append(f"no overall_accuracy in {path}")
    return value


def _sign_test(pairs: list[dict], a_key: str, b_key: str, lower_is_better: bool) -> dict:
    """Two-sided exact binomial sign test; ties excluded."""
    a_better = b_better = ties = 0
    for p in pairs:
        if p[a_key] == p[b_key]:
            ties += 1
        elif (p[a_key] < p[b_key]) == lower_is_better:
            a_better += 1
        else:
            b_better += 1
    n = a_better + b_better
    p_value = float(binomtest(a_better, n, 0.5, alternative="two-sided").pvalue) if n else None
    return {
        f"{a_key}_better": a_better,
        f"{b_key}_better": b_better,
        "ties": ties,
        "n_non_tied": n,
        "p_two_sided": p_value,
    }


def judge_comparison() -> dict:
    """(i) Pair two judges' accuracies over the 51 k=5 evaluations."""
    evaluations: list[tuple[str, str, str | None]] = []
    for generator in GENERATORS:
        for dataset in FREE_TEXT_DATASETS:
            for column in _text_columns(dataset):
                evaluations.append((dataset, generator, column))
        for dataset in FREE_TEXT_DATASETS:
            evaluations.append((dataset, generator, None))

    accuracy: dict[str, list[float | None]] = {}
    for judge in JUDGES:
        accuracy[judge] = []
        for dataset, generator, column in evaluations:
            label = f"{dataset} | {generator} | judge={judge} | {column or 'whole_row'}"
            accuracy[judge].append(_entry_accuracy(_judge_block(dataset, generator, judge), column, label))

    complete = [i for i in range(len(evaluations)) if all(accuracy[j][i] is not None for j in JUDGES)]
    n_pairs = len(list(itertools.combinations(JUDGES, 2)))
    alpha_corrected = ALPHA / n_pairs
    comparisons = []
    for judge_a, judge_b in itertools.combinations(JUDGES, 2):
        pairs = [{"a": accuracy[judge_a][i], "b": accuracy[judge_b][i]} for i in complete]
        # Higher accuracy is the stronger judge.
        test = _sign_test(pairs, "a", "b", lower_is_better=False)
        comparisons.append({
            "judge_a": judge_a,
            "judge_b": judge_b,
            "judge_a_wins": test["a_better"],
            "judge_b_wins": test["b_better"],
            "ties": test["ties"],
            "n_non_tied": test["n_non_tied"],
            "p_two_sided": test["p_two_sided"],
            "significant_bonferroni": test["p_two_sided"] is not None and test["p_two_sided"] < alpha_corrected,
        })
    return {
        "n_evaluations": len(evaluations),
        "n_evaluations_complete": len(complete),
        "bonferroni_alpha": alpha_corrected,
        "mean_accuracy": {
            j: sum(accuracy[j][i] for i in complete) / len(complete) if complete else None for j in JUDGES
        },
        "comparisons": comparisons,
    }


def paired_whole_row(datasets: list[str], gen_a: str, gen_b: str, name_a: str, name_b: str) -> dict:
    """(ii)/(iii) Pair whole-row accuracy of two configurations per (dataset, judge)."""
    pairs = []
    for dataset in datasets:
        for judge in JUDGES:
            acc_a = _whole_row(dataset, gen_a, judge)
            acc_b = _whole_row(dataset, gen_b, judge)
            if acc_a is not None and acc_b is not None:
                pairs.append({"dataset": dataset, "judge": judge, name_a: acc_a, name_b: acc_b})
    return {**_sign_test(pairs, name_a, name_b, lower_is_better=True), "n_pairs": len(pairs), "pairs": pairs}


def paired_fidelity() -> dict:
    """(iv) Pair overall structured fidelity of NIFTSy Gfl (k=5) and Qseven per dataset."""
    pairs = []
    for dataset in ALL_DATASETS:
        niftsy = _fidelity(dataset, NIFTSY_GFL)
        qseven = _fidelity(dataset, QSEVEN)
        if niftsy is not None and qseven is not None:
            pairs.append({"dataset": dataset, "niftsy": niftsy, "qseven": qseven})
    return {**_sign_test(pairs, "niftsy", "qseven", lower_is_better=False), "n_pairs": len(pairs), "pairs": pairs}


def main() -> int:
    output = {
        "metadata": {"generated_at": datetime.now(timezone.utc).isoformat()},
        "judge_comparison": judge_comparison(),
        "retrieval_ablation": paired_whole_row(FREE_TEXT_DATASETS, NIFTSY_GFL, NIFTSY_GFL_K0, "k5", "k0"),
        "qseven_whole_row": paired_whole_row(ALL_DATASETS, NIFTSY_GFL, QSEVEN, "niftsy", "qseven"),
        "qseven_structured_fidelity": paired_fidelity(),
    }
    output["quality_control"] = {
        "degenerate": [q for q in qc_seen.values() if q["majority_class_share"] >= DEGENERATE_SHARE],
        "parse_failures": [q for q in qc_seen.values() if q["parse_failure_share"] > PARSE_FAILURE_SHARE],
    }
    unique_missing = sorted(set(missing_inputs))
    output["metadata"]["missing_inputs"] = unique_missing

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, indent=2))
    print(f"Saved: {OUTPUT_JSON}")

    jc = output["judge_comparison"]
    print(f"(i)   judges: {jc['n_evaluations_complete']}/{jc['n_evaluations']} evaluations complete")
    for key in ("retrieval_ablation", "qseven_whole_row", "qseven_structured_fidelity"):
        r = output[key]
        print(f"      {key}: {r['n_pairs']} pairs, {r['ties']} ties, p={r['p_two_sided']}")
    if unique_missing:
        print(f"\nINCOMPLETE: {len(unique_missing)} expected inputs missing or errored:")
        for m in unique_missing:
            print(f"  - {m}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

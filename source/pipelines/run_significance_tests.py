"""
Statistical significance tests for LLM-as-judge results.

Reads all structured_metrics JSON reports and computes:
  - Per-evaluation: exact binomial test vs H0: p=0.5, Wilson CI, Cohen's h
  - Aggregate: sign test across all free-text columns, by judge and combined
  - Cross-judge: two-proportion z-test comparing judge models on matched pairs
  - Confusion matrix breakdown: sensitivity / specificity

Output: reports/significance_tests.json

No API calls required — all data is in the existing report files.
Handles gracefully:
  - Datasets with no free-text columns (empty or missing free_text_columns)
  - Error entries inside llm_judge ({"error": "..."})
  - Null/missing fields anywhere in the judge block
  - Reports with partial or no judge results
"""

import json
import math
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from scipy.stats import binomtest, norm

REPORTS_DIR = pathlib.Path("reports")
OUTPUT_JSON = pathlib.Path("reports/significance_tests.json")

# Only datasets that have free-text columns — k=0 vs k=5 is meaningless for structured-only.
FREE_TEXT_DATASETS = {
    "adult_with_three", "fake_job_postings", "harvard_reviews",
    "netflix_titles", "Kickstarter", "Airbnb", "consumer_complaint",
}


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class JudgeEval:
    report_file: str
    dataset: str
    generator: str
    judge: str
    eval_type: str      # "free_text" | "whole_row"
    column: str         # column name or "__whole_row__"
    n: int
    n_real: int
    n_synthetic: int
    correct: int
    tp: int             # true_real_predicted_real
    fn: int             # true_real_predicted_synthetic
    fp: int             # true_synthetic_predicted_real
    tn: int             # true_synthetic_predicted_synthetic
    failed_to_parse: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n > 0 else float("nan")

    @property
    def sensitivity(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else float("nan")

    @property
    def specificity(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom > 0 else float("nan")

    @property
    def is_degenerate(self) -> bool:
        predicted_real = self.tp + self.fp
        predicted_synthetic = self.fn + self.tn
        return predicted_real == 0 or predicted_synthetic == 0


# ---------------------------------------------------------------------------
# Robust report parsing
# ---------------------------------------------------------------------------

def _safe_int(d: dict, key: str) -> int:
    v = d.get(key, 0)
    return int(v) if isinstance(v, (int, float)) and not math.isnan(float(v)) else 0


def _safe_str(d: dict, key: str, default: str = "unknown") -> str:
    v = d.get(key)
    return str(v) if v is not None else default


def _parse_judge_entry(
    path: pathlib.Path,
    dataset: str,
    generator: str,
    judge: str,
    col_name: str,
    entry,
    eval_type: str,
) -> Optional[JudgeEval]:
    """Parse one judge result entry. Returns None on any problem."""
    if not isinstance(entry, dict):
        return None
    if "error" in entry:
        return None
    n = _safe_int(entry, "n_examples")
    if n == 0:
        return None
    # Skip if all examples failed to parse (no real predictions made)
    failed = _safe_int(entry, "failed_to_parse")
    if failed >= n:
        return None
    return JudgeEval(
        report_file=path.name,
        dataset=dataset,
        generator=generator,
        judge=judge,
        eval_type=eval_type,
        column=col_name,
        n=n,
        n_real=_safe_int(entry, "n_real"),
        n_synthetic=_safe_int(entry, "n_synthetic"),
        correct=_safe_int(entry, "correct"),
        tp=_safe_int(entry, "true_real_predicted_real"),
        fn=_safe_int(entry, "true_real_predicted_synthetic"),
        fp=_safe_int(entry, "true_synthetic_predicted_real"),
        tn=_safe_int(entry, "true_synthetic_predicted_synthetic"),
        failed_to_parse=failed,
    )


def load_all_evals(reports_dir: pathlib.Path) -> tuple[list[JudgeEval], list[dict]]:
    """
    Returns (evals, skipped_log).
    skipped_log collects reasons any entry was skipped, for transparency.
    """
    evals: list[JudgeEval] = []
    skipped: list[dict] = []

    for path in sorted(reports_dir.glob("*_structured_metrics.json")):
        try:
            with open(path) as f:
                report = json.load(f)
        except Exception as e:
            skipped.append({"file": path.name, "reason": f"JSON parse error: {e}"})
            continue

        if not isinstance(report, dict):
            skipped.append({"file": path.name, "reason": "report is not a dict"})
            continue

        judge_block = report.get("llm_judge")
        if not isinstance(judge_block, dict):
            skipped.append({"file": path.name, "reason": "no llm_judge block"})
            continue

        dataset = _safe_str(report, "dataset_name", path.stem)
        generator = _safe_str(report, "model")
        judge = _safe_str(judge_block, "judge_model")

        # Free-text columns
        ft_cols = judge_block.get("free_text_columns")
        if isinstance(ft_cols, dict):
            for col, entry in ft_cols.items():
                ev = _parse_judge_entry(path, dataset, generator, judge, col, entry, "free_text")
                if ev:
                    evals.append(ev)
                else:
                    reason = "error entry" if isinstance(entry, dict) and "error" in entry \
                        else "null/empty/fully-failed entry"
                    skipped.append({
                        "file": path.name, "column": col,
                        "type": "free_text", "reason": reason
                    })

        # Whole-row
        wr_entry = judge_block.get("whole_row")
        ev = _parse_judge_entry(path, dataset, generator, judge, "__whole_row__", wr_entry, "whole_row")
        if ev:
            evals.append(ev)
        elif wr_entry is not None:
            skipped.append({"file": path.name, "type": "whole_row", "reason": "unparseable"})

    return evals, skipped


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def _binom(correct: int, n: int, p0: float = 0.5) -> dict:
    r_two = binomtest(correct, n, p0, alternative="two-sided")
    r_gt  = binomtest(correct, n, p0, alternative="greater")
    return {
        "p_two":    round(r_two.pvalue, 6),
        "p_one_gt": round(r_gt.pvalue,  6),
        "sig_two_05": bool(r_two.pvalue < 0.05),
        "sig_one_05": bool(r_gt.pvalue  < 0.05),
        "sig_one_01": bool(r_gt.pvalue  < 0.01),
    }


def _wilson_ci(correct: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    z = norm.ppf(1 - alpha / 2)
    p = correct / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / d
    return round(max(0.0, c - h), 6), round(min(1.0, c + h), 6)


def _cohens_h(p: float, p0: float = 0.5) -> float:
    return round(2 * math.asin(math.sqrt(p)) - 2 * math.asin(math.sqrt(p0)), 6)


def _sign_test(values: list[float], null: float = 0.5) -> dict:
    above = sum(1 for v in values if v > null)
    below = sum(1 for v in values if v < null)
    n = above + below
    if n == 0:
        return {"above": 0, "below": 0, "n_ties": len(values), "n": 0,
                "p": None, "sig_05": False}
    p = binomtest(above, n, 0.5, alternative="greater").pvalue
    return {
        "above": above,
        "below": below,
        "n_ties": len(values) - n,
        "n": n,
        "p": round(p, 6),
        "sig_05": bool(p < 0.05),
    }


def _two_prop_z(c1: int, n1: int, c2: int, n2: int) -> dict:
    p1, p2 = c1 / n1, c2 / n2
    pp = (c1 + c2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"z": None, "p_two": None, "sig_05": False}
    z = (p1 - p2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return {"z": round(z, 4), "p_two": round(p, 6), "sig_05": bool(p < 0.05)}


# ---------------------------------------------------------------------------
# Build output sections
# ---------------------------------------------------------------------------

def _eval_record(ev: JudgeEval) -> dict:
    binom = _binom(ev.correct, ev.n)
    ci_lo, ci_hi = _wilson_ci(ev.correct, ev.n)
    acc = round(ev.accuracy, 6)
    return {
        "report_file":      ev.report_file,
        "dataset":          ev.dataset,
        "generator":        ev.generator,
        "judge":            ev.judge,
        "eval_type":        ev.eval_type,
        "column":           ev.column,
        "n":                ev.n,
        "n_real":           ev.n_real,
        "n_synthetic":      ev.n_synthetic,
        "correct":          ev.correct,
        "accuracy":         acc,
        "wilson_ci_95":     [ci_lo, ci_hi],
        "cohens_h":         _cohens_h(acc),
        "sensitivity":      round(ev.sensitivity, 6) if not math.isnan(ev.sensitivity) else None,
        "specificity":      round(ev.specificity, 6) if not math.isnan(ev.specificity) else None,
        "p_two":            binom["p_two"],
        "p_one_gt":         binom["p_one_gt"],
        "sig_two_05":       binom["sig_two_05"],
        "sig_one_05":       binom["sig_one_05"],
        "sig_one_01":       binom["sig_one_01"],
        "degenerate":       ev.is_degenerate,
        "failed_to_parse":  ev.failed_to_parse,
    }


def build_sign_tests(evals: list[JudgeEval]) -> dict:
    ft = [e for e in evals if e.eval_type == "free_text"]
    wr = [e for e in evals if e.eval_type == "whole_row"]

    by_judge: dict[str, dict] = {}
    for judge_name in sorted({e.judge for e in ft}):
        group = [e for e in ft if e.judge == judge_name]
        accs = [e.accuracy for e in group]
        st = _sign_test(accs)
        by_judge[judge_name] = {
            **st,
            "mean_accuracy":   round(sum(accs) / len(accs), 6),
            "median_accuracy": round(sorted(accs)[len(accs) // 2], 6),
            "n_individually_sig_one_05": sum(
                1 for e in group
                if _binom(e.correct, e.n)["sig_one_05"]
            ),
            "n_evals": len(group),
        }

    all_ft_accs = [e.accuracy for e in ft]
    all_wr_accs = [e.accuracy for e in wr]

    return {
        "free_text_by_judge": by_judge,
        "free_text_combined": {
            **_sign_test(all_ft_accs),
            "mean_accuracy":   round(sum(all_ft_accs) / len(all_ft_accs), 6) if all_ft_accs else None,
            "n_below_random":  sum(1 for a in all_ft_accs if a < 0.5),
            "n_evals":         len(all_ft_accs),
        },
        "whole_row_combined": {
            **_sign_test(all_wr_accs),
            "mean_accuracy":   round(sum(all_wr_accs) / len(all_wr_accs), 6) if all_wr_accs else None,
            "n_evals":         len(all_wr_accs),
        },
    }


def build_cross_judge(evals: list[JudgeEval]) -> list[dict]:
    index: dict[tuple, list[JudgeEval]] = defaultdict(list)
    for ev in evals:
        index[(ev.dataset, ev.generator, ev.column)].append(ev)

    rows = []
    for (dataset, generator, column), group in sorted(index.items()):
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                z = _two_prop_z(a.correct, a.n, b.correct, b.n)
                rows.append({
                    "dataset":        dataset,
                    "generator":      generator,
                    "column":         column,
                    "eval_type":      a.eval_type,
                    "judge_a":        a.judge,
                    "judge_b":        b.judge,
                    "acc_a":          round(a.accuracy, 6),
                    "acc_b":          round(b.accuracy, 6),
                    "diff_a_minus_b": round(a.accuracy - b.accuracy, 6),
                    "z":              z["z"],
                    "p_two":          z["p_two"],
                    "sig_05":         z["sig_05"],
                })
    return rows


# ---------------------------------------------------------------------------
# k=0 vs k=5 ablation
# ---------------------------------------------------------------------------

def build_k0_ablation(evals: list[JudgeEval]) -> dict:
    """
    Paired comparison of whole_row LLM judge accuracy: k=5 vs k=0 neighbors.
    Only uses FREE_TEXT_DATASETS — for structured-only datasets k=0 == k=5 by construction.
    Lower accuracy = synthetic less detectable = better quality.
    Sign test H1: k=5 accuracy < k=0 accuracy (neighbors help).
    """
    wr = [e for e in evals if e.eval_type == "whole_row" and e.dataset in FREE_TEXT_DATASETS]

    k0_evals = [e for e in wr if "_k0" in e.generator]
    k5_evals = [e for e in wr if "_k0" not in e.generator and "great" not in e.generator.lower()]

    # Index k5 by (dataset, judge, base_generator)
    k5_index: dict[tuple, JudgeEval] = {}
    for e in k5_evals:
        k5_index[(e.dataset, e.judge, e.generator)] = e

    pairs = []
    for e0 in k0_evals:
        base_gen = e0.generator.replace("_k0", "")
        e5 = k5_index.get((e0.dataset, e0.judge, base_gen))
        if e5 is None:
            continue
        pairs.append({
            "dataset":       e0.dataset,
            "judge":         e0.judge,
            "generator_base": base_gen,
            "acc_k5":        round(e5.accuracy, 6),
            "acc_k0":        round(e0.accuracy, 6),
            "diff_k5_minus_k0": round(e5.accuracy - e0.accuracy, 6),
            "k5_better":     bool(e5.accuracy < e0.accuracy),
        })

    n_k5_better = sum(1 for p in pairs if p["k5_better"])
    n_k0_better = len(pairs) - n_k5_better
    n = len(pairs)
    sign_p = binomtest(n_k5_better, n, 0.5, alternative="greater").pvalue if n > 0 else None

    return {
        "note": "Lower LLM judge accuracy = synthetic less detectable = better quality.",
        "datasets_considered": sorted(FREE_TEXT_DATASETS),
        "n_pairs": n,
        "n_k5_better": n_k5_better,
        "n_k0_better": n_k0_better,
        "sign_test_p": round(sign_p, 6) if sign_p is not None else None,
        "sig_05": bool(sign_p < 0.05) if sign_p is not None else False,
        "pairs": sorted(pairs, key=lambda x: x["dataset"]),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(reports_dir: pathlib.Path = REPORTS_DIR,
        output_json: pathlib.Path = OUTPUT_JSON) -> None:

    print(f"Loading reports from: {reports_dir.resolve()}")
    evals, skipped = load_all_evals(reports_dir)

    n_reports = len({e.report_file for e in evals})
    print(f"  {len(evals)} evaluations across {n_reports} reports "
          f"({len(skipped)} entries skipped)")
    if skipped:
        for s in skipped:
            print(f"    [skip] {s}")

    ft_evals = [e for e in evals if e.eval_type == "free_text"]
    wr_evals = [e for e in evals if e.eval_type == "whole_row"]

    per_eval    = [_eval_record(e) for e in evals]
    sign_tests  = build_sign_tests(evals)
    cross_judge = build_cross_judge(evals)
    k0_ablation = build_k0_ablation(evals)
    degenerate  = [_eval_record(e) for e in evals if e.is_degenerate]

    ft_accs = [e.accuracy for e in ft_evals]
    output = {
        "metadata": {
            "generated_at":    datetime.now(timezone.utc).isoformat(),
            "reports_dir":     str(reports_dir.resolve()),
            "n_reports_read":  n_reports,
            "n_evaluations":   len(evals),
            "n_free_text":     len(ft_evals),
            "n_whole_row":     len(wr_evals),
            "n_skipped":       len(skipped),
            "skipped_log":     skipped,
        },
        "per_evaluation":          per_eval,
        "sign_tests":              sign_tests,
        "cross_judge_comparisons": cross_judge,
        "k0_ablation":             k0_ablation,
        "degenerate_cases":        degenerate,
        "summary": {
            "free_text_mean_accuracy":      round(sum(ft_accs) / len(ft_accs), 6) if ft_accs else None,
            "free_text_n_below_random":     sum(1 for a in ft_accs if a < 0.5),
            "free_text_n_sig_one_05":       sum(1 for e in ft_evals
                                                if _binom(e.correct, e.n)["sig_one_05"]),
            "free_text_n_degenerate":       sum(1 for e in ft_evals if e.is_degenerate),
            "free_text_sign_test_combined": sign_tests["free_text_combined"],
            "whole_row_mean_accuracy":      sign_tests["whole_row_combined"]["mean_accuracy"],
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved → {output_json.resolve()}")

    # Quick console summary
    st = sign_tests["free_text_combined"]
    print(f"\n  Free-text sign test (combined): "
          f"{st['above']}/{st['n']} above 0.5, p={st['p']} "
          f"{'*' if st['sig_05'] else '(ns)'}")
    print(f"  Free-text mean accuracy: {output['summary']['free_text_mean_accuracy']}")
    print(f"  Individually significant (one-sided p<0.05): "
          f"{output['summary']['free_text_n_sig_one_05']}/{len(ft_evals)}")
    print(f"  Cross-judge matched pairs: {len(cross_judge)}")
    cross_sig = sum(1 for r in cross_judge if r["sig_05"])
    print(f"  Significant judge differences (p<0.05): {cross_sig}/{len(cross_judge)}")

    ab = k0_ablation
    print(f"\n  k=0 vs k=5 ablation ({ab['n_pairs']} paired datasets with free text):")
    print(f"  k=5 better (lower acc): {ab['n_k5_better']}/{ab['n_pairs']}, "
          f"sign-test p={ab['sign_test_p']} {'*' if ab['sig_05'] else '(ns)'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Statistical significance tests for LLM judge results. "
                    "Reads all *_structured_metrics.json in reports/ and writes "
                    "reports/significance_tests.json."
    )
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--output-json", default="reports/significance_tests.json")
    args = p.parse_args()
    run(pathlib.Path(args.reports_dir), pathlib.Path(args.output_json))

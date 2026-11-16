"""Generate the paper's tables, figures, and inline numbers from reports/.

Every number in the paper is computed here from the JSON reports written by the
experiment scripts, so the paper can be reproduced from reports/ alone.
Accuracy, sign tests, and evaluation sets use the same helpers as
source/metrics/run_significance_tests.py.

Usage:
    python -m source.paper.make_paper_assets

Output: everything is printed to stdout: table rows and inline numbers for
writing/main.tex, and figure coordinates for the standalone figures in
writing/images/*.tex, which are maintained by hand (writing/ lives in Overleaf).
"""
from __future__ import annotations

import json
import pathlib
import statistics

from source.metrics.run_significance_tests import (
    ALL_DATASETS,
    FREE_TEXT_DATASETS,
    GENERATORS,
    JUDGES,
    REPORTS_DIR,
    _entry_accuracy,
    _judge_block,
    _safe,
    _text_columns,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SIGNIFICANCE_JSON = REPO_ROOT / "reports" / "significance_tests.json"

# Labels as printed in the paper (writing/main.tex macros).
DATASET_LABEL = {
    "adult_with_three": "Adult-Text",
    "Airbnb": "Airbnb",
    "consumer_complaint": "CFPB Complaints",
    "fake_job_postings": "Fake Jobs",
    "Kickstarter": "Kickstarter",
    "netflix_titles": "Netflix",
    "english_teaching": "English Teaching",
    "Spotify": "Spotify",
    "student_satisfaction": "Student Satisfaction",
}
# Suffix of the \ds... macro for each dataset in writing/main.tex.
DATASET_MACRO = {
    "adult_with_three": "Adult",
    "Airbnb": "Airbnb",
    "consumer_complaint": "Complaint",
    "fake_job_postings": "Jobs",
    "Kickstarter": "Kick",
    "netflix_titles": "Netflix",
    "english_teaching": "English",
    "Spotify": "Spotify",
    "student_satisfaction": "Student",
}
JUDGE_MACRO = {
    "gemini-3.1-flash-lite": r"\Gfl",
    "gemini-4-argon": r"\Pro",
    "gpt-6-luna": r"\GPT",
    "Qwen/Qwen2.5-14B-Instruct": r"\Qfourteen",
}
JUDGE_LABEL = {
    "gemini-3.1-flash-lite": "Flash-Lite",
    "gemini-4-argon": "Argon",
    "gpt-6-luna": "Luna",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen-14B",
}
# Judge style names used in writing/images/judge_by_dataset.tex.
JUDGE_PLOT_STYLE = {
    "gemini-3.1-flash-lite": "gflash",
    "gemini-4-argon": "gpro",
    "gpt-6-luna": "gpt",
    "Qwen/Qwen2.5-14B-Instruct": "q14",
}


def _mean(values: list[float]) -> float:
    return statistics.mean(values)


def judge_accuracy_by_dataset() -> dict[str, dict[str, dict[str, float]]]:
    """Mean judge accuracy (%) per dataset for the k=5 NIFTSy generators.

    Column-level: mean over generators and the dataset's free-text columns.
    Whole-row: mean over generators. The "All" row averages every evaluation
    in the panel (33 column-level, 18 whole-row per judge).
    """
    panels: dict[str, dict[str, dict[str, float]]] = {"column": {}, "whole_row": {}}
    for judge in JUDGES:
        all_column, all_whole = [], []
        for dataset in FREE_TEXT_DATASETS:
            column_acc, whole_acc = [], []
            for generator in GENERATORS:
                block = _judge_block(dataset, generator, judge)
                label = f"{dataset} | {generator} | judge={judge}"
                for column in _text_columns(dataset):
                    column_acc.append(_entry_accuracy(block, column, f"{label} | {column}"))
                whole_acc.append(_entry_accuracy(block, None, f"{label} | whole_row"))
            panels["column"].setdefault(dataset, {})[judge] = 100 * _mean(column_acc)
            panels["whole_row"].setdefault(dataset, {})[judge] = 100 * _mean(whole_acc)
            all_column += column_acc
            all_whole += whole_acc
        panels["column"].setdefault("All", {})[judge] = 100 * _mean(all_column)
        panels["whole_row"].setdefault("All", {})[judge] = 100 * _mean(all_whole)
    return panels


def judge_by_dataset_coordinates() -> str:
    """Coordinates for writing/images/judge_by_dataset.tex: judge accuracy (%) by dataset, column-level
    and whole-row panels; y = 7 (first dataset) down to 1 ("All")."""
    panels = judge_accuracy_by_dataset()
    rows = FREE_TEXT_DATASETS + ["All"]
    y_of = {row: len(rows) - i for i, row in enumerate(rows)}
    out = []
    for name, title in (("column", "Column-level"), ("whole_row", "Whole-row")):
        out.append(f"% {title}")
        for judge in JUDGES:
            coords = " ".join(f"({panels[name][r][judge]:.1f},{y_of[r]})" for r in rows)
            out.append(f"\\addplot[{JUDGE_PLOT_STYLE[judge]}] coordinates {{{coords}}};")
    return "\n".join(out)


def _latex_p(p: float) -> str:
    """p-value as printed in tables: two significant digits, scientific below 0.001."""
    if p >= 0.001:
        return f"{p:.3f}"
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def judge_pairwise_table() -> str:
    """Pairwise judge sign tests (test (i)) from reports/significance_tests.json."""
    comparison = json.loads(SIGNIFICANCE_JSON.read_text())["judge_comparison"]
    n_eval = comparison["n_evaluations_complete"]
    rows = []
    for c in comparison["comparisons"]:
        a_wins, b_wins = c["judge_a_wins"], c["judge_b_wins"]
        a_cell = f"\\textbf{{{a_wins}}}" if a_wins > b_wins else str(a_wins)
        b_cell = f"\\textbf{{{b_wins}}}" if b_wins > a_wins else str(b_wins)
        rows.append(
            f"{JUDGE_MACRO[c['judge_a']]} & {JUDGE_MACRO[c['judge_b']]} & {a_cell} & {b_cell} & "
            f"{c['ties']} & {_latex_p(c['p_two_sided'])} \\\\"
        )
    return (
        "\\begin{table}[htbp]\n\\centering\n\\footnotesize\n"
        "\\caption{\\textbf{Pairwise judge comparison.} Each row pairs the accuracies of two judges over the "
        f"same ${n_eval}$ evaluations: the column-level and whole-row evaluations of the three $k=5$ NIFTSy "
        "generators (Tables~\\ref{tab:judge-bygen} and~\\ref{tab:judge-wholerow-bygen}). Counts give the "
        "evaluations in which each judge is more accurate; the larger count is in bold. $p$ is the two-sided "
        "exact binomial sign test over non-tied evaluations. All six comparisons are significant after a "
        "Bonferroni correction ($\\alpha\\approx0.0083$).}\n"
        "\\label{tab:judge-pairwise}\n"
        "\\begin{tabular}{llcccc}\n\\toprule\n"
        "\\textbf{Judge A} & \\textbf{Judge B} & \\textbf{A more accurate} & \\textbf{B more accurate} & "
        "\\textbf{Ties} & \\textbf{$p$} \\\\ \\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def _structured_report(dataset: str, generator: str) -> dict:
    path = REPORTS_DIR / f"{dataset}_{_safe(generator)}_structured_metrics.json"
    return json.loads(path.read_text())


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def structured_fidelity() -> dict[str, dict[str, tuple[float, float]]]:
    """Mean and SD across the three k=5 generators of each mostlyai-qa fidelity component."""
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for dataset in ALL_DATASETS:
        qa = [_structured_report(dataset, g)["metrics_qa"] for g in GENERATORS]
        out[dataset] = {
            key: _mean_sd([q[key] for q in qa])
            for key in ("overall_accuracy", "univariate_accuracy", "bivariate_accuracy", "trivariate_accuracy")
        }
    return out


def fidelity_coordinates() -> str:
    """Coordinates for writing/images/fidelity.tex: overall structured fidelity, mean of the three k=5 generators."""
    fidelity = structured_fidelity()
    return "\n".join(f"    ({fidelity[d]['overall_accuracy'][0]:.3f},{DATASET_LABEL[d]})" for d in ALL_DATASETS)


def _worst_case(auc: float) -> float:
    return max(auc, 1.0 - auc)


def joint_fidelity_table() -> str:
    """Joint discriminator worst-case AUROC: calibration and synthetic (mean +- SD over generators)."""
    rows = []
    for dataset in ALL_DATASETS:
        joint = [_structured_report(dataset, g)["joint_holdout_discriminator"] for g in GENERATORS]
        calibration = statistics.mean(_worst_case(j["auc_real_vs_holdout"]) for j in joint)
        synth_mean, synth_sd = _mean_sd([_worst_case(j["auc_synth_vs_holdout"]) for j in joint])
        rows.append(
            f"\\ds{DATASET_MACRO[dataset]:<12s} & {calibration:.3f} & ${synth_mean:.3f} \\pm {synth_sd:.3f}$ & "
            f"{synth_mean - calibration:.3f} \\\\"
        )
    return "\n".join(rows)


def appendix_fidelity_tables() -> str:
    """Appendix: fidelity components (mean over generators), overall fidelity per generator, and joint
    discriminator worst-case AUROC per generator with feature width and calibration (mean over the
    generators' runs; the calibration does not involve synthetic data)."""
    components, per_gen, joint_rows = [], [], []
    fidelity = structured_fidelity()
    for dataset in ALL_DATASETS:
        f = fidelity[dataset]
        components.append(f"\\ds{DATASET_MACRO[dataset]:<12s} & \\textbf{{{f['univariate_accuracy'][0]:.3f}}} & "
                          f"{f['bivariate_accuracy'][0]:.3f} & {f['trivariate_accuracy'][0]:.3f} \\\\")
        reports = [_structured_report(dataset, g) for g in GENERATORS]
        per_gen.append(f"\\ds{DATASET_MACRO[dataset]:<12s} & "
                       + " & ".join(f"{r['metrics_qa']['overall_accuracy']:.3f}" for r in reports) + " \\\\")
        joint = [r["joint_holdout_discriminator"] for r in reports]
        calib = statistics.mean(_worst_case(j["auc_real_vs_holdout"]) for j in joint)
        joint_rows.append(f"\\ds{DATASET_MACRO[dataset]:<12s} & {joint[0]['total_feature_width']:,} & {calib:.3f} & "
                          + " & ".join(f"{_worst_case(j['auc_synth_vs_holdout']):.3f}" for j in joint) + " \\\\")
    spread = max(max(r) - min(r) for r in
                 ([_structured_report(d, g)["metrics_qa"]["overall_accuracy"] for g in GENERATORS] for d in ALL_DATASETS))
    return ("\n".join(components) + "\n\n" + "\n".join(per_gen) + f"\nmax spread across generators: {spread:.3f}"
            + "\n\n" + "\n".join(joint_rows))


def linguistic_bygen_table() -> str:
    """Appendix: per-generator word count, MATTR, and Distinct-2 for each free-text column; the word count
    closest to the real value is in bold."""
    results = json.loads(LINGUISTIC_JSON.read_text())
    rows, notes = [], []
    for dataset in FREE_TEXT_DATASETS:
        for column in _text_columns(dataset):
            per = [results[dataset][column][g] for g in GENERATORS]
            real = per[0]["wc_mean_real"]
            words = [p["wc_mean_synthetic"] for p in per]
            best = min(round(abs(w - real), 1) for w in words)
            wcells = [f"\\textbf{{{w:.1f}}}" if round(abs(w - real), 1) == best else f"{w:.1f}" for w in words]
            cells = wcells + [f"{p['mattr_synthetic']:.3f}" for p in per] + [f"{p['distinct2_synthetic']:.3f}" for p in per]
            label = f"\\ds{DATASET_MACRO[dataset]} ({COLUMN_LABEL.get(column, column)})".replace("_", "\\_")
            rows.append(f"{label:<32s} & " + " & ".join(cells) + " \\\\")
            notes.append(f"{label}: real words {real:.1f}, distinct2 real {per[0]['distinct2_real']:.3f}, "
                         f"mattr real {per[0]['mattr_real']:.3f}")
    return "\n".join(rows) + "\n\n" + "\n".join(notes)


def fidelity_numbers() -> str:
    """Inline numbers for the Fidelity subsection."""
    fidelity = structured_fidelity()
    overall = {d: fidelity[d]["overall_accuracy"][0] for d in ALL_DATASETS}
    max_sd = max(sd for d in ALL_DATASETS for _, sd in fidelity[d].values())
    lines = [f"overall fidelity: min {min(overall.values()):.3f} ({min(overall, key=overall.get)}), "
             f"max {max(overall.values()):.3f} ({max(overall, key=overall.get)})",
             f"max SD across generators over all components and datasets: {max_sd:.3f}"]
    lines += [f"  {d:22s} " + "  ".join(f"{k.split('_')[0]}={m:.3f}+-{s:.3f}" for k, (m, s) in fidelity[d].items())
              for d in ALL_DATASETS]
    return "\n".join(lines)


def _llm_text_report(dataset: str, generator: str) -> dict:
    path = REPORTS_DIR / "llm_metrics" / f"{dataset}_{_safe(generator)}_llm_text_metrics.json"
    return json.loads(path.read_text())


def free_text_quality() -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    """Per (dataset, column): mean and SD over the k=5 generators of the Frechet and MMD
    ratios (synthetic vs real-train, divided by real-holdout vs real-train), BERTScore F1,
    and MAUVE; plus the number of non-empty real training rows."""
    out = {}
    for dataset in FREE_TEXT_DATASETS:
        for column in _text_columns(dataset):
            dist = [_structured_report(dataset, g)["free_text_distribution"][column] for g in GENERATORS]
            text = [_llm_text_report(dataset, g) for g in GENERATORS]
            out[(dataset, column)] = {
                "n_train": (dist[0]["n_real_train"], 0.0),
                "frechet": _mean_sd([d["frechet_real_train_vs_synthetic"] / d["frechet_real_train_vs_real_holdout"] for d in dist]),
                "mmd": _mean_sd([d["mmd_rbf_real_train_vs_synthetic"] / d["mmd_rbf_real_train_vs_real_holdout"] for d in dist]),
                "bert_f1": _mean_sd([t["bertscore"][column]["bertscore_f1_mean"] for t in text]),
                "mauve": _mean_sd([t["mauve"][column]["mauve_score"] for t in text]),
            }
    return out


COLUMN_LABEL = {"consumer_complaint_narrative": "narrative"}


def full_quality_table() -> str:
    rows = []
    for (dataset, column), m in free_text_quality().items():
        label = COLUMN_LABEL.get(column, column).replace("_", "\\_")
        rows.append(
            f"\\ds{DATASET_MACRO[dataset]} ({label})\n"
            f"    & {int(m['n_train'][0]):,} & {m['frechet'][0]:.1f} & {m['mmd'][0]:.1f} & "
            f"{m['bert_f1'][0]:.3f} & {m['mauve'][0]:.3f} \\\\".replace(",", "{,}", 1) if m["n_train"][0] >= 1000 else
            f"\\ds{DATASET_MACRO[dataset]} ({label})\n"
            f"    & {int(m['n_train'][0])} & {m['frechet'][0]:.1f} & {m['mmd'][0]:.1f} & "
            f"{m['bert_f1'][0]:.3f} & {m['mauve'][0]:.3f} \\\\"
        )
    return "\n".join(rows)


def free_text_numbers() -> str:
    q = free_text_quality()
    lines = []
    for key in ("frechet", "mmd", "bert_f1", "mauve"):
        lo = min(q, key=lambda c: q[c][key][0]); hi = max(q, key=lambda c: q[c][key][0])
        lines.append(f"{key}: min {q[lo][key][0]:.3f} {lo}, max {q[hi][key][0]:.3f} +- {q[hi][key][1]:.3f} {hi}")
    lines.append("SD across generators, max per metric: " + ", ".join(
        f"{k}={max(v[k][1] for v in q.values()):.3f}" for k in ("frechet", "mmd", "bert_f1", "mauve")))
    return "\n".join(lines)


LINGUISTIC_JSON = REPORTS_DIR / "linguistic_quality" / "results.json"
LINGUISTIC_FIELDS = (("wc_mean", 1), ("mattr", 3), ("distinct2", 3))  # (field, decimals)


def linguistic_quality() -> dict[tuple[str, str], dict[str, float]]:
    """Real holdout value and mean synthetic value over the k=5 generators, per (dataset, column)."""
    results = json.loads(LINGUISTIC_JSON.read_text())
    out = {}
    for dataset in FREE_TEXT_DATASETS:
        for column in _text_columns(dataset):
            per_gen = [results[dataset][column][g] for g in GENERATORS]
            row = {}
            for field, _ in LINGUISTIC_FIELDS:
                row[f"{field}_real"] = per_gen[0][f"{field}_real"]
                row[f"{field}_synth"] = statistics.mean(v[f"{field}_synthetic"] for v in per_gen)
            out[(dataset, column)] = row
    return out


def linguistic_quality_table() -> str:
    rows = []
    for (dataset, column), m in linguistic_quality().items():
        label = f"\\ds{DATASET_MACRO[dataset]} ({COLUMN_LABEL.get(column, column)})".replace("_", "\\_")
        cells = " & ".join(
            f"{m[f'{field}_real']:.{nd}f} & {m[f'{field}_synth']:.{nd}f}" for field, nd in LINGUISTIC_FIELDS
        )
        rows.append(f"{label:<32s} & {cells} \\\\")
    return "\n".join(rows)


def linguistic_numbers() -> str:
    q = linguistic_quality()
    more_repetitive = [c for c, m in q.items() if m["distinct2_synth"] < m["distinct2_real"]]
    higher_mattr = [c for c, m in q.items() if m["mattr_synth"] > m["mattr_real"]]
    return (f"Distinct-2 synthetic < real (more repetitive) in {len(more_repetitive)} of {len(q)}; "
            f"exceptions: {[c for c in q if c not in more_repetitive]}\n"
            f"MATTR synthetic > real in {len(higher_mattr)} of {len(q)}: {higher_mattr}")


def judge_accuracy_by_column() -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    """Column-level accuracy (%) per (dataset, column) and judge: mean and SD over the k=5 generators."""
    out = {}
    for dataset in FREE_TEXT_DATASETS:
        for column in _text_columns(dataset):
            out[(dataset, column)] = {}
            for judge in JUDGES:
                acc = [
                    100 * _entry_accuracy(_judge_block(dataset, g, judge), column, f"{dataset} | {g} | {judge} | {column}")
                    for g in GENERATORS
                ]
                out[(dataset, column)][judge] = _mean_sd(acc)
    return out


def judge_accuracy_table() -> str:
    rows = []
    for (dataset, column), by_judge in judge_accuracy_by_column().items():
        label = f"\\ds{DATASET_MACRO[dataset]} ({COLUMN_LABEL.get(column, column)})".replace("_", "\\_")
        best = max(m for m, _ in by_judge.values())
        cells = []
        for judge in JUDGES:
            m, sd = by_judge[judge]
            cell = f"{m:.1f} $\\pm$ {sd:.1f}"
            cells.append(f"\\textbf{{{cell}}}" if round(m, 1) == round(best, 1) else cell)
        rows.append(f"{label}\n    & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows)


def judge_accuracy_numbers() -> str:
    q = judge_accuracy_by_column()
    best_count = {j: 0 for j in JUDGES}
    for by_judge in q.values():
        best = max(m for m, _ in by_judge.values())
        for j in JUDGES:
            best_count[j] += round(by_judge[j][0], 1) == round(best, 1)
    top = max(((c, j, m) for c, bj in q.items() for j, (m, _) in bj.items()), key=lambda x: x[2])
    spread = sorted(((max(m for m, _ in bj.values()) - min(m for m, _ in bj.values()), c) for c, bj in q.items()), reverse=True)
    near_chance = [c for c, bj in q.items() if max(m for m, _ in bj.values()) < 60]
    return (f"most accurate judge per column (ties counted for each): {best_count}\n"
            f"highest cell: {top}\n"
            f"largest judge spread (max-min): {[(round(s,1), c) for s, c in spread[:4]]}\n"
            f"columns where every judge < 60%: {near_chance}")


def _by_generator_rows(rows: list[tuple[str, str, str | None]]) -> str:
    """Rows of (label, dataset, column or None for whole row): accuracy (%) of each judge,
    grouped by generator; the most accurate judge within each generator group is in bold."""
    lines = []
    for label, dataset, column in rows:
        cells = []
        for generator in GENERATORS:
            acc = [100 * _entry_accuracy(_judge_block(dataset, generator, j), column, f"{dataset}|{generator}|{j}")
                   for j in JUDGES]
            best = max(round(a, 1) for a in acc)
            cells += [f"\\textbf{{{a:.1f}}}" if round(a, 1) == best else f"{a:.1f}" for a in acc]
        lines.append(f"{label:<28s} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def judge_bygen_table() -> str:
    rows = [(f"\\ds{DATASET_MACRO[d]} ({COLUMN_LABEL.get(c, c)})".replace("_", "\\_"), d, c)
            for d in FREE_TEXT_DATASETS for c in _text_columns(d)]
    return _by_generator_rows(rows)


def judge_wholerow_bygen_table() -> str:
    return _by_generator_rows([(f"\\ds{DATASET_MACRO[d]}", d, None) for d in FREE_TEXT_DATASETS])


def retrieval_text_tables() -> str:
    """Retrieval ablation (Gfl, k=5 vs k=0) per free-text column. Main table: MMD ratio, BERTScore F1,
    and mean word count (real, k=5, k=0); appendix table: Frechet ratio and MAUVE. The better of k=5
    and k=0 is in bold (word count: closer to real). Sign tests are test (v) of significance_tests.json."""
    sig = json.loads(SIGNIFICANCE_JSON.read_text())["retrieval_text_quality"]
    pairs = {m: {(p["dataset"], p["column"]): (p["k5"], p["k0"]) for p in r["pairs"]} for m, r in sig.items()}
    linguistic = json.loads(LINGUISTIC_JSON.read_text())
    gfl = GENERATORS[0]

    def pair(metric: str, key: tuple[str, str], nd: int, lower_is_better: bool) -> str:
        a, b = pairs[metric][key]
        fa, fb = f"{a:.{nd}f}", f"{b:.{nd}f}"
        if fa == fb:  # equal at the printed precision: no bold
            return f"{fa} & {fb}"
        a_wins = (a < b) == lower_is_better
        return f"\\textbf{{{fa}}} & {fb}" if a_wins else f"{fa} & \\textbf{{{fb}}}"

    main_rows, app_rows = [], []
    for dataset in FREE_TEXT_DATASETS:
        for column in _text_columns(dataset):
            key = (dataset, column)
            label = f"\\ds{DATASET_MACRO[dataset]} ({COLUMN_LABEL.get(column, column)})".replace("_", "\\_")
            wc = linguistic[dataset][column]
            real, w5, w0 = (wc[gfl]["wc_mean_real"], wc[gfl]["wc_mean_synthetic"], wc[f"{gfl}_k0"]["wc_mean_synthetic"])
            words = (f"\\textbf{{{w5:.1f}}} & {w0:.1f}" if abs(w5 - real) <= abs(w0 - real)
                     else f"{w5:.1f} & \\textbf{{{w0:.1f}}}")
            main_rows.append(f"{label:<32s} & {pair('mmd', key, 1, True)} & {pair('bertscore', key, 3, False)} "
                             f"& {real:.1f} & {words} \\\\")
            app_rows.append(f"{label:<32s} & {pair('frechet', key, 1, True)} & {pair('mauve', key, 3, False)} \\\\")
    tests = ", ".join(f"{m}: {r['k5_better']} vs {r['k0_better']} (ties {r['ties']}), p={r['p_two_sided']:.3f}"
                      for m, r in sig.items())
    return "\n".join(main_rows) + "\n\n" + "\n".join(app_rows) + "\n\n" + tests


TIMING_JSON = REPORTS_DIR / "timing_benchmark_results.json"
TIMING_TRACKS = {"Gfl": "Gfl", "Gfl-8": "Gfl-8", "GPT": "gpt-6-luna", "Q14": "Qwen-14B"}
# Standard list prices (USD per million input, output tokens), for the generation cost of the hosted tracks.
PRICES = {"Gfl": (0.25, 1.50), "Gfl-8": (0.25, 1.50), "gpt-6-luna": (0.10, 0.50)}


def runtime_cost_tables() -> str:
    """n=500 benchmark: wall-clock time per track; requests, tokens, and cost per hosted track;
    Q14 tokens; judge token usage per k=5 evaluation (one dataset and generator) for each judge."""
    timing = json.loads(TIMING_JSON.read_text())
    time_rows, q14_rows = [], []
    totals = {t: 0.0 for t in TIMING_TRACKS}
    cost_rows = {key: [] for key in PRICES}
    cost_totals = {key: [0, 0, 0, 0.0] for key in PRICES}
    for dataset in FREE_TEXT_DATASETS:
        secs = {t: timing[f"{dataset}|{key}"]["timing"]["total_seconds"] for t, key in TIMING_TRACKS.items()}
        for t in totals:
            totals[t] += secs[t]
        time_rows.append(f"\\ds{DATASET_MACRO[dataset]:<12s} & " + " & ".join(f"{secs[t]:,.1f}" for t in TIMING_TRACKS) + " \\\\")
        for key, (p_in, p_out) in PRICES.items():
            u = timing[f"{dataset}|{key}"]["llm_usage"]
            # Reasoning (thinking) tokens are billed as output tokens.
            cost = (u["prompt_tokens"] * p_in + (u["completion_tokens"] + u.get("thinking_tokens", 0)) * p_out) / 1e6
            for i, v in enumerate((u["requests"], u["prompt_tokens"], u["completion_tokens"], cost)):
                cost_totals[key][i] += v
            cost_rows[key].append(f"\\ds{DATASET_MACRO[dataset]:<12s} & {u['requests']:,} & {u['prompt_tokens']:,} & "
                                  f"{u['completion_tokens']:,} & \\${cost:.2f} \\\\")
        v = timing[f"{dataset}|Qwen-14B"]["llm_usage"]
        q14_rows.append(f"\\ds{DATASET_MACRO[dataset]:<12s} & {v['prompt_tokens'] / 1e6:.2f}M & {v['completion_tokens'] / 1e3:.1f}K \\\\")
    time_rows.append("Total & " + " & ".join(f"{totals[t]:,.1f}" for t in TIMING_TRACKS) + " \\\\")
    cost_text = []
    for key, rows in cost_rows.items():
        r, i, o, c = cost_totals[key]
        rows.append(f"Total & {r:,} & {i:,} & {o:,} & \\${c:.2f} (\\${c / 3000:.4f} per row) \\\\")
        cost_text.append(f"cost {key}\n" + "\n".join(rows))

    judge_tokens = {}
    for judge in JUDGES:
        totals_j = [json.loads((REPORTS_DIR / "judges" / f"{d}_{_safe(g)}_{_safe(judge)}_judge.json").read_text())
                    ["llm_judge"]["usage"] for d in FREE_TEXT_DATASETS for g in GENERATORS]
        total = [u["total_tokens"] for u in totals_j]
        judge_tokens[JUDGE_LABEL[judge]] = (min(total), max(total), sum(total),
                                            sum(u.get("thinking_tokens", 0) for u in totals_j))
    return ("\n".join(time_rows) + "\n\n" + "\n\n".join(cost_text) + "\n\nQ14 tokens\n" + "\n".join(q14_rows)
            + "\n\njudge tokens per evaluation (min, max, total over 18, thinking total): " + str(judge_tokens))


PRIVACY_RUNS = (1, 2, 3)


def _privacy_report(dataset: str, run: int) -> dict:
    return json.loads((REPORTS_DIR / "privacy" / f"{dataset}_run{run}_privacy.json").read_text())


def privacy_tables() -> str:
    """Privacy evaluation, mean over the three NPGC runs fit on the training split.
    Main table: identical-match rate, DCR share, and worst-case MIA without high-cardinality label
    columns (dagger where label columns exist; the footnote gives the MIA on all columns).
    Appendix rows: DCR share, MIA (all columns), and MIA (without label columns) per run.
    Also: every mostlyai-qa repeat with identical matches, against its holdout reference."""
    def fmt(x: float | None) -> str:
        return "--" if x is None else f"{x:.3f}"

    main_rows, app_rows, footnote, repeats = [], [], [], []
    for dataset in ALL_DATASETS:
        runs = [_privacy_report(dataset, r) for r in PRIVACY_RUNS]
        labels = runs[0]["high_cardinality_label_columns"]
        imr = _mean([r["identical_matches"] for r in runs])
        dcr = [r["dcr_share"] for r in runs]
        mia_all = [r["mia"]["mia_auroc_worst_case"] for r in runs]
        mia_wo = [(r["mia_without_label_columns"] or r["mia"])["mia_auroc_worst_case"] for r in runs]
        dagger = "$^\\dagger$" if labels else ""
        main_rows.append(f"\\ds{DATASET_MACRO[dataset]:<12s} & {imr:.3f} & {_mean(dcr):.3f} & {_mean(mia_wo):.3f}{dagger} \\\\")
        app_rows.append(f"\\ds{DATASET_MACRO[dataset]:<12s} & " + " & ".join(fmt(v) for v in dcr + mia_all)
                        + " & " + " & ".join(fmt(v) if labels else "--" for v in mia_wo) + " \\\\")
        if labels:
            footnote.append(f"{DATASET_MACRO[dataset]} {_mean(mia_all):.3f} {labels}")
        for r in runs:
            for q in r["qa_runs"]:
                if q["identical_matches"] > 0 or q["identical_matches_holdout"] > 0:
                    repeats.append(f"{DATASET_MACRO[dataset]} run{r['run']}: training {q['identical_matches']}, "
                                   f"holdout {q['identical_matches_holdout']}")
    return ("\n".join(main_rows) + "\n\n" + "\n".join(app_rows)
            + "\n\nMIA on all columns (datasets with label columns): " + "; ".join(footnote)
            + "\n\nrepeats with identical matches:\n" + "\n".join(repeats))


def great_baseline_coordinates() -> str:
    """Coordinates for writing/images/great_baseline_comparison.tex: overall structured fidelity of
    Qseven (first plot) and NIFTSy Gfl k=5 (second plot), all nine datasets."""
    pairs = {p["dataset"]: p for p in json.loads(SIGNIFICANCE_JSON.read_text())["qseven_structured_fidelity"]["pairs"]}
    out = []
    for key, name in (("qseven", "Qseven"), ("niftsy", "NIFTSy")):
        out.append(f"% {name}")
        out += [f"    ({pairs[d][key]:.3f},{DATASET_LABEL[d]})" for d in ALL_DATASETS]
    return "\n".join(out)


def great_baseline_tables() -> str:
    """Whole-row judge accuracy, NIFTSy Gfl (k=5) vs Qseven: main table (mean ± SD over judges)
    and per-judge appendix rows (lower is better, in bold), plus the fidelity differences."""
    sig = json.loads(SIGNIFICANCE_JSON.read_text())
    pairs = {(p["dataset"], p["judge"]): (100 * p["niftsy"], 100 * p["qseven"]) for p in sig["qseven_whole_row"]["pairs"]}

    def bold(a: float, b: float) -> tuple[str, str]:
        fa, fb = f"{a:.1f}", f"{b:.1f}"
        return (f"\\textbf{{{fa}}}" if a <= b else fa, f"\\textbf{{{fb}}}" if b <= a else fb)

    main_rows, app_rows = [], []
    for dataset in ALL_DATASETS:
        n = _mean_sd([pairs[(dataset, j)][0] for j in JUDGES])
        q = _mean_sd([pairs[(dataset, j)][1] for j in JUDGES])
        bn, bq = bold(round(n[0], 1), round(q[0], 1))
        main_rows.append(f"\\ds{DATASET_MACRO[dataset]:<12s} & {bn} $\\pm$ {n[1]:.1f} & {bq} $\\pm$ {q[1]:.1f} "
                         f"& ${n[0] - q[0]:+.1f}$ \\\\")
        cells = [c for j in JUDGES for c in bold(round(pairs[(dataset, j)][0], 1), round(pairs[(dataset, j)][1], 1))]
        app_rows.append(f"\\ds{DATASET_MACRO[dataset]:<12s} & " + " & ".join(cells) + " \\\\")
    w = sig["qseven_whole_row"]
    qseven_wins = [(d, JUDGE_LABEL[j]) for (d, j), (a, b) in pairs.items() if b < a]
    fid = sig["qseven_structured_fidelity"]
    diffs = sorted((p["niftsy"] - p["qseven"], p["dataset"], p["niftsy"], p["qseven"]) for p in fid["pairs"])
    return ("\n".join(main_rows) + "\n\n" + "\n".join(app_rows)
            + f"\n\nwhole row: niftsy better {w['niftsy_better']}, qseven better {w['qseven_better']}, ties {w['ties']}, p={w['p_two_sided']:.2f}"
            + f"\nqseven better pairs: {qseven_wins}"
            + f"\nfidelity: niftsy better {fid['niftsy_better']} of {fid['n_non_tied']}, p={fid['p_two_sided']:.3f}"
            + "\nfidelity differences (niftsy - qseven): "
            + ", ".join(f"{d}: {x:+.3f} ({a:.3f} vs {b:.3f})" for x, d, a, b in diffs))


def retrieval_ablation_tables() -> str:
    """Main ablation table (mean ± SD over judges, k=5 vs k=0), per-judge appendix rows, and the
    word-count table (mean over each dataset's text columns), all for the Gfl generator."""
    ablation = json.loads(SIGNIFICANCE_JSON.read_text())["retrieval_ablation"]
    pairs = {(p["dataset"], p["judge"]): (100 * p["k5"], 100 * p["k0"]) for p in ablation["pairs"]}

    def bold(a: float, b: float) -> tuple[str, str]:
        fa, fb = f"{a:.1f}", f"{b:.1f}"
        return (f"\\textbf{{{fa}}}" if a <= b else fa, f"\\textbf{{{fb}}}" if b <= a else fb)

    main_rows, app_rows, wc_rows = [], [], []
    linguistic = json.loads(LINGUISTIC_JSON.read_text())
    gfl = GENERATORS[0]
    for dataset in FREE_TEXT_DATASETS:
        k5 = _mean_sd([pairs[(dataset, j)][0] for j in JUDGES])
        k0 = _mean_sd([pairs[(dataset, j)][1] for j in JUDGES])
        b5, b0 = bold(round(k5[0], 1), round(k0[0], 1))
        main_rows.append(f"\\ds{DATASET_MACRO[dataset]:<12s} & {b5} $\\pm$ {k5[1]:.1f} & {b0} $\\pm$ {k0[1]:.1f} "
                         f"& ${k5[0] - k0[0]:+.1f}$ \\\\")
        cells = [c for j in JUDGES for c in bold(round(pairs[(dataset, j)][0], 1), round(pairs[(dataset, j)][1], 1))]
        app_rows.append(f"\\ds{DATASET_MACRO[dataset]:<12s} & " + " & ".join(cells) + " \\\\")
        cols = [linguistic[dataset][c] for c in _text_columns(dataset)]
        real = _mean([c[gfl]["wc_mean_real"] for c in cols])
        s5 = _mean([c[gfl]["wc_mean_synthetic"] for c in cols])
        s0 = _mean([c[f"{gfl}_k0"]["wc_mean_synthetic"] for c in cols])
        w5, w0 = (f"\\textbf{{{s5:.1f}}}", f"{s0:.1f}") if abs(s5 - real) <= abs(s0 - real) else (f"{s5:.1f}", f"\\textbf{{{s0:.1f}}}")
        wc_rows.append(f"\\ds{DATASET_MACRO[dataset]:<12s} & {real:.1f} & {w5} & {real:.1f} & {w0} \\\\")
    by_judge = {JUDGE_LABEL[j]: (sum(pairs[(d, j)][0] < pairs[(d, j)][1] for d in FREE_TEXT_DATASETS),
                                 sum(pairs[(d, j)][0] > pairs[(d, j)][1] for d in FREE_TEXT_DATASETS))
                for j in JUDGES}
    summary = (f"k5 better {ablation['k5_better']}, k0 better {ablation['k0_better']}, ties {ablation['ties']}, "
               f"p={ablation['p_two_sided']:.2f}\nper judge (k5 better, k0 better): {by_judge}")
    return "\n".join(main_rows) + "\n\n" + "\n".join(app_rows) + "\n\n" + "\n".join(wc_rows) + "\n\n" + summary


TSTR_DATASETS = ["adult_with_three", "Airbnb", "fake_job_postings", "netflix_titles"]
TSTR_METRICS = ("roc_auc", "accuracy", "balanced_accuracy", "f1_weighted")


def tstr_real_baseline(dataset: str) -> dict:
    """Real-trained baseline: the TSTR classifier of run_structured_metrics.py trained on the real
    training split instead of synthetic data (same cap, split, features, and holdout)."""
    import pandas as pd
    import yaml
    from sklearn.model_selection import train_test_split

    from source.metrics.structured_metrics import evaluate_tstr

    cfg = yaml.safe_load((REPO_ROOT / "configs" / "metrics.yml").read_text())
    seed = int(cfg.get("random_state", 42))
    meta = json.loads((REPO_ROOT / ".metadata" / dataset / "metadata.json").read_text())
    real_df = pd.read_csv(REPO_ROOT / cfg["real_data_path"].format(dataset_name=dataset))
    if len(real_df) > 20_000:
        real_df = real_df.sample(20_000, random_state=seed).reset_index(drop=True)
    real_train, real_holdout = train_test_split(real_df, test_size=float(cfg.get("holdout_fraction", 0.2)),
                                                random_state=seed)
    return evaluate_tstr(
        train_df=real_train, synthetic_df=real_train, holdout_df=real_holdout,
        target_column=meta["target_column"], text_columns=meta["free_text_columns"],
        structured_weight=float(cfg.get("tstr_structured_weight", 1.0)),
        text_weight_default=float(cfg.get("tstr_text_weight_default", 0.01)),
        text_weight_by_column=cfg.get("tstr_text_weight_by_column") or {},
        text_embedding_dim=int(cfg.get("text_embedding_dim", 768)),
        text_max_features=int(cfg.get("text_max_features", 200_000)),
    )


def tstr_tables() -> str:
    """Main TSTR table (mean ± SD across generators vs real baseline), by-generator appendix rows,
    and the real baseline's accuracy/F1."""
    target_label = {"adult_with_three": "income", "Airbnb": "superhost", "fake_job_postings": "fraudulent",
                    "netflix_titles": "type"}
    main_rows, app_rows, real_text, spreads = [], [], [], {}
    for dataset in TSTR_DATASETS:
        tstr = {g: _structured_report(dataset, g)["tstr"] for g in GENERATORS}
        real = tstr_real_baseline(dataset)
        auc, bal = _mean_sd([t["roc_auc"] for t in tstr.values()]), _mean_sd([t["balanced_accuracy"] for t in tstr.values()])
        name = f"\\ds{DATASET_MACRO[dataset]} ({target_label[dataset]})"
        main_rows.append(f"{name:<28s} & ${auc[0]:.3f} \\pm {auc[1]:.3f}$ & {real['roc_auc']:.3f} & "
                         f"${bal[0]:.3f} \\pm {bal[1]:.3f}$ & {real['balanced_accuracy']:.3f} \\\\")
        cells = [f"{tstr[g][m]:.2f}" for g in GENERATORS for m in TSTR_METRICS]
        app_rows.append(f"\\ds{DATASET_MACRO[dataset]:<12s} & " + " & ".join(cells) + " \\\\")
        real_text.append(f"{DATASET_MACRO[dataset]} acc/F1 {real['accuracy']:.3f}/{real['f1_weighted']:.3f}, "
                         f"AUC gap {real['roc_auc'] - auc[0]:.3f}")
        spreads[DATASET_MACRO[dataset]] = round(max(t["roc_auc"] for t in tstr.values())
                                                - min(t["roc_auc"] for t in tstr.values()), 3)
    return ("\n".join(main_rows) + "\n\n" + "\n".join(app_rows) + "\n\n" + "\n".join(real_text)
            + f"\nAUC range across generators: {spreads}")


def self_judge_numbers() -> str:
    """Self-judge comparisons (whole row, k=5), single-class whole-row evaluations, and QC flag counts."""
    from scipy.stats import binomtest

    lines, higher_total, lower_total = [], 0, 0
    for judge in (g for g in GENERATORS if g in JUDGES):
        higher = lower = ties = 0
        for dataset in FREE_TEXT_DATASETS:
            acc = {g: 100 * _entry_accuracy(_judge_block(dataset, g, judge), None, f"{dataset}|{g}|{judge}")
                   for g in GENERATORS}
            own = round(acc[judge], 1)
            other = round(_mean([acc[g] for g in GENERATORS if g != judge]), 1)
            higher += own > other
            lower += own < other
            ties += own == other
        lines.append(f"{JUDGE_LABEL[judge]}: own higher {higher}, lower {lower}, ties {ties}")
        higher_total, lower_total = higher_total + higher, lower_total + lower
    p = binomtest(higher_total, higher_total + lower_total).pvalue
    lines.append(f"total: own higher {higher_total} of {higher_total + lower_total} non-tied, sign test p={p:.2f}")

    whole_row, single_class = 0, []
    for path in sorted((REPORTS_DIR / "judges").glob("*_judge.json")):
        entry = json.loads(path.read_text())["llm_judge"].get("whole_row")
        if not isinstance(entry, dict) or not entry.get("n_examples"):
            continue
        whole_row += 1
        if entry.get("predicted_real", 0) == 0 or entry.get("predicted_synthetic", 0) == 0:
            single_class.append(path.stem.removesuffix("_judge"))
    lines.append(f"whole-row evaluations: {whole_row}; single-class: {len(single_class)} {single_class}")

    qc = json.loads(SIGNIFICANCE_JSON.read_text())["quality_control"]
    degenerate = qc["degenerate"]
    lines.append(f"degenerate: {len(degenerate)} ({sum(e['evaluation'].startswith('adult_with_three') for e in degenerate)} on Adult); "
                 f"parse failures > 5%: {len(qc['parse_failures'])} {[e['evaluation'] for e in qc['parse_failures']]}")
    return "\n".join(lines)


def main() -> None:
    # Everything is printed: figure coordinates for writing/images/*.tex, table rows and
    # inline numbers for writing/main.tex.
    for make in (judge_by_dataset_coordinates, fidelity_coordinates, great_baseline_coordinates,
                 judge_pairwise_table, joint_fidelity_table, fidelity_numbers, full_quality_table, free_text_numbers,
                 linguistic_quality_table, linguistic_numbers, judge_accuracy_table, judge_accuracy_numbers,
                 judge_bygen_table, judge_wholerow_bygen_table, self_judge_numbers, tstr_tables,
                 retrieval_ablation_tables, great_baseline_tables, privacy_tables, runtime_cost_tables,
                 retrieval_text_tables, appendix_fidelity_tables, linguistic_bygen_table):
        print(f"\n% --- {make.__name__} ---\n{make()}")


if __name__ == "__main__":
    main()

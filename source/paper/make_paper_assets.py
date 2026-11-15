"""Generate the paper's tables, figures, and inline numbers from reports/.

Every number in the paper is computed here from the JSON reports written by the
experiment scripts, so the paper can be reproduced from reports/ alone.
Accuracy, sign tests, and evaluation sets use the same helpers as
source/metrics/run_significance_tests.py.

Usage:
    python -m source.paper.make_paper_assets

Output: writing/images/*.tex (standalone figures, compiled to PDF); tables are
printed to stdout for pasting into writing/main.tex.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import subprocess

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
IMAGES_DIR = REPO_ROOT / "writing" / "images"
SIGNIFICANCE_JSON = REPO_ROOT / "reports" / "significance_tests.json"
TINYTEX_BIN = pathlib.Path.home() / ".TinyTeX" / "bin" / "x86_64-linux"

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
    "gemini-3.1-flash-lite": "Gflash",
    "gemini-4-argon": "Gpro",
    "gpt-6-luna": "GPT",
    "Qwen/Qwen2.5-14B-Instruct": "Q14b",
}
# One marker and color per judge (colorblind-safe Okabe-Ito colors).
JUDGE_STYLE = {
    "gemini-3.1-flash-lite": ("o", "0072B2"),
    "gemini-4-argon": ("square*", "D55E00"),
    "gpt-6-luna": ("triangle*", "009E73"),
    "Qwen/Qwen2.5-14B-Instruct": ("diamond*", "CC79A7"),
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


def judge_by_dataset_figure() -> pathlib.Path:
    """Horizontal dot plot: judge accuracy by dataset, column-level and whole-row panels."""
    panels = judge_accuracy_by_dataset()
    rows = FREE_TEXT_DATASETS + ["All"]
    # Top-to-bottom order: datasets as listed, "All" last.
    y_of = {row: len(rows) - i for i, row in enumerate(rows)}
    y_labels = ",".join(DATASET_LABEL.get(r, r) if r != "All" else r"\textbf{All}" for r in rows)
    y_ticks = ",".join(str(y_of[r]) for r in rows)

    def panel(name: str, title: str, first: bool) -> str:
        plots = []
        for judge in JUDGES:
            mark, color = JUDGE_STYLE[judge]
            coords = " ".join(f"({panels[name][r][judge]:.1f},{y_of[r]})" for r in rows)
            legend = f"\\addlegendentry{{{JUDGE_LABEL[judge]}}}" if first else ""
            plots.append(
                f"\\addplot[only marks, mark={mark}, mark size=3.6pt, color={{rgb,255:red,{int(color[0:2],16)};"
                f"green,{int(color[2:4],16)};blue,{int(color[4:6],16)}}}] coordinates {{{coords}}};{legend}"
            )
        yticklabels = f"yticklabels={{{y_labels}}}," if first else "yticklabels={},"
        return (
            f"\\nextgroupplot[title={{{title}}}, {yticklabels}]\n"
            f"\\draw[dashed, gray] (axis cs:50,0.4) -- (axis cs:50,{len(rows) + 0.6});\n"
            f"\\draw[gray!40] (axis cs:30,1.5) -- (axis cs:100,1.5);\n"
            + "\n".join(plots)
        )

    tex = rf"""\documentclass[crop,tikz]{{standalone}}
\usepackage{{pgfplots}}
\usetikzlibrary{{pgfplots.groupplots}}
\pgfplotsset{{compat=1.15}}
% AUTO-GENERATED by source/paper/make_paper_assets.py from reports/judges -- do not edit.
\begin{{document}}
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{group size=2 by 1, horizontal sep=0.6cm}},
    width=8.2cm, height=6.2cm,
    xmin=30, xmax=100, ymin=0.4, ymax={len(rows) + 0.6},
    xtick={{30,40,50,60,70,80,90,100}},
    ytick={{{y_ticks}}},
    xlabel={{Judge accuracy (\%)}},
    xmajorgrids, grid style={{dashed, gray!25}},
    tick label style={{font=\small}}, title style={{font=\small\bfseries}},
    legend style={{font=\small, legend columns=4, at={{(1.03,1.16)}}, anchor=south, draw=none}},
]
{panel("column", "Column-level", True)}
{panel("whole_row", "Whole-row", False)}
\end{{groupplot}}
\end{{tikzpicture}}
\end{{document}}
"""
    path = IMAGES_DIR / "judge_by_dataset.tex"
    path.write_text(tex)
    return path


def compile_figure(tex_path: pathlib.Path) -> None:
    env = {"PATH": f"{TINYTEX_BIN}:/usr/bin:/bin"}
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tex_path.parent, env=env, check=True, stdout=subprocess.DEVNULL,
    )
    for ext in (".aux", ".log"):
        tex_path.with_suffix(ext).unlink(missing_ok=True)


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


def fidelity_figure() -> pathlib.Path:
    """Horizontal bar chart of overall structured fidelity (mean of the three k=5 generators)."""
    fidelity = structured_fidelity()
    labels = [DATASET_LABEL[d] for d in ALL_DATASETS]
    coords = "\n".join(f"    ({fidelity[d]['overall_accuracy'][0]:.3f},{DATASET_LABEL[d]})" for d in ALL_DATASETS)
    y_coords = ",\n        ".join(reversed(labels))  # first dataset at the top
    tex = rf"""\documentclass[crop,tikz]{{standalone}}
\usepackage[dvipsnames]{{xcolor}}
\usepackage{{pgfplots}}
\pgfplotsset{{
    compat=1.15,
    every axis label/.append style={{font=\small\bfseries}},
    every tick label/.append style={{font=\small}},
    legend style={{font=\small}},
    grid style={{dashed, gray!30}},
}}
\definecolor{{niftsyncolor}}{{HTML}}{{7B2CBF}}
% AUTO-GENERATED by source/paper/make_paper_assets.py from reports/*_structured_metrics.json -- do not edit.
\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
    xbar, bar width=6pt, xmin=0.0, xmax=1.00, width=13cm, height=8cm,
    grid=major, enlarge x limits=false,
    axis x line*=bottom, axis y line*=left,
    xlabel={{Overall Structured Fidelity}},
    symbolic y coords={{
        {y_coords}
    }},
    ytick=data,
    yticklabel style={{font=\normalsize}}, xticklabel style={{font=\normalsize}},
    legend image code/.code={{\draw[#1, draw=none] (0cm,-0.05cm) rectangle (0.3cm,0.1cm);}},
    legend style={{at={{(0.98,0.02)}}, anchor=south east, font=\small, draw=gray!50}},
]
\addplot[fill=niftsyncolor!70, draw=niftsyncolor] coordinates {{
{coords}
}};
\legend{{NIFTSy $k=5$}}
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""
    path = IMAGES_DIR / "fidelity.tex"
    path.write_text(tex)
    return path


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


def main() -> None:
    for make in (judge_by_dataset_figure, fidelity_figure):
        path = make()
        compile_figure(path)
        print(f"Wrote {path.relative_to(REPO_ROOT)} and its PDF")
    # Tables are printed for pasting into writing/main.tex.
    for make in (judge_pairwise_table, joint_fidelity_table, fidelity_numbers):
        print(f"\n% --- {make.__name__} ---\n{make()}")


if __name__ == "__main__":
    main()

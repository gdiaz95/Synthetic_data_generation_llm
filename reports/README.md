# Reports

Every number, table, and figure in the paper comes from the JSON files in this
directory. `source/paper/make_paper_assets.py` reads them and prints each table,
inline number, and figure coordinate:

```bash
poetry run python -m source.paper.make_paper_assets
```

Each block of its output starts with `% --- <function name> ---`; the table
below says which block belongs to which part of the paper.

## How the reports were produced

All reports are written by the two experiment scripts:

| Script | Steps |
|---|---|
| `experiments/run_metrics_billing.sh` | Hosted-API steps: Flash and Luna generation, Flash and Argon and Luna judges, hosted runtime benchmark |
| `experiments/run_metrics_no_billing.sh` | Local steps: Q-14B and GReaT (Q-7B) generation, Q-14B judge, structured metrics, privacy, BERTScore/MAUVE, linguistic quality, local runtime benchmark, significance tests |

Each step is skipped when its report already exists, so rerunning a script only
fills in what is missing. `run_manifest.csv` is the record of a final run of both
scripts over the complete reports: every step is `SKIP (already complete)`
except the significance tests, which always rerun (`PASS`). It shows that the
two scripts account for every report here.

## Files

Model names in file names: `gemini-3.1-flash-lite` (Flash), `gpt-6-luna`
(Luna), `Qwen-Qwen2.5-14B-Instruct` (Q-14B), `Qwen-Qwen2.5-7B-Instruct_great`
(the GReaT baseline, Q-7B), `gemini-4-argon` (Argon, judge only). The suffix
`_k0` marks the retrieval ablation (Flash without retrieval).

| Path | Contents |
|---|---|
| `{dataset}_{generator}_structured_metrics.json` | Structured fidelity (`metrics_qa`), joint discriminator, Fréchet/MMD distances (`free_text_distribution`), TSTR |
| `llm_metrics/{dataset}_{generator}_llm_text_metrics.json` | BERTScore and MAUVE per free-text column |
| `linguistic_quality/results.json` | Word count, MATTR, and Distinct-2 per dataset, column, and generator |
| `judges/{dataset}_{generator}_{judge}_judge.json` | LLM judge results (column-level and whole-row) and token usage |
| `privacy/{dataset}_run{1,2,3}_privacy.json` | Identical-match rate (with holdout references), DCR share, MIA, for NPGC fit on the training split |
| `timing_benchmark_results.json` | n=500 runtime benchmark: time and token usage per dataset and track (`Gfl`, `Gfl-8`, `gpt-6-luna`, `Qwen-14B`) |
| `significance_tests.json` | Sign tests (i)–(v) and the judge quality-control flags |
| `run_manifest.csv` | Final run record of both experiment scripts (see above) |

## Paper element → output block of `make_paper_assets.py`

| Paper | Block |
|---|---|
| Fidelity figure (`images/fidelity.tex`) | `fidelity_coordinates` |
| Fidelity numbers, joint-fidelity table | `fidelity_numbers`, `joint_fidelity_table` |
| Free-text quality table and numbers | `full_quality_table`, `free_text_numbers` |
| Linguistic quality table and numbers | `linguistic_quality_table`, `linguistic_numbers` |
| Downstream utility (TSTR) tables, including the real-trained baseline | `tstr_tables` |
| Judge figure (`images/judge_by_dataset.tex`) | `judge_by_dataset_coordinates` |
| Judge pairwise sign tests | `judge_pairwise_table` |
| Judge accuracy tables and numbers (appendix) | `judge_accuracy_table`, `judge_accuracy_numbers`, `judge_bygen_table`, `judge_wholerow_bygen_table` |
| Self-judge bias and quality control (appendix) | `self_judge_numbers` |
| Retrieval ablation: free-text table (main) and Fréchet/MAUVE table (appendix) | `retrieval_text_tables` |
| Retrieval ablation: judge table (appendix) | `retrieval_ablation_tables` |
| GReaT figure (`images/great_baseline_comparison.tex`) | `great_baseline_coordinates` |
| GReaT whole-row judge tables | `great_baseline_tables` |
| Privacy tables | `privacy_tables` |
| Runtime and cost tables, judge token usage | `runtime_cost_tables` |
| Fidelity components, fidelity by generator, joint fidelity by generator (appendix) | `appendix_fidelity_tables` |
| Linguistic quality by generator (appendix) | `linguistic_bygen_table` |

The TSTR real-trained baseline is the only value computed from the real data
rather than from a report: `tstr_tables` retrains the same classifier on the real
training split (same cap, split, and seed as `run_structured_metrics.py`), so it
needs `data/raw/`.

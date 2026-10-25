# Hyperparameters and implementation settings

This table mirrors the hyperparameters table that was previously reported in
the paper's appendix (`Appendix: Generator hyperparameters`). It is kept here
in full so the released configuration remains fully documented even though
the paper itself now only points here instead of reproducing the full table.

Values are fixed across datasets unless otherwise indicated. The maximum
output length $T_{\max}$ can be specified separately for each free-text
column (via dataset metadata).

| Component | Hyperparameter | Value |
|---|---|---|
| NPGC | DP budget ε | 1.0 |
| NPGC | enforce min/max bounds | true |
| Retrieval | top-k size | 5 |
| Retrieval | all text distance weights λ | 0.01 |
| Retrieval | top-k batch size | 512 synthetic rows |
| LLM generation | sampling temperature | 0.8 |
| LLM generation | nucleus top-p | 0.95 |
| LLM generation | maximum output tokens | 2048 |
| LLM generation | delay between API calls | 1.0 s |
| LLM generation | vLLM batch size | 64 (reduced to 32 for Adult, Jobs, and Netflix to avoid an out-of-memory error during the full-scale run; the n=500 runtime benchmark uses 64 uniformly since its shorter runs do not trigger this issue) |
| LLM generation | parallel API shards | 64 |
| LLM generation | failed-row retry passes | 1 |
| LLM generation | GPU memory utilization | 0.80 |
| LLM generation | local maximum model length | 4096 tokens |
| Prompt | maximum output words T_max | 80, per-column override allowed |
| Prompt | retrieved row excerpt cap | 20 words per example |
| Prompt | retrieved row reading budget | 250 words |
| Synthetic set | n_syn / n_train | 1.25 |
| Data split | holdout fraction | 0.20 |
| Randomization | random seed | 42 |
| Free-text metrics | TF-IDF maximum features | 200,000, including bigrams |
| Free-text metrics | SVD embedding dimension | 768 |
| Free-text metrics | MMD batch size | 4096 |
| LLM judge | examples per class | 128 real, 128 synthetic |
| LLM judge | batch size | 32 |
| LLM judge | thinking-model maximum output tokens | 4096 |
| LLM judge | non-thinking maximum output tokens | 48 |
| TSTR | structured block weight | 1.0 |
| TSTR | text block weight | 0.01 |
| MIA | sampled row cap | 20,000 per dataset |

We did not tune ε, k, any λ, the LLM sampling temperature, or the prompt
template separately for individual datasets. These values were selected
during pipeline development using an excluded debug dataset and
subsequently fixed for all reported experiments.

Feature weights were fixed to 1.0 for all structured columns in every
dataset except Netflix, where `country` uses w=5 to prioritize retrieved
rows with a similar country of origin, since Netflix titles span multiple
languages; this value was not tuned. For this one dataset, the structured
distance's [0,4] bound does not strictly hold.

The maximum output length T_max is the only parameter that varies across
free-text columns, set according to the natural length of each text field
rather than optimized against the evaluation metrics.

See `configs/tabular.yml`, `configs/llm_generation.yml`, and
`configs/metrics.yml` for the corresponding live configuration files, and
`docs_config_reference.md` for a field-by-field description of what each
config value controls in the code.

## GReaT (Q7G) baseline training configuration

This mirrors the training-configuration table that was previously reported
in the paper's GReaT ablation appendix. The baseline fine-tunes Qwen2.5-7B-Instruct
separately on row-serialized examples from each dataset
(`source/pipelines/run_great.py`), using a batch size of 16 and a maximum
sequence length of 256 tokens for all runs.

Training epochs are chosen from training-set size by a fixed schedule
(`_default_epochs` in `run_great.py`), not tuned per dataset: datasets with
at most 2,000 training rows use 20 epochs, datasets with more than 2,000 and
at most 5,000 rows use 15 epochs, and datasets with more than 5,000 rows use
10 epochs. Training data is capped at 20,000 rows for Airbnb, Kickstarter,
and Complaint (`_LARGE_DATASET_MAX_ROWS` in `run_great.py`) and at 5,000 rows
for Adult, Jobs, Netflix, and Spotify, to keep training times feasible.

| Dataset | n_train | Epochs | n_synthetic |
|---|---|---|---|
| Adult | 5,000 | 15 | 48,842 |
| Airbnb | 20,000 | 10 | 20,000 |
| Complaint | 20,000 | 10 | 20,000 |
| Jobs | 5,000 | 15 | 17,880 |
| Kickstarter | 20,000 | 10 | 20,000 |
| Netflix | 5,000 | 15 | 7,045 |
| English | 3,580 | 15 | 3,580 |
| Spotify | 5,000 | 15 | 114,000 |
| Student | 510 | 20 | 510 |

The Kickstarter run was repeated after the first attempt produced an overall
structured fidelity of 0.000; the second run obtained 0.0051, confirming the
low structured fidelity persists. Its row-serialized examples contain only
about 70-90 observed tokens within the fixed 256-token training window,
leaving a substantial fraction of each sequence as padding, which may
compound the difficulty of jointly learning the schema and row distribution
with the 7B model.

# Configuration Reference (One-by-One)

This project uses three YAML configuration files in `configs/` for the main generation pipeline, plus environment variables for API access.

## 1) `configs/tabular.yml`

Controls **tabular synthetic data generation** (structured columns).

### `input_csv`
- **What it is:** Path to the source CSV used as the real dataset.
- **Used by:** `run_full.py` before tabular/LLM stages begin.
- **Expected content:** Must include a `job_description` column, because the pipeline synthesizes structured fields separately and later regenerates free text.
- **Current value:** `data/small/adult_with_job_description.csv`

### `output_csv`
- **What it is:** Path where the final synthetic dataset is written.
- **Used by:** End of `run_full.py`.
- **Current value:** `data/processed/adults_synthetic_gemini_small.csv`

### `knn_k`
- **What it is:** Number of nearest real neighbors to retrieve for each synthetic tabular row in latent space.
- **Used by:** `run_tabular_generation(...)` in the KNN retrieval step.
- **Effect:** Higher values consider more real rows when mapping synthetic records back to similar original examples.
- **Also used by:** free-text prompt construction (`generate_free_text_column(...)`) as the number of neighbor snippets.
- **Current value:** `5`

### `epsilon`
- **What it is:** Privacy/noise control parameter passed into the tabular synthesizer (`NPGC_special`).
- **Used by:** Constructor in `run_tabular_generation(...)`.
- **Effect:** Governs perturbation/smoothing behavior in tabular synthesis (model-specific to `NPGC_special`).
- **Current value:** `1.0`

### `enforce_min_max_values`
- **What it is:** Boolean toggle to clamp generated numeric values into observed training min/max ranges.
- **Used by:** `NPGC_special` initialization.
- **Effect:** `true` helps prevent unrealistic out-of-range values in synthetic numeric columns.
- **Current value:** `true`

---

## 2) `configs/llm_generation.yml`

Controls **free-text generation** and model backend behavior.

### `model`
- **What it is:** Model identifier (local model path/name, Gemini model, or OpenAI model).
- **Used by:** `SingleGPUServer` initialization.
- **Auto-routing:** If `provider: auto`, the code infers provider from this string (e.g., names containing `gemini` route to Gemini).
- **Current value:** `gemini-2.5-flash`

### `provider`
- **What it is:** Backend type: `auto`, `local`, `gemini`, or `openai`.
- **Used by:** `SingleGPUServer._resolve_provider(...)`.
- **Current value:** `auto`

### `gpu_memory_utilization`
- **What it is:** Fraction of selected GPU VRAM reserved when running **local vLLM** backend.
- **Used by:** Local vLLM initialization (`gpu_memory_utilization=...`).
- **Note:** Ignored for remote API providers (Gemini/OpenAI).
- **Current value:** `0.80`

### `max_model_len`
- **What it is:** Maximum sequence length for **local vLLM** model context.
- **Used by:** Local vLLM initialization (`max_model_len=...`).
- **Note:** Not applied to Gemini/OpenAI API calls directly.
- **Current value:** `4096`

### `enforce_eager`
- **What it is:** Enables eager execution mode in **local vLLM**.
- **Used by:** Local vLLM initialization (`enforce_eager=...`).
- **Effect:** `false` is usually faster (better graph/cuda optimization). `true` can help compatibility/debugging but often lowers throughput.
- **Current value:** `false`

### `max_words`
- **What it is:** Hard cap on generated `job_description` word count.
- **Used by:** Prompt variable `{max_words}` and post-generation truncation (`cap_words`).
- **Current value:** `80`

### `temperature`
- **What it is:** Sampling randomness for text generation.
- **Used by:** `generate_batch(..., config={...})` passed to provider.
- **Effect:** Higher values increase variability/creativity.
- **Current value:** `0.8`

### `top_p`
- **What it is:** Nucleus sampling threshold.
- **Used by:** Provider generation config.
- **Effect:** Lower values constrain token choices to the highest-probability mass.
- **Current value:** `0.95`

### `max_tokens`
- **What it is:** Maximum output token budget per generation call.
- **Used by:** Provider generation config (`max_tokens` / mapped equivalents).
- **Current value:** `256`

### `sleep_sec`
- **What it is:** Delay between per-row generation calls.
- **Used by:** `time.sleep(...)` after each generated row.
- **Effect:** Helps reduce API throttling/rate-limit bursts.
- **Current value:** `0.1`

---

## 3) `configs/prompts.yml`

Controls prompt template text and neighbor snippet formatting.

### `neighbor_excerpt_words`
- **What it is:** Word limit per retrieved neighbor snippet included in the prompt.
- **Used by:** `build_prompt_from_structured_with_neighbors(...)` via `cap_words`.
- **Current value:** `20`

### `job_description_prompt`
- **What it is:** Main template used to generate each synthetic `job_description`.
- **Template placeholders:**
  - `{max_words}` from `llm_generation.yml:max_words`
  - `{structured_fields}` from each synthetic row
  - `{neighbor_block}` built from nearest-neighbor text snippets
- **Output contract:** Instructs model to return only a single prose job description with privacy/format constraints.

---

## 4) Environment variables (`.env`)

These are not in YAML, but they are required depending on provider.

### `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- Required when provider resolves to Gemini.

### `OPENAI_API_KEY`
- Required when provider resolves to OpenAI.

---

## 5) CLI overrides (highest precedence)

When running `source/pipelines/run_full.py`, these flags override YAML values:
- `--input-csv`, `--output-csv`
- `--model`, `--provider`
- `--gpu-memory-utilization`, `--max-model-len`

So precedence is generally:
1. CLI flag (if passed)
2. YAML config file value
3. Code default (for some fields)

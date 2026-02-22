# Synthetic Data Generation LLM

This project runs a vLLM server and includes GPU-aware model allocation.

## Why you were seeing virtualenv/workflow issues

Common causes in this repo:

- `README.md` was referenced in `pyproject.toml` but missing, which breaks `poetry check` and can affect install flow.
- Poetry virtualenv config was not in `poetry.toml`, so local/CI behavior could differ depending on user-global Poetry settings.

## Make local environment match this repo exactly

Use these commands from repo root:

```bash
# 1) Ensure Poetry uses Python 3.10 for this project
poetry env use 3.10

# 2) Install deps into the project-local venv (.venv)
poetry install

# 3) Verify env path (should be .../Synthetic_data_generation_llm/.venv)
poetry env info

# 4) Run tests via the managed environment
poetry run pytest -q
```

## Reset when environment gets broken

```bash
poetry env remove --all
poetry cache clear . --all
poetry install
```

## Running the server

```bash
poetry run python server_model.py --model mistralai/Mistral-7B-Instruct-v0.3 --tp 1
```

You can also provide:

- `--model-size-gb`
- `--gpus` (allowed GPU IDs)
- `--utilization`


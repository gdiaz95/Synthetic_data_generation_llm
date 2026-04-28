from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Default: 7B fits training on 47 GB (4-bit ~4 GB weights + ~18 GB activations ≈ 22 GB total).
# 14B (4-bit ~8 GB weights + ~38 GB activations) OOMs at any useful batch size on 47 GB.
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_METADATA_ROOT = Path(".metadata")
OUTPUT_DIR = Path("data/processed")
_LARGE_DATASET_MAX_ROWS = {"Kickstarter": 20_000, "Airbnb": 20_000, "consumer_complaint": 20_000}

# Attention is O(n²) in sequence length — keep this as small as possible.
# 256 covers structured-only rows and short text; consumer_complaint needs 512.
# Override per run with --max-seq-len if rows are truncated too aggressively.
MAX_SEQ_LEN = 256


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _serialize_row(row: dict[str, Any], columns: list[str]) -> str:
    parts = []
    for col in columns:
        val = row.get(col, "")
        if val is None or (isinstance(val, float) and pd.isna(val)):
            val = ""
        parts.append(f"{col} is {val}")
    return ", ".join(parts)


def _parse_row(text: str, columns: list[str]) -> dict[str, Any]:
    """Parse 'col1 is val1, col2 is val2, ...' back into a dict.
    Uses column names as delimiters so commas inside values are handled correctly.
    """
    result: dict[str, Any] = {}
    for i, col in enumerate(columns):
        marker = f"{col} is "
        start = text.find(marker)
        if start == -1:
            result[col] = None
            continue
        start += len(marker)
        end = len(text)
        for next_col in columns[i + 1:]:
            pos = text.find(f", {next_col} is ", start)
            if pos != -1:
                end = pos
                break
        result[col] = text[start:end].strip().rstrip(",").strip() or None
    return result


def _load_model_and_tokenizer(model_name: str, gpu_index: int):  # noqa: ARG001 — gpu_index used via CUDA_VISIBLE_DEVICES
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    # Disable gradient checkpointing — saves activation memory at cost of compute,
    # but with 47 GB GPU we have headroom. Disabling gives ~3-5x training speedup.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def _finetune(model, tokenizer, texts: list[str], epochs: int, batch_size: int) -> None:
    import bitsandbytes as bnb
    from torch.utils.data import DataLoader, Dataset

    class _RowDataset(Dataset):
        def __init__(self, ids, masks):
            self.ids = ids
            self.masks = masks
        def __len__(self):
            return len(self.ids)
        def __getitem__(self, i):
            return {"input_ids": self.ids[i], "attention_mask": self.masks[i], "labels": self.ids[i].clone()}

    print(f"Tokenizing {len(texts)} training rows (max_length={MAX_SEQ_LEN})...")
    enc = tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length", return_tensors="pt")
    dataset = _RowDataset(enc["input_ids"], enc["attention_mask"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 8-bit Adam: same convergence, uses ~75% less optimizer memory than fp32 Adam.
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=5e-4)
    device = next(model.parameters()).device

    from tqdm import tqdm
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"  Epoch {epoch}/{epochs} — loss: {total / len(loader):.4f}")


def _generate(model, tokenizer, columns: list[str], n_samples: int, batch_size: int) -> pd.DataFrame:
    import torch

    model.eval()
    device = next(model.parameters()).device
    prompt = f"{columns[0]} is "
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    rows: list[dict[str, Any]] = []
    generated = 0
    while generated < n_samples:
        cur = min(batch_size, n_samples - generated)
        with torch.no_grad():
            out = model.generate(
                prompt_ids.repeat(cur, 1),
                max_new_tokens=MAX_SEQ_LEN,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for seq in out:
            text = prompt + tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            rows.append(_parse_row(text, columns))
        generated += cur
        print(f"  Generated {min(generated, n_samples)}/{n_samples} rows", end="\r")
    print()
    return pd.DataFrame(rows, columns=columns)


def _default_epochs(n_train: int) -> int:
    if n_train <= 2_000:
        return 20
    if n_train <= 5_000:
        return 15
    return 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GReaT-style synthesis with Qwen2.5-14B + QLoRA.")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (auto if omitted).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        # GPU memory breakdown at seq_len=1024 on a 47 GB GPU:
        #   4-bit model weights:        ~8 GB
        #   LoRA adapters + gradients:  ~2 GB
        #   Activations @ batch=8:      ~4 GB  → ~14 GB total  (safe)
        #   Activations @ batch=16:     ~8 GB  → ~18 GB total  (safe, good speed)
        #   Activations @ batch=32:     ~16 GB → ~26 GB total  (safe, faster)
        #   Activations @ batch=64:     ~32 GB → ~42 GB total  (tight, may OOM)
        # Default=16 is a safe balance. Try 32 if you want ~2x speedup.
    )
    parser.add_argument("--max-train-rows", type=int, default=5_000)
    parser.add_argument("--max-seq-len", type=int, default=None, help="Override MAX_SEQ_LEN (default 256; use 512 for consumer_complaint).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Must set before any torch/transformers import
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    global MAX_SEQ_LEN
    if args.max_seq_len:
        MAX_SEQ_LEN = args.max_seq_len

    dataset_name = args.dataset_name

    meta_path = DEFAULT_METADATA_ROOT / dataset_name / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    free_text_cols: list[str] = meta.get("free_text_columns") or []

    tabular_cfg = _read_yaml(Path("configs/tabular.yml"))
    input_csv = Path(str(tabular_cfg["input_csv"]).format(dataset_name=dataset_name))
    if not input_csv.exists():
        input_csv = Path(f"data/raw/{dataset_name}.csv")

    print(f"\n=== GReaT synthesis: {dataset_name} ===")
    print(f"GPU      : {args.gpu_index}")
    print(f"Input    : {input_csv}")

    real_df = pd.read_csv(input_csv, low_memory=False)
    columns = list(real_df.columns)
    print(f"Shape    : {real_df.shape}  |  Free-text cols: {free_text_cols or '(none)'}")

    # Cap training rows for large datasets
    max_train = _LARGE_DATASET_MAX_ROWS.get(dataset_name, args.max_train_rows)
    train_df = real_df.sample(n=min(len(real_df), max_train), random_state=42)
    print(f"Training : {len(train_df)} rows")

    epochs = args.epochs or _default_epochs(len(train_df))
    print(f"Epochs   : {epochs}  |  Batch size: {args.batch_size}")

    # Determine output size (same cap logic as run_full.py)
    n_synthetic = _LARGE_DATASET_MAX_ROWS.get(dataset_name, len(real_df))
    print(f"Synthetic: {n_synthetic} rows to generate")

    texts = [_serialize_row(row, columns) for row in train_df.to_dict("records")]

    print(f"\nLoading {args.model} with QLoRA (4-bit)...")
    model, tokenizer = _load_model_and_tokenizer(args.model, args.gpu_index)

    print("\nFine-tuning...")
    _finetune(model, tokenizer, texts, epochs=epochs, batch_size=args.batch_size)

    print("\nGenerating synthetic rows...")
    synthetic_df = _generate(model, tokenizer, columns, n_samples=n_synthetic, batch_size=args.batch_size)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "-")
    out_path = OUTPUT_DIR / f"{dataset_name}_{safe_model}_great.csv"
    synthetic_df.to_csv(out_path, index=False)
    print(f"\n✅ Saved: {out_path}  ({len(synthetic_df)} rows × {len(columns)} cols)")


if __name__ == "__main__":
    main()

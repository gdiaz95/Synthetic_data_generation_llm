import os
import re
import argparse
import gc
import subprocess
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv


@dataclass
class GPUInfo:
    index: int
    total_gb: float
    free_gb: float


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Multi-GPU Server Script")

    # Model Settings
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3"),
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--model-size-gb",
        type=float,
        default=(float(os.getenv("MODEL_SIZE_GB")) if os.getenv("MODEL_SIZE_GB") else None),
        help=(
            "Approximate model size in GB. If omitted, an estimate is inferred from model name"
            " (e.g., 7B -> ~14GB in fp16)."
        ),
    )

    # Hardware Settings
    parser.add_argument(
        "--gpus",
        type=str,
        default=os.getenv("CUDA_VISIBLE_DEVICES", ""),
        help="Comma-separated GPU IDs to allow. Empty means auto-detect.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
        help="Tensor parallel size (number of GPUs to split model across)",
    )

    # Memory/Performance Settings
    parser.add_argument(
        "--utilization",
        type=float,
        default=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.80")),
        help="Fraction of GPU memory to reserve",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=int(os.getenv("MAX_MODEL_LEN", "4096")),
        help="Maximum sequence length",
    )

    return parser.parse_args()


def get_gpu_info() -> List[GPUInfo]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return []

    gpus: List[GPUInfo] = []
    for line in proc.stdout.strip().splitlines():
        if not line.strip():
            continue
        idx_s, total_mb_s, free_mb_s = [x.strip() for x in line.split(",")]
        gpus.append(
            GPUInfo(
                index=int(idx_s),
                total_gb=float(total_mb_s) / 1024.0,
                free_gb=float(free_mb_s) / 1024.0,
            )
        )
    return gpus


def infer_model_size_gb(model_name: str) -> Optional[float]:
    # e.g. "7B", "70b", "8x7B"
    experts = re.search(r"(\d+)x(\d+(?:\.\d+)?)b", model_name, flags=re.IGNORECASE)
    if experts:
        total_b = float(experts.group(1)) * float(experts.group(2))
        return total_b * 2.0  # fp16/bf16 ~2 bytes/param

    match = re.search(r"(\d+(?:\.\d+)?)b", model_name, flags=re.IGNORECASE)
    if match:
        params_b = float(match.group(1))
        return params_b * 2.0  # fp16/bf16

    return None


def select_gpus_for_model(
    model_name: str,
    tp_size: int,
    gpu_util: float,
    allowed_gpu_ids: Optional[List[int]] = None,
    model_size_gb: Optional[float] = None,
) -> (Optional[List[int]], str):
    all_gpus = get_gpu_info()
    if not all_gpus:
        return None, "no gpus"

    print("\n🖥️ GPU space on current machine:")
    for g in all_gpus:
        print(f"  - GPU {g.index}: free {g.free_gb:.2f} GB / total {g.total_gb:.2f} GB")

    candidate_gpus = all_gpus
    if allowed_gpu_ids:
        allowed_set = set(allowed_gpu_ids)
        candidate_gpus = [g for g in all_gpus if g.index in allowed_set]

    if not candidate_gpus:
        return None, "no free gpus"

    if len(candidate_gpus) < tp_size:
        return None, "no free gpus"

    if model_size_gb is None:
        model_size_gb = infer_model_size_gb(model_name)

    if model_size_gb is None:
        return None, "error loading model size"

    # reserve some runtime headroom
    required_total = model_size_gb * 1.20
    required_per_gpu = required_total / tp_size

    # free memory usable at requested utilization
    sorted_gpus = sorted(candidate_gpus, key=lambda g: g.free_gb, reverse=True)
    chosen = sorted_gpus[:tp_size]

    lacking = [g for g in chosen if g.free_gb * gpu_util < required_per_gpu]
    if lacking:
        return None, "no space"

    return [g.index for g in chosen], "ok"


class MultiGPUServer:
    def __init__(self, args):
        self.model_name = args.model
        self.tp_size = args.tp
        self.gpu_util = args.utilization
        self.max_len = args.max_len

        allowed = [int(g.strip()) for g in args.gpus.split(",") if g.strip() != ""]
        selected_gpus, status = select_gpus_for_model(
            model_name=self.model_name,
            tp_size=self.tp_size,
            gpu_util=self.gpu_util,
            allowed_gpu_ids=allowed if allowed else None,
            model_size_gb=args.model_size_gb,
        )

        if status != "ok":
            message = {
                "no gpus": "no gpus",
                "no free gpus": "no free gpus",
                "no space": "no space",
                "error loading model size": "error loading model size",
            }.get(status, "server could not be allocated")
            raise RuntimeError(message)

        self.selected_gpus = selected_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in selected_gpus)

        print("🚀 Initializing vLLM")
        print(f"📍 GPUs: {self.selected_gpus} (TP Size: {self.tp_size})")
        print(f"📦 Model: {self.model_name}")

        try:
            from vllm import LLM, SamplingParams
            from vllm.distributed.parallel_state import (
                destroy_model_parallel,
                destroy_distributed_environment,
            )

            self._SamplingParams = SamplingParams
            self._destroy_model_parallel = destroy_model_parallel
            self._destroy_distributed_environment = destroy_distributed_environment

            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tp_size,
                gpu_memory_utilization=self.gpu_util,
                max_model_len=self.max_len,
                trust_remote_code=True,
                enforce_eager=True,
            )
        except Exception:
            self.cleanup()
            raise

    def cleanup(self):
        """Forcefully clears GPU memory and destroys distributed processes."""
        print("\n🧹 Cleaning up GPU resources...")
        try:
            if hasattr(self, "_destroy_model_parallel"):
                self._destroy_model_parallel()
            if hasattr(self, "_destroy_distributed_environment"):
                self._destroy_distributed_environment()
        except Exception:
            pass

        if hasattr(self, "llm"):
            del self.llm

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        print("✅ Cleanup complete.")

    def generate_batch(self, prompts, config=None):
        if config is None:
            config = {}

        sampling_params = self._SamplingParams(
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.95),
            max_tokens=config.get("max_tokens", 512),
            repetition_penalty=config.get("repetition_penalty", 1.1),
            stop=["[INST]", "</s>", "[/INST]"],
        )

        outputs = self.llm.generate(prompts, sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]


if __name__ == "__main__":
    load_dotenv(override=True)
    args = parse_args()

    server = None
    try:
        server = MultiGPUServer(args)
        test_res = server.generate_batch(["[INST] Hello, who are you in one sentence? [/INST]"])
        print(f"Response: {test_res[0]}")
    except RuntimeError as err:
        print(str(err))
    finally:
        if server is not None:
            server.cleanup()

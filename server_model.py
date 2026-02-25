import argparse
import gc
import os
import subprocess
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class GPUInfo:
    index: int
    total_mb: int
    used_mb: int
    utilization: int


class GPUSelectionError(RuntimeError):
    """Raised when no usable GPU can be selected."""


class ModelFitError(RuntimeError):
    """Raised when the selected model does not fit on a single GPU."""


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Single-GPU Server Script")

    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3"),
        help="HuggingFace model name or path. CLI value has priority over .env.",
    )

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


def detect_gpus():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]

    try:
        output = subprocess.check_output(cmd, text=True).strip()
    except FileNotFoundError as exc:
        raise GPUSelectionError("No GPUs available: nvidia-smi is not installed.") from exc
    except subprocess.CalledProcessError as exc:
        raise GPUSelectionError("No GPUs available: failed to query nvidia-smi.") from exc

    if not output:
        raise GPUSelectionError("No GPUs available.")

    gpus = []
    for line in output.splitlines():
        idx, total, used, util = [x.strip() for x in line.split(",")]
        gpus.append(
            GPUInfo(
                index=int(idx),
                total_mb=int(total),
                used_mb=int(used),
                utilization=int(util),
            )
        )
    return sorted(gpus, key=lambda gpu: gpu.index)


def select_free_gpu(required_utilization):
    gpus = detect_gpus()

    free_candidates = []
    for gpu in gpus:
        # Conservative definition of "free": no active compute utilization and tiny memory use.
        if gpu.utilization == 0 and gpu.used_mb <= 256:
            free_candidates.append(gpu)

    if not free_candidates:
        raise GPUSelectionError("All GPUs are currently busy/full. No free GPU available.")

    selected = free_candidates[0]
    max_usable_mb = int(selected.total_mb * required_utilization)
    print(
        f"🖥️  Selected GPU {selected.index} | total={selected.total_mb}MB | "
        f"used={selected.used_mb}MB | target_utilization={required_utilization} "
        f"(~{max_usable_mb}MB usable)"
    )
    return selected


class SingleGPUServer:
    def __init__(self, args):
        self.model_name = args.model
        self.gpu_util = args.utilization
        self.max_len = args.max_len

        selected_gpu = select_free_gpu(self.gpu_util)

        # Runtime-only assignment to force vLLM onto one selected GPU.
        # We no longer require CUDA_VISIBLE_DEVICES in .env.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu.index)

        print("🚀 Initializing vLLM")
        print(f"📍 GPU: {selected_gpu.index} (single GPU mode)")
        print(f"📦 Model: {self.model_name}")

        try:
            from vllm import LLM, SamplingParams
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            self._SamplingParams = SamplingParams
            self._destroy_model_parallel = destroy_model_parallel
            self._destroy_distributed_environment = destroy_distributed_environment

            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=self.gpu_util,
                max_model_len=self.max_len,
                trust_remote_code=True,
                enforce_eager=True,
            )
        except Exception as exc:
            self.cleanup()
            message = str(exc).lower()
            if "out of memory" in message or "cuda" in message and "memory" in message:
                raise ModelFitError(
                    "Model does not fit in one GPU. Retry with another model."
                ) from exc
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
        server = SingleGPUServer(args)
        test_res = server.generate_batch(["[INST] Hello, who are you in one sentence? [/INST]"])
        print(f"Response: {test_res[0]}")
    except (GPUSelectionError, ModelFitError) as exc:
        print(f"❌ {exc}")
    finally:
        if server is not None:
            server.cleanup()

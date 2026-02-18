import os
import argparse
import gc
from dotenv import load_dotenv


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Multi-GPU Server Script")

    # Model Settings
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3"),
        help="HuggingFace model name or path",
    )

    # Hardware Settings
    parser.add_argument(
        "--gpus",
        type=str,
        default=os.getenv("CUDA_VISIBLE_DEVICES", "2"),
        help="Comma-separated GPU IDs (e.g., 0,2,5,7)",
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


class MultiGPUServer:
    def __init__(self, args):
        self.model_name = args.model
        self.tp_size = args.tp
        self.gpu_util = args.utilization
        self.max_len = args.max_len

        visible = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
        if self.tp_size > len(visible):
            raise ValueError(
                f"--tp ({self.tp_size}) cannot be > number of visible GPUs ({len(visible)}) "
                f"from --gpus='{args.gpus}'."
            )

        print("🚀 Initializing vLLM")
        print(f"📍 GPUs: {args.gpus} (TP Size: {self.tp_size})")
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
                enforce_eager=True,  # keep your setting
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
    # 1. Load defaults from .env if it exists
    load_dotenv(override=True)

    # 2. Parse command line arguments
    args = parse_args()

    # IMPORTANT: set CUDA_VISIBLE_DEVICES BEFORE importing vLLM/torch
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    try:
        server = MultiGPUServer(args)
        test_res = server.generate_batch(["[INST] Hello, who are you in one sentence? [/INST]"])
        print(f"Response: {test_res[0]}")
    finally:
        if server is not None:
            server.cleanup()

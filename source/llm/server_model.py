import argparse
import gc
import json
import os
import threading
import time
import subprocess
import urllib.error
import urllib.request
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

class APIModelConnectionError(RuntimeError):
   """Raised when a remote API model cannot be reached or authenticated."""


def _stringify_response_value(value):
   if value is None:
       return None
   if hasattr(value, "name"):
       return str(getattr(value, "name"))
   if hasattr(value, "value"):
       raw_value = getattr(value, "value")
       if raw_value is not None:
           return str(raw_value)
   return str(value)


def _extract_status_code(exc):
   """Best-effort status extraction for SDK errors (Gemini/OpenAI clients)."""
   for attr in ("status_code", "code", "status"):
       value = getattr(exc, attr, None)
       if isinstance(value, int):
           return value

   response = getattr(exc, "response", None)
   if response is not None:
       value = getattr(response, "status_code", None)
       if isinstance(value, int):
           return value
   return None

def parse_args():
   parser = argparse.ArgumentParser(description="vLLM Single-GPU Server Script")


   parser.add_argument(
       "--model",
       type=str,
       default="gemini-2.5-flash",
       help=(
           "Model name or path. Supports local vLLM models (HuggingFace/path) and "
           "API models like Gemini or GPT. CLI value has priority over .env."
       ),
   )
   parser.add_argument(
       "--provider",
       type=str,
       choices=["auto", "local", "gemini", "openai"],
       default="auto",
       help="Model backend provider. Defaults to auto detection.",
   )
   parser.add_argument(
       "--utilization",
       type=float,
       default=0.80,
       help="Fraction of GPU memory to reserve",
   )
   parser.add_argument(
       "--max-len",
       type=int,
       default=4096,
       help="Maximum sequence length",
   )
   parser.add_argument(
       "--enforce-eager",
       action="store_true",
       help=(
           "Force eager execution in vLLM. Leave disabled for better throughput on "
           "most local GPU runs."
       ),
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
       self.provider = self._resolve_provider(args.provider, self.model_name)
       self.gpu_util = args.utilization
       self.max_len = args.max_len
       self.enforce_eager = bool(getattr(args, "enforce_eager", False))
       self.llm = None
       self._usage_lock = threading.Lock()
       self._usage_totals = {
           "prompt_tokens": 0,
           "completion_tokens": 0,
           "requests": 0,
       }


       print("🚀 Initializing model server")
       print(f"📦 Model: {self.model_name}")
       print(f"🔌 Provider mode: {self.provider}")
       if self.provider == "local":
           print(f"⚙️  vLLM enforce_eager: {self.enforce_eager}")


       if self.provider == "local":
           self._init_local_vllm()
       elif self.provider == "gemini":
           self._init_gemini()
       elif self.provider == "openai":
           self._init_openai()
       else:
           raise ValueError(f"Unsupported provider '{self.provider}'.")


   @staticmethod
   def _resolve_provider(provider_arg, model_name):
       if provider_arg and provider_arg != "auto":
           return provider_arg


       model_lower = model_name.lower()
       if "gemini" in model_lower:
           return "gemini"
       if model_lower.startswith("gpt") or "openai" in model_lower:
           return "openai"
       return "local"


   def _init_local_vllm(self):
       selected_gpu = select_free_gpu(self.gpu_util)


       # Runtime-only assignment to force vLLM onto one selected GPU.
       # We no longer require CUDA_VISIBLE_DEVICES in .env.
       os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu.index)


       print("🚀 Initializing local vLLM backend")
       print(f"📍 GPU: {selected_gpu.index} (single GPU mode)")


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
               enforce_eager=self.enforce_eager,
           )
       except Exception as exc:
           self.cleanup()
           message = str(exc).lower()
           if "out of memory" in message or "cuda" in message and "memory" in message:
               raise ModelFitError(
                   "Model does not fit in one GPU. Retry with another model."
               ) from exc
           raise


   def _init_gemini(self):
       self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
       if not self.gemini_api_key:
           raise APIModelConnectionError(
               "Gemini provider selected but GEMINI_API_KEY/GOOGLE_API_KEY is missing. "
               "Add it to your .env file."
           )


       self._gemini_client = None
       try:
           from google import genai


           self._gemini_client = genai.Client(api_key=self.gemini_api_key)
       except (ModuleNotFoundError, ImportError):
           print(
               "⚠️  `google-genai` is not installed. Falling back to Gemini REST API mode."
           )


   def _init_openai(self):
       self.openai_api_key = os.getenv("OPENAI_API_KEY")
       if not self.openai_api_key:
           raise APIModelConnectionError(
               "OpenAI provider selected but OPENAI_API_KEY is missing. "
               "Add it to your .env file."
           )


   def _record_usage(self, prompt_tokens=0, completion_tokens=0, requests=1):
       prompt_tokens = int(prompt_tokens or 0)
       completion_tokens = int(completion_tokens or 0)
       requests = int(requests or 0)
       with self._usage_lock:
           self._usage_totals["prompt_tokens"] += max(0, prompt_tokens)
           self._usage_totals["completion_tokens"] += max(0, completion_tokens)
           self._usage_totals["requests"] += max(0, requests)


   def get_usage_summary(self):
       with self._usage_lock:
           prompt_tokens = int(self._usage_totals["prompt_tokens"])
           completion_tokens = int(self._usage_totals["completion_tokens"])
           requests = int(self._usage_totals["requests"])
       return {
           "prompt_tokens": prompt_tokens,
           "completion_tokens": completion_tokens,
           "total_tokens": prompt_tokens + completion_tokens,
           "requests": requests,
       }


   @staticmethod
   def _safe_read_http_error(exc):
       try:
           body = exc.read().decode("utf-8")
           return body[:500]
       except Exception:
           return str(exc)


   @staticmethod
   def _is_retryable_http_error(status_code):
       return status_code in {408, 429, 500, 502, 503, 504}


   @staticmethod
   def _gemini_empty_response_retry_attempts(config):
       return max(1, int(config.get("empty_response_retry_attempts", 3)))


   @staticmethod
   def _gemini_empty_response_retry_sleep(config):
       return max(0.0, float(config.get("empty_response_retry_sleep_sec", 1.0)))


   def _format_gemini_sdk_response_diagnostics(self, response):
       details = []
       candidates = getattr(response, "candidates", None) or []
       details.append(f"candidate_count={len(candidates)}")

       prompt_feedback = getattr(response, "prompt_feedback", None)
       if prompt_feedback is not None:
           block_reason = _stringify_response_value(
               getattr(prompt_feedback, "block_reason", None)
           )
           if block_reason:
               details.append(f"prompt_block_reason={block_reason}")

       for idx, candidate in enumerate(candidates[:2]):
           finish_reason = _stringify_response_value(
               getattr(candidate, "finish_reason", None)
           )
           if finish_reason:
               details.append(f"candidate_{idx}_finish_reason={finish_reason}")

           safety_ratings = getattr(candidate, "safety_ratings", None) or []
           if safety_ratings:
               condensed = []
               for rating in safety_ratings[:3]:
                   category = _stringify_response_value(
                       getattr(rating, "category", None)
                   )
                   probability = _stringify_response_value(
                       getattr(rating, "probability", None)
                   )
                   blocked = getattr(rating, "blocked", None)
                   condensed.append(
                       f"{category or 'unknown'}:{probability or 'unknown'}"
                       + (
                           f":blocked={blocked}"
                           if blocked is not None
                           else ""
                       )
                   )
               details.append(
                   f"candidate_{idx}_safety_ratings=[{', '.join(condensed)}]"
               )

       return "; ".join(details)


   def _format_gemini_rest_response_diagnostics(self, data):
       details = []
       candidates = data.get("candidates") or []
       details.append(f"candidate_count={len(candidates)}")

       prompt_feedback = data.get("promptFeedback") or {}
       if prompt_feedback.get("blockReason"):
           details.append(f"prompt_block_reason={prompt_feedback['blockReason']}")

       for idx, candidate in enumerate(candidates[:2]):
           finish_reason = candidate.get("finishReason")
           if finish_reason:
               details.append(f"candidate_{idx}_finish_reason={finish_reason}")

           safety_ratings = candidate.get("safetyRatings") or []
           if safety_ratings:
               condensed = []
               for rating in safety_ratings[:3]:
                   condensed.append(
                       f"{rating.get('category', 'unknown')}:"
                       f"{rating.get('probability', 'unknown')}"
                       + (
                           f":blocked={rating.get('blocked')}"
                           if rating.get("blocked") is not None
                           else ""
                       )
                   )
               details.append(
                   f"candidate_{idx}_safety_ratings=[{', '.join(condensed)}]"
               )

       return "; ".join(details)


   def _retry_gemini_request(self, make_request):
       max_attempts = 5
       base_wait_seconds = 2

       for attempt in range(1, max_attempts + 1):
           try:
               return make_request()
           except urllib.error.HTTPError as exc:
               details = self._safe_read_http_error(exc)
               if self._is_retryable_http_error(exc.code) and attempt < max_attempts:
                   wait_seconds = base_wait_seconds * attempt
                   print(
                       f"⚠️  Gemini request failed with HTTP {exc.code}. "
                       f"Retrying in {wait_seconds}s (attempt {attempt}/{max_attempts})..."
                   )
                   time.sleep(wait_seconds)
                   continue
               raise APIModelConnectionError(
                   "Couldn't connect to Gemini API. Check model name and API key. "
                   f"Details: {details}"
               ) from exc
           except urllib.error.URLError as exc:
               if attempt < max_attempts:
                   wait_seconds = base_wait_seconds * attempt
                   print(
                       "⚠️  Gemini request failed due to a network error. "
                       f"Retrying in {wait_seconds}s (attempt {attempt}/{max_attempts})..."
                   )
                   time.sleep(wait_seconds)
                   continue
               raise APIModelConnectionError(
                   "Couldn't connect to Gemini API. Check network access and API key."
               ) from exc
           except Exception as exc:
               status_code = _extract_status_code(exc)
               if status_code is not None:
                   details = str(exc)
                   if self._is_retryable_http_error(status_code) and attempt < max_attempts:
                       wait_seconds = base_wait_seconds * attempt
                       print(
                           f"⚠️  Gemini request failed with HTTP {status_code}. "
                           f"Retrying in {wait_seconds}s (attempt {attempt}/{max_attempts})..."
                       )
                       time.sleep(wait_seconds)
                       continue
                   raise APIModelConnectionError(
                       "Couldn't connect to Gemini API. Check model name and API key. "
                       f"Details: {details}"
                   ) from exc

               if attempt < max_attempts:
                   wait_seconds = base_wait_seconds * attempt
                   print(
                       "⚠️  Gemini request raised an unexpected error. "
                       f"Retrying in {wait_seconds}s (attempt {attempt}/{max_attempts})... "
                       f"{type(exc).__name__}: {exc}"
                   )
                   time.sleep(wait_seconds)
                   continue
               raise APIModelConnectionError(
                   "Couldn't connect to Gemini API. Check model name and API key. "
                   f"Details: {type(exc).__name__}: {exc}"
               ) from exc


   def _generate_with_gemini_rest(self, prompt, config, model_name):
       generation_cfg = {
       "temperature": config.get("temperature", 0.7),
       "topP": config.get("top_p", 0.95),
       "maxOutputTokens": config.get("max_tokens", 512),
       }

       payload = {
           "contents": [{"parts": [{"text": prompt}]}],
           "generationConfig": generation_cfg,
       }


       url = (
           "https://generativelanguage.googleapis.com/v1beta/"
           f"models/{model_name}:generateContent?key={self.gemini_api_key}"
       )
       request = urllib.request.Request(
           url,
           data=json.dumps(payload).encode("utf-8"),
           headers={"Content-Type": "application/json"},
           method="POST",
       )


       def _request():
           with urllib.request.urlopen(request, timeout=45) as response:
               return json.loads(response.read().decode("utf-8"))

       empty_response_attempts = self._gemini_empty_response_retry_attempts(config)
       empty_response_wait = self._gemini_empty_response_retry_sleep(config)
       diagnostics = "candidate_count=0"

       for attempt in range(1, empty_response_attempts + 1):
           data = self._retry_gemini_request(_request)

           usage = data.get("usageMetadata") or {}
           self._record_usage(
               prompt_tokens=usage.get("promptTokenCount", 0),
               completion_tokens=usage.get("candidatesTokenCount", 0),
               requests=1,
           )

           candidates = data.get("candidates") or []
           if candidates:
               parts = candidates[0].get("content", {}).get("parts", [])
               text_parts = [part.get("text", "") for part in parts if part.get("text")]
               text = "\n".join(text_parts).strip()
               if text:
                   return text

           diagnostics = self._format_gemini_rest_response_diagnostics(data)
           if attempt < empty_response_attempts:
               print(
                   "⚠️  Gemini returned an empty response body. "
                   f"Retrying in {empty_response_wait}s "
                   f"(attempt {attempt}/{empty_response_attempts})... {diagnostics}"
               )
               time.sleep(empty_response_wait)
               continue

       raise APIModelConnectionError(
           "Gemini returned an empty response after retries. "
           f"Details: {diagnostics}"
       )


   def _generate_with_gemini(self, prompt, config):
       model_name = self.model_name
       if model_name.startswith("models/"):
           model_name = model_name[len("models/") :]


       if self._gemini_client is None:
           return self._generate_with_gemini_rest(prompt, config, model_name)


       generation_config = {
           "temperature": config.get("temperature", 0.7),
           "top_p": config.get("top_p", 0.95),
           "max_output_tokens": config.get("max_tokens", 512),
       }

       def _request():
           return self._gemini_client.models.generate_content(
               model=model_name,
               contents=prompt,
               config=generation_config,
           )


       empty_response_attempts = self._gemini_empty_response_retry_attempts(config)
       empty_response_wait = self._gemini_empty_response_retry_sleep(config)
       diagnostics = "candidate_count=0"

       for attempt in range(1, empty_response_attempts + 1):
           response = self._retry_gemini_request(_request)

           usage = getattr(response, "usage_metadata", None)
           if usage is not None:
               self._record_usage(
                   prompt_tokens=getattr(usage, "prompt_token_count", 0),
                   completion_tokens=getattr(usage, "candidates_token_count", 0),
                   requests=1,
               )
           else:
               self._record_usage(requests=1)

           text = getattr(response, "text", None)
           if text:
               return text.strip()

           diagnostics = self._format_gemini_sdk_response_diagnostics(response)
           if attempt < empty_response_attempts:
               print(
                   "⚠️  Gemini returned an empty response body. "
                   f"Retrying in {empty_response_wait}s "
                   f"(attempt {attempt}/{empty_response_attempts})... {diagnostics}"
               )
               time.sleep(empty_response_wait)
               continue

       raise APIModelConnectionError(
           "Gemini returned an empty response after retries. "
           f"Details: {diagnostics}"
       )


   def _generate_with_openai(self, prompt, config):
       model_name = self.model_name
       if model_name.lower().startswith("openai/"):
           model_name = model_name.split("/", 1)[1]

       model_name_lower = model_name.lower()
       is_newer_model = any(newer_model in model_name_lower for newer_model in ["gpt-5", "gpt-o1"])

       # Use max_completion_tokens for newer models (gpt-5, gpt-o1, etc.)
       max_tokens_value = config.get("max_tokens", 512)
       if is_newer_model or "gpt-4o-mini" in model_name_lower:
           token_param = "max_completion_tokens"
       else:
           token_param = "max_tokens"

       # Models like gpt-5-mini only support temperature=1.0
       temp_value = config.get("temperature", 0.7)
       if is_newer_model:
           temp_value = 1.0

       # Build payload, excluding unsupported parameters for newer models
       payload = {
           "model": model_name,
           "messages": [{"role": "user", "content": prompt}],
           "temperature": temp_value,
           token_param: max_tokens_value,
       }
       
       # Newer models (gpt-5, gpt-o1) don't support top_p
       if not is_newer_model:
           payload["top_p"] = config.get("top_p", 0.95)


       request = urllib.request.Request(
           "https://api.openai.com/v1/chat/completions",
           data=json.dumps(payload).encode("utf-8"),
           headers={
               "Content-Type": "application/json",
               "Authorization": f"Bearer {self.openai_api_key}",
           },
           method="POST",
       )


       try:
           with urllib.request.urlopen(request, timeout=45) as response:
               data = json.loads(response.read().decode("utf-8"))
       except urllib.error.HTTPError as exc:
           details = self._safe_read_http_error(exc)
           raise APIModelConnectionError(
               "Couldn't connect to OpenAI API. Check model name and API key. "
               f"Details: {details}"
           ) from exc
       except urllib.error.URLError as exc:
           raise APIModelConnectionError(
               "Couldn't connect to OpenAI API. Check network access and API key."
           ) from exc

       usage = data.get("usage") or {}
       self._record_usage(
           prompt_tokens=usage.get("prompt_tokens", 0),
           completion_tokens=usage.get("completion_tokens", 0),
           requests=1,
       )


       choices = data.get("choices") or []
       if not choices:
           raise APIModelConnectionError(
               "OpenAI returned no choices. Check API key/model access."
           )


       message = choices[0].get("message", {})
       return (message.get("content") or "").strip()


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


   def _default_local_stop_sequences(self):
       """Conservative stop strings for common local chat templates."""
       return [
           "[INST]",
           "[/INST]",
           "</s>",
           "<|im_start|>",
           "<|im_end|>",
           "<|start_header_id|>",
           "<|end_header_id|>",
           "<|eot_id|>",
           "<|end_of_text|>",
       ]


   def generate_batch(self, prompts, config=None):
       if config is None:
           config = {}


       if self.provider == "gemini":
           return [self._generate_with_gemini(prompt, config) for prompt in prompts]


       if self.provider == "openai":
           return [self._generate_with_openai(prompt, config) for prompt in prompts]


       stop_sequences = config.get("stop_sequences")
       if stop_sequences is None:
           stop_sequences = self._default_local_stop_sequences()

       sampling_params = self._SamplingParams(
           temperature=config.get("temperature", 0.7),
           top_p=config.get("top_p", 0.95),
           max_tokens=config.get("max_tokens", 512),
           repetition_penalty=config.get("repetition_penalty", 1.1),
           stop=stop_sequences,
       )


       outputs = self.llm.generate(prompts, sampling_params)
       prompt_tokens = 0
       completion_tokens = 0
       for output in outputs:
           prompt_tokens += len(getattr(output, "prompt_token_ids", []) or [])
           first_output = output.outputs[0] if output.outputs else None
           completion_tokens += len(getattr(first_output, "token_ids", []) or [])
       self._record_usage(
           prompt_tokens=prompt_tokens,
           completion_tokens=completion_tokens,
           requests=len(prompts),
       )
       return [o.outputs[0].text.strip() for o in outputs]

if __name__ == "__main__":
   load_dotenv(override=True)
   args = parse_args()


   server = None
   try:
       server = SingleGPUServer(args)
       test_res = server.generate_batch(["[INST] Hello, who are you in one sentence? [/INST]"])
       print(f"Response: {test_res[0]}")
   except (GPUSelectionError, ModelFitError, APIModelConnectionError) as exc:
       print(f"❌ {exc}")
   finally:
       if server is not None and server.provider == "local":
           server.cleanup()

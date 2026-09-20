"""generate.py - Generate Ada code from base and fine-tuned models.

Loads the ada-eval expanded datasets, generates Ada code using both
the base Qwen3-8B model and the fine-tuned q3as model, and writes
generated solutions as packed ada-eval datasets (JSONL) so that
BUILD/TEST/PROVE evaluations can be run directly with ada-eval.

The base model defaults to the LOCAL download (models/qwen3-8b, i.e.
unsloth/Qwen3-8B) so that training and evaluation use exactly the same
downloaded base weights.

Usage:
    uv run python eval/generate.py --model outputs/q3as --max-samples 20
    uv run python eval/generate.py --model outputs/q3as --base-model models/qwen3-8b --max-samples 50
    uv run python eval/generate.py --model outputs/q3as --dataset spark_learn --max-samples 10 --temperature 0.3
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# The host may have less RAM than the model needs in bf16 (8B model -> ~16 GB
# of weights). Setting these before transformers is imported keeps the loader
# synchronous and bounds what accumulate in the allocator.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logger = logging.getLogger("q3as_generate")
GENERATED_DIR = Path("outputs/generated_solutions")
DEFAULT_BASE_MODEL = Path("models/qwen3-8b")
_ADA_EVAL_SRC = Path(__file__).resolve().parents[1].parent / "ada-eval" / "src"
if str(_ADA_EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(_ADA_EVAL_SRC))

# Long prompts push up KV-cache and activation memory at generation time; on
# a 7 GB host RAM / 8 GB VRAM box they are truncated before tokenization.
MAX_PROMPT_CHARS = 24000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Ada code with base and fine-tuned models.",
    )
    parser.add_argument(
        "--model", type=Path, default=Path("outputs/q3as"),
        help="Fine-tuned model checkpoint path.",
    )
    parser.add_argument(
        "--base-model", type=Path, default=DEFAULT_BASE_MODEL,
        help="Base model path (default: local models/qwen3-8b download of unsloth/Qwen3-8B).",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Specific dataset to generate.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=20,
        help="Max samples per dataset.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=1024,
        help="Max new tokens per sample.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--enable-thinking", action="store_true", default=False,
        help="Enable Qwen3 thinking mode during generation.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--worker", action="store_true", help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _worker_parser() -> argparse.ArgumentParser:
    """Argument parser for the single-model generation worker subprocess."""
    parser = argparse.ArgumentParser(description="Single-model generation worker.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--model-label", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _run_worker(args: argparse.Namespace) -> int:
    """Load one model, generate for all datasets, print a JSON summary, exit.

    Each model runs in its own process: a completed transformers/bitsandbytes
    load leaves the CUDA context oversized, so a second load in the same
    process can OOM even after empty_cache. A fresh process starts with a
    pristine GPU, which keeps dual-model runs inside 8 GB VRAM.
    """
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] [worker] %(message)s"
    )

    model, tokenizer = load_model(args.model_path, args.base_model_path)
    results = run_generation_for_model(
        model, tokenizer, args.model_label,
        find_expanded_datasets(args.dataset), args,
    )
    free_model_memory(model, tokenizer)
    print(json.dumps(results))
    return 0


def find_expanded_datasets(
    dataset_name: str | None,
) -> list[tuple[str, Path]]:
    """Find expanded dataset directories from ada-eval."""
    expanded_dir = Path("../ada-eval/data/base/expanded")
    if not expanded_dir.exists():
        logger.error("Expanded datasets not found at %s", expanded_dir)
        sys.exit(1)
    return [
        (d.name, d)
        for d in sorted(expanded_dir.iterdir())
        if d.is_dir() and (not dataset_name or d.name == dataset_name)
    ]


def load_sample_prompts(expanded_dir: Path) -> list[dict[str, Any]]:
    """Load sample names, prompts, and locations from an expanded dataset."""
    samples: list[dict[str, Any]] = []
    for sample_dir in sorted(expanded_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        other_path = sample_dir / "other.json"
        prompt_path = sample_dir / "prompt.md"
        if not other_path.exists() or not prompt_path.exists():
            continue
        try:
            with open(other_path) as f:
                other_data = json.load(f)
            prompt = prompt_path.read_text(encoding="utf-8")
            comments_path = sample_dir / "comments.md"
            comments = comments_path.read_text(encoding="utf-8") if comments_path.exists() else ""
            samples.append({
                "name": sample_dir.name,
                "dir": sample_dir,
                "prompt": prompt,
                "location": other_data.get("location", {}),
                "comments": comments,
                "canonical_evaluation_results": other_data.get("canonical_evaluation_results", []),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return samples


def extract_ada_code(generated_text: str) -> str:
    """Extract Ada code block from model output."""
    match = re.search(r"```ada\s*\n(.*?)```", generated_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", generated_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return generated_text.strip()


def checkpoint_is_prequantized_4bit(model_path: Path) -> bool:
    """True when the safetensors shards hold bitsandbytes-quantized weights.

    Such checkpoints cannot be re-loaded with a fresh quantization config
    (the loader would materialize the missing bf16 weights and OOM); the
    QLoRA adapter route (base model + lora_adapter) must be used instead.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return False
    for shard in sorted(model_path.glob("*.safetensors")):
        try:
            with safe_open(shard, framework="pt") as f:
                if any("quant_state" in k or "base_layer" in k for k in f):
                    return True
        except Exception:  # noqa: BLE001, S112  (probe is best-effort)
            continue
    return False


def load_model(model_path: Path, base_model_path: Path):
    """Load a model and tokenizer once for efficient reuse.

    Two checkpoint layouts are supported:

    - QLoRA checkpoints with a ``lora_adapter/`` subdirectory (what
      training/train_unsloth.py writes): the local base model is loaded in
      4-bit NF4 and the adapter is applied on top. This is standard QLoRA
      inference and matches training-time behavior exactly.
    - Truly merged checkpoints: loaded directly, quantized to 4-bit NF4
      on load.

    4-bit loading keeps the whole 8B model in GPU VRAM (~5.5 GB) instead
    of spilling bf16 weights into system RAM, which OOMs hosts with less
    RAM than a bf16 copy of the model needs. ``device_map="auto"`` is
    deliberately avoided: it offloads overflow weights to system RAM.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model: Any
    adapter_dir = model_path / "lora_adapter"
    if adapter_dir.is_dir():
        from peft import PeftModel

        logger.info(
            "Loading 4-bit base %s + LoRA adapter %s", base_model_path, adapter_dir
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    else:
        if checkpoint_is_prequantized_4bit(model_path):
            raise RuntimeError(
                f"{model_path} is a 4-bit serialized checkpoint without a "
                "lora_adapter/ subdirectory; it cannot be loaded directly. "
                "Re-run `make train` or pass a checkpoint that contains "
                "lora_adapter/."
            )
        logger.info("Loading merged model in 4-bit NF4: %s", model_path)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
    model.eval()
    return model, tokenizer


def free_model_memory(model, tokenizer) -> None:
    """Release a model's GPU and host memory before loading the next one."""
    import torch

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool = False,
) -> list[str]:
    """Generate code for a batch of prompts using the model's chat template."""
    import torch
    results: list[str] = []
    for prompt in prompts:
        try:
            # Cap prompt size: huge prompts blow up KV-cache and host memory
            # on small-RAM hosts.
            prompt = prompt[:MAX_PROMPT_CHARS]
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            results.append(extract_ada_code(generated))
        except Exception as exc:  # noqa: BLE001  (keep generating remaining prompts)
            logger.error("Generation failed: %s", exc)
            results.append("")
    return results


def run_generation_for_model(
    model,
    tokenizer,
    model_label: str,
    datasets: list[tuple[str, Path]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Generate solutions for all datasets and pack them as ada-eval JSONL."""
    from ada_eval.datasets.types import GENERATED_SAMPLE_TYPES
    from ada_eval.datasets.types.samples import ExitStatus, GenerationStats

    output_root = GENERATED_DIR / model_label
    output_root.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, int] = {}
    total = 0

    for dataset_name, expanded_dir in datasets:
        samples = load_sample_prompts(expanded_dir)[: args.max_samples]
        if not samples:
            by_dataset[dataset_name] = 0
            continue

        prompts = [s["prompt"] for s in samples]
        start = time.perf_counter()
        generated_codes = generate_batch(
            model, tokenizer, prompts,
            args.max_new_tokens, args.temperature, args.enable_thinking,
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        sample_kind = "spark" if dataset_name.startswith("spark") else "ada"
        sample_type = GENERATED_SAMPLE_TYPES[sample_kind]
        packed_samples = []
        count = 0
        for sample, ada_code in zip(samples, generated_codes):
            if not ada_code:
                logger.warning("%s/%s: empty generation, skipped", dataset_name, sample["name"])
                continue

            sample_dir: Path = sample["dir"]

            def read_dir_bytes(root: Path) -> dict[Path, bytes]:
                files: dict[Path, bytes] = {}
                if root.is_dir():
                    for file_path in sorted(root.rglob("*")):
                        if file_path.is_file():
                            files[file_path.relative_to(root)] = file_path.read_bytes()
                return files

            # generated_solution must be a complete buildable project: ada-eval
            # runs gprbuild/gnatformat inside it (BUILD) and proves it (PROVE).
            # Start from the base project tree and overlay the model's code at
            # the location the prompt asked to fill (the convention ada-eval's
            # own generation agents use).
            generated_solution = dict(sources)
            location = sample["location"]
            gen_path = location.get("path", "generated.adb") if isinstance(location, dict) else "generated.adb"
            generated_solution[Path(gen_path)] = ada_code.encode("utf-8")

            packed = sample_type(
                name=sample["name"],
                location=location,
                prompt=sample["prompt"],
                sources=sources,
                canonical_solution={},
                canonical_evaluation_results=sample.get("canonical_evaluation_results", []),
                comments=sample.get("comments", ""),
                generation_stats=GenerationStats(
                    exit_status=ExitStatus.SUCCESS,
                    stdout=ada_code,
                    stderr="",
                    runtime_ms=max(elapsed_ms // len(samples), 1),
                ),
                generated_solution=generated_solution,
                unit_tests=unit_tests,
            )
            packed_samples.append(packed)
            count += 1

        # Packed dataset file name must be "<kind>_<name>.jsonl" for ada-eval.
        out_file = output_root / f"{sample_kind}_{dataset_name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for packed in packed_samples:
                f.write(packed.model_dump_json(exclude_defaults=True) + "\n")
        logger.info("%s: %d/%d samples generated -> %s", dataset_name, count, len(samples), out_file)
        by_dataset[dataset_name] = count
        total += count

    return {"by_dataset": by_dataset, "total": total, "output_dir": str(output_root)}


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("Starting Ada code generation pipeline")
    logger.info("Fine-tuned model: %s", args.model)
    logger.info("Base model: %s", args.base_model)
    logger.info("Max samples per dataset: %d", args.max_samples)

    if not args.model.exists():
        logger.error(
            "Fine-tuned model not found at %s - run `make train` first.", args.model
        )
        sys.exit(1)
    if not args.base_model.exists():
        logger.error(
            "Base model not found at %s - run `make download` first "
            "(or pass --base-model with an existing local path).", args.base_model,
        )
        sys.exit(1)

    if not find_expanded_datasets(args.dataset):
        logger.error("No datasets found")
        sys.exit(1)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    # One subprocess per model: each from_pretrained gets a pristine GPU,
    # so the second model load cannot OOM against leftover CUDA context
    # state from the first.
    def spawn(label: str, model_path: Path) -> dict[str, Any]:
        cmd = [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--model-path", str(model_path),
            "--base-model-path", str(args.base_model),
            "--model-label", label,
            "--max-samples", str(args.max_samples),
            "--max-new-tokens", str(args.max_new_tokens),
            "--temperature", str(args.temperature),
        ]
        if args.dataset:
            cmd += ["--dataset", args.dataset]
        if args.enable_thinking:
            cmd.append("--enable-thinking")
        if args.verbose:
            cmd.append("--verbose")
        logger.info("Launching worker for %s: %s", label, model_path)
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            logger.error("Worker for %s failed with exit code %d", label, proc.returncode)
            return {"by_dataset": {}, "total": 0, "output_dir": str(GENERATED_DIR / label)}
        try:
            return json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            logger.error("Could not parse worker summary for %s", label)
            return {"by_dataset": {}, "total": 0, "output_dir": str(GENERATED_DIR / label)}

    base_label = f"base_{args.base_model.name}"
    summary: dict[str, Any] = {
        "fine_tuned": spawn("fine_tuned", args.model),
        base_label: spawn(base_label, args.base_model),
    }

    summary["max_samples"] = args.max_samples
    summary["dataset"] = args.dataset or "all"
    summary["base_model_path"] = str(args.base_model)

    with open(Path("outputs/generation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Generation complete: %d fine-tuned, %d base samples",
        summary["fine_tuned"]["total"], summary[base_label]["total"],
    )


if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.exit(_run_worker(_worker_parser().parse_args()))
    main()

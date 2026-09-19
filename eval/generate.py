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
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("q3as_generate")
GENERATED_DIR = Path("outputs/generated_solutions")
DEFAULT_BASE_MODEL = Path("models/qwen3-8b")
_ADA_EVAL_SRC = Path(__file__).resolve().parents[1].parent / "ada-eval" / "src"
if str(_ADA_EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(_ADA_EVAL_SRC))


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
    return parser.parse_args()


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


def load_model(model_path: Path):
    """Load a model and tokenizer once for efficient reuse."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_path = str(model_path)
    logger.info("Loading model: %s", load_path)
    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


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

            # Project files needed to build (main.gpr, src/, ...). The file at
            # location.path is replaced with the model's generation.
            sources = read_dir_bytes(sample_dir / "base")
            unit_tests = read_dir_bytes(sample_dir / "tests")

            location = sample["location"]
            gen_path = location.get("path", "generated.adb") if isinstance(location, dict) else "generated.adb"
            gen_filename = Path(gen_path).name
            if gen_filename.endswith(".ads"):
                gen_filename = gen_filename[:-4] + ".adb"
            generated_solution = {Path(gen_filename): ada_code.encode("utf-8")}

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

    datasets = find_expanded_datasets(args.dataset)
    if not datasets:
        logger.error("No datasets found")
        sys.exit(1)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    # Generate with fine-tuned model
    ft_model, ft_tokenizer = load_model(args.model)
    summary["fine_tuned"] = run_generation_for_model(
        ft_model, ft_tokenizer, "fine_tuned", datasets, args
    )
    del ft_model

    # Generate with base model (same local download used for training)
    base_model, base_tokenizer = load_model(args.base_model)
    base_label = f"base_{args.base_model.name}"
    summary[base_label] = run_generation_for_model(
        base_model, base_tokenizer, base_label, datasets, args
    )
    del base_model

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    main()

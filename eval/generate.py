"""generate.py - Generate Ada code from base and fine-tuned models.

Loads the ada-eval expanded datasets, generates Ada code using both
the base Qwen3-8B model and the fine-tuned q3as model, and writes
the generated solutions into ada-eval's expected directory structure
so that BUILD/TEST/PROVE evaluations can be run.

Each model is loaded once and used for all samples for efficiency.

Usage:
    uv run python eval/generate.py --model outputs/q3as --max-samples 20
    uv run python eval/generate.py --model outputs/q3as --base-model unsloth/Qwen3-8B --max-samples 50
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Ada code with base and fine-tuned models.",
    )
    parser.add_argument(
        "--model", type=Path, default=Path("outputs/q3as"),
        help="Fine-tuned model checkpoint path.",
    )
    parser.add_argument(
        "--base-model", type=str, default="unsloth/Qwen3-8B",
        help="Base model identifier for comparison.",
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
            samples.append({
                "name": sample_dir.name,
                "dir": sample_dir,
                "prompt": prompt,
                "location": other_data.get("location", {}),
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


def load_model(model_name: str | None, model_path: Path | None):
    """Load a model and tokenizer once for efficient reuse."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    load_path = str(model_path) if model_path and model_path.exists() else model_name
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
) -> list[str]:
    """Generate code for a batch of prompts."""
    import torch
    results: list[str] = []
    for prompt in prompts:
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            results.append(extract_ada_code(generated))
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            results.append("")
    return results


def write_generated_solution(
    sample: dict[str, Any],
    ada_code: str,
    output_dir: Path,
) -> None:
    """Write generated Ada code into the ada-eval expected directory structure."""
    gen_dir = output_dir / sample["name"] / "generated_solution"
    gen_dir.mkdir(parents=True, exist_ok=True)

    location = sample["location"]
    src_path = location.get("path", "solution.adb") if isinstance(location, dict) else "solution.adb"
    filename = Path(src_path).name

    # If source is .ads, generate .adb
    if filename.endswith(".ads"):
        filename = filename.rsplit(".", 1)[0] + ".adb"

    (gen_dir / filename).write_text(ada_code, encoding="utf-8")

    # Update other.json with generation info
    other_path = output_dir / sample["name"] / "other.json"
    if other_path.exists():
        with open(other_path) as f:
            other_data = json.load(f)
    else:
        other_data = {}
    other_data["generation_stats"] = {"exit_status": "success"}
    other_path.write_text(json.dumps(other_data, indent=4) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("Starting Ada code generation pipeline")
    logger.info("Fine-tuned model: %s", args.model)
    logger.info("Base model: %s", args.base_model)
    logger.info("Max samples per dataset: %d", args.max_samples)

    datasets = find_expanded_datasets(args.dataset)
    if not datasets:
        logger.error("No datasets found")
        sys.exit(1)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    # Generate with fine-tuned model
    ft_model, ft_tokenizer = load_model(None, args.model)
    ft_results: dict[str, int] = {}
    ft_total = 0
    for dataset_name, expanded_dir in datasets:
        samples = load_sample_prompts(expanded_dir)[:args.max_samples]
        if not samples:
            ft_results[dataset_name] = 0
            continue
        prompts = [s["prompt"] for s in samples]
        generated_codes = generate_batch(
            ft_model, ft_tokenizer, prompts,
            args.max_new_tokens, args.temperature,
        )
        output_dir = GENERATED_DIR / "fine_tuned" / args.model.name
        count = 0
        for sample, ada_code in zip(samples, generated_codes):
            if ada_code:
                write_generated_solution(sample, ada_code, output_dir)
                count += 1
        ft_results[dataset_name] = count
        ft_total += count
        logger.info("%s: %d/%d samples generated", dataset_name, count, len(samples))
    summary["fine_tuned"] = {"by_dataset": ft_results, "total": ft_total}

    # Generate with base model
    if args.base_model:
        base_model, base_tokenizer = load_model(args.base_model, None)
        base_results: dict[str, int] = {}
        base_total = 0
        for dataset_name, expanded_dir in datasets:
            samples = load_sample_prompts(expanded_dir)[:args.max_samples]
            if not samples:
                base_results[dataset_name] = 0
                continue
            prompts = [s["prompt"] for s in samples]
            generated_codes = generate_batch(
                base_model, base_tokenizer, prompts,
                args.max_new_tokens, args.temperature,
            )
            output_dir = GENERATED_DIR / "base" / args.base_model
            count = 0
            for sample, ada_code in zip(samples, generated_codes):
                if ada_code:
                    write_generated_solution(sample, ada_code, output_dir)
                    count += 1
            base_results[dataset_name] = count
            base_total += count
            logger.info("%s: %d/%d samples generated", dataset_name, count, len(samples))
        summary["base"] = {"by_dataset": base_results, "total": base_total}
        del base_model

    del ft_model
    if __import__("torch").cuda.is_available():
        __import__("torch").cuda.empty_cache()

    summary["max_samples"] = args.max_samples
    summary["dataset"] = args.dataset or "all"

    with open(Path("outputs/generation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Generation complete: %d fine-tuned, %d base samples",
                summary["fine_tuned"]["total"], summary.get("base", {}).get("total", 0))


if __name__ == "__main__":
    main()

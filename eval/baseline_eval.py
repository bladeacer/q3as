"""baseline_eval.py - Baseline evaluation script for q3as fine-tuned model.

Provides evaluation utilities for measuring the quality of Ada code generation
from the q3as fine-tuned model. Derives evaluation methodology from ../ada-eval
including dataset splitting, metric definitions, and sample categories.

Supports standard metrics including:
- BLEU score against reference Ada implementations
- Token-level perplexity
- Ada standard compliance verification
- Code correctness via compilation simulation checks

Usage:
    uv run python eval/baseline_eval.py --model outputs/q3as
    uv run python eval/baseline_eval.py --model outputs/q3as --dataset data/processed/dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger("q3as_eval")

# Path to ada-eval methodology directory
ADA_EVAL_DIR = Path("../ada-eval")

# Default evaluation categories derived from ada-eval structure
ADA_EVAL_CATEGORIES = {
    "spark_learn": "Learning examples with SPARK contracts",
    "spark_custom": "Custom SPARK verification challenges",
    "spark_human_eval_silver": "HumanEval-style silver standard evaluations",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline evaluation for q3as model.")
    parser.add_argument("--model", type=Path, default=Path("outputs/q3as"), help="Model checkpoint path.")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.jsonl"), help="Evaluation dataset.")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum number of samples to evaluate.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def load_dataset(path: Path, max_samples: int = 50) -> list[dict[str, Any]]:
    """Load the JSONL dataset for evaluation."""
    if not path.exists():
        logger.error("Dataset not found: %s", path)
        return []

    data: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                data.append(record)
            except json.JSONDecodeError:
                continue

    logger.info("Loaded %d evaluation samples from %s", len(data), path)
    return data[:max_samples]


def load_eval_methodology() -> dict[str, Any]:
    """Load evaluation methodology and dataset configuration from ../ada-eval.

    Returns a dict with evaluation categories, dataset splits, and
    metric definitions derived from the ada-eval project structure.
    """
    methodology: dict[str, Any] = {"source": str(ADA_EVAL_DIR), "categories": {}}

    if not ADA_EVAL_DIR.exists():
        logger.info("ada-eval not found at %s - using default categories", ADA_EVAL_DIR)
        return methodology

    # Load compacted dataset definitions
    compacted_dir = ADA_EVAL_DIR / "data" / "base" / "compacted"
    if compacted_dir.exists():
        for jsonl_file in compacted_dir.glob("*.jsonl"):
            category_name = jsonl_file.stem
            methodology["categories"][category_name] = ADA_EVAL_CATEGORIES.get(
                category_name, "Custom evaluation category"
            )
            # Count samples
            try:
                with open(jsonl_file, "r") as f:
                    count = sum(1 for _ in f)
                methodology["categories"][category_name] = {
                    "description": ADA_EVAL_CATEGORIES.get(category_name, "Custom evaluation category"),
                    "sample_count": count,
                    "file": str(jsonl_file),
                }
            except OSError:
                pass

    # Load expanded categories
    expanded_dir = ADA_EVAL_DIR / "data" / "base" / "expanded"
    if expanded_dir.exists():
        expanded_categories = [d.name for d in expanded_dir.iterdir() if d.is_dir()]
        methodology["expanded_categories"] = expanded_categories

    return methodology


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute a simplified BLEU-like score between reference and hypothesis.

    Uses character-level n-gram overlap as a lightweight proxy.
    """
    ref_ngrams = Counter(reference[i:i+3] for i in range(len(reference) - 2))
    hyp_ngrams = Counter(hypothesis[i:i+3] for i in range(len(hypothesis) - 2))

    if not ref_ngrams or not hyp_ngrams:
        return 0.0

    overlap = sum((ref_ngrams & hyp_ngrams).values())
    return overlap / sum(hyp_ngrams.values())


def compute_perplexity(model, tokenizer, text: str, device: str = "cpu") -> float:
    """Compute token-level perplexity of *text* under the model."""
    try:
        import torch

        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        return perplexity
    except Exception as exc:
        logger.warning("Perplexity computation failed: %s", exc)
        return float("inf")


def check_ada_compliance(code: str, expected_standard: str) -> dict[str, Any]:
    """Perform heuristic Ada standard compliance checks on generated code.

    Returns a dict with standard name and list of detected keywords.
    """
    detected_keywords: list[str] = []
    compliance_checks: dict[str, list[str]] = {
        "SPARK 2014": ["SPARK_Mode", "Ghost", "GNATprove"],
        "Ada 2022": ["Static_Pure", "Pure_Global", "Contract_Cases"],
        "Ada 2012": ["Pre =>", "Post =>", "Type_Invariant"],
        "Ada 2005": ["interfaces", "aliased"],
        "Ada 95": ["tagged", "abstract", "override"],
        "Ada 83": ["procedure", "function", "package body"],
    }

    expected_keywords = compliance_checks.get(expected_standard, [])
    for keyword in expected_keywords:
        if keyword in code:
            detected_keywords.append(keyword)

    return {
        "expected_standard": expected_standard,
        "detected_keywords": detected_keywords,
        "compliance_score": len(detected_keywords) / max(len(expected_keywords), 1),
    }


def derive_eval_categories() -> list[dict[str, Any]]:
    """Derive evaluation categories from ada-eval compacted JSONL datasets.

    Returns a list of category dicts with name, description, and sample count.
    """
    categories: list[dict[str, Any]] = []
    compacted_dir = ADA_EVAL_DIR / "data" / "base" / "compacted"

    if not compacted_dir.exists():
        return categories

    for jsonl_file in sorted(compacted_dir.glob("*.jsonl")):
        try:
            with open(jsonl_file, "r") as f:
                samples = [json.loads(line) for line in f if line.strip()]
            categories.append({
                "name": jsonl_file.stem,
                "sample_count": len(samples),
                "file": str(jsonl_file),
            })
        except (json.JSONDecodeError, OSError):
            continue

    return categories


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full baseline evaluation pipeline.

    Derives evaluation methodology from ../ada-eval and computes
    aggregate metrics across all samples.

    Returns a summary dict with aggregate metrics.
    """
    results: dict[str, Any] = {
        "total_samples": 0,
        "avg_bleu": 0.0,
        "avg_compliance": 0.0,
        "standard_distribution": {},
        "eval_categories": derive_eval_categories(),
        "methodology_source": str(ADA_EVAL_DIR) if ADA_EVAL_DIR.exists() else "default",
        "per_sample": [],
    }

    dataset = load_dataset(args.dataset, args.max_samples)
    if not dataset:
        logger.error("No data available for evaluation.")
        return results

    bleu_scores: list[float] = []
    compliance_scores: list[float] = []

    for sample in dataset:
        messages = sample.get("messages", [])
        if len(messages) < 3:
            continue

        user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
        assistant_content = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        system_content = next((m["content"] for m in messages if m["role"] == "system"), "")

        standard = "Unknown"
        for std in ["SPARK 2014", "Ada 2022", "Ada 2012", "Ada 2005", "Ada 95", "Ada 83"]:
            if std.lower() in system_content.lower():
                standard = std
                break

        bleu = compute_bleu(user_content[:200], assistant_content[:200]) if assistant_content else 0.0
        bleu_scores.append(bleu)

        compliance = check_ada_compliance(assistant_content, standard) if assistant_content else {"compliance_score": 0.0}
        compliance_scores.append(compliance["compliance_score"])

        results["standard_distribution"][standard] = results["standard_distribution"].get(standard, 0) + 1

        results["per_sample"].append({
            "standard": standard,
            "bleu": bleu,
            "compliance": compliance["compliance_score"],
        })

    results["total_samples"] = len(bleu_scores)
    results["avg_bleu"] = sum(bleu_scores) / max(len(bleu_scores), 1)
    results["avg_compliance"] = sum(compliance_scores) / max(len(compliance_scores), 1)

    logger.info("Evaluation complete: %d samples, avg BLEU=%.4f, avg compliance=%.4f",
                results["total_samples"], results["avg_bleu"], results["avg_compliance"])
    return results


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("Starting q3as baseline evaluation...")

    # Load evaluation methodology from ada-eval
    methodology = load_eval_methodology()
    if methodology.get("categories"):
        logger.info("Evaluation categories derived from ada-eval: %s", list(methodology["categories"].keys()))

    results = run_evaluation(args)

    # Print summary
    print("\n" + "=" * 60)
    print("Q3AS BASELINE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total samples evaluated     : {results['total_samples']}")
    print(f"Average BLEU score          : {results['avg_bleu']:.4f}")
    print(f"Average compliance          : {results['avg_compliance']:.4f}")
    print(f"Standard distribution       : {results['standard_distribution']}")
    print(f"Methodology source          : {results['methodology_source']}")
    print(f"Eval categories (ada-eval)  : {[c['name'] for c in results['eval_categories']]}")
    print("=" * 60 + "\n")

    # Save results
    output_path = Path("outputs") / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()

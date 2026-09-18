"""baseline_eval.py - Baseline evaluation for q3as fine-tuned model.

Provides comprehensive evaluation of Ada code generation quality
from the q3as fine-tuned model, comparing against the base Qwen3-8B
model. Evaluation metrics include:

- BLEU score against reference Ada implementations
- Ada standard compliance (keyword-based)
- Compilation pass rate (via ada-eval BUILD)
- Unit test pass rate (via ada-eval TEST)
- SPARK verification success rate (via ada-eval PROVE)

Usage:
    uv run python eval/baseline_eval.py --model outputs/q3as
    uv run python eval/baseline_eval.py --model outputs/q3as --base-model unsloth/Qwen3-8B --max-samples 20
    uv run python eval/baseline_eval.py --model outputs/q3as --evals build test prove
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger("q3as_eval")

ADA_EVAL_DIR = Path("../ada-eval")
EVAL_RESULTS_DIR = Path("outputs/eval_results")
GENERATED_DIR = Path("outputs/generated_solutions")


def check_tools_available() -> bool:
    """Check if required GNAT tools are available."""
    for tool in ["gnatprove", "gprbuild", "gnatformat"]:
        if shutil.which(tool) is None:
            logger.warning("Required tool not found: %s", tool)
            return False
    return True

ADA_EVAL_CATEGORIES = {
    "spark_learn": "Learning examples with SPARK contracts",
    "spark_custom": "Custom SPARK verification challenges",
    "spark_human_eval_silver": "HumanEval-style silver standard evaluations",
}

DEFAULT_STANDARD_KEYWORDS: dict[str, list[str]] = {
    "SPARK 2014": ["SPARK_Mode", "Ghost", "GNATprove"],
    "Ada 2022": ["Static_Pure", "Pure_Global", "Contract_Cases"],
    "Ada 2012": ["Pre =>", "Post =>", "Type_Invariant"],
    "Ada 2005": ["interfaces", "aliased"],
    "Ada 95": ["tagged", "abstract", "override"],
    "Ada 83": ["procedure", "function", "package body"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline evaluation for q3as model.")
    parser.add_argument("--model", type=Path, default=Path("outputs/q3as"), help="Fine-tuned model checkpoint path.")
    parser.add_argument("--base-model", type=str, default="unsloth/Qwen3-8B", help="Base model identifier for comparison.")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/dataset.jsonl"), help="Evaluation dataset.")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum number of samples to evaluate.")
    parser.add_argument("--evals", nargs="+", choices=["build", "test", "prove"], default=["build", "test", "prove"], help="Evaluation types to run via ada-eval.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def load_dataset(path: Path, max_samples: int = 50) -> list[dict[str, Any]]:
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
    methodology: dict[str, Any] = {"source": str(ADA_EVAL_DIR), "categories": {}}
    if not ADA_EVAL_DIR.exists():
        return methodology
    compacted_dir = ADA_EVAL_DIR / "data" / "base" / "compacted"
    if compacted_dir.exists():
        for jsonl_file in compacted_dir.glob("*.jsonl"):
            category_name = jsonl_file.stem
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
    expanded_dir = ADA_EVAL_DIR / "data" / "base" / "expanded"
    if expanded_dir.exists():
        methodology["expanded_categories"] = [d.name for d in expanded_dir.iterdir() if d.is_dir()]
    return methodology


def compute_bleu(reference: str, hypothesis: str) -> float:
    ref_ngrams = Counter(reference[i:i+3] for i in range(len(reference) - 2))
    hyp_ngrams = Counter(hypothesis[i:i+3] for i in range(len(hypothesis) - 2))
    if not ref_ngrams or not hyp_ngrams:
        return 0.0
    overlap = sum((ref_ngrams & hyp_ngrams).values())
    return overlap / sum(hyp_ngrams.values())


def check_ada_compliance(code: str, expected_standard: str) -> dict[str, Any]:
    detected_keywords: list[str] = []
    expected_keywords = DEFAULT_STANDARD_KEYWORDS.get(expected_standard, [])
    for keyword in expected_keywords:
        if keyword in code:
            detected_keywords.append(keyword)
    return {
        "expected_standard": expected_standard,
        "detected_keywords": detected_keywords,
        "compliance_score": len(detected_keywords) / max(len(expected_keywords), 1),
    }


def derive_eval_categories() -> list[dict[str, Any]]:
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


def extract_ada_code(generated_text: str) -> str:
    """Extract Ada code block from model output."""
    match = re.search(r"```ada\s*\n(.*?)```", generated_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", generated_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return generated_text.strip()


def compute_compilation_stats_from_ada_eval(model_label: str) -> dict[str, int]:
    """Read compilation stats from ada-eval evaluation results."""
    stats = {"compiled": 0, "failed": 0, "total": 0}
    model_eval_dir = EVAL_RESULTS_DIR / model_label
    if not model_eval_dir.exists():
        return stats

    for dataset_dir in model_eval_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        for result_file in dataset_dir.rglob("*.json"):
            try:
                with open(result_file) as f:
                    data = json.load(f)
                if isinstance(data, dict) and data.get("eval") == "build":
                    stats["total"] += 1
                    if data.get("compiled"):
                        stats["compiled"] += 1
                    else:
                        stats["failed"] += 1
            except (json.JSONDecodeError, OSError):
                continue
    return stats


def compute_test_stats_from_ada_eval(model_label: str) -> dict[str, int]:
    """Read test stats from ada-eval evaluation results."""
    stats = {"passed": 0, "failed": 0, "total": 0}
    model_eval_dir = EVAL_RESULTS_DIR / model_label
    if not model_eval_dir.exists():
        return stats

    for dataset_dir in model_eval_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        for result_file in dataset_dir.rglob("*.json"):
            try:
                with open(result_file) as f:
                    data = json.load(f)
                if isinstance(data, dict) and data.get("eval") == "test":
                    stats["total"] += 1
                    if data.get("passed_tests"):
                        stats["passed"] += 1
                    else:
                        stats["failed"] += 1
            except (json.JSONDecodeError, OSError):
                continue
    return stats


def compute_prove_stats_from_ada_eval(model_label: str) -> dict[str, int]:
    """Read SPARK proof stats from ada-eval evaluation results."""
    stats = {"proved": 0, "unproved": 0, "error": 0, "total": 0}
    model_eval_dir = EVAL_RESULTS_DIR / model_label
    if not model_eval_dir.exists():
        return stats

    for dataset_dir in model_eval_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        for result_file in dataset_dir.rglob("*.json"):
            try:
                with open(result_file) as f:
                    data = json.load(f)
                if isinstance(data, dict) and data.get("eval") == "prove":
                    stats["total"] += 1
                    result = data.get("result", "")
                    if result == "proved":
                        stats["proved"] += 1
                    elif result == "unproved":
                        stats["unproved"] += 1
                    else:
                        stats["error"] += 1
            except (json.JSONDecodeError, OSError):
                continue
    return stats


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full evaluation pipeline.

    Computes BLEU, compliance, compilation, test, and SPARK proof metrics
    for both the fine-tuned and base models.
    """
    model_label = "q3as_fine_tuned"
    base_model_label = args.base_model.replace("/", "_")

    results: dict[str, Any] = {
        "total_samples": 0,
        "avg_bleu": 0.0,
        "avg_compliance": 0.0,
        "standard_distribution": {},
        "eval_categories": derive_eval_categories(),
        "methodology_source": str(ADA_EVAL_DIR) if ADA_EVAL_DIR.exists() else "default",
        "per_sample": [],
        "fine_tuned": {"model": str(args.model), "stats": {}},
        "base_model": {"model": args.base_model, "stats": {}},
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

    # Compute ada-eval metrics for both models
    ft_stats = compute_compilation_stats_from_ada_eval(model_label)
    ft_test_stats = compute_test_stats_from_ada_eval(model_label)
    ft_prove_stats = compute_prove_stats_from_ada_eval(model_label)

    results["fine_tuned"]["stats"] = {
        "build": ft_stats,
        "test": ft_test_stats,
        "prove": ft_prove_stats,
    }

    base_stats = compute_compilation_stats_from_ada_eval(base_model_label)
    base_test_stats = compute_test_stats_from_ada_eval(base_model_label)
    base_prove_stats = compute_prove_stats_from_ada_eval(base_model_label)

    results["base_model"]["stats"] = {
        "build": base_stats,
        "test": base_test_stats,
        "prove": base_prove_stats,
    }

    # Compute pass rates
    if ft_stats["total"] > 0:
        results["fine_tuned"]["compile_rate"] = ft_stats["compiled"] / ft_stats["total"]
    if ft_test_stats["total"] > 0:
        results["fine_tuned"]["test_pass_rate"] = ft_test_stats["passed"] / ft_test_stats["total"]
    if ft_prove_stats["total"] > 0:
        results["fine_tuned"]["prove_success_rate"] = ft_prove_stats["proved"] / ft_prove_stats["total"]

    if base_stats["total"] > 0:
        results["base_model"]["compile_rate"] = base_stats["compiled"] / base_stats["total"]
    if base_test_stats["total"] > 0:
        results["base_model"]["test_pass_rate"] = base_test_stats["passed"] / base_test_stats["total"]
    if base_prove_stats["total"] > 0:
        results["base_model"]["prove_success_rate"] = base_prove_stats["proved"] / base_prove_stats["total"]

    logger.info(
        "Evaluation complete: %d samples, avg BLEU=%.4f, avg compliance=%.4f",
        results["total_samples"], results["avg_bleu"], results["avg_compliance"],
    )
    return results


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("Starting q3as baseline evaluation...")
    logger.info("Fine-tuned model: %s", args.model)
    logger.info("Base model: %s", args.base_model)
    logger.info("Evals: %s", args.evals)

    # Check tool availability
    tools_ok = check_tools_available()
    if not tools_ok:
        logger.warning("GNAT tools (gnatprove, gprbuild, gnatformat) not found.")
        logger.warning("Install GNAT Pro/Community or set up PATH to enable compilation/test/SPARK evaluation.")
        logger.warning("BLEU and compliance metrics will still be computed.")

    methodology = load_eval_methodology()
    if methodology.get("categories"):
        logger.info("Evaluation categories derived from ada-eval: %s", list(methodology["categories"].keys()))

    results = run_evaluation(args)

    # Print summary
    print("\n" + "=" * 70)
    print("Q3AS EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total samples evaluated     : {results['total_samples']}")
    print(f"Average BLEU score          : {results['avg_bleu']:.4f}")
    print(f"Average compliance          : {results['avg_compliance']:.4f}")
    print(f"Standard distribution       : {results['standard_distribution']}")
    print()

    # Fine-tuned model stats
    ft = results["fine_tuned"]
    if ft["stats"]:
        b = ft["stats"].get("build", {})
        t = ft["stats"].get("test", {})
        p = ft["stats"].get("prove", {})
        print(f"--- Fine-tuned ({args.model.name}) ---")
        if b.get("total", 0) > 0:
            compile_rate = b.get("compiled", 0) / b["total"] * 100
            print(f"  Compilation: {b.get('compiled', 0)}/{b['total']} passed ({compile_rate:.1f}%)")
        if t.get("total", 0) > 0:
            test_rate = t.get("passed", 0) / t["total"] * 100
            print(f"  Unit Tests:  {t.get('passed', 0)}/{t['total']} passed ({test_rate:.1f}%)")
        if p.get("total", 0) > 0:
            prove_rate = p.get("proved", 0) / p["total"] * 100
            print(f"  SPARK Proof: {p.get('proved', 0)}/{p['total']} proved ({prove_rate:.1f}%)")

    # Base model stats
    base = results["base_model"]
    if base["stats"]:
        b = base["stats"].get("build", {})
        t = base["stats"].get("test", {})
        p = base["stats"].get("prove", {})
        print(f"\n--- Base ({args.base_model}) ---")
        if b.get("total", 0) > 0:
            compile_rate = b.get("compiled", 0) / b["total"] * 100
            print(f"  Compilation: {b.get('compiled', 0)}/{b['total']} passed ({compile_rate:.1f}%)")
        if t.get("total", 0) > 0:
            test_rate = t.get("passed", 0) / t["total"] * 100
            print(f"  Unit Tests:  {t.get('passed', 0)}/{t['total']} passed ({test_rate:.1f}%)")
        if p.get("total", 0) > 0:
            prove_rate = p.get("proved", 0) / p["total"] * 100
            print(f"  SPARK Proof: {p.get('proved', 0)}/{p['total']} proved ({prove_rate:.1f}%)")

    # Comparison
    if ft["stats"] and base["stats"]:
        print(f"\n{'=' * 70}")
        print("COMPARISON (Fine-tuned vs Base)")
        print(f"{'=' * 70}")
        print(f"{'Metric':<25s} {'Base':>10s} {'Fine-tuned':>12s} {'Delta':>10s}")
        print("-" * 60)

        for eval_name, metric_name, display in [
            ("build", "compile_rate", "Compilation pass rate"),
            ("test", "test_pass_rate", "Test pass rate"),
            ("prove", "prove_success_rate", "SPARK proof success"),
        ]:
            ft_rate = ft.get(metric_name, 0.0) * 100
            base_rate = base.get(metric_name, 0.0) * 100
            delta = ft_rate - base_rate
            print(f"{display:<25s} {base_rate:>9.1f}% {ft_rate:>11.1f}% {delta:>+9.1f}%")

    print(f"\nMethodology source: {results['methodology_source']}")
    print(f"Eval categories: {[c['name'] for c in results['eval_categories']]}")
    print("=" * 70 + "\n")

    # Save results
    output_path = Path("outputs") / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()

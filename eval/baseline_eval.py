"""baseline_eval.py - Reference-based evaluation of q3as generation quality.

Scores what the models actually produced. `make generate` writes one record
per benchmark sample to ``outputs/generated_solutions/<label>/<dataset>.jsonl``,
each carrying the model's ``generated_solution`` as a base64 file map. This
module joins those against the ada-eval canonical solutions (by dataset and
sample name) and reports, per model:

- BLEU-4 against the canonical solution (correct implementation: clipped
  n-gram precisions, brevity penalty, add-one smoothing for the high-order
  precisions that short Ada files usually zero out)
- exact-match rate of the subprogram's own source file
- file-set match rate (did the model produce the reference's project layout)
- Ada standard compliance, scored against the standard detected on the
  *canonical* solution
- compilation, unit-test, and SPARK proof rates (from ada-eval results)

It deliberately does not read the training dataset: scoring the gold answers
against themselves measures the corpus, not the model. The base model
defaults to the LOCAL download (models/qwen3-8b, i.e. Qwen/Qwen3-8B) - the
same weights used for fine-tuning - so the comparison isolates the effect of
fine-tuning.

The comparison is only as good as the generations on disk. With no
``make generate`` output this exits non-zero rather than reporting zeros,
because a table of 0.0% rates reads as a measurement.

Usage:
    uv run python eval/baseline_eval.py --model outputs/q3as
    uv run python eval/baseline_eval.py --model outputs/q3as --max-samples 20
    uv run python eval/baseline_eval.py --model outputs/q3as --evals build prove
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add scripts/ so the Alire environment helper is importable
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
_PROCESSING_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "data" / "processing_scripts"
if str(_PROCESSING_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PROCESSING_SCRIPTS_DIR))

from ada_eval_common import aggregate_eval_results
from alire_env import has_tool

logger = logging.getLogger("q3as_eval")

# ada-eval lives in the source cache (fetch_repos.py); fall back to the
# legacy sibling layout for checkouts that have not re-run setup yet.
# The dataset builder owns Ada standard detection; reuse it rather than
# re-deriving standards here (its comment-stripped weighted matcher is the
# one the training data was built with).
import build_dataset as bd
import source_paths

ADA_EVAL_DIR = source_paths.resolve("ada-eval") or Path("data/raw_repos/_missing/ada-eval")
EVAL_RESULTS_DIR = Path("outputs/eval_results")
GENERATED_DIR = Path("outputs/generated_solutions")
DEFAULT_BASE_MODEL = Path("models/qwen3-8b")
BASE_MODEL_LABEL = "base_qwen3-8b"
FINE_TUNED_LABEL = "fine_tuned"

# The tools each ada-eval kind needs. Honoured by --evals, so asking for
# only BUILD does not fail because gnatprove is missing.
EVAL_TOOLS: dict[str, tuple[str, ...]] = {
    "build": ("gprbuild",),
    "test": ("gprbuild", "gprclean"),
    "prove": ("gprbuild", "gnatprove"),
}

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

# Ada-aware tokenisation: identifiers, numbers, character literals, strings,
# and multi-character operators. Whitespace splitting would glue operators to
# operands and make n-gram overlap meaningless.
_TOKEN_RE = re.compile(
    r"""
    '(?:[^']|'')+'          # character literal
    | "(?:[^"]|"")*"        # string literal
    | [A-Za-z][A-Za-z0-9_]* # identifier
    | \d+(?:\.\d+)?(?:[eE][-+]?\d+)?   # numeric literal
    | \*\*|:=|<=|>=|/=|<>|=>|\.\.|<<|>>  # multi-character operators
    | [^\s]                 # any single remaining character
    """,
    re.VERBOSE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference-based evaluation of q3as generation quality."
    )
    parser.add_argument(
        "--model", type=Path, default=Path("outputs/q3as"),
        help="Fine-tuned model checkpoint path (used for the label and a presence check).",
    )
    parser.add_argument(
        "--base-model", type=Path, default=DEFAULT_BASE_MODEL,
        help="Base model path (default: local models/qwen3-8b download of Qwen/Qwen3-8B).",
    )
    parser.add_argument(
        "--generated-dir", type=Path, default=GENERATED_DIR,
        help="Root of the per-model generations written by `make generate`.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Cap on scored samples per model (0 = all).",
    )
    parser.add_argument(
        "--evals", nargs="+", choices=sorted(EVAL_TOOLS), default=sorted(EVAL_TOOLS),
        help="ada-eval result kinds to summarise.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Generation / reference loading
# --------------------------------------------------------------------------- #


def _dataset_of(path: Path) -> str:
    """Recover the ada-eval dataset name from a generated file name.

    generate.py writes ``spark_<dataset>.jsonl``; the ada-eval dataset is
    already prefixed with ``spark_``, which makes the doubled prefix easy to
    mis-parse. Stripping one leading ``spark_`` recovers the real name.
    """
    stem = path.stem
    return stem.removeprefix("spark_")


def load_reference_index(ada_eval_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Index the canonical solutions by (dataset, sample name)."""
    compacted = ada_eval_dir / "data" / "base" / "compacted"
    index: dict[tuple[str, str], dict[str, Any]] = {}
    if not compacted.exists():
        logger.warning("No ada-eval reference data at %s", compacted)
        return index
    for jsonl_file in sorted(compacted.glob("*.jsonl")):
        dataset = jsonl_file.stem
        try:
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    index[(dataset, sample.get("name", ""))] = sample
        except OSError as exc:
            logger.warning("Cannot read reference %s: %s", jsonl_file, exc)
    return index


def load_generated(generated_dir: Path, label: str) -> list[tuple[str, dict[str, Any]]]:
    """Load one model's generations as (dataset, record) pairs."""
    model_dir = generated_dir / label
    if not model_dir.exists():
        logger.warning("No generations for %s at %s", label, model_dir)
        return []
    records: list[tuple[str, dict[str, Any]]] = []
    for jsonl_file in sorted(model_dir.glob("*.jsonl")):
        dataset = _dataset_of(jsonl_file)
        try:
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        records.append((dataset, json.loads(line)))
                    except json.JSONDecodeError:
                        continue
        except OSError as exc:
            logger.warning("Cannot read generations %s: %s", jsonl_file, exc)
    return records


def decode_files(solution: Any) -> dict[str, str]:
    """Decode an ada-eval file map ({name: base64}) into {name: text}."""
    if not isinstance(solution, dict):
        return {}
    decoded: dict[str, str] = {}
    for name, blob in solution.items():
        if not isinstance(blob, str):
            continue
        try:
            decoded[name] = base64.b64decode(blob, validate=False).decode("utf-8", "replace")
        except (ValueError, TypeError) as exc:
            logger.debug("Cannot decode %s: %s", name, exc)
    return decoded


def normalise_code(text: str) -> str:
    """Normalise line endings and trailing whitespace for comparison."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def tokenize(code: str) -> list[str]:
    """Split Ada source into comparable tokens."""
    return _TOKEN_RE.findall(code)


def bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Corpus BLEU-4 for a single reference/hypothesis pair.

    Standard BLEU: geometric mean of clipped n-gram precisions for n=1..4,
    multiplied by a brevity penalty. Short Ada files almost always produce a
    zero 3- or 4-gram precision, which would zero the whole score, so each
    higher-order precision gets add-one smoothing on its denominator. The
    result is comparable across samples; it is not corpus BLEU (no
    multi-reference aggregation).
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        if len(hyp_tokens) < n:
            # Not enough tokens to form any n-gram: treat as smoothed 0.
            precisions.append(0.0)
            continue
        ref_ngrams = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
        hyp_ngrams = Counter(tuple(hyp_tokens[i:i + n]) for i in range(len(hyp_tokens) - n + 1))
        clipped = sum((ref_ngrams & hyp_ngrams).values())
        total = sum(hyp_ngrams.values())
        precisions.append((clipped + 1) / (total + 1))

    if min(precisions) <= 0.0:
        # A zero precision (smoothing cannot rescue a 0/0) must not be
        # smoothed into a positive score, but it also must not poison the
        # geometric mean into a hard zero for an otherwise close match.
        precisions = [p if p > 0.0 else 1e-9 for p in precisions]

    log_mean = sum(math.log(p) for p in precisions) / len(precisions)
    brevity_penalty = (
        1.0
        if len(hyp_tokens) > len(ref_tokens)
        else math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))
    )
    return brevity_penalty * math.exp(log_mean)


def check_ada_compliance(code: str, expected_standard: str) -> dict[str, Any]:
    """Score a generated file against the markers of its target standard."""
    detected_keywords = [kw for kw in DEFAULT_STANDARD_KEYWORDS.get(expected_standard, []) if kw in code]
    expected = DEFAULT_STANDARD_KEYWORDS.get(expected_standard, [])
    return {
        "expected_standard": expected_standard,
        "detected_keywords": detected_keywords,
        "compliance_score": len(detected_keywords) / max(len(expected), 1),
    }


# --------------------------------------------------------------------------- #
# Per-model scoring
# --------------------------------------------------------------------------- #


def score_model(
    label: str,
    generated_dir: Path,
    references: dict[tuple[str, str], dict[str, Any]],
    max_samples: int = 0,
) -> dict[str, Any]:
    """Score one model's generations against the canonical solutions.

    Reports coverage as well as scores: the benchmark has 19 samples but a
    generation run may only have produced some of them, and a rate computed
    over 7 samples must not read like one computed over 19.
    """
    generated = load_generated(generated_dir, label)
    if max_samples > 0:
        generated = generated[:max_samples]

    per_sample: list[dict[str, Any]] = []
    unmatched = 0
    for dataset, record in generated:
        name = record.get("name", "")
        reference = references.get((dataset, name))
        if reference is None:
            unmatched += 1
            logger.warning("No canonical solution for %s/%s; skipping", dataset, name)
            continue

        primary = (record.get("location") or {}).get("path", "")
        gen_files = decode_files(record.get("generated_solution"))
        ref_files = decode_files(reference.get("canonical_solution"))
        if not gen_files:
            logger.warning("Empty generated_solution for %s/%s", dataset, name)
            continue

        gen_primary = normalise_code(gen_files.get(primary, ""))
        ref_primary = normalise_code(ref_files.get(primary, ""))
        # Standards are a property of the reference; detecting them on the
        # model output would grade the model on its own guess.
        standard = bd.detect_ada_standard(ref_primary) if ref_primary else "Unknown"
        compliance = check_ada_compliance(gen_primary, standard)

        per_sample.append({
            "dataset": dataset,
            "name": name,
            "primary_file": primary,
            "standard": standard,
            "bleu": bleu(ref_primary, gen_primary),
            "exact_match": bool(ref_primary) and gen_primary == ref_primary,
            "file_set_match": set(gen_files) == set(ref_files),
            "compliance": compliance["compliance_score"],
        })

    scored = len(per_sample)
    result: dict[str, Any] = {
        "label": label,
        "samples_generated": len(generated),
        "samples_scored": scored,
        "samples_unmatched": unmatched,
        "reference_total": len(references),
        "per_sample": per_sample,
    }
    if scored:
        result.update({
            "bleu": sum(s["bleu"] for s in per_sample) / scored,
            "exact_match_rate": sum(1 for s in per_sample if s["exact_match"]) / scored,
            "file_set_match_rate": sum(1 for s in per_sample if s["file_set_match"]) / scored,
            "compliance": sum(s["compliance"] for s in per_sample) / scored,
            "standard_distribution": dict(Counter(s["standard"] for s in per_sample)),
        })
    return result


# --------------------------------------------------------------------------- #
# ada-eval BUILD / TEST / PROVE results
# --------------------------------------------------------------------------- #


def compute_stats_from_ada_eval(model_label: str, evals: list[str] | None = None) -> dict[str, Any]:
    """Read build/test/prove stats for one model from the ada-eval results."""
    return aggregate_eval_results(EVAL_RESULTS_DIR / model_label, evals)


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #


def check_tools_available(evals: list[str]) -> bool:
    """Check the tools the requested evals need, in the Alire environment."""
    ok = True
    for tool in sorted({t for kind in evals for t in EVAL_TOOLS.get(kind, ())}):
        if not has_tool(tool):
            logger.warning("Required tool not found in the Alire environment: %s", tool)
            ok = False
    return ok


def load_eval_methodology() -> dict[str, Any]:
    methodology: dict[str, Any] = {"source": str(ADA_EVAL_DIR), "categories": {}}
    if not ADA_EVAL_DIR.exists():
        return methodology
    compacted_dir = ADA_EVAL_DIR / "data" / "base" / "compacted"
    if compacted_dir.exists():
        for jsonl_file in compacted_dir.glob("*.jsonl"):
            try:
                with open(jsonl_file, encoding="utf-8") as f:
                    count = sum(1 for _ in f)
                methodology["categories"][jsonl_file.stem] = {
                    "description": ADA_EVAL_CATEGORIES.get(jsonl_file.stem, "Custom category"),
                    "sample_count": count,
                    "file": str(jsonl_file),
                }
            except OSError:
                continue
    return methodology


def print_stats_block(stats: dict[str, Any]) -> None:
    for key, label, numerator in (
        ("build", "Compilation", "compiled"),
        ("test", "Unit Tests", "passed"),
        ("prove", "SPARK Proof", "proved"),
    ):
        block = stats.get(key, {})
        total = block.get("total", 0)
        if total > 0:
            print(f"  {label}: {block.get(numerator, 0)}/{total} ({block[numerator] / total * 100:.1f}%)")


def main() -> int:
    args = parse_args()
    # basicConfig must come first: it sets the root level when no handlers
    # exist, and would otherwise overwrite a setLevel(DEBUG) applied before it.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Starting q3as reference-based evaluation")
    logger.info("Fine-tuned model: %s", args.model)
    logger.info("Base model: %s", args.base_model)
    logger.info("Evals: %s", args.evals)
    logger.info("Generations root: %s", args.generated_dir)

    if not args.model.exists():
        logger.error("Fine-tuned model not found at %s - run `make train` first.", args.model)
        return 1

    if not check_tools_available(args.evals):
        logger.warning(
            "Some tools for %s are missing; run `make prove` for the full "
            "ada-eval pipeline. Generation metrics are unaffected.",
            args.evals,
        )

    references = load_reference_index(ADA_EVAL_DIR)
    if not references:
        logger.error(
            "No ada-eval canonical solutions at %s; run `make fetch-sources`.", ADA_EVAL_DIR
        )
        return 1
    logger.info("Indexed %d canonical solutions", len(references))

    labels = [(FINE_TUNED_LABEL, str(args.model)), (BASE_MODEL_LABEL, str(args.base_model))]
    per_model: dict[str, Any] = {}
    for label, model_path in labels:
        scored = score_model(label, args.generated_dir, references, args.max_samples)
        scored["model"] = model_path
        scored["ada_eval"] = compute_stats_from_ada_eval(label, args.evals)
        per_model[label] = scored
        logger.info(
            "%s: %d/%d samples scored (benchmark has %d), BLEU=%.4f, exact=%d",
            label, scored.get("samples_scored", 0), scored.get("samples_generated", 0),
            scored.get("reference_total", 0), scored.get("bleu", 0.0),
            sum(1 for s in scored["per_sample"] if s["exact_match"]),
        )
        scored_n, benchmark_n = scored.get("samples_scored", 0), scored.get("reference_total", 0)
        if benchmark_n and scored_n < benchmark_n:
            logger.warning(
                "%s covers only %d of the %d benchmark samples. Rates below cover the "
                "generated subset only; re-run `make generate` for full coverage.",
                label, scored_n, benchmark_n,
            )

    scored_any = [m for m in per_model.values() if m.get("samples_scored")]
    if not scored_any:
        logger.error(
            "No generated solutions found under %s. Run `make generate` first; "
            "there is nothing to score.",
            args.generated_dir,
        )
        return 1

    results: dict[str, Any] = {
        # Key names are the contract scripts/gen_eval_report.py reads.
        "bleu": per_model[FINE_TUNED_LABEL].get("bleu", 0.0),
        "compliance": per_model[FINE_TUNED_LABEL].get("compliance", 0.0),
        "per_model": per_model,
        "generated": {
            "root": str(args.generated_dir),
            "reference_total": len(references),
            "samples": {label: m.get("samples_scored", 0) for label, m in per_model.items()},
        },
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "evals": args.evals,
        "eval_categories": sorted(ADA_EVAL_CATEGORIES),
        "methodology_source": str(ADA_EVAL_DIR),
        "metric_notes": {
            "bleu": "BLEU-4, add-one smoothed, brevity penalty, single reference",
            "exact_match": "normalised text equality on the subprogram's own source file",
            "compliance": "standard detected on the canonical solution, not on the model output",
        },
    }

    print("\n" + "=" * 70)
    print("Q3AS EVALUATION SUMMARY")
    print("=" * 70)
    for label, model in per_model.items():
        print(f"--- {label} ({model['model']}) ---")
        benchmark = model.get("reference_total", 0)
        scored_count = model.get("samples_scored", 0)
        coverage = f"{scored_count}/{benchmark} of the benchmark" if benchmark else str(scored_count)
        print(f"  Samples scored  : {coverage} (generated {model.get('samples_generated', 0)})")
        if benchmark and scored_count < benchmark:
            print(
                f"  ** partial coverage: {benchmark - scored_count} benchmark samples "
                "have no generation **"
            )
        if model.get("samples_scored"):
            print(f"  BLEU-4          : {model['bleu']:.4f}")
            print(
                f"  Exact match     : {model['exact_match_rate'] * 100:.1f}%"
                f"  ({sum(1 for s in model['per_sample'] if s['exact_match'])}/{model['samples_scored']})"
            )
            print(f"  File-set match  : {model['file_set_match_rate'] * 100:.1f}%")
            print(f"  Std compliance  : {model['compliance']:.4f}")
            print(f"  Standards       : {model.get('standard_distribution', {})}")
        print_stats_block(model["ada_eval"])

    ft, base = per_model[FINE_TUNED_LABEL], per_model[BASE_MODEL_LABEL]
    if ft.get("samples_scored") and base.get("samples_scored"):
        print(f"\n{'=' * 70}")
        print("COMPARISON (Fine-tuned vs Base)")
        print(f"{'=' * 70}")
        print(f"{'Metric':<22s} {'Base':>10s} {'Fine-tuned':>12s} {'Delta':>10s}")
        print("-" * 56)
        for key, display in (
            ("bleu", "BLEU-4"),
            ("exact_match_rate", "Exact match"),
            ("file_set_match_rate", "File-set match"),
            ("compliance", "Std compliance"),
        ):
            b, f = base.get(key, 0.0), ft.get(key, 0.0)
            print(f"{display:<22s} {b:>9.1%} {f:>11.1%} {f - b:>+9.1%}")
        print(f"\n{'=' * 70}")

    print(f"Methodology source: {results['methodology_source']}")
    print("=" * 70 + "\n")

    output_path = Path("outputs") / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results saved to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

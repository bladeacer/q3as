#!/usr/bin/env python3
"""gen_eval_report.py - versioned evaluation reports for q3as.

After every eval run (`make eval-report`, called by `make all`), this tool
writes:

- ``docs/results/result-data-vX.Y.Z.json``   the key metrics (machine readable)
- ``docs/results/result-vX.Y.Z.md``          a concise human-readable summary
- ``docs/results/README.md``                 an index linking every version,
                                             with a comparison table of the
                                             last three versions

The version comes from ``alire.toml`` (single source, via bump_version).
Re-running for the same version overwrites that version's files in place.

Inputs (all optional; the report notes whatever is missing):
- ``outputs/eval_results/<model>/<dataset>/*.jsonl``   ada-eval per-sample results
- ``outputs/eval_results.json``                        baseline_eval aggregate (BLEU etc.)
- ``outputs/comparison_report.txt``                    eval_pipeline text report
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bump_version import read_version

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "docs" / "results"
EVAL_RESULTS_DIR = ROOT / "outputs" / "eval_results"
BASELINE_JSON = ROOT / "outputs" / "eval_results.json"
PIPELINE_REPORT = ROOT / "outputs" / "comparison_report.txt"
TRAINING_SUMMARY = ROOT / "outputs" / "q3as" / "training_summary.json"

MODELS = {"base_qwen3-8b": "base", "fine_tuned": "fine-tuned"}


# --------------------------------------------------------------------------- #
# Metric extraction
# --------------------------------------------------------------------------- #

def _percent(hits: int, total: int) -> float | None:
    return round(100.0 * hits / total, 1) if total else None


def collect_ada_eval_metrics(eval_dir: Path) -> dict[str, Any]:
    """Aggregate ada-eval JSONL per-sample results per model and dataset."""
    models: dict[str, Any] = {}
    for model_dir in sorted(eval_dir.iterdir()) if eval_dir.exists() else []:
        if not model_dir.is_dir():
            continue
        samples = build = test = proved = unproved = errors = 0
        datasets: dict[str, dict[str, int]] = {}
        unproved_checks: Counter[str] = Counter()
        proved_checks: Counter[str] = Counter()
        for result_file in sorted(model_dir.rglob("*.jsonl")):
            dataset = result_file.parent.name
            ds = datasets.setdefault(dataset, {"samples": 0, "build": 0, "test": 0})
            for line in result_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for entry in record.get("evaluation_results") or []:
                    kind = entry.get("eval")
                    if kind == "build":
                        samples += 1
                        ds["samples"] += 1
                        compiled = bool(entry.get("compiled"))
                        build += compiled
                        ds["build"] += compiled
                    elif kind == "test":
                        passed = bool(entry.get("passed_tests"))
                        test += passed
                        ds["test"] += passed
                    elif kind == "prove":
                        result = entry.get("result")
                        if result == "proved":
                            proved += 1
                        elif result == "unproved":
                            unproved += 1
                            unproved_checks.update(entry.get("unproved_checks") or {})
                            proved_checks.update(entry.get("proved_checks") or {})
                        else:
                            errors += 1
        if samples == 0:
            continue
        models[model_dir.name] = {
            "samples": samples,
            "build": build,
            "build_pct": _percent(build, samples),
            "test": test,
            "test_pct": _percent(test, samples),
            "proved": proved,
            "unproved": unproved,
            "prove_errors": errors,
            "proved_checks": dict(proved_checks.most_common()),
            "unproved_checks": dict(unproved_checks.most_common(8)),
            "datasets": datasets,
        }
    return models


def collect_baseline_metrics(path: Path) -> dict[str, Any]:
    """Pass through the baseline_eval aggregate, trimmed to known keys."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        key: payload[key]
        for key in ("bleu", "compliance", "per_model", "generated", "timestamp")
        if key in payload
    }


def collect_pipeline_note(path: Path) -> str | None:
    """First lines of the human pipeline report, when present."""
    if not path.exists():
        return None
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return "\n".join(lines[:6]) if lines else None


# --------------------------------------------------------------------------- #
# Training metrics: losses, perplexity, and trend health
# --------------------------------------------------------------------------- #

def _perplexity(loss: float) -> float:
    """exp(loss), the standard LM reporting unit."""
    return round(math.exp(loss), 3)


def _trend_diagnosis(
    eval_history: list[dict[str, float]],
    train_history: list[dict[str, float]],
) -> tuple[str, list[str]]:
    """Judge whether training was smooth and had a healthy trend.

    Verdicts are plain factual statements built from measurable signals:

    - healthy:  val loss reaches its minimum in the second half of the
                eval steps and the final value is well below the first,
    - plateau:  val loss stops improving (the early-stopping callback did
                its job),
    - overfit:  val loss rises late while train loss keeps falling,
    - unstable: any val-loss step jumps upward by more than 25 percent,
    - sparse:   too few evaluation points to judge.

    Returns (verdict, list of findings). Findings name the numbers they
    come from, so the markdown stays auditable.
    """
    findings: list[str] = []
    steps = [p["step"] for p in eval_history]
    losses = [p["eval_loss"] for p in eval_history]
    if len(losses) < 2:
        return ("sparse", ["Fewer than two evaluation points recorded; no trend to judge."])

    first, last, best = losses[0], losses[-1], min(losses)
    best_idx = losses.index(best)
    findings.append(
        f"validation loss {first:.4f} at step {steps[0]} to {last:.4f} at step {steps[-1]}; "
        f"best {best:.4f} at step {steps[best_idx]}"
    )

    verdict = "healthy"
    # Unstable: a single-step jump upward of more than 25 percent.
    for prev, cur, step in zip(losses, losses[1:], steps[1:]):
        if prev > 0 and (cur - prev) / prev > 0.25:
            verdict = "unstable"
            findings.append(f"validation loss jumped {100 * (cur - prev) / prev:.0f}% at step {step}")

    # Overfit: best occurs strictly before the last step and val rose from
    # the best by more than 5 percent while train loss still fell.
    rose_from_best = (last - best) / best if best > 0 else 0.0
    if best_idx < len(losses) - 1 and rose_from_best > 0.05:
        train_last = train_history[-1]["loss"] if train_history else None
        train_at_best = next(
            (p["loss"] for p in reversed(train_history) if p["step"] <= steps[best_idx]),
            None,
        )
        if train_last is not None and train_at_best is not None and train_last < train_at_best:
            verdict = "overfit"
            findings.append(
                f"validation loss rose {100 * rose_from_best:.0f}% after step {steps[best_idx]} "
                f"while train loss kept falling ({train_at_best:.4f} to {train_last:.4f})"
            )
        elif verdict == "healthy":
            verdict = "plateau"
            findings.append(f"validation loss stopped improving after step {steps[best_idx]}")

    if verdict == "healthy":
        improvement = (first - last) / first if first > 0 else 0.0
        findings.append(f"total validation-loss improvement {100 * improvement:.0f}%")
        if best_idx <= len(steps) // 2 and best_idx < len(steps) - 1:
            verdict = "plateau"
            findings.append(
                f"best loss arrived early (step {steps[best_idx]} of {steps[-1]}); "
                "later evaluations did not improve it"
            )
    return (verdict, findings)


def collect_training_metrics(path: Path) -> dict[str, Any]:
    """Read training_summary.json into the report's loss-metrics block.

    Missing file or fields degrade to None entries; the report renders an
    explicit gap instead of inventing numbers.
    """
    if not path.exists():
        return {}
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return {}
    if not isinstance(summary, dict):
        return {}

    train_history: list[dict[str, float]] = summary.get("train_loss_history") or []
    eval_history: list[dict[str, float]] = summary.get("eval_loss_history") or []
    test_metrics = summary.get("test_metrics") or {}

    train_first = train_history[0]["loss"] if train_history else None
    train_last = summary.get("train_loss_final")
    if train_last is None and train_history:
        train_last = train_history[-1]["loss"]
    eval_first = eval_history[0]["eval_loss"] if eval_history else None
    eval_last = eval_history[-1]["eval_loss"] if eval_history else None
    eval_best = min((p["eval_loss"] for p in eval_history), default=None)

    verdict, findings = _trend_diagnosis(eval_history, train_history)

    return {
        "base_model": summary.get("base_model"),
        "steps": (train_history[-1]["step"] if train_history else None),
        "train_loss": {"first": train_first, "last": train_last},
        "val_loss": {"first": eval_first, "last": eval_last, "best": eval_best},
        "test_loss": test_metrics.get("test_eval_loss"),
        "perplexity": {
            "train_last": _perplexity(train_last) if train_last is not None else None,
            "val_best": _perplexity(eval_best) if eval_best is not None else None,
            "test": _perplexity(test_metrics["test_eval_loss"]) if test_metrics.get("test_eval_loss") else None,
        },
        "train_loss_history": train_history,
        "eval_loss_history": eval_history,
        "trend": {"verdict": verdict, "findings": findings},
    }


def render_training_markdown(training: dict[str, Any]) -> list[str]:
    """Render the loss table, trend verdict, and per-step val curve."""
    if not training:
        return [
            "## Training metrics",
            "",
            (
                "_No training metrics found; `outputs/q3as/training_summary.json` is missing. "
                "Run `make train` first._"
            ),
            "",
        ]

    def fmt(value: float | None) -> str:
        return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"

    lines: list[str] = ["## Training metrics", ""]
    tl, vl = training["train_loss"], training["val_loss"]
    ppl = training["perplexity"]
    lines.append("| Metric | Train | Validation (best) | Test (held out) |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Loss | {fmt(tl.get('last'))} | {fmt(vl.get('best'))} | {fmt(training.get('test_loss'))} |"
    )
    lines.append(
        f"| Perplexity | {fmt(ppl.get('train_last'))} | {fmt(ppl.get('val_best'))} | {fmt(ppl.get('test'))} |"
    )
    steps = training.get("steps")
    if steps:
        lines.append("")
        lines.append(f"Loss history over {steps} training steps.")
    lines.append("")

    trend = training.get("trend") or {}
    lines.append(f"**Trend: {trend.get('verdict', 'n/a')}**")
    lines.append("")
    for finding in trend.get("findings") or []:
        lines.append(f"- {finding}")
    lines.append("")

    eval_history = training.get("eval_loss_history") or []
    if eval_history:
        lines.append("Validation loss per evaluation:")
        lines.append("")
        lines.append("| Step | Val loss |")
        lines.append("|---|---|")
        for point in eval_history:
            lines.append(f"| {point['step']} | {point['eval_loss']:.4f} |")
        lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #

def _fmt_pct(value: float | None) -> str:
    return f"{value:.1f}%" if value is not None else "n/a"


def render_markdown(version: str, data: dict[str, Any]) -> str:
    """Render the per-version summary. Concise: tables, findings, artifacts."""
    lines: list[str] = []
    title = f"Results v{version}"
    lines.append(f"# {title}")
    lines.append("")
    ran = data.get("generated_at", "")
    lines.append(f"Eval run captured {ran}. Data: [`result-data-v{version}.json`](result-data-v{version}.json).")
    lines.append("")

    ada = data.get("ada_eval") or {}
    base = ada.get("base_qwen3-8b")
    ft = ada.get("fine_tuned")
    if base and ft:
        lines.append("## Headline (ada-eval)")
        lines.append("")
        lines.append("| Metric | Base | Fine-tuned | Δ |")
        lines.append("|---|---|---|---|")
        n = ft.get("samples", 0)
        lines.append(f"| Samples | {base.get('samples', 0)} | {n} | |")
        lines.append(
            f"| Build | {base.get('build', 0)}/{base.get('samples', 0)} ({_fmt_pct(base.get('build_pct'))}) "
            f"| {ft.get('build', 0)}/{n} ({_fmt_pct(ft.get('build_pct'))}) "
            f"| {round((ft.get('build_pct') or 0) - (base.get('build_pct') or 0), 1)} pts |"
        )
        lines.append(
            f"| Unit tests | {base.get('test', 0)}/{base.get('samples', 0)} ({_fmt_pct(base.get('test_pct'))}) "
            f"| {ft.get('test', 0)}/{n} ({_fmt_pct(ft.get('test_pct'))}) "
            f"| {round((ft.get('test_pct') or 0) - (base.get('test_pct') or 0), 1)} pts |"
        )
        lines.append(
            f"| SPARK proved | {base.get('proved', 0)} | {ft.get('proved', 0)} | |"
        )
        lines.append(
            f"| Prove errors | {base.get('prove_errors', 0)} | {ft.get('prove_errors', 0)} | |"
        )
        lines.append("")

        ft_unproved = ft.get("unproved_checks") or {}
        if ft_unproved:
            lines.append("Fine-tuned model proof blockers (check kinds left unproved):")
            lines.append("")
            for kind, count in list(ft_unproved.items())[:5]:
                lines.append(f"- `{kind}` x{count}")
            lines.append("")

        per_dataset = ft.get("datasets") or {}
        if per_dataset:
            lines.append("## Per dataset (fine-tuned)")
            lines.append("")
            lines.append("| Dataset | Samples | Build | Test |")
            lines.append("|---|---|---|---|")
            for name, ds in sorted(per_dataset.items()):
                lines.append(
                    f"| {name} | {ds.get('samples', 0)} | {ds.get('build', 0)} | {ds.get('test', 0)} |"
                )
            lines.append("")
    else:
        lines.append("## ada-eval results")
        lines.append("")
        lines.append("_No ada-eval JSONL results found under `outputs/eval_results/`._")
        lines.append("")

    baseline = data.get("baseline") or {}
    if baseline:
        lines.append("## baseline_eval aggregates")
        lines.append("")
        for key in ("bleu", "compliance"):
            if key in baseline:
                lines.append(f"- **{key}**: `{json.dumps(baseline[key])[:200]}`")
        lines.append("")

    note = data.get("pipeline_report_note")
    if note:
        lines.append("## comparison_report.txt excerpt")
        lines.append("")
        lines.append("```text")
        lines.append(note)
        lines.append("```")
        lines.append("")

    lines.extend(render_training_markdown(data.get("training") or {}))

    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `outputs/eval_results/<model>/<dataset>/*.jsonl` (per-sample results)")
    lines.append("- `outputs/generated_solutions/<label>/` (model generations)")
    lines.append("- `outputs/q3as/training_summary.json` (loss history; rendered above)")
    lines.append("")
    lines.append("See the [results index](README.md) for the version comparison table.")
    lines.append("")
    lines.append("[\u2190 Back to results index](README.md)")
    lines.append("")
    return "\n".join(lines)


def fmt_loss(value: Any) -> str:
    """Format an optional loss value for the index table."""
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


def render_index(versions: list[dict[str, Any]]) -> str:
    """Render docs/results/README.md: index + last-3 comparison table."""
    lines: list[str] = []
    lines.append("# Evaluation Results")
    lines.append("")
    lines.append(
        "Per-version summaries of every `make eval` / `make eval-pipeline` run. "
        "Metrics JSON lives beside each markdown file. Regenerate with "
        "`make eval-report` after a run; the version comes from `alire.toml`."
    )
    lines.append("")
    lines.append("## All versions")
    lines.append("")
    lines.append("| Version | Captured | Build (base → fine-tuned) | Test (base → fine-tuned) | Report |")
    lines.append("|---|---|---|---|---|")
    for entry in versions:
        ada = entry.get("ada_eval") or {}
        base = ada.get("base_qwen3-8b") or {}
        ft = ada.get("fine_tuned") or {}
        v = entry["version"]
        lines.append(
            f"| v{v} | {entry.get('generated_at', '?')} "
            f"| {base.get('build', 0)} ({_fmt_pct(base.get('build_pct'))}) → "
            f"{ft.get('build', 0)} ({_fmt_pct(ft.get('build_pct'))}) "
            f"| {base.get('test', 0)} ({_fmt_pct(base.get('test_pct'))}) → "
            f"{ft.get('test', 0)} ({_fmt_pct(ft.get('test_pct'))}) "
            f"| [result-v{v}.md](result-v{v}.md) |"
        )
    lines.append("")

    recent = versions[:3]
    if len(recent) > 1:
        lines.append("## Last " + str(len(recent)) + " versions compared")
        lines.append("")
        header = "| Metric | " + " | ".join(f"v{e['version']}" for e in recent) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(recent) + 1))
        rows: list[tuple[str, list[str]]] = []
        metrics = [
            ("Fine-tuned build %", lambda e: _fmt_pct(((e.get("ada_eval") or {}).get("fine_tuned") or {}).get("build_pct"))),
            ("Fine-tuned test %", lambda e: _fmt_pct(((e.get("ada_eval") or {}).get("fine_tuned") or {}).get("test_pct"))),
            ("Base build %", lambda e: _fmt_pct(((e.get("ada_eval") or {}).get("base_qwen3-8b") or {}).get("build_pct"))),
            ("Proved samples (FT)", lambda e: str(((e.get("ada_eval") or {}).get("fine_tuned") or {}).get("proved", 0))),
            ("Test loss (FT)", lambda e: fmt_loss((e.get("training") or {}).get("test_loss"))),
            ("Training trend", lambda e: str(((e.get("training") or {}).get("trend") or {}).get("verdict", "n/a"))),
        ]
        for label, getter in metrics:
            rows.append((label, [getter(e) for e in recent]))
        for label, cells in rows:
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("_Version order: newest first._")
        lines.append("")

    lines.append("Navigation: [project README](../../README.md) · [architecture](../architecture.md) · [evaluation guide](../evaluation.md)")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def load_existing(version: str) -> dict[str, Any]:
    """Load an existing data file for *version* (for in-place regeneration)."""
    path = RESULTS_DIR / f"result-data-v{version}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def gather(version: str) -> dict[str, Any]:
    data: dict[str, Any] = load_existing(version)
    data["version"] = version
    data["generated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    data["ada_eval"] = collect_ada_eval_metrics(EVAL_RESULTS_DIR)
    baseline = collect_baseline_metrics(BASELINE_JSON)
    if baseline:
        data["baseline"] = baseline
    note = collect_pipeline_note(PIPELINE_REPORT)
    if note:
        data["pipeline_report_note"] = note
    training = collect_training_metrics(TRAINING_SUMMARY)
    if training:
        data["training"] = training
    else:
        data["training"] = {}
    return data


def write_all(version: str, data: dict[str, Any]) -> list[Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data_path = RESULTS_DIR / f"result-data-v{version}.json"
    md_path = RESULTS_DIR / f"result-v{version}.md"
    data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(version, data), encoding="utf-8")

    versions = list_all_versions()
    (RESULTS_DIR / "README.md").write_text(render_index(versions), encoding="utf-8")
    return [data_path, md_path, RESULTS_DIR / "README.md"]


def list_all_versions() -> list[dict[str, Any]]:
    """Every version with a data file, newest first (semver sort)."""
    entries: list[dict[str, Any]] = []
    if RESULTS_DIR.exists():
        for path in RESULTS_DIR.glob("result-data-v*.json"):
            match = re.fullmatch(r"result-data-v(\d+\.\d+\.\d+)\.json", path.name)
            if not match:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            entries.append(payload)
    entries.sort(key=lambda e: [int(x) for x in e["version"].split(".")], reverse=True)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate versioned eval reports under docs/results/.")
    parser.add_argument("--version", help="Override version (default: read from alire.toml).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    version = args.version or read_version()
    data = gather(version)
    paths = write_all(version, data)
    for path in paths:
        logger.info("wrote %s", path)
    ada = data.get("ada_eval") or {}
    if ada:
        for model, metrics in sorted(ada.items()):
            logger.info(
                "%s: build %s (%s%%), test %s (%s%%)",
                model, metrics.get("build"), metrics.get("build_pct"),
                metrics.get("test"), metrics.get("test_pct"),
            )
    else:
        logger.warning("no ada-eval metrics found - report records the gap")
    return 0


if __name__ == "__main__":
    sys.exit(main())

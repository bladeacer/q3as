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

    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `outputs/eval_results/<model>/<dataset>/*.jsonl` (per-sample results)")
    lines.append("- `outputs/generated_solutions/<label>/` (model generations)")
    lines.append("")
    lines.append("See the [results index](README.md) for the version comparison table.")
    lines.append("")
    lines.append("[\u2190 Back to results index](README.md)")
    lines.append("")
    return "\n".join(lines)


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

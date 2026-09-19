"""validate_defects.py - Compile-check defect pairs against the real GNAT.

Runs the dataset build's defect injector over real Ada sources, writes the
broken variants to a temp project, compiles each with `gnat compile`, and
compares the compiler output against the diagnosis the dataset claims.
Also verifies the corrected originals compile clean.

The claimed compiler messages are "error substrings": a family passes when
its expected substring appears in the real output (or, for the contract
family, when the expected warning appears). A broken variant that compiles
clean is a hard failure: the defect must break the build.

Usage:
    uv run python scripts/validate_defects.py                     # sample mode
    uv run python scripts/validate_defects.py --limit 50          # fewer pairs
    uv run python scripts/validate_defects.py --source ../adacovex
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "processing_scripts"))

import build_dataset as bd

# Per-family check: (expected_substring, kind). kind is "error" (must appear
# in gnat output) or "warning" (must appear; a clean compile is a failure).
FAMILY_CHECKS: dict[str, tuple[str, str]] = {
    "syntax": ('error:', "error"),
    "context": ('missing "with', "error"),
    "visibility": ("is not visible", "error"),
    "contract": ("is not a valid aspect identifier", "warning"),
    "mismatch": ("not type conformant", "error"),
}


def unit_and_filename(code: str) -> tuple[str, str] | None:
    """Derive (gnat_unit_name, file_name) for the FIRST unit in a snippet.

    GNAT requires the file name to match the unit name (lowercase), and a
    file can hold only one compilation unit. The dataset build concatenates
    spec and body into one snippet, so this splits the snippet into its
    units and returns the first one with its proper file name. The units
    are emitted in spec-then-body order so both files can be written.
    """
    units = split_units(code)
    if not units:
        return None
    name, fname, _text = units[0]
    return name, fname


def split_units(code: str) -> list[tuple[str, str, str]]:
    """Split a concatenated Ada snippet into (unit_name, file_name, text).

    Recognizes package specs (including 'with aspects' like
    'package CRDT.Rgas with SPARK_Mode is'), package bodies, and library
    subprograms. Each matched unit must reach its own 'end <last-part>;'
    so nested subprograms and instantiations are not mistaken for units.
    """
    # (pos, short_name, is_body, full_name)
    unit_starts: list[tuple[int, str, bool, str]] = []
    pattern = re.compile(
        r"\bpackage\s+(body\s+)?((?:\w+)(?:\.\w+)*)"
        r"(?:\s+with\s+[^\n]*?)?\s+is\b"
    )
    for m in pattern.finditer(code):
        is_body = bool(m.group(1))
        full = m.group(2)
        unit_starts.append((m.start(), full.split(".")[-1], is_body, full))
    for m in re.finditer(r"^\s*procedure\s+(\w+)\s+is\b", code, re.MULTILINE):
        unit_starts.append((m.start(), m.group(1), False, m.group(1)))
    if not unit_starts:
        return []
    unit_starts.sort()
    units: list[tuple[str, str, str]] = []
    for i, entry in enumerate(unit_starts):
        pos, short, is_body, full = entry
        end = unit_starts[i + 1][0] if i + 1 < len(unit_starts) else len(code)
        text = code[pos:end].strip()
        if not re.search(r"\bend\s+" + re.escape(short) + r"\s*;", text):
            continue
        # Decide the file extension: body keyword wins; otherwise a spec
        # already emitted for this unit name means this is its body.
        already_spec = any(u[0] == short and u[1].endswith(".ads") for u in units)
        ext = ".adb" if (is_body or already_spec) else ".ads"
        units.append((short, short.lower() + ext, text))
    return units


def compile_snippet(workdir: Path, code: str, name: str, fname: str) -> tuple[bool, str]:
    """Write the snippet's units to *workdir* and compile. Returns (ok, output).

    The snippet is split into its units and each unit is written under its
    own file name (spec -> <name>.ads, body -> <name>.adb), then the entry
    file is compiled with -k so the compiler follows to the other units.
    ok=False means the compiler reported errors. Output has the file-name
    warning lines stripped (they are an artifact of the harness).
    """
    units = split_units(code)
    if not units:
        return False, "error: no compilable unit found in snippet"
    for unit_name, unit_fname, unit_text in units:
        (workdir / unit_fname).write_text(unit_text + "\n", encoding="utf-8")
    stem = fname.removesuffix(".ads").removesuffix(".adb")
    for suffix in (".o", ".ali"):
        (workdir / (stem + suffix)).unlink(missing_ok=True)
    try:
        proc = subprocess.run(
            ["gnat", "compile", "-q", "-k", "-j0", fname],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "error: harness compile timeout"
    output = (proc.stdout + proc.stderr).replace(str(workdir) + "/", "")
    output = "\n".join(
        ln for ln in output.splitlines()
        if "file name does not match unit name" not in ln
    )
    return "error:" not in output, output


def validate_source(
    source_dir: Path,
    limit: int,
    workdir: Path,
) -> dict[str, object]:
    """Run defect validation over Ada pairs found under source_dir."""
    discovered = bd.discover_files(source_dir)
    pairs = bd.pair_files(discovered)
    results: dict[str, dict[str, int]] = {
        family: {"pass": 0, "fail": 0, "skip": 0} for family in FAMILY_CHECKS
    }
    failures: list[str] = []
    originals_bad = 0
    checked = 0

    for pair in pairs:
        if checked >= limit:
            break
        spec = pair["spec"]
        impl = pair["impl"]
        code = ""
        if isinstance(spec, Path):
            code = bd.read_and_sanitize(spec) or ""
        if isinstance(impl, Path):
            code += "\n" + (bd.read_and_sanitize(impl) or "")
        code = code.strip()
        if not (60 <= len(code) <= 6000):
            continue

        # Baseline: the original must compile clean. It often does not in
        # isolation (project context missing), so we only count it when the
        # compiler reports errors *inside the unit itself*.
        named = unit_and_filename(code)
        if named is None:
            continue
        name, fname = named
        original_ok, original_out = compile_snippet(workdir, code, name, fname)
        if not original_ok and "error:" in original_out:
            # The untouched source fails to compile: skip this pair, the
            # defect result would be confounded.
            originals_bad += 1
            continue

        package = pair.get("package")
        package = str(package) if package else name
        checked += 1

        for family, (expected, kind) in FAMILY_CHECKS.items():
            result = bd.inject_defect(code, family)
            if result is None:
                results[family]["skip"] += 1
                continue
            broken, _diagnosis, claimed = result
            units_broken = split_units(broken)
            if not units_broken:
                results[family]["skip"] += 1
                continue
            bname, bfname, _text = units_broken[0]
            broken_ok, out = compile_snippet(workdir, broken, bname, bfname)

            if kind == "error" and broken_ok:
                # The defect did not break the build: hard failure.
                results[family]["fail"] += 1
                failures.append(
                    f"{family}/{package}: broken snippet compiles CLEAN "
                    f"(claimed: {claimed!r})"
                )
            elif expected.split(": ", 1)[-1].split('"')[0].strip() and expected not in out:
                results[family]["fail"] += 1
                failures.append(
                    f"{family}/{package}: expected {expected!r} not in output; "
                    f"got: {out.strip().splitlines()[:2]!r}"
                )
            else:
                results[family]["pass"] += 1

    return {
        "checked": checked,
        "skipped_originals_failing": originals_bad,
        "results": results,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile-check q3as defect pairs with the real GNAT.",
    )
    parser.add_argument(
        "--source", type=Path, action="append", default=[],
        help="Ada source tree to draw pairs from (repeatable). "
             "Default: ../adacovex ../Ada_CRDT ../ada-eval",
    )
    parser.add_argument(
        "--limit", type=int, default=30,
        help="Maximum number of source pairs to check.",
    )
    args = parser.parse_args()

    sources = args.source or [Path("../adacovex"), Path("../Ada_CRDT"), Path("../ada-eval")]
    if shutil.which("gnat") is None:
        print("gnat not found on PATH. Install GNAT or run under Alire:", file=sys.stderr)
        print("  alr exec -- python scripts/validate_defects.py", file=sys.stderr)
        sys.exit(2)

    report: dict[str, object] = {"checked": 0, "skipped_originals_failing": 0, "results": {}, "failures": []}
    with tempfile.TemporaryDirectory(prefix="q3as-defects-") as tmp:
        workdir = Path(tmp)
        for source in sources:
            if not source.exists():
                print(f"source not found, skipping: {source}")
                continue
            print(f"== validating defects from {source} ==")
            sub = validate_source(source, args.limit - int(report["checked"]), workdir)  # type: ignore[arg-type]
            _merge(report, sub)
            if int(report["checked"]) >= args.limit:  # type: ignore[arg-type]
                break

    print()
    print(f"source pairs checked: {report['checked']}")
    print(f"pairs skipped (original does not compile standalone): {report['skipped_originals_failing']}")
    for family, counts in report["results"].items():  # type: ignore[union-attr]
        print(f"  {family:<11} pass={counts['pass']:<4} fail={counts['fail']:<4} skip={counts['skip']}")
    failures: list[str] = report["failures"]  # type: ignore[assignment]
    if failures:
        print()
        print(f"FAILURES ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print()
    print("All checked defect families behave as claimed.")


def _merge(total: dict[str, object], sub: dict[str, object]) -> None:
    total["checked"] = int(total["checked"]) + int(sub["checked"])  # type: ignore[arg-type]
    total["skipped_originals_failing"] = int(total["skipped_originals_failing"]) + int(sub["skipped_originals_failing"])  # type: ignore[arg-type]
    results = total["results"]
    sub_results = sub["results"]
    assert isinstance(results, dict) and isinstance(sub_results, dict)
    for family, counts in sub_results.items():
        assert isinstance(counts, dict)
        bucket: dict[str, int] = results.setdefault(family, {"pass": 0, "fail": 0, "skip": 0})  # type: ignore[arg-type]
        for key, value in counts.items():
            bucket[key] = bucket.get(key, 0) + value
    total["failures"] = total["failures"] + sub["failures"]  # type: ignore[operator]


if __name__ == "__main__":
    main()

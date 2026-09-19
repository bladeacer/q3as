"""validate_defects.py - Compile-check defect pairs against the real GNAT.

Runs the dataset build's defect injector over real Ada sources, writes the
broken variants to a temp project, compiles each with `gnat compile`, and
compares the compiler output against the message the dataset claims. Also
verifies the corrected originals compile clean.

Method notes, from GNAT 14 experiments:

- A snippet holds a spec and often a body. `gnat compile -k` on the spec
  never compiles the body, so defects injected into the body (mismatch,
  body syntax) stay invisible. The entry file is therefore the body when
  one exists, else the spec.
- Spec-only units cannot generate code in a full compile ("cannot generate
  code for file ... (package spec)"). The flag -gnatc keeps full syntax and
  semantic checks without code generation, so spec-only snippets check the
  same error classes.
- The claimed message comes straight from inject_defect. A family passes
  when the claimed substring appears in the real output. An "error:" claim
  must also break the build: a broken variant that compiles clean is a
  hard failure.

Usage:
    uv run python scripts/validate_defects.py                     # sample mode
    uv run python scripts/validate_defects.py --limit 50          # fewer pairs
    uv run python scripts/validate_defects.py --source ../adacovex
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR.parent / "data" / "processing_scripts"))

import build_dataset as bd
from alire_env import ToolNotAvailable, alire_env_path, find_tool

_DEFECT_FAMILIES: tuple[str, ...] = bd._DEFECT_FAMILIES


def split_units(code: str) -> list[tuple[str, str, str]]:
    """Split a concatenated Ada snippet into (unit_name, file_name, text).

    Only library-level units count, and those start at column 0 in this
    corpus. Anchoring at column 0 keeps nested (indented) procedures and
    packages from being mistaken for compilation units: with the loose
    anchor a package spec plus one nested procedure split into just the
    nested fragment, which compiles clean on its own and masks every
    defect. Each matched unit must reach its own 'end <name>;' so
    instantiations ('package V is new ...;') are not mistaken for units.
    """
    starts: list[tuple[int, str, bool, str]] = []
    pkg = re.compile(
        r"^package\s+(body\s+)?((\w+)(?:\.\w+)*)"
        r"(?:\s+with\s+[^\n]*?)?[ \t]*\n?\s*is\b",
        re.MULTILINE,
    )
    for m in pkg.finditer(code):
        is_body = bool(m.group(1))
        full = m.group(2)
        starts.append((m.start(), full.split(".")[-1], is_body, full))
    for m in re.finditer(
        r"^(?:procedure|function)\s+(\w+)\s*[^\n]*?\s+is\b", code, re.MULTILINE,
    ):
        starts.append((m.start(), m.group(1), True, m.group(1)))
    if not starts:
        return []
    starts.sort()
    units: list[tuple[str, str, str]] = []
    for i, (pos, short, is_body, full) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(code)
        text = code[pos:end].strip()
        # Dotted unit names end as 'end Adacovex.IR_Bounds;' (the full
        # dotted name, optionally). Accept both forms.
        end_ok = re.search(
            r"\bend\s+(" + re.escape(full) + r"|" + re.escape(short) + r")\s*;",
            text,
        )
        if not end_ok:
            continue
        # Decide the file extension: body keyword wins; otherwise a spec
        # already emitted for this unit name means this is its body.
        already_spec = any(u[0] == short and u[1].endswith(".ads") for u in units)
        ext = ".adb" if (is_body or already_spec) else ".ads"
        # Child units follow GNAT's kid naming: CRDT.Test_Support lives in
        # crdt-test_support.ads. Writing test_support.ads would only earn
        # a file-name warning and confuse the source search order.
        fname = full.lower().replace(".", "-") + ext
        units.append((short, fname, text))
    # Recover an intentionally broken FIRST unit, such as the syntax
    # family's dropped 'is': the strict pattern needs 'is', so the broken
    # unit fails to match and its text (up to the next strict start) is
    # silently dropped from the emitted files. The broken unit would never
    # be compiled and the snippet would report a false clean build. The
    # fragment still has a name and its own 'end <name>;', which is enough
    # to identify and write it.
    first_start = starts[0][0] if starts else len(code)
    prefix = code[:first_start]
    head = re.match(r"\s*package\s+(body\s+)?((\w+)(?:\.\w+)*)", prefix)
    if head:
        is_body_h = bool(head.group(1))
        full_h = head.group(2)
        short_h = full_h.split(".")[-1]
        if re.search(
            r"\bend\s+(" + re.escape(full_h) + r"|" + re.escape(short_h) + r")\s*;",
            prefix,
        ) and not any(u[0] == short_h and u[1].endswith(".ads") for u in units):
            ext_h = ".adb" if is_body_h else ".ads"
            units.insert(0, (short_h, full_h.lower().replace(".", "-") + ext_h, prefix.strip()))
    if units:
        return units
    # Tolerant fallback for intentionally broken snippets, such as the
    # syntax family's dropped 'is': the strict pattern needs 'is' but the
    # broken unit still has a name and an 'end <name>;'. Without the
    # fallback the harness would silently compile nothing or the wrong
    # fragment and report a false clean build.
    m = re.search(r"^package\s+(body\s+)?((\w+)(?:\.\w+)*)", code, re.MULTILINE)
    if m:
        is_body = bool(m.group(1))
        full = m.group(2)
        short = full.split(".")[-1]
        end_m = re.search(r"\bend\s+(" + re.escape(full) + r"|" + re.escape(short) + r")\s*;", code)
        if end_m:
            ext = ".adb" if is_body else ".ads"
            return [(short, full.lower().replace(".", "-") + ext, code[:end_m.end()].strip())]
    return []


def entry_file(units: list[tuple[str, str, str]]) -> str:
    """Pick the file gnat must compile to see every defect.

    Bodies win: `gnat compile -k` on a spec does not compile the body,
    and most defect families break the body or need body errors to show.
    A library procedure is its own body, so .adb is right for it too.
    """
    bodies = [fname for _n, fname, _t in units if fname.endswith(".adb")]
    return bodies[0] if bodies else units[0][1]


_GNAT: str | None = None


def _gnat() -> str:
    """Resolve gnat through the Alire environment (cached)."""
    global _GNAT
    if _GNAT is None:
        try:
            _GNAT = str(find_tool("gnat"))
        except ToolNotAvailable as exc:
            print(str(exc), file=sys.stderr)
            print("  make prove", file=sys.stderr)
            sys.exit(2)
    return _GNAT


def _alire_subprocess_env() -> dict[str, str]:
    """Child env with the Alire toolchain first on PATH.

    The gnat driver spawns gcc and other tool binaries itself, so they must
    resolve to the Alire versions too, not just the top-level gnat.
    """
    env = dict(os.environ)
    env["PATH"] = alire_env_path()
    return env


def compile_snippet(
    workdir: Path,
    code: str,
    include_dirs: list[Path] | None = None,
) -> tuple[bool, str]:
    """Write the snippet's units to *workdir* and compile. Returns (ok, output).

    The snippet is split into its units, each unit is written under its
    own file name, and EVERY unit is compiled explicitly with -gnatc (no
    code generation, so spec-only units work). Compiling every unit is
    essential: with -gnatc, gnatmake -k traces the closure only through
    explicit with clauses, and a package body never withs its own spec,
    so a broken spec would otherwise go unread and compile clean. -I dirs
    let units find sibling specs they reference, which keeps the
    original-vs-broken comparison meaningful for snippets that are not
    self-contained. ok=False means the compiler reported errors anywhere.
    Output has the file-name warning lines stripped (harness artifact).
    """
    units = split_units(code)
    if not units:
        return False, "error: no compilable unit found in snippet"
    # Clear previous snippets first: a stale spec from an earlier pair in
    # the shared workdir satisfies a with clause and fakes a clean baseline.
    for stale in workdir.iterdir():
        if stale.suffix in (".ads", ".adb", ".ali", ".o"):
            stale.unlink()
    for _unit_name, unit_fname, unit_text in units:
        (workdir / unit_fname).write_text(unit_text + "\n", encoding="utf-8")
    for _unit_name, unit_fname, _t in units:
        stem = unit_fname.removesuffix(".ads").removesuffix(".adb")
        for suffix in (".o", ".ali"):
            (workdir / (stem + suffix)).unlink(missing_ok=True)
    all_output: list[str] = []
    any_error = False
    for _unit_name, unit_fname, _t in units:
        cmd = [_gnat(), "compile", "-q", "-gnatc", "-j0"]
        for include_dir in include_dirs or []:
            # gnatmake wants -I and the directory in one token: a separated
            # "-I" "dir" aborts with 'missing source directory name' (which
            # carries no 'error:' substring and would fake a clean build).
            cmd.append(f"-I{include_dir}")
        cmd.append(unit_fname)
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                env=_alire_subprocess_env(),
            )
        except subprocess.TimeoutExpired:
            return False, "error: harness compile timeout"
        out = (proc.stdout + proc.stderr).replace(str(workdir) + "/", "")
        out = "\n".join(
            ln for ln in out.splitlines()
            if "file name does not match unit name" not in ln
        )
        all_output.append(out)
        if "error:" in out:
            any_error = True
    return not any_error, "\n".join(all_output)


def validate_source(
    source_dir: Path,
    limit: int,
    workdir: Path,
) -> dict[str, object]:
    """Run defect validation over Ada pairs found under source_dir."""
    discovered = bd.discover_files(source_dir)
    pairs = bd.pair_files(discovered)
    # Every source dir of the tree goes on -I: a pair that withs a third
    # file (a parent package, a support spec) must resolve it, or the
    # original fails standalone and the pair is skipped. Those skips bias
    # the sample toward trivial snippets.
    tree_dirs = sorted({p.parent.resolve() for p in discovered if isinstance(p, Path)})
    results: dict[str, dict[str, int]] = {
        family: {"pass": 0, "fail": 0, "skip": 0} for family in _DEFECT_FAMILIES
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
        paths: list[Path] = []
        if isinstance(spec, Path):
            code = bd.read_and_sanitize(spec) or ""
            paths.append(spec)
        if isinstance(impl, Path):
            code += "\n" + (bd.read_and_sanitize(impl) or "")
            paths.append(impl)
        code = code.strip()
        if not (60 <= len(code) <= 6000):
            continue
        if not split_units(code):
            continue

        # The pair's own dirs first, then the whole tree: GNAT searches
        # -I dirs in order, and the nearest definition should win.
        include_dirs = [p.parent.resolve() for p in paths] + tree_dirs

        # Baseline: the original must compile clean. It often does not in
        # isolation (project context missing), so we only count it when the
        # compiler reports errors *inside the unit itself*.
        original_ok, _original_out = compile_snippet(workdir, code, include_dirs)
        if not original_ok:
            # The untouched source fails to compile: skip this pair, the
            # defect result would be confounded.
            originals_bad += 1
            continue

        package = pair.get("package")
        package = str(package) if package else "pair"
        checked += 1

        for family in _DEFECT_FAMILIES:
            result = bd.inject_defect(code, family)
            if result is None:
                results[family]["skip"] += 1
                continue
            broken, _diagnosis, claimed = result
            if not split_units(broken):
                results[family]["skip"] += 1
                continue
            broken_ok, out = compile_snippet(workdir, broken, include_dirs)
            is_warning_claim = claimed.startswith("warning:")

            if not is_warning_claim and broken_ok:
                # The defect did not break the build: hard failure.
                results[family]["fail"] += 1
                failures.append(
                    f"{family}/{package}: broken snippet compiles CLEAN "
                    f"(claimed: {claimed!r})"
                )
            elif claimed not in out:
                results[family]["fail"] += 1
                failures.append(
                    f"{family}/{package}: claimed {claimed!r} not in output; "
                    f"got: {out.strip().splitlines()[:3]!r}"
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
    # Resolve gnat up front (exits with install instructions when missing).
    _gnat()

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

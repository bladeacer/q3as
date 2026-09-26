"""alire_env.py - Resolve tools through the Alire-managed Ada toolchain.

q3as manages its Ada tool dependencies (gnat, gprbuild, gnatprove, gnatdoc,
gnatformat, gprclean, gprls) with Alire instead of system packages: the dev
manifest alire-dev.toml declares them, and `alr exec` (wrapped by
scripts/ada_env.sh) puts their binaries on PATH.

This module gives Python callers (eval pipelines, the defect validator) the
same environment: it asks the wrapper for the Alire PATH once, caches it,
and finds tool binaries there. Call `alire_env_path()` first and pass the
result as `env` to subprocess calls, or use `find_tool()` to get an
absolute binary path.

Usage:
    from alire_env import find_tool, alire_env_path
    gnatprove = find_tool("gnatprove")
    subprocess.run([str(gnatprove), "--version"], env=alire_env_path())
"""

from __future__ import annotations

import argparse
import functools
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADA_ENV_SH = PROJECT_ROOT / "scripts" / "ada_env.sh"

# Tools q3as resolves through Alire. The Alire environment is preferred;
# when a tool is only on the system PATH (an Alire 'external' install such
# as a distribution gnatprove), it is used with a one-time warning.
ALIRE_TOOLS = ("gnat", "gprbuild", "gnatprove", "gnatdoc", "gnatformat", "gprclean", "gprls")

# The tools q3as actually invokes, and the ones the eval pipeline refuses to
# run without (see the "prove"/"build"/"test" tables in eval/eval_pipeline.py).
# gnatdoc is deliberately absent from ALIRE_TOOLS' required set: alire-dev.toml
# does not depend on it, so it is never present and must not fail a check.
REQUIRED_TOOLS = ("gnat", "gprbuild", "gnatprove", "gnatformat", "gprclean", "gprls")


class ToolNotAvailable(RuntimeError):
    """A tool is not present in the Alire environment or on PATH."""


@functools.lru_cache(maxsize=1)
def _alire_path_entry() -> str | None:
    """Return the PATH entry prefix `alr exec` provides, or None.

    Runs `scripts/ada_env.sh printenv PATH` once and extracts the leading
    Alire directories (crate bin dir plus the Alire dependency cache bins).
    None means Alire is unavailable and callers should fall back to os.environ.
    """
    if not ADA_ENV_SH.exists():
        return None
    try:
        proc = subprocess.run(
            ["bash", str(ADA_ENV_SH), "printenv", "PATH"],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    # printenv writes to stdout; alr's sync notes and warnings may land on
    # either stream, so only stdout lines are candidates.
    system_path = os.environ.get("PATH", "")
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    full_path = ""
    for ln in reversed(lines):
        # The real printenv line is the PATH value: it extends the outer
        # PATH (alr only prepends). A note like 'warn: ...' never does.
        if ln.endswith(system_path) and len(ln) > len(system_path):
            full_path = ln
            break
    if not full_path:
        return None
    prefix = full_path[: len(full_path) - len(system_path)]
    return prefix or None


@functools.lru_cache(maxsize=1)
def alire_env_path() -> str:
    """Return a PATH value with the Alire toolchain ahead of the system PATH."""
    prefix = _alire_path_entry()
    system = os.environ.get("PATH", "")
    if prefix:
        return f"{prefix}{system}"
    return system


def find_tool(name: str) -> Path:
    """Return the absolute path of *name*, Alire environment first.

    Search order: the Alire PATH prefix (the managed toolchain), then the
    remaining PATH. The latter covers Alire external installs (distribution
    packages alr detects) and dev machines without the full manifest
    resolution; every system-PATH hit is reported once so mixed
    environments stay visible.

    The warning keys off whether the hit is in the Alire prefix, not off
    which of the two searches found it. When no Alire environment exists the
    prefix is empty, so the two searches would be the same and a
    distribution-provided tool would be reported as managed.
    """
    prefix = _alire_path_entry() or ""
    alire_dirs = [d for d in prefix.split(os.pathsep) if d]
    system_dirs = [d for d in os.environ.get("PATH", "").split(os.pathsep) if d]
    for directory in alire_dirs + system_dirs:
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            if directory not in alire_dirs and name in ALIRE_TOOLS:
                _warn_system_tool(name, candidate)
            return candidate
    raise ToolNotAvailable(
        f"{name} not found in the Alire environment or PATH. "
        "Run `make prove` (Alire dev workspace build) to install it."
    )


@functools.cache
def _warn_system_tool(name: str, resolved: Path) -> None:
    """Warn once per tool that only a system install was found."""
    print(
        f"warning: {name} resolved from the system PATH ({resolved}), "
        "not the Alire environment; run `make prove` to install the managed version",
        file=sys.stderr,
    )


def has_tool(name: str) -> bool:
    """True when *name* resolves through the Alire environment."""
    try:
        find_tool(name)
    except ToolNotAvailable:
        return False
    return True


def main() -> int:
    """Report where each Alire-managed tool resolves, for `make prove` checks.

    Prints one line per tool and exits non-zero when a tool q3as actually
    invokes is missing, so this is usable as a verification step and not only
    as a debugging aid. Tools outside REQUIRED_TOOLS (gnatdoc) are reported but
    never fail the check.
    """
    parser = argparse.ArgumentParser(description="Resolve the Alire-managed Ada toolchain.")
    parser.add_argument(
        "--quiet", action="store_true", help="only report missing tools"
    )
    args = parser.parse_args()

    prefix = _alire_path_entry()
    if prefix is None:
        print("Alire environment: unavailable (run `make prove`)")

    missing: list[str] = []
    for tool in ALIRE_TOOLS:
        required = tool in REQUIRED_TOOLS
        try:
            resolved = find_tool(tool)
        except ToolNotAvailable:
            mark = "MISSING" if required else "missing (optional)"
            if required:
                missing.append(tool)
            print(f"  {tool:<12} {mark}")
            continue
        if args.quiet:
            continue
        alire_dirs = prefix.split(os.pathsep) if prefix else []
        origin = "" if str(Path(resolved).parent) in alire_dirs else "  (system PATH)"
        print(f"  {tool:<12} {resolved}{origin}")

    if missing:
        print(f"\nmissing required tool(s): {', '.join(missing)}", file=sys.stderr)
        print("Run `make prove` to fetch the managed toolchain.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

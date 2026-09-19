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
    """
    path_value = alire_env_path()
    for searched, source in ((path_value, "alire"), (os.environ.get("PATH", ""), "system")):
        for directory in searched.split(os.pathsep):
            if not directory:
                continue
            candidate = Path(directory) / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                if source == "system" and name in ALIRE_TOOLS:
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

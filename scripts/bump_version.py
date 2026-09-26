#!/usr/bin/env python3
"""bump_version.py - single-source version for q3as.

The version lives in ``alire.toml`` and is mirrored into ``alire-dev.toml``
(the publishing manifest and the dev manifest describe the same crate) and
into ``alire-ast.toml`` (the manifest that resolves libadalang for the
dataset AST parser). All three must carry the same version.
``pyproject.toml`` carries the same version for the Python tooling and is
synced best-effort when it exists and has a version line. This module reads
the current version for other tooling (eval reports use it to name result
files) and bumps the files together.

Usage:
    python scripts/bump_version.py get
    python scripts/bump_version.py set 0.2.0
    python scripts/bump_version.py bump [major|minor|patch]   # default: patch

Exit codes: 0 on success, 1 on invalid input, 2 on missing manifests.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# All three manifests describe the same q3as release: the publishing manifest,
# the dev manifest (SPARK toolchain) and the AST manifest (libadalang). They
# must never drift apart.
MANIFESTS = (ROOT / "alire.toml", ROOT / "alire-dev.toml", ROOT / "alire-ast.toml")
# Other files carrying the crate version, synced best-effort: a missing
# file or a missing version line is skipped, never an error.
EXTRA_VERSION_FILES = (ROOT / "pyproject.toml",)

_VERSION_RE = re.compile(r'^(version\s*=\s*")([^"]+)(")', re.MULTILINE)


def read_version(manifest: Path | None = None) -> str:
    """Return the version string from *manifest* (default: alire.toml)."""
    manifest = manifest or MANIFESTS[0]
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    text = manifest.read_text(encoding="utf-8")
    match = _VERSION_RE.search(text)
    if not match:
        raise ValueError(f"no version = \"...\" line in {manifest}")
    return match.group(2)


def _set_version_in(manifest: Path, version: str) -> bool:
    """Rewrite the version line in *manifest*. Returns True when changed."""
    text = manifest.read_text(encoding="utf-8")
    new_text, count = _VERSION_RE.subn(rf"\g<1>{version}\g<3>", text, count=1)
    if count == 0:
        raise ValueError(f"no version = \"...\" line in {manifest}")
    if new_text == text:
        return False
    manifest.write_text(new_text, encoding="utf-8")
    return True


def set_version(version: str) -> tuple[str, list[tuple[Path, bool]]]:
    """Set *version* in both manifests plus any extra version files.

    Returns (old, [(path, changed)]) covering every touched file.
    """
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(f"version must be x.y.z (got: {version})")
    old = read_version()
    results = [(manifest, _set_version_in(manifest, version)) for manifest in MANIFESTS]
    for extra in EXTRA_VERSION_FILES:
        if extra.exists() and _VERSION_RE.search(extra.read_text(encoding="utf-8")):
            results.append((extra, _set_version_in(extra, version)))
    return old, results


def bump_version(part: str = "patch") -> tuple[str, str]:
    """Bump one component of the current version. Returns (old, new)."""
    if part not in {"major", "minor", "patch"}:
        raise ValueError(f"part must be major|minor|patch (got: {part})")
    old = read_version()
    major, minor, patch = (int(x) for x in old.split("."))
    if part == "major":
        new = f"{major + 1}.0.0"
    elif part == "minor":
        new = f"{major}.{minor + 1}.0"
    else:
        new = f"{major}.{minor}.{patch + 1}"
    set_version(new)
    return old, new


def main() -> int:
    parser = argparse.ArgumentParser(description="Read or bump the q3as crate version.")
    parser.add_argument("command", choices=["get", "set", "bump"])
    parser.add_argument("value", nargs="?", help="x.y.z for set, major|minor|patch for bump")
    args = parser.parse_args()

    try:
        if args.command == "get":
            print(read_version())
            return 0
        if args.command == "set":
            if not args.value:
                parser.error("set requires a version argument")
            old, results = set_version(args.value)
            for manifest, changed in results:
                state = "updated" if changed else "already set"
                print(f"  {manifest.name}: {state} -> {args.value}")
            print(f"version: {old} -> {args.value}")
            return 0
        # bump
        old, new = bump_version(args.value or "patch")
        for manifest in MANIFESTS:
            print(f"  {manifest.name}: -> {new}")
        for extra in EXTRA_VERSION_FILES:
            if extra.exists() and _VERSION_RE.search(extra.read_text(encoding="utf-8")):
                print(f"  {extra.name}: -> {new}")
        print(f"version: {old} -> {new}")
        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, FileNotFoundError) else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""check-links.py - Check every markdown link in the repo resolves.

Adapted from ../adacovex/tools/check-links.py (MIT). Scans all ``.md``
files under the repository root and verifies that:

- relative links point at files that exist (resolved against the
  containing file's directory);
- links carrying an ``#anchor`` point at a real GitHub-style heading slug
  in the target markdown file (lowercased, punctuation stripped, spaces
  hyphenated);
- no file carries a control character, which is what a half-applied
  find-and-replace leaves behind (a link rewritten to a run of ``\\x01``);
- external links (http/https/mailto) are not verified.

A link that resolves on this machine can still be a dead link in the
repository a reader clones: anything git ignores (``outputs/``,
``data/processed/``, ``models/``, ``.alire-*/``) is absent from the clone,
so those targets are reported too. The ignore check needs git, and is
skipped when git is unavailable or this is not a working copy.

Fenced code blocks are stripped before link extraction so code samples
that happen to contain ``[x](y)``-looking text are not treated as links.

Usage:
  python3 tools/check-links.py            # check the whole repo; exit 1 on breaks
  python3 tools/check-links.py --dry-run  # print the scanned files, no checks
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent

# Directories whose contents are never scanned (build output, model
# downloads, Alire state, vendored index copies, fetched source cache).
SKIP_DIRS: tuple[str, ...] = (
    ".git",
    ".venv",
    "obj",
    "alire",
    "index",
    "config",
    "node_modules",
    ".pytest_cache",
    "raw_repos",
    "models",
    "outputs",
    "q3as-local-index",
)

LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
# Control characters other than tab: never legitimate in a text file.
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


@lru_cache(maxsize=1)
def repo_paths() -> tuple[frozenset[str], frozenset[str]] | None:
    """(tracked, ignored) repository-relative file paths, or None if unknowable.

    ``outputs/`` and friends exist on a built machine, so a plain existence
    check calls them fine; a reader who clones the repository gets a 404.
    Ignored means git would not keep the file, which is exactly what a clone
    lacks. Empty means "not a git working copy": the checks that need this
    are skipped rather than guessed at.
    """
    def run(*args: str) -> set[str]:
        try:
            done = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True,
                text=True, check=True, timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            return set()
        return set(done.stdout.split())

    tracked = run("ls-files", "--cached", "--others", "--exclude-standard")
    if not tracked:
        return None
    ignored = run("ls-files", "--others", "--ignored", "--exclude-standard")
    return frozenset(tracked), frozenset(ignored - tracked)


def is_absent_from_clone(rel_target: str, paths: tuple[frozenset[str], ...]) -> bool:
    """True when *rel_target* would not exist in a fresh clone.

    An ignored file, or a directory whose every file is ignored, is gone for
    a reader. A directory that also holds tracked files (``data/``, which
    has both the scripts and the generated splits) still resolves, so it is
    left alone.
    """
    tracked, ignored = paths
    if rel_target in ignored:
        return True
    prefix = f"{rel_target}/"
    if any(p.startswith(prefix) for p in tracked):
        return False
    return any(p.startswith(prefix) for p in ignored)




def slugify(heading: str) -> str:
    """GitHub-faithful anchor slug for a markdown heading.

    Mirrors github/html-pipeline's TableOfContentsFilter: lowercase, drop
    everything that is not a word character (letters, digits, underscore),
    hyphen, or space, then replace *each* space with a hyphen (consecutive
    spaces yield consecutive hyphens, e.g. ``a  b`` -> ``a--b``).
    """
    s: str = heading.strip().lower()
    s = re.sub(r"^#{1,6}\s+", "", s)          # drop leading markdown markers
    s = re.sub(r"[^\w\- ]", "", s)            # remove punctuation
    s = s.strip()                             # drop space left by markers
    s = s.replace(" ", "-")                   # each space -> hyphen
    return s


def headings_slugs(path: Path) -> list[str]:
    """Return the GitHub anchor slugs of all headings in a markdown file.

    Duplicate headings get GitHub's ``-1``, ``-2``, ... suffixes (the first
    occurrence keeps the bare slug).
    """
    slugs: list[str] = []
    seen: dict[str, int] = {}
    try:
        text: str = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return slugs
    for line in text.splitlines():
        m = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if m:
            slug: str = slugify(m.group(1))
            n: int = seen.get(slug, 0)
            seen[slug] = n + 1
            slugs.append(slug if n == 0 else f"{slug}-{n}")
    return slugs


def strip_code_fences(text: str) -> str:
    """Blank out fenced code blocks so their contents are not link-checked."""
    lines: list[str] = text.splitlines()
    out: list[str] = []
    in_fence: bool = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append("")
        elif in_fence:
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def md_files() -> list[Path]:
    """All markdown files to check, in a stable order."""
    files: list[Path] = []
    skip_parts: set = set(SKIP_DIRS)
    for path in sorted(ROOT.rglob("*.md")):
        rel: Path = path.relative_to(ROOT)
        if any(part in skip_parts for part in rel.parts):
            continue
        files.append(path)
    return files


def check_file(path: Path, slug_cache: dict[Path, list[str]],
               errors: list[str]) -> None:
    """Verify every link target in one markdown file."""
    try:
        raw: str = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        errors.append(f"{path}: unreadable: {exc}")
        return

    rel: Path = path.relative_to(ROOT)
    for line_no, line in enumerate(raw.splitlines(), start=1):
        if (hit := CONTROL_RE.search(line)) is not None:
            errors.append(f"{rel}:{line_no}: control character "
                          f"U+{ord(hit.group()):04X} (a rewritten link?)")

    ignored = repo_paths()
    text: str = strip_code_fences(raw)
    for line_no, line in enumerate(text.splitlines(), start=1):
        for target in LINK_RE.findall(line):
            target = target.strip()
            if not target or target.startswith("#"):
                continue
            if target.startswith(("http://", "https://", "mailto:", "ftp://")):
                continue
            if " " in target:  # unescaped spaces are not valid link targets
                errors.append(f"{rel}:{line_no}: link target contains a "
                              f"space: {target!r}")
                continue
            file_part, _, anchor = target.partition("#")
            if file_part == "":
                continue
            resolved: Path = (path.parent / file_part).resolve()
            if not resolved.exists():
                errors.append(f"{rel}:{line_no}: broken link: {target!r} "
                              f"(no such file {resolved})")
                continue
            if ignored and resolved.is_relative_to(ROOT):
                rel_target = resolved.relative_to(ROOT).as_posix()
                if is_absent_from_clone(rel_target, ignored):
                    errors.append(f"{rel}:{line_no}: link target is not in the "
                                  f"repository: {target!r} (git ignores it, so "
                                  f"a reader who clones gets a 404)")
                    continue
            if anchor and resolved.suffix == ".md":
                if resolved not in slug_cache:
                    slug_cache[resolved] = headings_slugs(resolved)
                if anchor not in slug_cache[resolved]:
                    errors.append(f"{rel}:{line_no}: broken anchor "
                                  f"{anchor!r} in {resolved.relative_to(ROOT)}")


def main(argv: list[str]) -> int:
    ap: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="list the scanned files without checking")
    args: argparse.Namespace = ap.parse_args(argv)

    files: list[Path] = md_files()
    if args.dry_run:
        for f in files:
            print(f.relative_to(ROOT))
        return 0

    errors: list[str] = []
    slug_cache: dict[Path, list[str]] = {}
    for f in files:
        check_file(f, slug_cache, errors)

    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        print(f"  Link check FAILED ({len(errors)} problem(s))")
        return 1
    print(f"  All links resolve across {len(files)} markdown files.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

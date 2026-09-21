"""gen_agents_tree.py - Generate the project file tree inside AGENTS.md.

Walks the repository (respecting .gitignore and skipping junk directories),
renders a markdown file tree, and replaces the section between the
AGENTS:TREE-BEGIN / AGENTS:TREE-END markers in AGENTS.md. Run via
`make agents-tree`.

The markers stay in the file: reruns only rewrite the content between them.

Usage:
    uv run python scripts/gen_agents_tree.py            # update AGENTS.md
    uv run python scripts/gen_agents_tree.py --stdout   # print only
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENTS_MD = PROJECT_ROOT / "AGENTS.md"
BEGIN_MARKER = "<!-- AGENTS:TREE-BEGIN -->"
END_MARKER = "<!-- AGENTS:TREE-END -->"

# Directories never shown (state, caches, virtualenvs, sibling data dumps).
SKIP_DIRS = {
    ".git", ".venv", ".ruff_cache", ".mypy_cache", ".pytest_cache",
    "__pycache__", ".alire-dev", "alire", "node_modules",
    ".unsloth", "unsloth_compiled_cache", "_unsloth_temporary_saved_buffers",
    ".agents", ".kilo", ".idea", ".vscode",
    "raw_repos",
}
# Shown as a collapsed one-liner instead of walked, with a per-dir note.
COLLAPSE_DIRS = {
    "models": "model weights, not shown",
    "outputs": "generated data, not shown",
    "config": "alr-generated project config, not shown",
    "raw_repos": "fetched source repositories (archive cache), not shown",
}
# File patterns never shown.
SKIP_FILES = {"*.pyc", "*.log", ".env"}

# Name of the entry the tree is rooted at.
ROOT_LABEL = "q3as/"


def load_gitignore_patterns() -> list[str]:
    """Read .gitignore and turn its lines into matchable patterns."""
    patterns: list[str] = []
    gitignore = PROJECT_ROOT / ".gitignore"
    if gitignore.exists():
        for raw in gitignore.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line.rstrip("/"))
    return patterns


def is_ignored(rel_path: str, name: str, gitignore_patterns: list[str]) -> bool:
    """True when *rel_path* (project-relative, no leading ./) is ignored.

    Matches the basename against file patterns and every path segment
    against directory patterns, mirroring gitignore semantics closely
    enough for a display tree.
    """
    for pattern in gitignore_patterns:
        if pattern in SKIP_DIRS:
            continue
        base = fnmatch.fnmatch(name, pattern)
        segment = any(fnmatch.fnmatch(part, pattern) for part in rel_path.split("/"))
        if base or segment:
            return True
    return False


def render_dir(
    directory: Path,
    prefix: str,
    gitignore_patterns: list[str],
    lines: list[str],
    depth_limit: int = 6,
) -> None:
    """Append tree lines for *directory* to *lines*."""
    entries = sorted(
        directory.iterdir(),
        key=lambda p: (p.is_file(), p.name.lower()),
    )
    # Directories first in display order: sort with dirs before files.
    entries = sorted(entries, key=lambda p: (not p.is_dir(), p.name.lower()))
    for entry in entries:
        rel = entry.relative_to(PROJECT_ROOT).as_posix()
        name = entry.name
        if entry.is_dir():
            if name in SKIP_DIRS or is_ignored(rel, name, gitignore_patterns):
                continue
            if name in COLLAPSE_DIRS:
                lines.append(f"{prefix}{name}/   ({COLLAPSE_DIRS[name]})")
                continue
            lines.append(f"{prefix}{name}/")
            if prefix.count("/") < depth_limit:
                render_dir(entry, f"{prefix}    ", gitignore_patterns, lines, depth_limit)
        else:
            if any(fnmatch.fnmatch(name, pattern) for pattern in SKIP_FILES):
                continue
            if is_ignored(rel, name, gitignore_patterns):
                continue
            lines.append(f"{prefix}{name}")


def build_tree() -> str:
    """Return the fenced markdown tree of the project."""
    gitignore_patterns = load_gitignore_patterns()
    lines: list[str] = [ROOT_LABEL]
    render_dir(PROJECT_ROOT, "   ", gitignore_patterns, lines)
    return "\n".join(lines)


def update_agents_md(tree: str, check_only: bool) -> int:
    """Replace the marker-delimited section in AGENTS.md."""
    if not AGENTS_MD.exists():
        print(f"AGENTS.md not found at {AGENTS_MD}", file=sys.stderr)
        return 2
    text = AGENTS_MD.read_text(encoding="utf-8")
    begin_idx = text.find(BEGIN_MARKER)
    end_idx = text.find(END_MARKER)
    if begin_idx == -1 or end_idx == -1 or end_idx < begin_idx:
        print(
            f"AGENTS.md must contain {BEGIN_MARKER} and {END_MARKER} markers",
            file=sys.stderr,
        )
        return 2
    new_section = f"{BEGIN_MARKER}\n```text\n{tree}\n```\n{END_MARKER}"
    updated = text[:begin_idx] + new_section + text[end_idx + len(END_MARKER):]
    if updated == text:
        print("AGENTS.md tree is up to date.")
        return 0
    if check_only:
        print("AGENTS.md tree is out of date (run `make agents-tree`).", file=sys.stderr)
        return 1
    AGENTS_MD.write_text(updated, encoding="utf-8")
    line_count = tree.count("\n") + 1
    print(f"AGENTS.md tree updated ({line_count} lines).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", action="store_true", help="Print the tree, do not write.")
    parser.add_argument("--check", action="store_true", help="Fail when the tree is stale.")
    args = parser.parse_args()

    tree = build_tree()
    if args.stdout:
        print(tree)
        return 0
    return update_agents_md(tree, check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())

"""parse_docs.py - Heading-aware chunking of Markdown/RST docs into STE JSONL.

Raw doc dumps lump unrelated concepts into one context window. This module
splits each document at its structural headings, so every chunk pairs one
topic heading with its explanatory body:

- Markdown: ATX headings (# .. ######) are parsed with markdown-it-py when
  installed (token line maps give exact section boundaries) and fall back to
  a line regex otherwise.
- reStructuredText: section titles via adornment underlines (optionally
  overlined). Levels derive from the order each adornment character first
  appears. docutils is used when installed; the regex scanner covers the
  corpus without it.

Prose between code fences is rewritten to the STE style distilled in
build_dataset.py (slop swaps, no em-dashes, no hedges); code fences stay
byte-exact. Each section becomes one chat turn: "Explain <Heading>" ->
<section body>. Output is a JSONL of records with a meta block (source,
heading path, level, format) beside the messages.

Usage:
    uv run python data/processing_scripts/parse_docs.py --input-dir ../learn \
        --output data/processed/docs_chunks.jsonl --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import re
import sys
import zlib
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import build_dataset as bd

logger = logging.getLogger("q3as_parse_docs")

DEFAULT_OUTPUT = Path("data/processed/docs_chunks.jsonl")
MIN_SECTION_CHARS = 120
MAX_SECTION_CHARS = 6000

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_RST_ADORNMENT = re.compile(r"^([=\-~^\"'`#*+:._])\1{2,}\s*$")
_LEADING_RULE = re.compile(r"^(?:[-=~^_#*+]{3,}|\s)+$")


# --------------------------------------------------------------------------- #
# Section splitting
# --------------------------------------------------------------------------- #


def split_markdown(text: str) -> list[dict[str, Any]]:
    """Split *text* into sections at ATX headings.

    Uses markdown-it-py's token line maps when importable (exact boundaries
    even with tricky fenced blocks) and a line regex otherwise. Each section
    dict carries level, title, the ancestor title path, and the raw body.
    """
    headings: list[tuple[int, int, str]] = []
    try:
        from markdown_it import MarkdownIt

        tokens = MarkdownIt().parse(text)
        for idx, token in enumerate(tokens):
            if token.type == "heading_open" and token.map:
                # heading_open carries only the tag (h1..h6); the title text
                # lives in the inline token that follows it.
                title = ""
                if idx + 1 < len(tokens) and tokens[idx + 1].type == "inline":
                    title = tokens[idx + 1].content.strip()
                headings.append((token.map[0], int(token.tag[1:]), title))
    except ImportError:
        for line_idx, line in enumerate(text.splitlines()):
            m = _MD_HEADING.match(line)
            if m:
                headings.append((line_idx, len(m.group(1)), m.group(2).strip()))

    if not headings:
        return [{"level": 0, "title": "", "path": [], "body": text}]
    return _sections_from_headings(text, headings)


def split_rst(text: str) -> list[dict[str, Any]]:
    """Split *text* into sections at reStructuredText titles.

    A title is a non-indented line adorned by an underline of repeated
    punctuation (optionally an identical overline above). Adornment
    characters map to levels in order of first appearance, mirroring how
    docutils assigns hierarchy.
    """
    lines = text.splitlines()
    headings: list[tuple[int, int, str]] = []
    adornment_levels: dict[str, int] = {}

    def level_for(char: str) -> int:
        if char not in adornment_levels:
            adornment_levels[char] = len(adornment_levels)
        return adornment_levels[char]

    for idx, line in enumerate(lines):
        stripped = line.rstrip()
        if not stripped or stripped != line or line[:1].isspace():
            continue
        if _RST_ADORNMENT.match(stripped):
            continue
        # Overline style: adornment, title, same adornment.
        if (
            idx >= 1
            and idx + 1 < len(lines)
            and _RST_ADORNMENT.match(lines[idx - 1].strip())
            and _RST_ADORNMENT.match(lines[idx + 1].strip())
            and lines[idx - 1].strip() == lines[idx + 1].strip()
        ):
            headings.append((idx, level_for(lines[idx + 1].strip()[0]), stripped))
            continue
        # Underline style.
        if idx + 1 < len(lines) and _RST_ADORNMENT.match(lines[idx + 1].strip()):
            headings.append((idx, level_for(lines[idx + 1].strip()[0]), stripped))

    if not headings:
        return [{"level": 0, "title": "", "path": [], "body": text}]
    return _sections_from_headings(text, headings)


def _sections_from_headings(
    text: str,
    headings: list[tuple[int, int, str]],
) -> list[dict[str, Any]]:
    """Slice *text* at each heading; attach the ancestor title path."""
    lines = text.splitlines(keepends=True)
    sections: list[dict[str, Any]] = []
    # Preamble before the first heading becomes its own section.
    if headings[0][0] > 0:
        preamble = "".join(lines[: headings[0][0]]).strip()
        if preamble:
            sections.append({"level": 0, "title": "", "path": [], "body": preamble})
    path: list[tuple[int, str]] = []
    for i, (line_idx, level, title) in enumerate(headings):
        while path and path[-1][0] >= level:
            path.pop()
        end = headings[i + 1][0] if i + 1 < len(headings) else len(lines)
        body = "".join(lines[line_idx + 1 : end])
        sections.append({
            "level": level,
            "title": title,
            "path": [t for _lvl, t in path] + [title],
            "body": body,
        })
        path.append((level, title))
    return sections


def parse_document(text: str, fmt: str, source: str) -> list[dict[str, Any]]:
    """Split one document and attach source/format metadata to each section."""
    splitter = split_markdown if fmt == "md" else split_rst
    sections = splitter(text)
    for section in sections:
        section["source"] = source
        section["format"] = fmt
    return sections


def _parse_file_task(task: tuple[str, str]) -> list[dict[str, Any]]:
    """Worker: read one documentation file and split it into sections."""
    path_str, fmt = task
    try:
        text = Path(path_str).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    if not text.strip():
        return []
    return parse_document(text, fmt, path_str)


# --------------------------------------------------------------------------- #
# Turn building
# --------------------------------------------------------------------------- #


def _clean_section_body(body: str) -> str:
    """STE-clean a section body without touching its code fences."""
    cleaned = bd._ste_clean_markdown(body)
    # Drop leading rule lines and blank padding: redundant lead-in prose.
    lines = cleaned.splitlines()
    while lines and (not lines[0].strip() or _LEADING_RULE.match(lines[0])):
        lines.pop(0)
    return "\n".join(lines).strip()


# The same doc-section request phrased several ways. All variants ask for
# an STE-compliant explanation; only the natural-language wrapper differs.
# Ada code inside the answers is never rephrased: syntax is fixed.
_DOC_USER_PHRASINGS = (
    ('Explain the section "{title}" from {source}. Follow Simplified '
    'Technical English rules and define technical terms at first use.'),
    ('What does the section "{title}" in {source} say? Write the answer in '
    'Simplified Technical English and define technical terms at first use.'),
    ('Summarize "{title}" from {source} for an engineer who is new to the '
    'topic. Use Simplified Technical English rules.'),
    ('Give a Simplified Technical English explanation of "{title}" from '
    '{source}. Define each technical term at its first use.'),
    ('Teach the topic "{title}" from {source}. Write short active sentences '
    'and follow Simplified Technical English rules.'),
)


def build_section_turn(
    section: dict[str, Any],
    min_chars: int = MIN_SECTION_CHARS,
    max_chars: int = MAX_SECTION_CHARS,
) -> dict[str, Any] | None:
    """Turn one section into an STE explanation chat record (or None).

    The user asks about the heading; the assistant delivers the section
    body, prose rewritten to ASD-STE100 rules, code fences untouched.
    """
    body = _clean_section_body(section.get("body", ""))
    title_path = [t for t in section.get("path", []) if t]
    if not title_path or not body:
        return None
    if not (min_chars <= len(body) <= max_chars):
        return None
    title_str = " > ".join(title_path)
    source = str(section.get("source", ""))
    source = source.replace("\\", "/").rsplit("/", 1)[-1] if source else source
    # The same ask, phrased several ways. Selection is content-derived
    # (crc32 of the section title), so the choice is stable across builds
    # and worker counts. Prose answers stay STE; only the question varies.
    phrasing = _DOC_USER_PHRASINGS[
        zlib.crc32(title_str.encode("utf-8")) % len(_DOC_USER_PHRASINGS)
    ]
    user_msg = phrasing.format(title=title_str, source=source or "the AdaCore course material")
    system_msg = bd.STE_RULE_BLOCK + "\n\n" + bd._TECH_TERM_DEFINITION_PROMPT
    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": body},
        ],
        "meta": {
            "kind": "doc_section",
            "source": source,
            "section": title_str,
            "level": section.get("level", 0),
            "format": section.get("format", ""),
        },
    }


def build_section_turns(
    sections: list[dict[str, Any]],
    min_chars: int = MIN_SECTION_CHARS,
    max_chars: int = MAX_SECTION_CHARS,
) -> list[dict[str, Any]]:
    """Build deduplicated STE turns for every usable section."""
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for section in sections:
        record = build_section_turn(section, min_chars, max_chars)
        if record is None:
            continue
        key = zlib.crc32(
            (record["meta"]["section"] + "\x00" + record["messages"][2]["content"]).encode("utf-8")
        )
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
    return records


# --------------------------------------------------------------------------- #
# File discovery and CLI
# --------------------------------------------------------------------------- #


def collect_doc_files(input_dirs: list[Path]) -> list[tuple[str, str]]:
    """List (path, format) pairs for every .md/.rst file under the roots."""
    tasks: list[tuple[str, str]] = []
    for root in input_dirs:
        if not root.exists():
            logger.warning("Input directory does not exist, skipping: %s", root)
            continue
        for pattern, fmt in (("*.md", "md"), ("*.rst", "rst")):
            for path in sorted(root.rglob(pattern)):
                tasks.append((str(path), fmt))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk Markdown/RST docs by headings into STE JSONL turns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=Path, action="append", default=[],
        help="Documentation tree to chunk (repeatable).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument("--min-chars", type=int, default=MIN_SECTION_CHARS, help="Skip sections shorter than this.")
    parser.add_argument("--max-chars", type=int, default=MAX_SECTION_CHARS, help="Skip sections longer than this.")
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Worker processes (0 = one per CPU core, 1 = serial).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_dirs = args.input_dir or [Path("../learn")]
    tasks = collect_doc_files(input_dirs)
    if not tasks:
        logger.error("No .md/.rst files found under %s", input_dirs)
        sys.exit(1)

    workers = args.workers if args.workers > 0 else (multiprocessing.cpu_count() or 1)
    sections: list[dict[str, Any]] = []
    if workers > 1 and len(tasks) >= 8:
        try:
            ctx = multiprocessing.get_context("fork")
        except ValueError:
            ctx = None
        if ctx is not None:
            with ctx.Pool(processes=min(workers, len(tasks))) as pool:
                for file_sections in pool.imap(_parse_file_task, tasks, chunksize=4):
                    sections.extend(file_sections)
    if not sections:
        for task in tasks:
            sections.extend(_parse_file_task(task))

    records = build_section_turns(sections, args.min_chars, args.max_chars)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    logger.info("Wrote %d doc-section turns to %s", len(records), args.output)


if __name__ == "__main__":
    main()

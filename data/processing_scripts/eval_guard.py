"""Eval-integrity guard: keeps ada-eval evaluation content out of training data.

Everything under ``../ada-eval/data`` is eval-proper: the compacted and
expanded datasets are the same 19 samples q3as is scored on, and
``data/generated`` / ``data/evaluated`` hold completions for those very
prompts. Training on any of it - base project, canonical solution, tests,
or prompt text - would let the model memorize the answers, so the scores
stop measuring anything.

The guard builds a blocklist from the eval data and checks every chat
record against it at two levels:

1. **Normalized text hashing** - comments stripped, string literals
   canonicalized, case folded (Ada is case insensitive), whitespace
   collapsed. Catches verbatim and re-formatted copies.
2. **Structural token hashing** - the normalized token sequence with every
   user identifier alpha-renamed by first occurrence and every number
   collapsed to one symbol. Catches copies that renamed identifiers or
   changed numeric spellings while keeping the logic identical.

Prompt text from the eval samples is additionally checked as a substring,
so a training record that embeds an eval task statement is caught even
when its code block differs.

Subprogram-level signatures come from the same extractors the dataset
builder uses (parse_ada_ast), so the guard sees exactly the units a
generated turn can contain. Short structural signatures are ignored:
generic two-line subprograms would match half the corpus and over-block.

If ``../ada-eval`` is missing the guard degrades to a no-op with a loud
warning; builds on a machine without the sibling repo are never blocked,
but ``make check-integrity`` reports the degraded state.

CLI contract (mirrors scripts/validate_defects.py): exit 0 when the given
JSONL files contain no eval content, exit 1 when any is found.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_ada_ast
import source_paths

logger = logging.getLogger(__name__)

# ada-eval lives in the source cache; fall back to the legacy sibling.
_ADA_EVAL_RESOLVED = source_paths.resolve("ada-eval")
ADA_EVAL_DIR = _ADA_EVAL_RESOLVED or Path("data/raw_repos/_missing/ada-eval")
ADA_EVAL_DATA = ADA_EVAL_DIR / "data"

# Structural signatures shorter than this are too generic to block on.
MIN_STRUCTURAL_LEN = 40

# Ada reserved words (RM 2.9, plus a few Annex flags) - never alpha-renamed.
ADA_KEYWORDS = frozenset([
    "abort", "abs", "abstract", "accept", "access", "aliased", "all", "and",
    "array", "at", "begin", "body", "case", "constant", "declare", "delay",
    "delta", "digits", "do", "else", "elsif", "end", "entry", "exception",
    "exit", "for", "function", "generic", "goto", "if", "in", "interface",
    "is", "limited", "loop", "mod", "new", "not", "null", "of", "or",
    "others", "out", "overriding", "package", "parallel", "pragma", "private",
    "procedure", "protected", "raise", "range", "record", "rem", "renames",
    "requeue", "return", "reverse", "select", "separate", "some", "subtype",
    "synchronized", "tagged", "task", "terminate", "then", "type", "until",
    "use", "when", "while", "with", "xor",
])

_TOKEN_RE = re.compile(
    r'"(?:[^"]|"")*"'  # string literal (doubled-quote escapes)
    r"|--[^\n]*"  # comment
    r"|'(?:[^'\n]|'')'"  # character literal
    r"|[A-Za-z][A-Za-z0-9_]*"  # identifier / keyword
    r"|\d[\d_]*(?:\.\d[\d_]*)?"  # number (decimal, underscores)
    r"|\.\.|:=|=>|<=|>=|/=|<<|>>|<>|\*\*"  # multi-char operators
    r"|\S",  # any single remaining character
)


def _tokens(text: str) -> list[str]:
    """Tokenize Ada text, dropping comments, canonicalizing strings."""
    out: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        tok = match.group(0)
        if tok.startswith("--"):
            continue
        if tok.startswith('"') or (tok.startswith("'") and len(tok) <= 3):
            out.append("S")
        elif tok[0].isalpha() or tok[0] == "_":
            out.append(tok.lower())
        else:
            out.append(tok)
    return out


def normalized_text(text: str) -> str:
    """Case-folded, comment-free, whitespace-collapsed token string."""
    return " ".join(_tokens(text))


def structural_text(text: str) -> str:
    """Alpha-renamed token string - rename-invariant, logic-sensitive shape.

    Identifiers that are not reserved words become ``N1, N2, ...`` in order
    of first occurrence. Numbers are kept: a changed bound or constant is a
    logic change and must produce a different signature. Two subprograms
    produce the same structural text exactly when they differ only in
    comments, layout, case, or identifier spelling.
    """
    seen: dict[str, str] = {}
    out: list[str] = []
    for tok in _tokens(text):
        if tok in ADA_KEYWORDS:
            out.append(tok)
        elif tok[0].isalpha() or tok[0] == "_":
            out.append(seen.setdefault(tok, f"N{len(seen) + 1}"))
        else:
            out.append(tok)
    return " ".join(out)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class EvalSignatures:
    """Everything that must never appear in training data."""

    exact_subprograms: set[str] = field(default_factory=set)
    structural_subprograms: set[str] = field(default_factory=set)
    exact_prompts: set[str] = field(default_factory=set)
    structural_prompts: set[str] = field(default_factory=set)
    degraded: bool = True

    def __len__(self) -> int:
        return (
            len(self.exact_subprograms)
            + len(self.structural_subprograms)
            + len(self.exact_prompts)
            + len(self.structural_prompts)
        )


def _add_ada_text(sigs: EvalSignatures, text: str) -> None:
    """Hash one Ada source text into both signature families."""
    if not text.strip():
        return
    norm = normalized_text(text)
    if norm:
        sigs.exact_subprograms.add(_sha(norm))
    struct = structural_text(text)
    if len(struct) >= MIN_STRUCTURAL_LEN:
        sigs.structural_subprograms.add(_sha(struct))


def _add_file(sigs: EvalSignatures, path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("Cannot read %s: %s", path, exc)
        return
    if path.suffix.lower() in {".ads", ".adb"}:
        # Subprogram-level signatures via the corpus extractors.
        for unit in parse_ada_ast.extract_spec_subprograms(text):
            _add_ada_text(sigs, str(unit.get("text") or ""))
        for unit in parse_ada_ast.extract_body_subprograms(text):
            _add_ada_text(sigs, str(unit.get("text") or ""))
    _add_ada_text(sigs, text)
    if path.name == "prompt.md":
        norm = normalized_text(text)
        if norm:
            sigs.exact_prompts.add(norm)
            struct = structural_text(text)
            if len(struct) >= MIN_STRUCTURAL_LEN:
                sigs.structural_prompts.add(struct)


def _add_compacted_record(sigs: EvalSignatures, record: dict[str, Any]) -> None:
    """Hash one compacted-JSONL record (canonical solution, sources, prompt)."""
    for key in ("prompt", "comments"):
        text = record.get(key)
        if isinstance(text, str) and text.strip():
            norm = normalized_text(text)
            sigs.exact_prompts.add(norm)
            struct = structural_text(text)
            if len(struct) >= MIN_STRUCTURAL_LEN:
                sigs.structural_prompts.add(struct)

    sources = record.get("sources")
    if isinstance(sources, dict):
        for content in sources.values():
            if isinstance(content, str) and content.strip():
                decoded = _decode_maybe_b64(content)
                _add_ada_text(sigs, decoded)

    solution = record.get("canonical_solution")
    if isinstance(solution, dict):
        for content in solution.values():
            if isinstance(content, str):
                _add_ada_text(sigs, content)
    elif isinstance(solution, str):
        _add_ada_text(sigs, solution)

    tests = record.get("unit_tests")
    if isinstance(tests, dict):
        for content in tests.values():
            if isinstance(content, str):
                _add_ada_text(sigs, content)


def _decode_maybe_b64(content: str) -> str:
    import base64

    try:
        return base64.b64decode(content, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return content


def load_eval_signatures(ada_eval_dir: Path | None = None) -> EvalSignatures:
    """Collect every blocked signature from the ada-eval data tree."""
    root = (ada_eval_dir or ADA_EVAL_DIR) / "data"
    sigs = EvalSignatures()
    if not root.exists():
        logger.warning(
            "ada-eval data not found at %s - integrity guard is DEGRADED "
            "and cannot verify the corpus against the eval suite",
            root,
        )
        return sigs

    sigs.degraded = False

    # Expanded tree: base projects, canonical solutions, tests, prompts.
    for dataset_dir in sorted((root / "base" / "expanded").glob("*")) if (root / "base" / "expanded").exists() else []:
        for path in sorted(dataset_dir.rglob("*")):
            if path.is_file():
                _add_file(sigs, path)

    # Compacted JSONL: same samples, different shape.
    compacted = root / "base" / "compacted"
    if compacted.exists():
        for jsonl in sorted(compacted.glob("*.jsonl")):
            try:
                for line in jsonl.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            _add_compacted_record(sigs, json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON in %s", jsonl)
            except OSError as exc:
                logger.warning("Cannot read %s: %s", jsonl, exc)

    # Generated completions and evaluation artifacts - prompts for the same
    # eval samples, so they must never train either.
    for extra in ("generated", "evaluated"):
        extra_dir = root / extra
        if extra_dir.exists():
            for path in sorted(extra_dir.rglob("*")):
                if path.is_file():
                    _add_file(sigs, path)

    logger.info(
        "Eval blocklist: %d subprogram signatures (%d exact, %d structural), "
        "%d prompts, degraded=%s",
        len(sigs.exact_subprograms) + len(sigs.structural_subprograms),
        len(sigs.exact_subprograms),
        len(sigs.structural_subprograms),
        len(sigs.exact_prompts),
        sigs.degraded,
    )
    return sigs


_FENCE_RE = re.compile(r"```(?:ada)?\n(.*?)```", re.DOTALL)


def _contamination_reason(record: dict[str, Any], sigs: EvalSignatures) -> str | None:
    """Return why this chat record is blocked, or None when clean."""
    for message in record.get("messages") or []:
        content = str(message.get("content") or "")
        norm = normalized_text(content)
        # Prompt containment: an eval task statement inside any message.
        for prompt in sigs.exact_prompts:
            if len(prompt) >= 20 and prompt in norm:
                return "contains eval prompt text"
        # Code fences: subprogram signatures, both families.
        for block in _FENCE_RE.findall(content):
            for unit in parse_ada_ast.extract_spec_subprograms(block):
                text = str(unit.get("text") or "")
                if text.strip() and _sha(normalized_text(text)) in sigs.exact_subprograms:
                    return f"eval subprogram `{unit.get('name')}` (normalized)"
                struct = structural_text(text)
                if len(struct) >= MIN_STRUCTURAL_LEN and _sha(struct) in sigs.structural_subprograms:
                    return f"eval subprogram `{unit.get('name')}` (structural)"
            for unit in parse_ada_ast.extract_body_subprograms(block):
                text = str(unit.get("text") or "")
                if text.strip() and _sha(normalized_text(text)) in sigs.exact_subprograms:
                    return f"eval subprogram `{unit.get('name')}` body (normalized)"
                struct = structural_text(text)
                if len(struct) >= MIN_STRUCTURAL_LEN and _sha(struct) in sigs.structural_subprograms:
                    return f"eval subprogram `{unit.get('name')}` body (structural)"
            # Whole-fence fallback: code the regex extractors cannot parse
            # (dedented, machine-mangled) still cannot hide as a single block.
            if block.strip():
                struct = structural_text(block)
                if len(struct) >= MIN_STRUCTURAL_LEN and _sha(struct) in sigs.structural_subprograms:
                    return "eval code block (structural, whole fence)"
    return None


def contaminated_groups(
    grouped: list[tuple[str, dict[str, list[dict[str, str]]]]],
    sigs: EvalSignatures | None = None,
) -> tuple[list[tuple[str, dict[str, list[dict[str, str]]]]], int, list[str]]:
    """Drop whole groups containing any eval content.

    One bad turn poisons its group: the sibling turns of the same unit are
    paraphrases of the same eval content, so the group goes as a whole.
    Returns (kept, dropped_count, reasons).
    """
    sigs = sigs if sigs is not None else load_eval_signatures()
    if sigs.degraded:
        return grouped, 0, ["guard degraded: ada-eval data missing, nothing blocked"]
    bad_groups: set[str] = set()
    for group, turn in grouped:
        if _contamination_reason(turn, sigs):
            bad_groups.add(group)
    if not bad_groups:
        return grouped, 0, []
    kept = [(group, turn) for group, turn in grouped if group not in bad_groups]
    return kept, len(grouped) - len(kept), sorted(bad_groups)


def check_jsonl(path: Path, sigs: EvalSignatures | None = None) -> list[str]:
    """Check one chat JSONL file; returns human-readable violations."""
    sigs = sigs if sigs is not None else load_eval_signatures()
    violations: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        reason = _contamination_reason(record, sigs)
        if reason:
            violations.append(f"{path.name}:{lineno}: {reason}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check chat JSONL files for ada-eval evaluation content.",
    )
    parser.add_argument(
        "datasets",
        type=Path,
        nargs="+",
        help="Chat JSONL files to check (e.g. data/processed/dataset_train.jsonl).",
    )
    parser.add_argument("--ada-eval-dir", type=Path, default=ADA_EVAL_DIR)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    sigs = load_eval_signatures(args.ada_eval_dir)
    if sigs.degraded:
        print("WARNING: ada-eval data missing - check is degraded, nothing to compare against.")
        return 2

    all_violations: list[str] = []
    for dataset in args.datasets:
        if not dataset.exists():
            print(f"ERROR: {dataset} not found")
            return 2
        all_violations.extend(check_jsonl(dataset, sigs))

    if all_violations:
        print(f"FAIL: {len(all_violations)} record(s) contain eval content:")
        for violation in all_violations:
            print(f"  {violation}")
        return 1
    print("OK: no eval content found in", ", ".join(str(d) for d in args.datasets))
    return 0


if __name__ == "__main__":
    sys.exit(main())

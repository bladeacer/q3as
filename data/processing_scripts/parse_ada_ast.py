"""parse_ada_ast.py - AST-driven Ada extraction for dataset building.

Instead of pairing whole .ads/.adb files, this module extracts precise
semantic units: subprogram declarations with their attached aspects
(Pre/Post/Global/Depends), their matching bodies, and constrained type
declarations. From those units it builds high-signal QA pairs:

- body-from-spec turns   the spec declaration (contracts included) asks
                         for the implementation
- contract QA turns      "How is this subprogram constrained?" answered
                         from the extracted aspect clauses
- type QA turns          "How is this type constrained?" answered from
                         range/digits/delta/mod constraints

libadalang is used when its Python bindings are installed (exact AST
extraction plus parse diagnostics for syntax validity). Without it, a
structural scanner does the extraction: it pairs each subprogram
declaration with the `end <name>;` aligned at the same indentation, which
is exact for GNAT-formatted sources. Either way the output schema is
identical, and a libadalang failure degrades to the scanner, never to a
crash.

Usage:
    uv run python data/processing_scripts/parse_ada_ast.py \
        --input-dir ../adacovex --output data/processed/ada_ast_units.jsonl
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

logger = logging.getLogger("q3as_parse_ada_ast")

DEFAULT_OUTPUT = Path("data/processed/ada_ast_units.jsonl")

try:  # libadalang is optional; the structural scanner is the fallback.
    import libadalang as lal  # type: ignore[import-not-found]

    HAS_LIBADALANG = True
except ImportError:
    lal = None  # type: ignore[assignment]
    HAS_LIBADALANG = False

_SPEC_SUBP = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<prefix>(?:not\s+)?overriding\s+)?"
    r"(?P<kind>procedure|function)\s+"
    r"(?P<name>\w+)"
    r"(?P<params>\s*\([^;]*?\))?"
    r"(?P<ret>\s+return\s+[\w.]+(\s+range\s+[^;]+)?)?"
    r"(?P<aspects>(?:\s*with[^;]*?)?)"
    r"\s*;",
    re.MULTILINE,
)
_BODY_SUBP = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<prefix>(?:not\s+)?overriding\s+)?"
    r"(?P<kind>procedure|function)\s+"
    r"(?P<name>\w+)"
    r"(?P<rest>[^;\n]*?)\bis\b(?P<after>[^\n]*)",
    re.MULTILINE,
)
_END_NAME = re.compile(r"^(?P<indent>[ \t]*)end\s+(?P<name>\w+)\s*;", re.MULTILINE)
_PACKAGE_NAME = re.compile(r"^\s*package\s+(?:body\s+)?([\w.]+)\s+is\b", re.MULTILINE)
_TYPE_DECL = re.compile(r"^\s*type\s+(?P<name>\w+)\s+is\s+(?P<def>[^\n;]+)", re.MULTILINE)
_ASPECT = re.compile(r"(?P<name>\w+)\s*=>")
_CONSTRAINT_WORDS = ("range", "digits", "delta", "mod")

# Expression functions: `function F (...) return T is (expr);`
_EXPR_BODY = re.compile(r"^\s*function\s+(?P<name>\w+)[^;\n]*\bis\s*\([^;]*\)\s*;", re.MULTILINE)


def _extract_aspects(aspect_text: str) -> dict[str, str]:
    """Split an aspect clause into {name: expression} (expressions kept raw)."""
    aspects: dict[str, str] = {}
    matches = list(_ASPECT.finditer(aspect_text))
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(aspect_text)
        expr = aspect_text[start:end].strip().rstrip(",").strip()
        if expr:
            aspects[match.group("name")] = expr
    return aspects


def _is_generic_instantiation(text: str) -> bool:
    return bool(re.search(r"\bis\s+new\b", text))


def extract_spec_subprograms(text: str) -> list[dict[str, Any]]:
    """Extract subprogram declarations (contracts included) from a spec file."""
    units: list[dict[str, Any]] = []
    for match in _SPEC_SUBP.finditer(text):
        if _is_generic_instantiation(match.group(0)):
            continue
        units.append({
            "name": match.group("name"),
            "kind": match.group("kind"),
            "text": match.group(0).strip(),
            "params": (match.group("params") or "").strip(),
            "returns": (match.group("ret") or "").replace("return", "", 1).strip(),
            "aspects": _extract_aspects(match.group("aspects") or ""),
        })
    return units


def extract_body_subprograms(text: str) -> list[dict[str, Any]]:
    """Extract subprogram bodies from a body file.

    Each declaration pairs with the first `end <name>;` aligned at the same
    indentation (GNAT formatting aligns the end with the declaration
    keyword), or with a trailing `);` for expression functions.
    """
    units: list[dict[str, Any]] = []
    ends = list(_END_NAME.finditer(text))
    used_ends: set[int] = set()

    for match in _BODY_SUBP.finditer(text):
        if _is_generic_instantiation(match.group(0)):
            continue
        name = match.group("name")
        indent = match.group("indent")
        after_is = match.group("after").strip()
        # Expression function: `... is (expr);` on the declaration line.
        expr_m = _EXPR_BODY.search(text[match.start():match.start() + len(match.group(0)) + 200])
        if after_is.startswith("(") and expr_m:
            units.append({
                "name": name,
                "kind": "function",
                "text": expr_m.group(0).strip(),
                "aspects": {},
            })
            continue
        for end_idx, end_match in enumerate(ends):
            if end_idx in used_ends:
                continue
            if end_match.group("name") != name or end_match.group("indent") != indent:
                continue
            used_ends.add(end_idx)
            units.append({
                "name": name,
                "kind": match.group("kind"),
                "text": text[match.start():end_match.end()].strip(),
                "aspects": {},
            })
            break
    return units


def extract_type_decls(text: str) -> list[dict[str, Any]]:
    """Extract scalar type declarations that carry a real constraint."""
    units: list[dict[str, Any]] = []
    for match in _TYPE_DECL.finditer(text):
        definition = match.group("def").strip()
        lowered = definition.lower()
        if any(word in lowered for word in _CONSTRAINT_WORDS):
            units.append({
                "name": match.group("name"),
                "text": f"type {match.group('name')} is {definition.rstrip()};",
                "definition": definition,
            })
    return units


def _package_of(text: str) -> str:
    match = _PACKAGE_NAME.search(text)
    return match.group(1) if match else ""


def pair_subprograms(
    specs: list[dict[str, Any]],
    bodies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Match spec declarations to bodies by (package, kind, name)."""
    body_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for body in bodies:
        key = (body.get("package", "").lower(), body["kind"].lower(), body["name"].lower())
        body_index.setdefault(key, body)

    pairs: list[dict[str, Any]] = []
    for spec in specs:
        key = (spec.get("package", "").lower(), spec["kind"].lower(), spec["name"].lower())
        matched: dict[str, Any] | None = body_index.get(key)
        pairs.append({
            "package": spec.get("package", ""),
            "name": spec["name"],
            "kind": spec["kind"],
            "spec_text": spec["text"],
            "params": spec.get("params", ""),
            "returns": spec.get("returns", ""),
            "aspects": spec.get("aspects", {}),
            "body_text": matched["text"] if matched else "",
            "body_file": matched.get("file", "") if matched else "",
            "file": spec.get("file", ""),
            "has_body": matched is not None,
        })
    return pairs


# --------------------------------------------------------------------------- #
# libadalang extraction (optional, exact)
# --------------------------------------------------------------------------- #


def _lal_extract_file(path_str: str) -> dict[str, Any]:  # pragma: no cover
    """Extract units via libadalang. Requires the lal Python bindings."""
    assert lal is not None
    ctx = lal.AnalysisContext()
    unit = ctx.get_from_file(path_str)
    valid = not unit.diagnostics
    specs: list[dict[str, Any]] = []
    bodies: list[dict[str, Any]] = []
    types: list[dict[str, Any]] = []

    def package_of_node(node: Any) -> str:
        for ancestor in node.parents:
            if isinstance(ancestor, lal.BasePackageDecl):
                name_node = ancestor.f_name
                return name_node.text
        return ""

    for node in unit.root.findall(lal.SubpSpec):
        name_node = node.f_subp_name
        name = name_node.text
        parent = node.parent
        is_body = isinstance(parent, lal.SubpBody)
        aspects: dict[str, str] = {}
        aspects_node = getattr(node, "f_aspects", None)
        if aspects_node is not None:
            for assoc in aspects_node.f_aspects:
                aspects[str(assoc.f_id.text)] = str(assoc.f_expr.text)
        entry = {
            "name": name,
            "kind": "function" if "function" in str(node.f_subp_kind.text).lower() else "procedure",
            "text": node.parent.source_text.strip() if parent is not None else "",
            "aspects": aspects,
            "package": package_of_node(node),
            "valid": valid,
            "file": path_str,
        }
        (bodies if is_body else specs).append(entry)

    for node in unit.root.findall(lal.TypeDecl):
        text_snippet = node.source_text.strip()
        types.append({"name": node.f_name.text, "text": text_snippet, "package": "", "file": path_str})

    return {"specs": specs, "bodies": bodies, "types": types, "valid": valid}


# --------------------------------------------------------------------------- #
# Worker and turn building
# --------------------------------------------------------------------------- #


def extract_file_task(task: tuple[str, str]) -> dict[str, Any]:
    """Worker: extract semantic units from one Ada file.

    Returns {"specs": [...], "bodies": [...], "types": [...], "file": str}.
    Uses libadalang when importable, else the structural scanner. The lal
    path is wrapped so any binding failure degrades to the scanner.
    """
    path_str, kind = task
    path = Path(path_str)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {"specs": [], "bodies": [], "types": [], "file": path_str}
    if not text.strip():
        return {"specs": [], "bodies": [], "types": [], "file": path_str}

    if HAS_LIBADALANG:
        try:
            return _lal_extract_file(path_str)  # pragma: no cover
        except Exception as exc:  # noqa: BLE001  (fall back to the scanner)
            logger.warning("libadalang failed on %s (%s); using the scanner", path_str, exc)

    package = _package_of(text)
    if kind == "ads":
        specs = extract_spec_subprograms(text)
        for spec in specs:
            spec["package"] = package
            spec["file"] = path_str
        return {"specs": specs, "bodies": [], "types": extract_type_decls(text), "file": path_str}
    bodies = extract_body_subprograms(text)
    for body in bodies:
        body["package"] = package
        body["file"] = path_str
    return {"specs": [], "bodies": bodies, "types": [], "file": path_str}


def _standard_label(code: str) -> str:
    return bd._standard_label(bd.detect_ada_standard(code))


def _source_label(path_str: str) -> str:
    """Machine-independent label: final path component only."""
    return path_str.replace("\\", "/").rsplit("/", 1)[-1] if path_str else path_str


def _strip_aspects(spec_text: str) -> str:
    """Remove the aspect clause (``with Pre => ... ;``) from a declaration.

    Aspect expressions contain no semicolons, so everything from the aspect
    marker ``with`` to the terminating semicolon is the clause. A declaration
    without aspects is returned unchanged, which callers use as the signal
    that no contract-completion turn can be built.
    """
    return re.sub(r"\s+\bwith\b[^;]*(?=;)", "", spec_text, count=1, flags=re.DOTALL)


def build_ada_ast_turns(
    specs: list[dict[str, Any]],
    bodies: list[dict[str, Any]],
    types: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build QA records from paired semantic units (parent-side, ordered)."""
    records: list[dict[str, Any]] = []
    for pair in pair_subprograms(specs, bodies):
        name = pair["name"]
        spec_text = pair["spec_text"]
        if not spec_text:
            continue
        standard = _standard_label(spec_text)
        source = _source_label(pair.get("file", ""))
        if pair["has_body"]:
            records.append({
                "messages": [
                    {"role": "user", "content": (
                        f"Please provide the {standard} body for this "
                        f"subprogram declaration of `{name}`.\n\n```ada\n{spec_text}\n```"
                    )},
                    {"role": "assistant", "content": f"```ada\n{pair['body_text']}\n```"},
                ],
                "meta": {
                    "kind": "ast_impl", "unit": name, "source": source,
                    "group": f"ast:{zlib.crc32(spec_text.encode('utf-8'))}",
                },
            })
        if pair["aspects"]:
            prose = bd.sanitize_prose(
                f"The declaration of {name} carries "
                f"{len(pair['aspects'])} aspect(s). They constrain the "
                "subprogram as follows."
            )
            lines = [prose, ""]
            for aspect_name, expr in pair["aspects"].items():
                lines.append(f"- {aspect_name} => {expr}")
            lines += ["", "The exact declaration:", "", f"```ada\n{spec_text}\n```"]
            records.append({
                "messages": [
                    {"role": "user", "content": (
                        f"How is the subprogram `{name}` constrained in this "
                        "specification?"
                    )},
                    {"role": "assistant", "content": "\n".join(lines)},
                ],
                "meta": {
                    "kind": "ast_contract", "unit": name, "source": source,
                    "group": f"ast:{zlib.crc32(spec_text.encode('utf-8'))}",
                },
            })

            # Spec-to-contract completion: show the bare declaration, ask for
            # the contract. This is the write-side skill the eval measures:
            # choosing Pre/Post/Global/Depends that make a subprogram provable.
            # The contract must be earned from real code, so blocklist hits are
            # removed by the eval guard before any split is written.
            bare = _strip_aspects(spec_text)
            if bare != spec_text:
                # Rebuild the full declaration: bare signature minus its
                # terminating semicolon, one aspect clause, comma-separated
                # aspects (repeated `with` markers are not valid Ada).
                decl = bare.rstrip()
                if decl.endswith(";"):
                    decl = decl[:-1].rstrip()
                aspects_str = ",\n        ".join(
                    f"{aspect_name} => {expr}"
                    for aspect_name, expr in pair["aspects"].items()
                )
                contract_decl = f"{decl}\n   with {aspects_str};"
                records.append({
                    "messages": [
                        {"role": "user", "content": (
                            f"Write the SPARK contract for this {standard} "
                            f"subprogram `{name}`. Give the declaration with "
                            "its aspect clauses (Pre, Post, Global, Depends) "
                            "only.\n\n```ada\n" + bare + "\n```"
                        )},
                        {"role": "assistant", "content": f"```ada\n{contract_decl}\n```"},
                    ],
                    "meta": {
                        "kind": "ast_contract_write", "unit": name, "source": source,
                        "group": f"ast:{zlib.crc32(spec_text.encode('utf-8'))}",
                    },
                })
    for type_unit in types:
        name = type_unit["name"]
        text = type_unit["text"]
        if not text:
            continue
        records.append({
            "messages": [
                {"role": "user", "content": f"How is the type `{name}` constrained?"},
                {"role": "assistant", "content": (
                    f"{_standard_label(text)} constrains the type in its "
                    f"definition:\n\n```ada\n{text}\n```"
                )},
            ],
            "meta": {
                "kind": "ast_type", "unit": name,
                "source": _source_label(type_unit.get("file", "")),
                "group": f"ast:{zlib.crc32(text.encode('utf-8'))}",
            },
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Ada semantic units (libadalang or structural) into JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=Path, action="append", default=[],
        help="Ada source tree to extract from (repeatable).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Worker processes (0 = one per CPU core, 1 = serial).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("libadalang available: %s", HAS_LIBADALANG)

    if args.input_dir:
        input_dirs = args.input_dir
    else:
        import source_paths

        input_dirs = source_paths.default_code_dirs() or [Path("../adacovex")]
    tasks: list[tuple[str, str]] = []
    for root in input_dirs:
        if not root.exists():
            logger.warning("Input directory does not exist, skipping: %s", root)
            continue
        for pattern, kind in (("*.ads", "ads"), ("*.adb", "adb")):
            for path in sorted(root.rglob(pattern)):
                tasks.append((str(path), kind))
    if not tasks:
        logger.error("No .ads/.adb files found under %s", input_dirs)
        sys.exit(1)

    workers = args.workers if args.workers > 0 else (multiprocessing.cpu_count() or 1)
    results: list[dict[str, Any]] = []
    if workers > 1 and len(tasks) >= 8:
        try:
            ctx = multiprocessing.get_context("fork")
        except ValueError:
            ctx = None
        if ctx is not None:
            with ctx.Pool(processes=min(workers, len(tasks))) as pool:
                results = list(pool.imap(extract_file_task, tasks, chunksize=4))
    if not results:
        results = [extract_file_task(task) for task in tasks]

    specs = [unit for result in results for unit in result["specs"]]
    bodies = [unit for result in results for unit in result["bodies"]]
    types = [unit for result in results for unit in result["types"]]
    logger.info(
        "Extracted %d spec subprograms, %d bodies, %d constrained types from %d files",
        len(specs), len(bodies), len(types), len(results),
    )

    records = build_ada_ast_turns(specs, bodies, types)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    logger.info("Wrote %d AST-derived turns to %s", len(records), args.output)


if __name__ == "__main__":
    main()

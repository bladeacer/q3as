"""code_variants.py - correctness-preserving Ada variants for data variety.

This module is part of the dataset builder, not a standalone tool: the
correct-variant turns it produces are wired into extra-turn ingestion in
build_dataset.py.

Ada's fixed syntax means natural-language paraphrasing has no code
equivalent, but structure-preserving transformations create genuine
variety without changing meaning:

- **Identifier renaming** - declared locals, parameters, and package-level
  objects get fresh names (deterministic per snippet, never touching
  predefined units like ``Ada.Text_IO``, attributes, or strings). A renamed
  snippet is a new training surface with identical semantics.
- **Statement reordering** - independent consecutive assignment statements
  in a body can swap when neither reads the other's target (verified with
  a data-flow check on the statement text). Produces an equivalent body.
- **Typed contract variants** - a contract proven over one type family
  yields sibling declarations over the other integer types; the shape of
  the obligation is the lesson, not the spelling of the type.

All transformations are deterministic (content-derived choices), so
dataset bytes stay identical across builds and worker counts.
"""

from __future__ import annotations

import itertools
import re
import zlib
from typing import Any

# Ada reserved words (subset relevant to renaming): never renamed.
_KEYWORDS = frozenset([
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

_IDENTIFIER = re.compile(r"\b[A-Za-z]\w*\b")
# Dotted prefixes of the predefined library: rename the *last* component
# never - in Ada.Text_IO every component is library-defined.
_PREDEF_PREFIX = re.compile(r"\b(?:Ada|System|GNAT|Interfaces|Standard)(?:\.\w+)+")
# 'Attribute references: X'Old, X'Result, T'Last ...
_ATTRIBUTE = re.compile(r"'(?:Old|Result|Last|First|Range|Length|Access|Class|Image|Value|Size|Min|Max|Succ|Pred|Pos|Val)\b")


def _protected_spans(code: str) -> list[tuple[int, int]]:
    """Comments and string literals: identifiers there are not renamed."""
    spans = [(m.start(), m.end()) for m in re.finditer(r"--[^\n]*", code)]
    spans.extend((m.start(), m.end()) for m in re.finditer(r'"(?:[^"]|"")*"', code))
    return spans


def _declared_names(code: str) -> list[str]:
    """Declared identifiers safe to rename (locals, params, objects).

    Returns names in declaration order. Speculative: only identifiers that
    are (a) not keywords, (b) not part of a predefined dotted unit,
    (c) not the enclosing package/subprogram name, and (d) declared with a
    ``name :`` or ``name : in`` shape.
    """
    names: list[str] = []
    seen: set[str] = set()
    # Object declarations: "Name : [aliased|constant] Type"
    for match in re.finditer(r"^\s*([A-Za-z]\w*)\s*:\s*(?:constant\s+|aliased\s+)?", code, re.MULTILINE):
        name = match.group(1)
        if name not in _KEYWORDS and name not in seen:
            seen.add(name)
            names.append(name)
    # Parameter names in parameter lists: "Name : [mode] Type"
    for match in re.finditer(r"[;(,]\s*([A-Za-z]\w*)\s*:\s", code):
        name = match.group(1)
        if name not in _KEYWORDS and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _replacement_name(name: str, index: int) -> str:
    """Fresh name preserving the original's length feel and casing."""
    skeleton = "Var_" if name[0].isupper() else "var_"
    return f"{skeleton}{index + 1}"


def rename_identifiers(code: str, salt: str = "") -> str:
    """Rename declared identifiers to deterministic fresh names.

    Predefined units, attributes, keywords, comments, and strings are
    preserved. The mapping is content-derived (crc32 of code+salt), so the
    same snippet always yields the same renamed snippet.
    """
    names = _declared_names(code)
    if not names:
        return code
    mapping: dict[str, str] = {}
    base = zlib.crc32((code + salt).encode("utf-8"))
    for i, name in enumerate(names):
        mapping[name] = _replacement_name(name, (base + i) % 997)
    # Ensure uniqueness within this snippet.
    used: set[str] = set()
    for name in names:
        target = mapping[name]
        while target in used:
            mapping[name] = target + "_a"
            target = mapping[name]
        used.add(target)

    spans = _protected_spans(code)
    out: list[str] = []
    pos = 0
    for match in _IDENTIFIER.finditer(code):
        start, end = match.start(), match.end()
        name = match.group(0)
        if name not in mapping or any(s <= start < e for s, e in spans):
            continue
        # Predefined dotted unit: skip every component.
        if any(ps <= start < pe for ps, pe in _predef_spans(code)):
            continue
        # Attribute context: "Name'Old" - the attribute itself is no
        # identifier, but the prefix is the user's name and must rename.
        out.append(code[pos:start])
        out.append(mapping[name])
        pos = end
    out.append(code[pos:])
    return "".join(out)


def _predef_spans(code: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _PREDEF_PREFIX.finditer(code)]


def reorder_statements(code: str) -> str | None:
    """Swap two adjacent independent assignment statements, or None.

    Independence check (conservative): two statements swap only when each
    is a plain ``Target := <expr>;`` whose Target is not mentioned in the
    other statement. This excludes read-write conflicts; calls, control
    flow, and attribute writes never swap.
    """
    body_m = re.search(r"\bbegin\b", code)
    if not body_m:
        return None
    spans = _protected_spans(code)
    stmts = list(re.finditer(r"^[ \t]*[A-Za-z]\w*\s*:=\s*[^;]+;", code[body_m.end():], re.MULTILINE))
    for first, second in itertools.pairwise(stmts):
        # Must be adjacent in the text (only whitespace between them).
        gap = code[body_m.end() + first.end():body_m.end() + second.start()]
        if gap.strip():
            continue
        f_start = body_m.end() + first.start()
        f_end = body_m.end() + first.end()
        s_start = body_m.end() + second.start()
        s_end = body_m.end() + second.end()
        if any(max(f_start, s_start) <= p < max(f_end, s_end) for p, _e in [(s, e) for s, e in spans] for p in range(max(f_start, s_start), max(f_end, s_end))):
            continue
        f_target = re.match(r"[ \t]*([A-Za-z]\w*)\s*:=", first.group(0)).group(1)  # type: ignore[union-attr]
        s_target = re.match(r"[ \t]*([A-Za-z]\w*)\s*:=", second.group(0)).group(1)  # type: ignore[union-attr]
        f_text, s_text = first.group(0), second.group(0)
        # Data flow: no target appears inside the other statement.
        if re.search(rf"\b{re.escape(f_target)}\b", s_text):
            continue
        if re.search(rf"\b{re.escape(s_target)}\b", f_text):
            continue
        # Exchange the two statement slices. The match text spans the whole
        # line (indent included, regex anchors on ^), so swapping the exact
        # slices keeps indentation and surrounding newlines intact.
        reordered = (
            code[:f_start]
            + s_text
            + code[f_end:s_start]
            + f_text
            + code[s_end:]
        )
        return reordered if reordered != code else None
    return None


# --------------------------------------------------------------------------- #
# Typed contract variants
# --------------------------------------------------------------------------- #

# Integer families whose obligations transfer: same contract shape, new type.
_TYPE_SIBLINGS = {
    "Integer": ("Natural", "Positive"),
    "Natural": ("Integer", "Positive"),
    "Positive": ("Natural", "Integer"),
}

_INT_TYPE = re.compile(r"\b(Integer|Natural|Positive|Long_Integer)\b")


def typed_contract_variants(decl: str) -> list[str]:
    """Sibling declarations of *decl* over other integer types.

    ``procedure Bump (X : in out Natural) with Pre => X <= Natural'Last - 1``
    yields the Integer and Positive spellings, which train the *shape* of
    the overflow obligation rather than one type's spelling.
    """
    typ = _INT_TYPE.search(decl)
    if not typ:
        return []
    root = typ.group(1)
    variants: list[str] = []
    for sibling in _TYPE_SIBLINGS.get(root, ()):
        variants.append(_INT_TYPE.sub(sibling, decl, count=0))
    return variants


# --------------------------------------------------------------------------- #
# Integration: correct-variant turns from a chat record's code
# --------------------------------------------------------------------------- #

_FENCE = re.compile(r"```(?:ada)?\n(.*?)```", re.DOTALL)


def variant_turns(
    record: dict[str, Any],
    kind_prefix: str = "variant",
) -> list[tuple[str, dict[str, Any]]]:
    """Build (group, record) correct-variant turns for one chat record.

    Mirrors the original record's user ask but presents the variant code,
    so the model sees the same task on differently spelled, semantically
    identical code. Returns [] when the record has no Ada fence or no
    transformation applies. Never applied to defect turns (their broken
    code must stay tied to the original).
    """
    messages = record.get("messages") or []
    assistant = next((m for m in messages if m.get("role") == "assistant"), None)
    if assistant is None:
        return []
    blocks = _FENCE.findall(str(assistant.get("content", "")))
    if not blocks:
        return []
    code = blocks[0]
    meta = record.get("meta") or {}
    group = str(meta.get("group") or "variant")
    out: list[tuple[str, dict[str, Any]]] = []

    renamed = rename_identifiers(code, salt=meta.get("kind", ""))
    if renamed != code:
        user = next((m for m in messages if m.get("role") == "user"), None)
        user_text = str(user.get("content", "")) if user else "Provide the same code with different names."
        # Also rename identifiers inside the user's fence, if any, for
        # consistency of the pair.
        new_user = user_text
        for fence in _FENCE.findall(user_text):
            new_user = new_user.replace(fence, rename_identifiers(fence, salt=meta.get("kind", "")))
        out.append((group, {
            "messages": [
                {"role": "user", "content": new_user},
                {"role": "assistant", "content": str(assistant.get("content", "")).replace(code, renamed)},
            ],
            "meta": {**meta, "kind": f"{kind_prefix}_renamed"},
        }))

    reordered = reorder_statements(code)
    if reordered is not None:
        user = next((m for m in messages if m.get("role") == "user"), None)
        user_text = str(user.get("content", "")) if user else ""
        out.append((group, {
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": str(assistant.get("content", "")).replace(code, reordered)},
            ],
            "meta": {**meta, "kind": f"{kind_prefix}_reordered"},
        }))
    return out

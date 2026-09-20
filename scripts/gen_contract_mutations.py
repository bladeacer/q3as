#!/usr/bin/env python3
"""gen_contract_mutations.py - gnatprove-verified synthetic contract data.

The eval showed the fine-tuned model writes near-correct code whose
contracts do not discharge (``VC_OVERFLOW_CHECK`` x8 and friends). Real
corpora contain few *provable* contracts, so this tool synthesizes them:

1. Each template below (guarded increment, even halve, swap with
   ``Depends``, clamp range, guarded subtract) is instantiated over
   parameterized names, types, and bounds.
2. Every instance is compiled and proved with the real ``gnatprove`` in a
   temporary GNAT project. Only instances whose whole unit comes out
   ``proved`` are kept - a contract that does not prove never trains.
3. Wrong variants (a deliberately weakened contract) are kept only when
   gnatprove reports them ``not proved``, so broken examples carry a
   machine-checked proof failure, not a guess.

Output: ``data/processed/contract_mutations.jsonl`` in the chat-record
shape the dataset builder ingests (``--extra-turns``), with a ``meta``
block recording the verification result. The eval guard still applies to
this file like any other source.

Cached: when the output file exists the tool does nothing (``--force``
regenerates).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from alire_env import alire_env_path, find_tool

OUTPUT = ROOT / "data" / "processed" / "contract_mutations.jsonl"

# --------------------------------------------------------------------------- #
# Templates (each shape verified against gnatprove 16, --level=1, before
# release; see the probe notes in docs/datasets-and-training.md)
# --------------------------------------------------------------------------- #

TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "name": "guarded_increment",
        "decl": "procedure {unit} (X : in out {typ})",
        "contract": "Pre => X <= {typ}'Last - 1",
        "body": (
            "   procedure {unit} (X : in out {typ}) is\n"
            "   begin\n      X := X + 1;\n   end {unit};"
        ),
        "types": ("Integer", "Natural", "Positive"),
        # (old fragment, replacement) - makes the contract too weak to prove.
        "weaken": (r"X <= {typ}'Last - 1", "X >= 0"),
    },
    {
        "name": "guarded_subtract",
        "decl": "procedure {unit} (A : in {typ}; B : in {typ}; R : out {typ})",
        "contract": "Pre => A >= B",
        "body": (
            "   procedure {unit} (A : in {typ}; B : in {typ}; R : out {typ}) is\n"
            "   begin\n      R := A - B;\n   end {unit};"
        ),
        "types": ("Natural", "Positive"),
        "weaken": (r"A >= B", "A > 0"),
    },
    {
        "name": "even_halve",
        "decl": "procedure {unit} (X : in Positive; Y : out {typ})",
        "contract": "Pre => X mod 2 = 0, Post => Y * 2 = X",
        "body": (
            "   procedure {unit} (X : in Positive; Y : out {typ}) is\n"
            "   begin\n      Y := X / 2;\n   end {unit};"
        ),
        "types": ("Natural",),
        "weaken": (r"X mod 2 = 0", "X > 0"),
    },
    {
        "name": "swap_depends",
        "decl": "procedure {unit} (A : in out {typ}; B : in out {typ})",
        "contract": "Depends => (A => B, B => A), Post => A = B'Old and B = A'Old",
        "body": (
            "   procedure {unit} (A : in out {typ}; B : in out {typ}) is\n"
            "      T : {typ} := A;\n   begin\n      A := B;\n      B := T;\n   end {unit};"
        ),
        "types": ("Integer", "Natural"),
        "weaken": (r"Post => A = B'Old and B = A'Old", "Post => A = A'Old"),
    },
    {
        "name": "clamp_range",
        "decl": "procedure {unit} (X : in {typ}; Lo : in {typ}; Hi : in {typ}; Y : out {typ})",
        "contract": "Pre => Lo <= Hi, Post => Y in Lo .. Hi",
        "body": (
            "   procedure {unit} (X : in {typ}; Lo : in {typ}; Hi : in {typ}; Y : out {typ}) is\n"
            "   begin\n"
            "      if X < Lo then\n         Y := Lo;\n"
            "      elsif X > Hi then\n         Y := Hi;\n"
            "      else\n         Y := X;\n      end if;\n   end {unit};"
        ),
        "types": ("Integer",),
        "weaken": (r"Pre => Lo <= Hi, ", ""),
    },
)

_UNIT_NAMES = (
    "Bump_Value", "Step_Up", "Advance", "Raise_By_One", "Shift_Positive",
    "Take_Half", "Split_Even", "Halve_Input", "Exchange", "Trade_Slots",
    "Swap_Sides", "Confine", "Bound_Value", "Restrain_To", "Clip_To_Range",
    "Safe_Diff", "Subtract_Guarded", "Take_Away", "Diminish_By",
)


def _pick(seq: tuple[str, ...], key: str) -> str:
    return seq[zlib.crc32(key.encode("utf-8")) % len(seq)]


def instantiate(template: dict[str, Any], index: int) -> dict[str, Any]:
    """Build one instance dict (unit, typ, spec, body, template)."""
    unit = _pick(_UNIT_NAMES, template["name"] + str(index))
    typ = template["types"][index % len(template["types"])]
    subs = {"unit": unit, "typ": typ}
    decl = template["decl"].format(**subs)
    contract = template["contract"].format(**subs)
    spec = (
        "pragma SPARK_Mode (On);\n\n"
        f"package Ctr_{index} is\n\n"
        f"   {decl}\n     with {contract};\n\n"
        f"end Ctr_{index};\n"
    )
    body = (
        "pragma SPARK_Mode (On);\n\n"
        f"package body Ctr_{index} is\n\n"
        f"{template['body'].format(**subs)}\n\n"
        f"end Ctr_{index};\n"
    )
    return {"index": index, "unit": unit, "typ": typ, "decl": decl, "spec": spec, "body": body, "template": template}


def weaken(instance: dict[str, Any]) -> str | None:
    """Break the contract per the template rule (for wrong-example turns)."""
    old, new = instance["template"]["weaken"]
    old_f = old.format(typ=instance["typ"])
    new_f = new.format(typ=instance["typ"])
    broken = instance["spec"].replace(old_f, new_f)
    return broken if broken != instance["spec"] else None


# --------------------------------------------------------------------------- #
# gnatprove verification
# --------------------------------------------------------------------------- #

_GPR = """project Prv is
   for Languages use ("Ada");
   for Source_Dirs use ("src");
   for Object_Dir use "obj";
end Prv;
"""


def _alire_env() -> dict[str, str]:
    """Full environment with the Alire toolchain prepended to PATH."""
    return {**os.environ, "PATH": alire_env_path()}


def run_gnatprove(project_dir: Path) -> tuple[bool, str]:
    """Prove everything in the project. Returns (all_proved, summary)."""
    gpr = project_dir / "main.gpr"
    if not gpr.exists():
        gpr.write_text(_GPR, encoding="utf-8")
    gnatprove = find_tool("gnatprove")
    cmd = [str(gnatprove), f"-P{gpr}", "-j0", "--level=1"]
    proc = subprocess.run(
        cmd, cwd=project_dir, capture_output=True, text=True, timeout=600,
        env=_alire_env(), check=False,
    )
    out_file = project_dir / "obj" / "gnatprove" / "gnatprove.out"
    summary = out_file.read_text(encoding="utf-8", errors="replace") if out_file.exists() else ""
    output = proc.stdout + proc.stderr
    all_proved = proc.returncode == 0 and "not proved" not in summary and "error:" not in output
    return all_proved, summary


def _packages(units: list[dict[str, Any]]) -> set[str]:
    return {re.search(r"package\s+(?:body\s+)?(\w+)", u["spec"]).group(1) for u in units}  # type: ignore[union-attr]


def batch_verify(batch: list[dict[str, Any]], workdir: Path) -> dict[int, bool]:
    """Verify a batch in one gnatprove run. Returns {index: all_proved}."""
    src = workdir / "src"
    src.mkdir(parents=True, exist_ok=True)
    for unit in batch:
        package = re.search(r"package\s+(?:body\s+)?(\w+)", unit["spec"]).group(1)  # type: ignore[union-attr]
        (src / f"{package.lower()}.ads").write_text(unit["spec"], encoding="utf-8")
        (src / f"{package.lower()}.adb").write_text(unit["body"], encoding="utf-8")
    all_proved, summary = run_gnatprove(workdir)
    flags: dict[int, bool] = {}
    for unit in batch:
        package = re.search(r"package\s+(?:body\s+)?(\w+)", unit["spec"]).group(1)  # type: ignore[union-attr]
        if all_proved:
            flags[unit["index"]] = True
            continue
        # Partial failure: this unit proves only if its summary lines all
        # say "and proved" (package and each subprogram).
        lines = re.findall(rf"^\s*{package}(?:\.\w+)? at \S+ .*$", summary, re.MULTILINE)
        flags[unit["index"]] = bool(lines) and all("and proved" in ln and "not proved" not in ln for ln in lines)
    return flags


def verify_unproved(spec: str, body: str, workdir: Path) -> bool:
    """True when gnatprove reports the unit NOT proved (wrong variants)."""
    src = workdir / "src"
    src.mkdir(parents=True, exist_ok=True)
    for path in src.glob("*"):
        path.unlink()
    package = re.search(r"package\s+(?:body\s+)?(\w+)", spec).group(1)  # type: ignore[union-attr]
    (src / f"{package.lower()}.ads").write_text(spec, encoding="utf-8")
    (src / f"{package.lower()}.adb").write_text(body, encoding="utf-8")
    all_proved, summary = run_gnatprove(workdir)
    return (not all_proved) or "not proved" in summary


# --------------------------------------------------------------------------- #
# Turn building
# --------------------------------------------------------------------------- #

def _explain(unit: str, typ: str, spec: str) -> str:
    """Short STE explanation (template-checked, not model-generated)."""
    contract = re.search(r"with ([^;]+);", spec, re.DOTALL)
    text = " ".join(contract.group(1).split()) if contract else "the contract"
    return (
        f"The body of {unit} performs arithmetic on {typ} values. Without a "
        "constraint the arithmetic can overflow or violate the subtype "
        f"range. The contract {text} rules out those inputs, so gnatprove "
        "discharges every check."
    )


def _decl_with_contract(spec: str, unit: str) -> str:
    """Extract the declaration plus aspect clause lines for a unit."""
    match = re.search(rf"^\s*(?:procedure|function)\s+{unit}\b[^;]*;", spec, re.DOTALL | re.MULTILINE)
    return " ".join(match.group(0).split()) if match else unit


def build_turns(limit: int) -> list[dict[str, Any]]:
    """Instantiate, verify, and emit chat records."""
    records: list[dict[str, Any]] = []
    per_template = max(1, limit // len(TEMPLATES))
    instances: list[dict[str, Any]] = []
    index = 0
    for template in TEMPLATES:
        for _ in range(per_template):
            instances.append(instantiate(template, index))
            index += 1

    batch_size = 6
    for start in range(0, len(instances), batch_size):
        batch = instances[start:start + batch_size]
        with tempfile.TemporaryDirectory(prefix="q3as-prove-") as tmp:
            try:
                flags = batch_verify(batch, Path(tmp))
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
                logger.warning("gnatprove batch failed (%s); skipping %d units", exc, len(batch))
                continue
        for instance in batch:
            if not flags.get(instance["index"]):
                logger.info("unit %s did not prove; dropped", instance["unit"])
                continue
            spec, unit, typ = instance["spec"], instance["unit"], instance["typ"]
            group = f"contract-synth:{zlib.crc32(spec.encode('utf-8'))}"
            verified = "gnatprove proved"
            full_decl = _decl_with_contract(spec, unit)
            bare = re.sub(r"\s*with [^;]+;", ";", full_decl, count=1)

            records.append({
                "messages": [
                    {"role": "user", "content": (
                        f"Write a SPARK contract that makes {unit} provable "
                        "for this declaration. Give the declaration with its "
                        "aspect clauses only.\n\n```ada\n" + bare + "\n```"
                    )},
                    {"role": "assistant", "content": "```ada\n" + full_decl + "\n```"},
                ],
                "meta": {
                    "kind": "contract_synth_write", "unit": unit,
                    "template": instance["template"]["name"], "verified": verified,
                    "group": group,
                },
            })
            records.append({
                "messages": [
                    {"role": "user", "content": (
                        f"Why does the contract of `{unit}` over {typ} make "
                        "the body provable?"
                    )},
                    {"role": "assistant", "content": _explain(unit, typ, spec)},
                ],
                "meta": {
                    "kind": "contract_synth_why", "unit": unit,
                    "template": instance["template"]["name"], "verified": verified,
                    "group": group,
                },
            })

            broken_spec = weaken(instance)
            if broken_spec is None:
                continue
            with tempfile.TemporaryDirectory(prefix="q3as-unpr-") as tmp2:
                try:
                    still_proves = not verify_unproved(broken_spec, instance["body"], Path(tmp2))
                except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
                    logger.warning("unproved-check failed (%s); skipping wrong turn", exc)
                    continue
            if still_proves:
                logger.info("weakened %s still proved; dropped wrong turn", unit)
                continue
            broken_decl = _decl_with_contract(broken_spec, unit)
            records.append({
                "messages": [
                    {"role": "user", "content": (
                        f"gnatprove cannot prove {unit} with this contract. "
                        "Fix the contract so the body proves.\n\n```ada\n"
                        + broken_decl + "\n```"
                    )},
                    {"role": "assistant", "content": (
                        "The contract is too weak. It does not rule out the "
                        "inputs that make the body violate its checks. The "
                        "provable contract constrains the inputs before the "
                        "arithmetic runs.\n\n```ada\n" + full_decl + "\n```"
                    )},
                ],
                "meta": {
                    "kind": "contract_synth_fix", "unit": unit,
                    "template": instance["template"]["name"],
                    "verified": "gnatprove unproved before fix",
                    "group": group,
                },
            })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate gnatprove-verified contract training turns.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int, default=30, help="approximate number of instances to verify")
    parser.add_argument("--force", action="store_true", help="regenerate even when the output exists")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    if args.output.exists() and not args.force:
        logger.info("cached output exists: %s (use --force to regenerate)", args.output)
        return 0

    records = build_turns(args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    kinds: dict[str, int] = {}
    for record in records:
        kinds[record["meta"]["kind"]] = kinds.get(record["meta"]["kind"], 0) + 1
    logger.info("wrote %d gnatprove-verified turns to %s (%s)", len(records), args.output, kinds)
    return 0 if records else 1


if __name__ == "__main__":
    sys.exit(main())

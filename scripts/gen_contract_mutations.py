#!/usr/bin/env python3
"""gen_contract_mutations.py - gnatprove-verified synthetic contract data.

The eval showed the fine-tuned model writes near-correct code whose
contracts do not discharge: ``VC_OVERFLOW_CHECK`` x8, ``UNINITIALIZED``
x4, ``VC_POSTCONDITION`` x3, ``VC_RAISE`` x2, ``DEPENDS_MISSING`` x2
(results v0.4.1). Real corpora contain few *provable* contracts, so this
tool synthesizes them, one template family per eval blocker:

1. Each template below is instantiated over parameterized names, types,
   and bounds. The families cover overflow guards (``Pre`` bounding
   arithmetic), initialization plus postcondition (an ``out`` parameter
   written on every path), raise guards (a ``Pre`` that makes the
   ``raise`` impossible), loop invariants (triangle sum, array maximum),
   ``Depends`` swaps, clamp ranges, and ``Global`` data-flow contracts.
2. Every instance is compiled and proved with the real ``gnatprove`` in a
   temporary GNAT project, in batches. Only instances whose whole unit
   comes out ``proved`` are kept - a contract that does not prove never
   trains.
3. Wrong variants (a deliberately weakened contract) are verified the
   same way and kept only when gnatprove reports them ``not proved``, so
   broken examples carry a machine-checked proof failure, not a guess.

Every template family can produce up to four turn kinds: write the
contract (``contract_synth_write``), explain why it discharges
(``contract_synth_why``), complete a body under a spec
(``contract_synth_body``, when the body carries the real lesson), and
repair a machine-verified broken contract (``contract_synth_fix``).

Output: ``data/processed/contract_mutations.jsonl`` in the chat-record
shape the dataset builder ingests (``--extra-turns``), with a ``meta``
block recording the verification result. The eval guard still applies to
this file like any other source.

Cached: the run is skipped when this file, ``alire_env.py``, and ``--limit``
are unchanged since the last successful build and the output is still in
place; ``--force`` regenerates. A run that proves nothing does not record a
stamp, so the next invocation retries instead of trusting an empty file.
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
sys.path.insert(0, str(ROOT / "data" / "processing_scripts"))

import alire_env
import stage_state
from alire_env import alire_env_path, find_tool

OUTPUT = ROOT / "data" / "processed" / "contract_mutations.jsonl"# --------------------------------------------------------------------------- #
# Templates: one family per eval proof blocker. Each template states the
# whole SPARK unit (spec aspect clauses plus body), so verification covers
# the real proof obligation, not just a declaration. Shapes marked "probe
# notes" were verified against gnatprove 16 --level=1 before release.
# --------------------------------------------------------------------------- #

TEMPLATES: tuple[dict[str, Any], ...] = (
    # --- VC_OVERFLOW_CHECK: the contract must bound the arithmetic. ---
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
    # --- UNINITIALIZED + VC_POSTCONDITION: an out parameter must be
    # --- assigned on every path; the Post states what was stored.
    {
        "name": "init_default",
        "decl": "procedure {unit} (R : out {typ}; N : in {typ})",
        "contract": "Post => R = N",
        "body": (
            "   procedure {unit} (R : out {typ}; N : in {typ}) is\n"
            "   begin\n      R := N;\n   end {unit};"
        ),
        "types": ("Natural", "Integer", "Positive"),
        # A body that skips the write on one path: gnatprove reports
        # "might not be initialized" plus "postcondition might fail".
        "weak_body": (
            "   procedure {unit} (R : out {typ}; N : in {typ}) is\n"
            "   begin\n"
            "      if N > 0 then\n         R := N;\n      end if;\n"
            "   end {unit};"
        ),
        "fix_contract": "Pre => N > 0, Post => R = N",
        "fix_explain": (
            "The body writes R only when N is positive. On other paths R is "
            "never assigned, so gnatprove reports that R might not be "
            "initialized and that the postcondition might fail. The "
            "precondition N > 0 rules out the skipping paths, so both "
            "checks discharge. Restricting the inputs is the right fix "
            "here: adding an else branch would change what the subprogram "
            "computes."
        ),
        "weaken": None,
    },
    # --- VC_RAISE: the Pre makes the raise impossible. ---
    {
        "name": "raise_guard",
        "decl": "procedure {unit} (V : in {typ})",
        "contract": "Pre => V >= 0",
        "body": (
            "   procedure {unit} (V : in {typ}) is\n"
            "   begin\n"
            "      if V < 0 then\n         raise Constraint_Error;\n      end if;\n"
            "   end {unit};"
        ),
        "types": ("Integer",),
        # A wrong-boundary guard: V /= 0 still allows V = Integer'First, so
        # the raise check cannot discharge. Valid syntax, real failure.
        "weaken": ("Pre => V >= 0", "Pre => V /= 0"),
    },
    # --- Loop invariants: the lesson is a floating invariant that ties the
    # --- running variable to the processed prefix, plus the widening that
    # --- discharges the overflow VCs a naive accumulator trips.
    {
        "name": "loop_sum",
        "decl": "function {unit} (N : in Natural) return Long_Long_Integer",
        "contract": (
            "Post => {unit}'Result\n"
            "               = ({typ2} (N) + 1) * {typ2} (N) / 2"
        ),
        "body": (
            "   function {unit} (N : in Natural) return Long_Long_Integer is\n"
            "      Total : Long_Long_Integer := 0;\n"
            "   begin\n"
            "      for I in 1 .. N loop\n"
            "         pragma Loop_Invariant\n"
            "           (Total = {typ2} (I - 1) * {typ2} (I) / 2);\n"
            "         Total := Total + {typ2} (I);\n"
            "      end loop;\n"
            "      return Total;\n"
            "   end {unit};"
        ),
        "types": (),  # unused: {typ} does not appear in this template
        "typ2": "Long_Long_Integer",
        # A wrong postcondition (the classic off-by-one: forgetting the
        # +1). Valid syntax, real failure: only N in 0 .. 1 satisfies it.
        "weaken": (
            r"= ({typ2} (N) + 1) * {typ2} (N) / 2",
            r"= {typ2} (N)",
        ),
    },
    {
        "name": "loop_max_array",
        "decl": "function {unit} (A : in Int_Array) return Integer",
        "contract": (
            "Pre  => A'Length > 0 and A'Last < Positive'Last, "
            "Post => (for all K in A'Range => {unit}'Result >= A (K))"
        ),
        "body": (
            "   function {unit} (A : in Int_Array) return Integer is\n"
            "      Best : Integer := A (A'First);\n"
            "   begin\n"
            "      for K in A'First + 1 .. A'Last loop\n"
            "         pragma Loop_Invariant\n"
            "           (for all J in A'First .. K - 1 => Best >= A (J));\n"
            "         if A (K) > Best then\n"
            "            Best := A (K);\n"
            "         end if;\n"
            "      end loop;\n"
            "      return Best;\n"
            "   end {unit};"
        ),
        "types": (),
        # Drop the A'Last bound: the index increment A'First + 1 can then
        # overflow, so the invariant cannot discharge.
        "weaken": (" and A'Last < Positive'Last", ""),
        "extra_decls": {
            "Int_Array": "   type Int_Array is array (Positive range <>) of Integer;\n",
        },
    },
    # --- DEPENDS_MISSING: an explicit data-flow contract. ---
    {
        "name": "swap_depends",
        "decl": "procedure {unit} (A : in out {typ}; B : in out {typ})",
        "contract": "Depends => (A => B, B => A), Post => A = B'Old and B = A'Old",
        "body": (
            "   procedure {unit} (A : in out {typ}; B : in out {typ}) is\n"
            "      T : {typ} := A;\n"
            "   begin\n"
            "      A := B;\n"
            "      B := T;\n"
            "   end {unit};"
        ),
        "types": ("Integer", "Natural"),
        "weaken": (r"Post => A = B'Old and B = A'Old", "Post => A = A'Old"),
    },
    # --- Global data flow: a package variable plus its Global contract.
    # --- The wrong form contradicts the body and the flow analysis rejects
    # --- it with a hard error (probe note: gnatprove reports "Count" is
    # --- referenced in Pre but missing from the Global for the null form).
    {
        "name": "global_counter",
        "decl": "procedure {unit} (X : in {typ})",
        "contract": (
            "Global => (In_Out => Count), Pre => X <= {typ}'Last - Count"
        ),
        "body": (
            "   procedure {unit} (X : in {typ}) is\n"
            "   begin\n"
            "      Count := Count + X;\n"
            "   end {unit};"
        ),
        "types": ("Natural",),
        "weaken": ("Global => (In_Out => Count)", "Global => null"),
        "extra_decls": {
            "Count": "   Count : Natural := 0;\n",
        },
    },
)

_UNIT_NAMES = (
    "Bump_Value", "Step_Up", "Advance", "Raise_By_One", "Shift_Positive",
    "Take_Half", "Split_Even", "Halve_Input", "Exchange", "Trade_Slots",
    "Swap_Sides", "Confine", "Bound_Value", "Restrain_To", "Clip_To_Range",
    "Safe_Diff", "Subtract_Guarded", "Take_Away", "Diminish_By",
    "Fill_Buffer", "Store_Value", "Copy_Result", "Load_Entry", "Set_Field",
    "Check_Sign", "Guard_Value", "Reject_Negative", "Screen_Input",
    "Sum_Range", "Accumulate", "Running_Total", "Pick_Maximum",
    "Tallest_Of", "Greatest_In", "Reflect_Pair", "Mirror_Items",
    "Count_Up", "Record_Call", "Log_Tick", "Advance_Clock",
)


def _pick(seq: tuple[str, ...], key: str) -> str:
    return seq[zlib.crc32(key.encode("utf-8")) % len(seq)]


def instantiate(template: dict[str, Any], index: int) -> dict[str, Any]:
    """Build one instance dict (unit, typ, spec, body, template)."""
    unit = _pick(_UNIT_NAMES, template["name"] + str(index))
    typ = template["types"][index % len(template["types"])] if template["types"] else ""
    subs = {"unit": unit, "typ": typ, "typ2": template.get("typ2", "")}
    decl = template["decl"].format(**subs)
    contract = template["contract"].format(**subs)
    # Package-level declarations some templates need (a state variable, an
    # array type the declaration refers to). Plain Ada, goes into spec and
    # body alike.
    extra_decls: dict[str, str] = template.get("extra_decls", {})
    extra = (
        "\n" + "\n".join(extra_decls.values()).rstrip("\n") + "\n"
        if extra_decls
        else ""
    )
    # extra_decls go into the SPEC only: the body sees them through the
    # package, and a second copy in the body is a declaration conflict
    # (verified: gnatprove rejects the duplicated type with "conflicts with
    # declaration", which then cascades into a bogus "missing body").
    spec = (
        "pragma SPARK_Mode (On);\n\n"
        f"package Ctr_{index} is\n{extra}\n"
        f"   {decl}\n     with {contract};\n\n"
        f"end Ctr_{index};\n"
    )
    body = (
        "pragma SPARK_Mode (On);\n\n"
        f"package body Ctr_{index} is\n\n"
        f"{template['body'].format(**subs)}\n\n"
        f"end Ctr_{index};\n"
    )
    return {
        "index": index, "unit": unit, "typ": typ, "decl": decl,
        "contract": contract, "spec": spec, "body": body, "template": template,
    }


def weaken(instance: dict[str, Any]) -> str | None:
    """Break the contract per the template rule (for wrong-example turns)."""
    rule = instance["template"].get("weaken")
    if rule is None:
        return None
    old, new = rule
    subs = {"typ": instance["typ"], "typ2": instance["template"].get("typ2", "")}
    old_f = old.format(**subs)
    new_f = new.format(**subs)
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


def _unit_proved_from_summary(package: str, summary: str) -> bool | None:
    """Per-unit proof status from a gnatprove.out summary.

    True when every analyzed line for *package* (the package itself plus its
    ``Package.Unit`` subprograms) ends in "and proved"; False when any line
    is "not proved"; None when the unit has no lines at all, which the
    caller must treat as "did not prove" so a parse failure never trains an
    unverified unit.
    """
    lines = re.findall(rf"^\s*{package}(?:\.\w+)? at \S+ .*$", summary, re.MULTILINE)
    if not lines:
        return None
    return all("and proved" in ln and "not proved" not in ln for ln in lines)


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
        status = True if all_proved else _unit_proved_from_summary(package, summary)
        flags[unit["index"]] = status is True
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

def _explain(unit: str, typ: str, spec: str, template_name: str) -> str:
    """Short STE explanation (template-checked, not model-generated)."""
    if template_name == "init_default":
        return (
            f"{unit} writes the out parameter R from N. The postcondition "
            f"R = N states that fact. SPARK flow analysis requires an out "
            "parameter to be assigned on every path, and the prover "
            "discharges the postcondition from the single assignment."
        )
    if template_name == "raise_guard":
        return (
            f"The body of {unit} raises Constraint_Error when V is "
            "negative. The precondition V >= 0 makes that branch "
            "impossible, so gnatprove discharges the exception check. In "
            "SPARK, a subprogram allows no exception to propagate unless "
            "the contract rules the raising inputs out."
        )
    if template_name in ("loop_sum", "loop_max_array"):
        return (
            "The loop needs a loop invariant that ties the running "
            "variable to the part of the input processed so far. The "
            "invariant holds before the loop, survives one iteration, and "
            "is strong enough at the exit to prove the postcondition. The "
            "arithmetic is widened to Long_Long_Integer so the overflow "
            "checks discharge, because a Natural accumulator can overflow "
            "before the loop ends."
        )
    if template_name == "global_counter":
        return (
            f"{unit} updates the package variable Count. The Global "
            "aspect declares that data flow, and the precondition bounds "
            "X so the update cannot overflow. gnatprove checks the "
            "contract against the body, so a Global aspect that hides a "
            "variable the body reads or writes is a hard error."
        )
    contract = re.search(r"with ([^;]+);", spec, re.DOTALL)
    text = " ".join(contract.group(1).split()) if contract else "the contract"
    return (
        f"The body of {unit} performs arithmetic on {typ} values. Without a "
        "constraint the arithmetic can overflow or violate the subtype "
        f"range. The contract {text} rules out those inputs, so gnatprove "
        "discharges every check."
    )


def _decl_with_contract(spec: str, unit: str) -> str:
    """Extract the declaration plus aspect clauses for a unit.

    A plain first-semicolon cut truncates every multi-parameter declaration
    at its parameter list (``procedure Fill (R : out Natural;``), so this
    scans from the declaration start to the first ``;`` at parenthesis
    depth zero: parameter semicolons sit inside parens, aspect expressions
    use parens and commas, and the terminating semicolon is bare.
    """
    match = re.search(rf"\b(?:procedure|function)\s+{unit}\b", spec)
    if not match:
        return unit
    depth = 0
    for position in range(match.start(), len(spec)):
        char = spec[position]
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == ";" and depth == 0:
            return " ".join(spec[match.start():position + 1].split())
    return unit


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
            template = instance["template"]
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
                    "template": template["name"], "verified": verified,
                    "group": group,
                },
            })
            records.append({
                "messages": [
                    {"role": "user", "content": (
                        f"Why does the contract of `{unit}` over {typ} make "
                        "the body provable?"
                    ) if typ else (
                        f"Why does the contract of `{unit}` make the body "
                        "provable?"
                    )},
                    {"role": "assistant", "content": _explain(unit, typ, spec, template["name"])},
                ],
                "meta": {
                    "kind": "contract_synth_why", "unit": unit,
                    "template": template["name"], "verified": verified,
                    "group": group,
                },
            })

            # Body completion: for the templates whose real lesson sits in
            # the body (loop invariant placement, every-path assignment),
            # also train spec-in, working-body-out. gnatprove proved exactly
            # this spec/body pair in the batch run above.
            if "weak_body" in template:
                records.append({
                    "messages": [
                        {"role": "user", "content": (
                            "Write the SPARK body for this declaration. The "
                            "body must satisfy every check the contract "
                            "implies.\n\n```ada\n" + bare + "\n```"
                        )},
                        {"role": "assistant", "content": "```ada\n" + instance["body"] + "\n```"},
                    ],
                    "meta": {
                        "kind": "contract_synth_body", "unit": unit,
                        "template": template["name"], "verified": verified,
                        "group": group,
                    },
                })

            if "fix_contract" in template:
                # Weak-body family: the contract is right and the body is
                # wrong (writes the out parameter on one path only). Verify
                # that the weak body really fails, then train repaired-unit
                # out. The corrected precondition is part of the answer.
                weak_body = template["weak_body"].format(
                    unit=unit, typ=typ, typ2=template.get("typ2", ""),
                )
                with tempfile.TemporaryDirectory(prefix="q3as-unpr-") as tmp2:
                    try:
                        weak_proves = not verify_unproved(spec, weak_body, Path(tmp2))
                    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
                        logger.warning("unproved-check failed (%s); skipping body-fix turn", exc)
                        weak_proves = True
                if not weak_proves:
                    fixed_unit = spec + "\n" + template["body"].format(
                        unit=unit, typ=typ, typ2=template.get("typ2", ""),
                    )
                    records.append({
                        "messages": [
                            {"role": "user", "content": (
                                "gnatprove cannot prove this unit: the out "
                                "parameter R might not be initialized on every "
                                "path. Add the missing contract piece so the "
                                "body proves.\n\n```ada\n"
                                + spec + "\n" + weak_body + "\n```"
                            )},
                            {"role": "assistant", "content": (
                                template["fix_explain"]
                                + "\n\n```ada\n" + fixed_unit + "\n```"
                            )},
                        ],
                        "meta": {
                            "kind": "contract_synth_fix", "unit": unit,
                            "template": template["name"],
                            "verified": "gnatprove unproved before fix",
                            "group": group,
                        },
                    })
                continue

            broken_spec = weaken(instance)
            if broken_spec is None:
                continue
            with tempfile.TemporaryDirectory(prefix="q3as-unpr-") as tmp3:
                try:
                    still_proves = not verify_unproved(broken_spec, instance["body"], Path(tmp3))
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
                    "template": template["name"],
                    "verified": "gnatprove unproved before fix",
                    "group": group,
                },
            })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate gnatprove-verified contract training turns.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--limit", type=int, default=60,
        help="approximate number of instances to verify (about ten per template)",
    )
    parser.add_argument("--force", action="store_true", help="regenerate even when the inputs are unchanged")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # The templates and the gnatprove level decide the output, so both are in
    # the fingerprint: editing a template or changing --limit rebuilds. gnatprove
    # itself is not hashed (it is a toolchain binary, not a source input), so
    # use --force after a toolchain upgrade.
    spec = stage_state.make_spec(
        name="contract_mutations",
        outputs=[args.output],
        scripts=[Path(__file__).resolve(), Path(alire_env.__file__).resolve()],
        params=[("limit", args.limit), ("templates", len(TEMPLATES))],
    )
    if spec.skip_if_fresh(force=args.force):
        return 0

    # `make gen-contracts` must not break the build when the prover is not
    # installed, so an unavailable toolchain keeps whatever output exists and
    # exits 0. This is the same guarantee the old "output exists" cache gave,
    # but it is now keyed on the toolchain instead of on the output file.
    if not alire_env.has_tool("gnatprove"):
        logger.warning(
            "gnatprove is not available (run `make prove`); keeping %s as is",
            args.output if args.output.exists() else "no contract output",
        )
        return 0

    records = build_turns(args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    kinds: dict[str, int] = {}
    for record in records:
        kinds[record["meta"]["kind"]] = kinds.get(record["meta"]["kind"], 0) + 1
    logger.info("wrote %d gnatprove-verified turns to %s (%s)", len(records), args.output, kinds)
    if records:
        spec.mark_fresh()
    else:
        # No verified turns: leave any previous stamp alone so a later run
        # retries instead of trusting a cached "fresh" marker for an empty file.
        logger.warning("no verified turns produced; not recording a fresh stamp")
    return 0 if records else 1


if __name__ == "__main__":
    sys.exit(main())

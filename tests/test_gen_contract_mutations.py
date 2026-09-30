"""Tests for scripts/gen_contract_mutations.py.

The generator's contract with the dataset: every template family targets an
eval proof blocker (results v0.4.1: VC_OVERFLOW_CHECK, UNINITIALIZED,
VC_POSTCONDITION, VC_RAISE, DEPENDS_MISSING), every emitted contract is
gnatprove-verified before it trains, and the declaration extractor keeps the
whole declaration (the old first-semicolon cut truncated multi-parameter
declarations at their parameter list). The gnatprove invocation itself is
exercised by `make gen-contracts`; these tests pin the pure logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import gen_contract_mutations as gcm

# --------------------------------------------------------------------------- #
# Template families
# --------------------------------------------------------------------------- #


def test_template_families_are_unique_and_named() -> None:
    names = [t["name"] for t in gcm.TEMPLATES]
    assert len(names) == len(set(names))
    assert set(names) == {
        "guarded_increment",
        "guarded_subtract",
        "even_halve",
        "clamp_range",
        "init_default",
        "raise_guard",
        "loop_sum",
        "loop_max_array",
        "swap_depends",
        "global_counter",
    }


def test_every_template_instantiates_to_spark_package_pair() -> None:
    for template in gcm.TEMPLATES:
        instance = gcm.instantiate(template, 3)
        assert "pragma SPARK_Mode (On);" in instance["spec"]
        assert "package Ctr_3 is" in instance["spec"]
        assert "package body Ctr_3 is" in instance["body"]
        assert instance["unit"] in instance["spec"]
        assert instance["unit"] in instance["body"]
        # The declared unit must appear as a declaration in the spec, and the
        # body must implement the same unit.
        assert f"{instance['unit']} (" in instance["spec"]
        assert f"end {instance['unit']};" in instance["body"]


def test_extra_decls_land_in_spec_not_body() -> None:
    """A duplicated declaration in body and spec is a compile error."""
    for template in gcm.TEMPLATES:
        instance = gcm.instantiate(template, 0)
        for decl in template.get("extra_decls", {}).values():
            name = decl.strip().split()[1]
            assert name in instance["spec"]
            # The body redeclares nothing: it sees the spec's declarations.
            assert decl.strip() not in instance["body"]


def test_templates_carry_verification_metadata() -> None:
    for template in gcm.TEMPLATES:
        # Every family must state its verification gate: either a weaken rule
        # (wrong variants are checked to still fail) or a verified weak-body
        # fixture with its repair explanation.
        has_weaken = template.get("weaken") is not None
        has_body_fix = "fix_contract" in template and "fix_explain" in template
        assert has_weaken or has_body_fix, template["name"]


# --------------------------------------------------------------------------- #
# Declaration extraction (the first-semicolon truncation bug)
# --------------------------------------------------------------------------- #


def test_decl_with_contract_keeps_multi_parameter_declaration() -> None:
    spec = (
        "package Ctr is\n"
        "   procedure Fill (R : out Natural; N : in Natural)\n"
        "     with Pre => N > 0, Post => R = N;\n"
        "end Ctr;\n"
    )
    decl = gcm._decl_with_contract(spec, "Fill")
    assert decl == (
        "procedure Fill (R : out Natural; N : in Natural)"
        " with Pre => N > 0, Post => R = N;"
    )


def test_decl_with_contract_keeps_quantified_postcondition() -> None:
    spec = (
        "package Ctr is\n"
        "   function Max_Of (A : in Int_Array) return Integer\n"
        "     with Pre => A'Length > 0,"
        " Post => (for all K in A'Range => Max_Of'Result >= A (K));\n"
        "end Ctr;\n"
    )
    decl = gcm._decl_with_contract(spec, "Max_Of")
    assert decl.endswith(";")
    assert "for all K in A'Range" in decl
    # Semicolons inside the quantified expression must not cut the result.
    assert decl.count(";") == 1


def test_decl_with_contract_missing_unit_returns_name() -> None:
    assert gcm._decl_with_contract("package Ctr is\nend Ctr;\n", "Nope") == "Nope"


# --------------------------------------------------------------------------- #
# Weakening (wrong variants)
# --------------------------------------------------------------------------- #


def test_weaken_changes_every_rule_based_spec() -> None:
    for template in gcm.TEMPLATES:
        if template.get("weaken") is None:
            continue
        instance = gcm.instantiate(template, 5)
        broken = gcm.weaken(instance)
        assert broken is not None
        assert broken != instance["spec"]
        # The broken spec must stay syntactically plausible Ada: a contract
        # part is still present. (verify_unproved confirms the real syntax
        # and unproved status during generation.)


def test_weaken_handles_typ2_templates() -> None:
    template = next(t for t in gcm.TEMPLATES if t["name"] == "loop_sum")
    instance = gcm.instantiate(template, 2)
    broken = gcm.weaken(instance)
    assert broken is not None
    assert "Long_Long_Integer (N)" in broken


def test_weaken_returns_none_without_rule() -> None:
    template = next(t for t in gcm.TEMPLATES if t["name"] == "init_default")
    instance = gcm.instantiate(template, 1)
    assert gcm.weaken(instance) is None


# --------------------------------------------------------------------------- #
# Explanations (STE style)
# --------------------------------------------------------------------------- #


def test_explanations_are_template_specific_and_ste_clean() -> None:
    for template in gcm.TEMPLATES:
        instance = gcm.instantiate(template, 4)
        text = gcm._explain(instance["unit"], instance["typ"], instance["spec"], template["name"])
        assert len(text) > 60
        assert "--" not in text
        # STE: no hedging, no em-dashes, active voice.
        for banned in ("—", "maybe", "perhaps", "might want"):
            assert banned not in text


def test_init_default_explanation_covers_initialization() -> None:
    template = next(t for t in gcm.TEMPLATES if t["name"] == "init_default")
    instance = gcm.instantiate(template, 0)
    text = gcm._explain(instance["unit"], instance["typ"], instance["spec"], "init_default")
    assert "out parameter" in text
    assert "every path" in text


# --------------------------------------------------------------------------- #
# Batch parsing (how per-unit results are read back)
# --------------------------------------------------------------------------- #


def test_batch_flags_read_proved_lines() -> None:
    summary = (
        "in unit ctr_0, 2 subprograms and packages out of 2 analyzed\n"
        "  Ctr_0 at ctr_0.ads:3 flow analyzed (0 errors) and proved (0 checks)\n"
        "  Ctr_0.Fill at ctr_0.ads:5 flow analyzed (0 errors) and proved (1 checks)\n"
    )
    assert gcm._unit_proved_from_summary("Ctr_0", summary) is True


def test_batch_flags_read_unproved_lines() -> None:
    summary = (
        "in unit ctr_0, 2 subprograms and packages out of 2 analyzed\n"
        "  Ctr_0 at ctr_0.ads:3 flow analyzed (0 errors) and proved (0 checks)\n"
        "  Ctr_0.Fill at ctr_0.ads:5 flow analyzed (0 errors) and not proved, 0 checks out of 1 proved\n"
    )
    assert gcm._unit_proved_from_summary("Ctr_0", summary) is False


def test_batch_flags_unknown_when_unit_absent() -> None:
    assert gcm._unit_proved_from_summary("Ctr_9", "no matching lines") is None

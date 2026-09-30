"""Tests for the lab prompt/answer extraction and the contract-synth dedup exemption.

Training-material labs ship a ``prompt/`` skeleton tree beside an ``answer/``
solution tree; the extraction turns those twins into completion turns in the
shape of the evaluation set. The dedup exemption keeps the gnatprove-verified
contract curriculum from being thinned by the AST-structural cap, which exists
to balance *unbounded* extractor families, not a 700-turn verified one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "processing_scripts"))

import build_dataset as bd

# --------------------------------------------------------------------------- #
# Lab pair discovery
# --------------------------------------------------------------------------- #


def _make_lab(root: Path, lab: str, files: dict[str, tuple[str, str]]) -> None:
    """Write one prompt/answer lab twin under *root*."""
    for name, (prompt, answer) in files.items():
        for side, text in (("prompt", prompt), ("answer", answer)):
            path = root / lab / side / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")


PROMPT_ADB = """\
procedure Calc is
begin
   null;  -- TODO implement
end Calc;
"""
ANSWER_ADB = """\
procedure Calc is
   X : Integer := 41;
begin
   X := X + 1;
end Calc;
"""


def test_discover_lab_pairs_finds_prompt_answer_twins(tmp_path: Path) -> None:
    _make_lab(tmp_path, "my_lab", {"calc.adb": (PROMPT_ADB, ANSWER_ADB)})
    pairs = bd.discover_lab_pairs([tmp_path])
    assert len(pairs) == 1
    assert pairs[0]["lab"] == "my_lab"
    assert pairs[0]["name"] == "calc.adb"
    assert "TODO" in pairs[0]["prompt"]
    assert "X + 1" in pairs[0]["answer"]


def test_discover_lab_pairs_requires_answer_tree(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "half_lab" / "prompt"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "calc.adb").write_text(PROMPT_ADB, encoding="utf-8")
    assert bd.discover_lab_pairs([tmp_path]) == []


def test_lab_pair_turns_skip_identical_and_non_ada(tmp_path: Path) -> None:
    _make_lab(tmp_path, "no_op_lab", {"same.adb": (PROMPT_ADB, PROMPT_ADB)})
    _make_lab(tmp_path, "notes_lab", {"readme.txt": ("buy gnats", "sell gnats")})
    _make_lab(tmp_path, "real_lab", {"calc.adb": (PROMPT_ADB, ANSWER_ADB)})
    pairs = bd.discover_lab_pairs([tmp_path])
    assert len(pairs) == 3  # discovery is format-blind
    turns = bd.build_lab_pair_turns(pairs)
    # One completion turn plus one diff-derived explanation turn.
    assert len(turns) == 2
    completion = turns[0][1]
    user = completion["messages"][0]["content"]
    assistant = completion["messages"][1]["content"]
    assert user.startswith("Complete the Ada source file calc.adb.")
    assert "TODO" in user
    assert "X + 1" in assistant
    explanation = turns[1][1]
    assert "Explain what the solution adds" in explanation["messages"][0]["content"]
    # The explanation is derived from the diff, so it quotes the added code
    # (no subprogram boundary in this diff, so the first added line shows).
    assert "X : Integer := 41" in explanation["messages"][1]["content"]
    assert not explanation["messages"][1]["content"].lstrip().startswith("+")


def test_lab_turns_share_one_group_per_lab(tmp_path: Path) -> None:
    _make_lab(tmp_path, "wide_lab", {
        "calc.adb": (PROMPT_ADB, ANSWER_ADB),
        "calc.ads": (
            "procedure Calc;\n",
            "procedure Calc;\n",
        ),
    })
    pairs = bd.discover_lab_pairs([tmp_path])
    turns = bd.build_lab_pair_turns(pairs)
    groups = {g for g, _t in turns}
    assert len(groups) == 1  # spec, body, and explanations of one lab split together


def test_lab_explanation_names_added_subprograms(tmp_path: Path) -> None:
    answer = PROMPT_ADB.replace(
        "   null;  -- TODO implement",
        "   X : Integer := 1;\n   X := X + 1;",
    ) + "\nprocedure Extra_Step is\nbegin\n   null;\nend Extra_Step;\n"
    _make_lab(tmp_path, "named_lab", {"calc.adb": (PROMPT_ADB, answer)})
    turns = bd.build_lab_pair_turns(bd.discover_lab_pairs([tmp_path]))
    explanations = [
        t for _g, t in turns if "Explain what the solution adds" in t["messages"][0]["content"]
    ]
    assert len(explanations) == 1
    text = explanations[0]["messages"][1]["content"]
    assert "Extra_Step" in text
    assert "maybe" not in text and "—" not in text


# --------------------------------------------------------------------------- #
# Contract-synth dedup exemption
# --------------------------------------------------------------------------- #


def _synth_turn(unit: str) -> dict[str, list[dict[str, str]]]:
    return {
        "messages": [
            {"role": "user", "content": f"Write a SPARK contract for {unit}."},
            {"role": "assistant", "content": f"```ada\nprocedure {unit} (X : in out Natural) with Pre => X <= Natural'Last - 1;\n```"},
        ],
    }


def test_contract_synth_groups_are_not_structurally_capped() -> None:
    groups = [
        (f"contract-synth:{i}", _synth_turn(f"Unit_{i}"))
        for i in range(bd.AST_STRUCTURAL_CAP + 5)
    ]
    deduped, dropped, detail = bd.dedup_grouped(groups, ast_structural_cap=bd.AST_STRUCTURAL_CAP)
    # Every verified instance survives; only byte-identical records drop,
    # and these all differ in the unit name.
    assert dropped == 0
    assert len(deduped) == len(groups)
    assert detail["ast_structural_capped"] == 0


def test_contract_synth_exact_duplicates_still_drop() -> None:
    groups = [("contract-synth:1", _synth_turn("Same")), ("contract-synth:2", _synth_turn("Same"))]
    deduped, dropped, _detail = bd.dedup_grouped(groups, ast_structural_cap=bd.AST_STRUCTURAL_CAP)
    assert dropped == 1
    assert len(deduped) == 1


def test_regular_code_groups_still_capped() -> None:
    def plain(unit: str) -> dict[str, list[dict[str, str]]]:
        return {
            "messages": [
                {"role": "user", "content": f"Implement {unit}."},
                {"role": "assistant", "content": f"```ada\nprocedure {unit} (X : in out Natural) is\nbegin\n   X := X + 1;\nend {unit};\n```"},
            ],
        }

    groups = [(f"ast:{i}", plain(f"Unit_{i}")) for i in range(bd.AST_STRUCTURAL_CAP + 5)]
    _deduped, dropped, detail = bd.dedup_grouped(groups, ast_structural_cap=bd.AST_STRUCTURAL_CAP)
    assert detail["ast_structural_capped"] == 5
    assert dropped == 5

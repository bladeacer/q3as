"""Tests for eval/generate.py: prompt construction and reply parsing.

The generation pipeline feeds the model the task plus the full base project
tree, then parses the reply into project-file overlays. These tests pin the
prompt contract (sources shown, target called out, non-source files
protected) and the parser's fallbacks (multi-file entries, single fenced
block, raw text) plus its path-safety rules.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import generate as gen

# --------------------------------------------------------------------------- #
# build_user_prompt
# --------------------------------------------------------------------------- #


def make_sample(**overrides):
    sample = {
        "prompt": "Please make Absolute_Value provable.",
        "location": {"path": "src/integer_utils.ads", "subprogram_name": "Absolute_Value"},
        "sources_text": {
            Path("main.gpr"): "project Main is end Main;",
            Path("src/integer_utils.ads"): "package Integer_Utils is end Integer_Utils;",
        },
    }
    sample.update(overrides)
    return sample


class TestBuildUserPrompt:
    def test_includes_task_and_all_sources(self):
        prompt = gen.build_user_prompt(make_sample())
        assert "Please make Absolute_Value provable." in prompt
        assert "File: main.gpr" in prompt
        assert "File: src/integer_utils.ads" in prompt
        assert "project Main is end Main;" in prompt

    def test_calls_out_target_file(self):
        prompt = gen.build_user_prompt(make_sample())
        assert "Always include the file src/integer_utils.ads in the reply." in prompt

    def test_restricts_changes_to_target_directory(self):
        prompt = gen.build_user_prompt(make_sample())
        assert "Only files in src may change." in prompt

    def test_target_in_project_root_wording(self):
        sample = make_sample(location={"path": "main.adb"})
        prompt = gen.build_user_prompt(sample)
        assert "Only files in the project root may change." in prompt

    def test_missing_location_falls_back_to_generated_adb(self):
        sample = make_sample(location={})
        prompt = gen.build_user_prompt(sample)
        assert "Always include the file generated.adb in the reply." in prompt

    def test_reply_format_instruction_present(self):
        prompt = gen.build_user_prompt(make_sample())
        assert 'write a line "File: <path>"' in prompt
        assert "```ada" in prompt

    def test_sources_sorted_deterministically(self):
        sample = make_sample(sources_text={
            Path("b.adb"): "b",
            Path("a.ads"): "a",
        })
        prompt = gen.build_user_prompt(sample)
        assert prompt.index("File: a.ads") < prompt.index("File: b.adb")


# --------------------------------------------------------------------------- #
# parse_generated_files
# --------------------------------------------------------------------------- #


MULTI_FILE_REPLY = """\
File: src/foo.ads
```ada
package Foo is
   procedure P (X : Integer)
   with Pre => X > 0;
end Foo;
```

File: src/foo.adb
```ada
package body Foo is
   procedure P (X : Integer) is
   begin
      null;
   end P;
end Foo;
```"""


class TestParseGeneratedFiles:
    def test_multi_file_reply(self):
        files = gen.parse_generated_files(MULTI_FILE_REPLY, Path("src/foo.ads"))
        assert set(files) == {Path("src/foo.ads"), Path("src/foo.adb")}
        assert "Pre => X > 0" in files[Path("src/foo.ads")]

    def test_entries_outside_target_dir_dropped(self):
        files = gen.parse_generated_files(MULTI_FILE_REPLY, Path("src/other.ads"))
        # foo.ads/foo.adb are not next to other.ads... they are (src/), so
        # build a reply whose only entries live elsewhere.
        reply = MULTI_FILE_REPLY + "\n\nFile: main.gpr\n```ada\nproject M is end M;\n```"
        files = gen.parse_generated_files(reply, Path("src/foo.ads"))
        assert Path("main.gpr") not in files

    def test_single_fenced_block_falls_back_to_target(self):
        reply = "Here is the fix:\n```ada\npackage Foo is end Foo;\n```"
        files = gen.parse_generated_files(reply, Path("src/foo.ads"))
        assert set(files) == {Path("src/foo.ads")}
        assert "package Foo is end Foo;" in files[Path("src/foo.ads")]

    def test_raw_text_falls_back_to_target(self):
        reply = "The warning says R might be uninitialized."
        files = gen.parse_generated_files(reply, Path("src/foo.ads"))
        assert set(files) == {Path("src/foo.ads")}
        assert "uninitialized" in files[Path("src/foo.ads")]

    def test_trailing_punctuation_stripped_from_path(self):
        reply = "File: src/foo.ads:\n```ada\npackage Foo is end Foo;\n```"
        files = gen.parse_generated_files(reply, Path("src/foo.ads"))
        assert Path("src/foo.ads") in files

    def test_absolute_path_rejected(self):
        reply = "File: /etc/passwd\n```ada\nx\n```"
        files = gen.parse_generated_files(reply, Path("src/foo.ads"))
        assert Path("/etc/passwd") not in files
        # The reply still has a fenced block, so it lands at the target path.
        assert Path("src/foo.ads") in files

    def test_parent_traversal_rejected(self):
        reply = "File: ../evil.ads\n```ada\nx\n```"
        files = gen.parse_generated_files(reply, Path("src/foo.ads"))
        assert Path("../evil.ads") not in files

    def test_non_source_suffix_rejected(self):
        reply = "File: src/notes.txt\n```ada\nx\n```"
        files = gen.parse_generated_files(reply, Path("src/foo.ads"))
        assert Path("src/notes.txt") not in files

    def test_empty_reply_yields_empty_map(self):
        assert gen.parse_generated_files("", Path("src/foo.ads")) == {}
        assert gen.parse_generated_files("   \n  ", Path("src/foo.ads")) == {}


# --------------------------------------------------------------------------- #
# plan_batches
# --------------------------------------------------------------------------- #


class TestPlanBatches:
    def test_empty_input(self):
        assert gen.plan_batches([], 512, 0, 100_000) == []

    def test_no_budget_falls_back_to_singletons(self):
        # budget <= 0 means "could not measure VRAM": one prompt per call.
        assert gen.plan_batches([10, 20, 30], 5, 0, 0) == [[0], [1], [2]]

    def test_explicit_cap_wins_over_budget(self):
        # Budget would allow far more, but max_batch_size clamps it.
        assert gen.plan_batches([100, 100, 100, 100], 10, 2, 10_000_000) == [[0, 1], [2, 3]]

    def test_covered_indices_and_prompt_order_preserved(self):
        # Lengths deliberately unsorted: the plan must still return every
        # index exactly once so results stay in prompt order.
        lengths = [900, 10, 500, 20]
        batches = gen.plan_batches(lengths, 5, 0, 10_000_000)
        flat = [i for b in batches for i in b]
        assert sorted(flat) == list(range(len(lengths)))
        assert len(flat) == len(set(flat))

    def test_batches_respect_token_budget(self):
        # 3 prompts of 1000 tokens + 100 new tokens = 3300 <= 4000 budget.
        batches = gen.plan_batches([1000, 1000, 1000], 100, 0, 4000)
        assert [len(b) for b in batches] == [3]

    def test_long_prompt_splits_the_budget(self):
        # 2 prompts of 2000+100 = 4200 > 4000, so only one fits at a time.
        batches = gen.plan_batches([2000, 2000], 100, 0, 4000)
        assert [len(b) for b in batches] == [1, 1]

    def test_shortest_first_to_limit_padding(self):
        # Sorted by length, so the long prompt never pads a short one.
        batches = gen.plan_batches([2000, 10, 10], 100, 0, 2200)
        assert [len(b) for b in batches] == [2, 1]
        assert batches[0] == [1, 2]
        assert batches[1] == [0]


class TestKvBytesPerToken:
    def test_grouped_query_attention_shrinks_the_cache(self):
        cfg = types.SimpleNamespace(
            num_hidden_layers=36,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
        )
        # 36 layers * 2 (K,V) * 8 kv heads * 128 head_dim * 2 bytes.
        assert gen._kv_bytes_per_token(cfg) == 36 * 2 * 8 * 128 * 2

    def test_head_dim_derived_when_absent(self):
        cfg = types.SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=2,
        )
        assert gen._kv_bytes_per_token(cfg) == 2 * 2 * 2 * 64 * 2

    def test_incomplete_config_is_unknown(self):
        assert gen._kv_bytes_per_token(types.SimpleNamespace()) == 0
        assert gen._kv_bytes_per_token(None) == 0


class TestIsOutOfMemory:
    def test_recognises_cuda_oom(self):
        assert gen._is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate"))

    def test_recognises_by_class_name(self):
        class OutOfMemoryError(Exception):
            pass

        assert gen._is_out_of_memory(OutOfMemoryError("no detail given"))

    def test_ordinary_error_is_not_oom(self):
        assert not gen._is_out_of_memory(ValueError("bad prompt"))


# --------------------------------------------------------------------------- #
# _safe_source_path
# --------------------------------------------------------------------------- #


class TestSafeSourcePath:
    @pytest.mark.parametrize("raw,expected", [
        ("src/foo.ads", "src/foo.ads"),
        (" main.gpr ", "main.gpr"),
        ("`src/foo.adb`", "src/foo.adb"),
    ])
    def test_accepts(self, raw, expected):
        assert gen._safe_source_path(raw) == Path(expected)

    @pytest.mark.parametrize("raw", [
        "/abs/foo.ads",
        "../foo.ads",
        "src/../../foo.ads",
        "notes.txt",
        "src/foo.adbx",
    ])
    def test_rejects(self, raw):
        assert gen._safe_source_path(raw) is None


# --------------------------------------------------------------------------- #
# set_generation_seed (transformers stubbed out to keep the test fast)
# --------------------------------------------------------------------------- #


class TestSetGenerationSeed:
    def test_seeds_python_random_and_calls_hf(self, monkeypatch):
        stub = types.ModuleType("transformers")
        calls = []
        stub.set_seed = lambda s: calls.append(s)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", stub)
        gen.set_generation_seed(42)
        assert calls == [42]

    def test_is_deterministic_call_order(self, monkeypatch):
        stub = types.ModuleType("transformers")
        calls = []
        stub.set_seed = lambda s: calls.append(s)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", stub)
        gen.set_generation_seed(7)
        gen.set_generation_seed(7)
        assert calls == [7, 7]


# --------------------------------------------------------------------------- #
# body_sibling: the spec/body routing repair
# --------------------------------------------------------------------------- #
#
# Every ada-eval sample targets a spec (.ads) file. Before the repair, a reply
# carrying the unit's body was written into that spec, producing
# "package body ... end Foo;" inside a package spec, which GNAT rejects: the
# sample failed BUILD for a routing mistake rather than for wrong Ada. On the
# v0.7.0 run six of nineteen fine-tuned replies did exactly this and all six
# failed to build, while the base model (which used the requested
# "File: <path>" format on every sample) never triggered it.

BODY_REPLY = """\
```ada
package body Foo is

   function Count (Str : String; Char : Character) return Natural is
      Result : Natural := 0;
   begin
      return Result;
   end Count;

end Foo;
```"""

SPEC_REPLY = """\
```ada
package Foo is

   function Count (Str : String; Char : Character) return Natural;

end Foo;
```"""

PROJECT_WITH_BODY = frozenset({Path("src/foo.ads"), Path("src/foo.adb"), Path("main.gpr")})


class TestBodySibling:
    def test_spec_with_a_body_in_the_tree(self):
        assert gen.body_sibling(Path("src/foo.ads"), PROJECT_WITH_BODY) == Path("src/foo.adb")

    def test_body_target_is_left_alone(self):
        # Already the right file; routing again would be a no-op at best.
        assert gen.body_sibling(Path("src/foo.adb"), PROJECT_WITH_BODY) is None

    def test_spec_without_a_body_in_the_tree(self):
        # Never invent a file the project did not have.
        assert gen.body_sibling(Path("src/foo.ads"), frozenset({Path("src/foo.ads")})) is None

    def test_unknown_tree_gets_no_repair(self):
        assert gen.body_sibling(Path("src/foo.ads"), None) is None

    def test_non_ada_target(self):
        assert gen.body_sibling(Path("main.gpr"), PROJECT_WITH_BODY) is None


class TestParseGeneratedFilesRouting:
    def test_body_reply_goes_to_the_body(self):
        files = gen.parse_generated_files(BODY_REPLY, Path("src/foo.ads"), PROJECT_WITH_BODY)
        assert set(files) == {Path("src/foo.adb")}
        assert "package body Foo is" in files[Path("src/foo.adb")]

    def test_spec_reply_stays_in_the_spec(self):
        files = gen.parse_generated_files(SPEC_REPLY, Path("src/foo.ads"), PROJECT_WITH_BODY)
        assert set(files) == {Path("src/foo.ads")}

    def test_no_repair_without_the_project_tree(self):
        # Passing project_files is opt-in: the old behaviour is intact.
        files = gen.parse_generated_files(BODY_REPLY, Path("src/foo.ads"))
        assert set(files) == {Path("src/foo.ads")}

    def test_no_repair_when_the_body_is_absent(self):
        tree = frozenset({Path("src/foo.ads")})
        files = gen.parse_generated_files(BODY_REPLY, Path("src/foo.ads"), tree)
        assert set(files) == {Path("src/foo.ads")}

    def test_explicit_file_entries_win_over_the_repair(self):
        # The model named the file itself; its choice is authoritative even
        # when the block happens to open a package body.
        reply = "File: src/foo.ads\n```ada\npackage body Foo is\nend Foo;\n```"
        files = gen.parse_generated_files(reply, Path("src/foo.ads"), PROJECT_WITH_BODY)
        assert set(files) == {Path("src/foo.ads")}

    def test_body_reply_at_a_body_target_is_untouched(self):
        files = gen.parse_generated_files(BODY_REPLY, Path("src/foo.adb"), PROJECT_WITH_BODY)
        assert set(files) == {Path("src/foo.adb")}

    def test_leading_comment_before_the_body_still_routes(self):
        # gnatdoc-style preambles are common in the training corpus.
        reply = "```ada\n--  Counter for Foo.\npackage body Foo is\nend Foo;\n```"
        files = gen.parse_generated_files(reply, Path("src/foo.ads"), PROJECT_WITH_BODY)
        assert set(files) == {Path("src/foo.adb")}

    def test_bare_package_body_line_is_routed(self):
        reply = "```ada\npackage body Foo is end Foo;\n```"
        files = gen.parse_generated_files(reply, Path("src/foo.ads"), PROJECT_WITH_BODY)
        assert set(files) == {Path("src/foo.adb")}

    def test_project_dump_keeps_its_spec_first(self):
        # A whole-project reply opens with the spec; routing it to the body
        # would lose the spec the sample is scored on.
        reply = (
            "```ada\npackage Foo is\n"
            "   function Count (Str : String) return Natural;\n"
            "end Foo;\n\npackage body Foo is\nend Foo;\n```"
        )
        files = gen.parse_generated_files(reply, Path("src/foo.ads"), PROJECT_WITH_BODY)
        assert set(files) == {Path("src/foo.ads")}

    def test_generic_instantiation_is_not_mistaken_for_a_body(self):
        # "package body" must be the unit's own clause, not any mention of it.
        reply = "```ada\nfunction Make return Integer_Utils.Placeholder;\n```"
        files = gen.parse_generated_files(reply, Path("src/foo.ads"), PROJECT_WITH_BODY)
        assert set(files) == {Path("src/foo.ads")}


# --------------------------------------------------------------------------- #
# reply_format
# --------------------------------------------------------------------------- #


class TestReplyFormat:
    def test_file_blocks_is_the_requested_contract(self):
        assert gen.reply_format(MULTI_FILE_REPLY) == "file_blocks"

    def test_single_fenced_block(self):
        assert gen.reply_format(SPEC_REPLY) == "fenced_block"

    def test_prose_without_a_fence(self):
        assert gen.reply_format("The warning says R might be uninitialized.") == "raw_text"

    def test_a_rejected_path_does_not_count_as_the_contract(self):
        # A "File:" line naming a non-source file is dropped by the parser,
        # so the reply falls back and must not be reported as compliant.
        reply = "File: notes.txt\n```ada\npackage Foo is end Foo;\n```"
        assert gen.reply_format(reply) == "fenced_block"

    def test_backticked_path_still_counts_as_the_contract(self):
        # _safe_source_path strips backticks, so this reply is usable as-is
        # and the harness never falls back.
        reply = "File: `src/foo.ads`\n```ada\npackage Foo is end Foo;\n```"
        assert gen.reply_format(reply) == "file_blocks"

    def test_one_valid_entry_among_rejected_ones_counts(self):
        reply = (
            "File: notes.txt\n```ada\nx\n```\n\n"
            "File: src/foo.ads\n```ada\npackage Foo is end Foo;\n```"
        )
        assert gen.reply_format(reply) == "file_blocks"


# --------------------------------------------------------------------------- #
# run_generation_for_model: the overlay, the echo skip and the sidecar
# --------------------------------------------------------------------------- #
#
# This function builds the project ada-eval then compiles, tests and proves.
# Two of its decisions are invisible in the scores unless they are pinned: a
# reply that only echoes the base file leaves the base project in place (so
# the sample is graded on the base tree, not on model output), and the reply
# shape is recorded in a sidecar because it cannot go in the ada-eval sample
# schema.

SPEC_TEXT = """\
package Integer_Utils is

   function Absolute_Value (Value : Integer) return Integer
     with Post => Absolute_Value'Result >= 0;

end Integer_Utils;
"""

BODY_TEXT = """\
package body Integer_Utils is

   function Absolute_Value (Value : Integer) return Integer is
   begin
      if Value < 0 then
         return -Value;
      end if;
      return Value;
   end Absolute_Value;

end Integer_Utils;
"""


def _write_sample_tree(root: Path, *, body: str = BODY_TEXT, spec: str = SPEC_TEXT) -> Path:
    """An expanded ada-eval sample directory (base/, tests/, prompt.md)."""
    sample = root / "spark_learn" / "absolute_value"
    (sample / "base" / "src").mkdir(parents=True)
    (sample / "base" / "main.gpr").write_text("project Main is end Main;\n", encoding="utf-8")
    (sample / "base" / "src" / "integer_utils.ads").write_text(spec, encoding="utf-8")
    (sample / "base" / "src" / "integer_utils.adb").write_text(body, encoding="utf-8")
    (sample / "tests").mkdir()
    (sample / "tests" / "tests.adb").write_text("-- tests\n", encoding="utf-8")
    (sample / "prompt.md").write_text("Make Absolute_Value provable.", encoding="utf-8")
    (sample / "other.json").write_text(
        json.dumps({
            "location": {"path": "src/integer_utils.ads", "subprogram_name": "Absolute_Value"},
            "canonical_evaluation_results": [],
        }),
        encoding="utf-8",
    )
    return sample


@pytest.fixture
def overlay_env(tmp_path, monkeypatch):
    """Stub the ada-eval sample types and the model call.

    run_generation_for_model imports GENERATED_SAMPLE_TYPES lazily and calls
    generate_batch; both are replaced so the overlay can be exercised without
    a model or an ada-eval install.
    """
    class _Sample:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def model_dump_json(self, exclude_defaults=True):
            return json.dumps({
                "name": self.kwargs["name"],
                "generated_solution": {
                    str(k): base64.b64encode(v).decode()
                    for k, v in self.kwargs["generated_solution"].items()
                },
                "location": self.kwargs["location"],
            })

    class ExitStatus:
        SUCCESS = "success"

    class GenerationStats:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    types_mod = types.ModuleType("ada_eval.datasets.types")
    types_mod.GENERATED_SAMPLE_TYPES = {"spark": _Sample, "ada": _Sample}
    samples_mod = types.ModuleType("ada_eval.datasets.types.samples")
    samples_mod.ExitStatus = ExitStatus
    samples_mod.GenerationStats = GenerationStats
    for name in ("ada_eval", "ada_eval.datasets"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "ada_eval.datasets.types", types_mod)
    monkeypatch.setitem(sys.modules, "ada_eval.datasets.types.samples", samples_mod)

    monkeypatch.setattr(gen, "GENERATED_DIR", tmp_path / "generated_solutions")
    return tmp_path


def _run_overlay(tmp_path, monkeypatch, expanded, reply, max_samples=5):
    """Run one sample through the real overlay with a fixed reply."""
    monkeypatch.setattr(gen, "generate_batch", lambda *a, **k: [reply] * max_samples)
    args = argparse.Namespace(
        max_samples=max_samples,
        max_new_tokens=64,
        temperature=0.0,
        enable_thinking=False,
        system_prompt="",
        max_prompt_chars=12000,
        batch_size=1,
    )
    return gen.run_generation_for_model(None, None, "fine_tuned", [("spark_learn", expanded)], args)


def _packed(tmp_path, label="fine_tuned"):
    path = tmp_path / "generated_solutions" / label / "spark_spark_learn.jsonl"
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    return {
        Path(k): base64.b64decode(v).decode()
        for k, v in record["generated_solution"].items()
    }


class TestOverlay:
    def test_base_tree_is_the_starting_point(self, tmp_path, monkeypatch, overlay_env):
        sample = _write_sample_tree(tmp_path)
        _run_overlay(tmp_path, monkeypatch, sample.parent, f"```ada\n{SPEC_TEXT}```")
        files = _packed(tmp_path)
        # main.gpr is never a target, so it must survive untouched.
        assert files[Path("main.gpr")] == "project Main is end Main;\n"

    def test_verbatim_echo_leaves_the_base_project(self, tmp_path, monkeypatch, overlay_env):
        sample = _write_sample_tree(tmp_path)
        _run_overlay(tmp_path, monkeypatch, sample.parent, f"```ada\n{SPEC_TEXT}```")
        files = _packed(tmp_path)
        assert files[Path("src/integer_utils.ads")] == SPEC_TEXT

    def test_body_reply_routes_to_the_sibling_body(self, tmp_path, monkeypatch, overlay_env):
        sample = _write_sample_tree(tmp_path)
        _run_overlay(tmp_path, monkeypatch, sample.parent, f"```ada\n{BODY_TEXT}```")
        files = _packed(tmp_path)
        # The spec keeps its base content; the body carries the model's code.
        assert files[Path("src/integer_utils.ads")] == SPEC_TEXT
        assert files[Path("src/integer_utils.adb")] == BODY_TEXT

    def test_packed_file_name_follows_the_ada_eval_rule(self, tmp_path, monkeypatch, overlay_env):
        sample = _write_sample_tree(tmp_path)
        _run_overlay(tmp_path, monkeypatch, sample.parent, "```ada\nx\n```")
        out = tmp_path / "generated_solutions" / "fine_tuned"
        assert (out / "spark_spark_learn.jsonl").exists()

    def test_empty_reply_is_skipped(self, tmp_path, monkeypatch, overlay_env):
        sample = _write_sample_tree(tmp_path)
        result = _run_overlay(tmp_path, monkeypatch, sample.parent, "")
        assert result["total"] == 0

    def test_summary_counts_and_output_dir(self, tmp_path, monkeypatch, overlay_env):
        sample = _write_sample_tree(tmp_path)
        result = _run_overlay(tmp_path, monkeypatch, sample.parent, "```ada\nx\n```")
        assert result["by_dataset"] == {"spark_learn": 1}
        assert result["total"] == 1
        assert result["output_dir"].endswith("fine_tuned")


class TestGenerationMetaSidecar:
    def _meta(self, tmp_path, label="fine_tuned"):
        path = tmp_path / "generated_solutions" / label / "generation_meta.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_records_the_requested_format(self, tmp_path, monkeypatch, overlay_env):
        sample = _write_sample_tree(tmp_path)
        reply = "File: src/integer_utils.ads\n```ada\n" + SPEC_TEXT + "```"
        _run_overlay(tmp_path, monkeypatch, sample.parent, reply)
        assert self._meta(tmp_path)["absolute_value"]["reply_format"] == "file_blocks"

    def test_records_a_fallback_format(self, tmp_path, monkeypatch, overlay_env):
        sample = _write_sample_tree(tmp_path)
        _run_overlay(tmp_path, monkeypatch, sample.parent, f"```ada\n{SPEC_TEXT}```")
        assert self._meta(tmp_path)["absolute_value"]["reply_format"] == "fenced_block"

    def test_unchanged_project_is_counted_as_untouched(self, tmp_path, monkeypatch, overlay_env):
        # The echo leaves the base tree in place, so BUILD measures the base
        # project. That has to be visible next to the score.
        sample = _write_sample_tree(tmp_path)
        _run_overlay(tmp_path, monkeypatch, sample.parent, f"```ada\n{SPEC_TEXT}```")
        assert self._meta(tmp_path)["absolute_value"]["changed_files"] == 0

    def test_a_real_edit_counts_as_changed(self, tmp_path, monkeypatch, overlay_env):
        # A reply identical to the base file is an echo (0); one that differs
        # is an edit (1).
        sample = _write_sample_tree(tmp_path)
        edited = BODY_TEXT.replace("return Value;", "return Natural (Value);")
        _run_overlay(tmp_path, monkeypatch, sample.parent, f"```ada\n{edited}```")
        assert self._meta(tmp_path)["absolute_value"]["changed_files"] == 1

    def test_routing_is_recorded(self, tmp_path, monkeypatch, overlay_env):
        # A repaired sample must be identifiable as repaired, so a build
        # credit can be read for what it was.
        sample = _write_sample_tree(tmp_path)
        _run_overlay(tmp_path, monkeypatch, sample.parent, f"```ada\n{BODY_TEXT}```")
        assert self._meta(tmp_path)["absolute_value"]["routed_body"] == [
            "src/integer_utils.adb"
        ]

    def test_no_routing_recorded_for_a_spec_reply(self, tmp_path, monkeypatch, overlay_env):
        sample = _write_sample_tree(tmp_path)
        edited = SPEC_TEXT.replace("Post =>", "Pre =>")
        _run_overlay(tmp_path, monkeypatch, sample.parent, f"```ada\n{edited}```")
        assert self._meta(tmp_path)["absolute_value"]["routed_body"] == []

    def test_sidecar_is_valid_json_for_every_sample(self, tmp_path, monkeypatch, overlay_env):
        sample = _write_sample_tree(tmp_path)
        _run_overlay(tmp_path, monkeypatch, sample.parent, "```ada\nx\n```")
        meta = self._meta(tmp_path)
        assert set(meta) == {"absolute_value"}
        assert set(meta["absolute_value"]) == {
            "reply_format", "changed_files", "routed_body",
        }

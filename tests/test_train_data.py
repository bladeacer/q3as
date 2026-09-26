"""Tests for the split-loading path in training/train_unsloth.py.

The trainer streams each split from JSONL into a cached Arrow table instead of
holding parsed records, templated strings, and the table in memory at once.
These tests pin the two things that makes that safe: the streaming loader and
the in-memory fallback must agree on which records are trainable, and the
fingerprint that keys the Arrow cache must change whenever anything that
affects the produced text changes.

unsloth is stubbed: importing the real one costs ~1.3 GB of host RAM and a
patched transformers, neither of which a unit test should need.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

if "unsloth" not in sys.modules:
    sys.modules["unsloth"] = types.ModuleType("unsloth")

import train_unsloth as tr


class FakeTokenizer:
    """Minimal stand-in: records the template it was given."""

    def __init__(self, chat_template: str = "TPL:{{ text }}", name: str = "fake/tok",
                 vocab: dict | None = None):
        self.chat_template = chat_template
        self.name_or_path = name
        self._vocab = vocab if vocab is not None else {f"tok{i}": i for i in range(8)}

    def get_vocab(self):
        return self._vocab

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is False
        return self.chat_template.replace(
            "{{ text }}", "|".join(m["content"] for m in messages)
        )

    def __call__(self, text, truncation=False, max_length=None, **kwargs):
        # One id per character is enough to test truncation and ordering.
        ids = [ord(c) % 1000 for c in text]
        if truncation and max_length:
            ids = ids[:max_length]
        return types.SimpleNamespace(input_ids=ids)


def record(user: str = "q", assistant: str = "a") -> dict:
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]}


def write_jsonl(path: Path, rows: list) -> Path:
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------------- #
# usable_record
# --------------------------------------------------------------------------- #


class TestUsableRecord:
    def test_accepts_a_well_formed_record(self):
        row = record()
        assert tr.usable_record(row, 1) is row

    def test_accepts_a_multiturn_record(self):
        row = {"messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]}
        assert tr.usable_record(row, 1) is row

    @pytest.mark.parametrize("row", [
        {},
        {"messages": []},
        {"messages": [{"role": "user", "content": "only one"}]},
        {"messages": "not a list"},
        "not a dict",
        None,
        42,
    ])
    def test_rejects_malformed(self, row):
        assert tr.usable_record(row, 1) is None

    def test_rejects_an_empty_assistant_reply(self):
        # Training on these teaches the model to emit EOS immediately.
        assert tr.usable_record(record(assistant=""), 1) is None
        assert tr.usable_record(record(assistant="   \n "), 1) is None

    def test_allows_an_empty_user_turn(self):
        # Only the assistant reply must be non-empty.
        row = record(user="", assistant="answer")
        assert tr.usable_record(row, 1) is row


# --------------------------------------------------------------------------- #
# ChatTemplateJsonl
# --------------------------------------------------------------------------- #


class TestChatTemplateJsonl:
    def test_renders_each_record_with_the_chat_template(self, tmp_path):
        path = write_jsonl(tmp_path / "s.jsonl", [record("q1", "a1"), record("q2", "a2")])
        out = list(tr.ChatTemplateJsonl(path, FakeTokenizer())())
        assert out == [{"text": "TPL:q1|a1"}, {"text": "TPL:q2|a2"}]

    def test_skips_the_same_records_as_the_in_memory_loader(self, tmp_path):
        rows = [
            record("keep1", "a1"),
            record("drop-empty", ""),
            {"messages": [{"role": "user", "content": "lonely"}]},
            record("keep2", "a2"),
        ]
        path = write_jsonl(tmp_path / "s.jsonl", rows)
        path.write_text(
            path.read_text(encoding="utf-8") + "{not json}\n\n", encoding="utf-8"
        )
        tok = FakeTokenizer()
        streamed = [e["text"] for e in tr.ChatTemplateJsonl(path, tok)()]
        in_memory = [
            tok.apply_chat_template(r["messages"], tokenize=False,
                                    add_generation_prompt=False)
            for r in tr.load_jsonl_records(path)
        ]
        assert streamed == in_memory == ["TPL:keep1|a1", "TPL:keep2|a2"]


# --------------------------------------------------------------------------- #
# text_fingerprint
# --------------------------------------------------------------------------- #


class TestTextFingerprint:
    @pytest.fixture(autouse=True)
    def _reset_digest_memo(self):
        tr._TOKENIZER_DIGEST = None
        yield
        tr._TOKENIZER_DIGEST = None

    def test_is_stable_for_identical_inputs(self, tmp_path):
        path = write_jsonl(tmp_path / "s.jsonl", [record()])
        tok = FakeTokenizer()
        assert tr.text_fingerprint(path, tok) == tr.text_fingerprint(path, tok)

    def test_changes_when_the_split_content_changes(self, tmp_path):
        path = write_jsonl(tmp_path / "s.jsonl", [record("q1", "a1")])
        tok = FakeTokenizer()
        before = tr.text_fingerprint(path, tok)
        write_jsonl(path, [record("q1", "a1"), record("q2", "a2")])
        assert tr.text_fingerprint(path, tok) != before

    def test_changes_when_a_record_is_edited_in_place(self, tmp_path):
        # Same size, same line count: only the content hash can catch this.
        path = write_jsonl(tmp_path / "s.jsonl", [record("aaaa", "a1")])
        tok = FakeTokenizer()
        before = tr.text_fingerprint(path, tok)
        write_jsonl(path, [record("bbbb", "a1")])
        assert tr.text_fingerprint(path, tok) != before

    def test_changes_when_the_chat_template_changes(self, tmp_path):
        path = write_jsonl(tmp_path / "s.jsonl", [record()])
        before = tr.text_fingerprint(path, FakeTokenizer("A:{{ text }}"))
        after = tr.text_fingerprint(path, FakeTokenizer("B:{{ text }}"))
        assert before != after

    def test_ignores_the_tokenizer_name(self, tmp_path):
        # The key follows the vocabulary, not the model name: the produced ids
        # are identical either way, so a rename must not needlessly invalidate
        # the cache (and a same-name republish with a new vocab must).
        path = write_jsonl(tmp_path / "s.jsonl", [record()])
        tr._TOKENIZER_DIGEST = None
        before = tr.text_fingerprint(path, FakeTokenizer(name="org/one"))
        tr._TOKENIZER_DIGEST = None
        after = tr.text_fingerprint(path, FakeTokenizer(name="org/two"))
        assert before == after

    def test_changes_when_the_vocabulary_changes(self, tmp_path):
        # Same name, different vocab: the ids are then wrong, not just stale.
        path = write_jsonl(tmp_path / "s.jsonl", [record()])
        tr._TOKENIZER_DIGEST = None
        before = tr.text_fingerprint(path, FakeTokenizer())
        tr._TOKENIZER_DIGEST = None
        after = tr.text_fingerprint(path, FakeTokenizer(vocab={"only": 1}))
        assert before != after

    def test_separates_the_text_and_tokenized_tables(self, tmp_path):
        # They are different columns, so one cache key must not serve both.
        path = write_jsonl(tmp_path / "s.jsonl", [record()])
        tok = FakeTokenizer()
        tr._TOKENIZER_DIGEST = None
        text_fp = tr.text_fingerprint(path, tok, 0, kind="text")
        ids_fp = tr.text_fingerprint(path, tok, 1024, kind="ids")
        assert text_fp != ids_fp

    def test_tokenized_fingerprint_depends_on_max_seq_length(self, tmp_path):
        # Truncation length changes the ids, so it has to be in the key.
        path = write_jsonl(tmp_path / "s.jsonl", [record()])
        tok = FakeTokenizer()
        tr._TOKENIZER_DIGEST = None
        assert (tr.text_fingerprint(path, tok, 512, kind="ids")
                != tr.text_fingerprint(path, tok, 1024, kind="ids"))

    def test_changes_when_the_pipeline_version_bumps(self, tmp_path, monkeypatch):
        # Guards against a stale cache outliving a change to how text is built.
        path = write_jsonl(tmp_path / "s.jsonl", [record()])
        tok = FakeTokenizer()
        before = tr.text_fingerprint(path, tok)
        monkeypatch.setattr(tr, "TEXT_PIPELINE_VERSION", tr.TEXT_PIPELINE_VERSION + 1)
        assert tr.text_fingerprint(path, tok) != before

    def test_handles_a_missing_file_without_raising(self, tmp_path):
        # Recorded as absent so a later-created split cannot match an old cache.
        fingerprint = tr.text_fingerprint(tmp_path / "gone.jsonl", FakeTokenizer())
        assert isinstance(fingerprint, str) and fingerprint


# --------------------------------------------------------------------------- #
# split_records (the fallback carve still used when no val file exists)
# --------------------------------------------------------------------------- #


class TestSplitRecords:
    def test_is_deterministic_and_disjoint(self):
        data = [record(f"q{i}", f"a{i}") for i in range(200)]
        train, val, test = tr.split_records(data, seed=42)
        again = tr.split_records(data, seed=42)
        assert [r["messages"][0]["content"] for r in val] == \
               [r["messages"][0]["content"] for r in again[1]]
        names = [
            [r["messages"][0]["content"] for r in group] for group in (train, val, test)
        ]
        assert sum(len(g) for g in names) == len(data)
        flat = [n for g in names for n in g]
        assert len(set(flat)) == len(flat)

    def test_a_different_seed_carves_differently(self):
        data = [record(f"q{i}", f"a{i}") for i in range(200)]
        a = [r["messages"][0]["content"] for r in tr.split_records(data, seed=1)[1]]
        b = [r["messages"][0]["content"] for r in tr.split_records(data, seed=2)[1]]
        assert a != b


# --------------------------------------------------------------------------- #
# TokenizedJsonl / build_ids_dataset
# --------------------------------------------------------------------------- #


class TestTokenizedJsonl:
    def test_matches_the_tokenizer_on_the_rendered_text(self, tmp_path):
        path = write_jsonl(tmp_path / "s.jsonl", [record("q1", "a1"), record("q2", "a2")])
        tok = FakeTokenizer()
        rows = list(tr.TokenizedJsonl(path, tok, 1024)())
        expected = [
            {"input_ids": tok(tok.apply_chat_template(r["messages"])).input_ids}
            for r in tr.load_jsonl_records(path)
        ]
        assert rows == expected

    def test_truncates_to_max_seq_length(self, tmp_path):
        path = write_jsonl(tmp_path / "s.jsonl", [record("x" * 50, "y" * 50)])
        long_rows = list(tr.TokenizedJsonl(path, FakeTokenizer(), 1024)())
        short_rows = list(tr.TokenizedJsonl(path, FakeTokenizer(), 16)())
        assert len(long_rows[0]["input_ids"]) > 16
        assert len(short_rows[0]["input_ids"]) == 16
        # Truncation is a prefix cut, so the short ids start the long ones.
        assert long_rows[0]["input_ids"][:16] == short_rows[0]["input_ids"]

    def test_applies_the_same_skip_rules(self, tmp_path):
        rows = [record("keep", "a"), record("drop", ""), {"messages": []}]
        path = write_jsonl(tmp_path / "s.jsonl", rows)
        assert len(list(tr.TokenizedJsonl(path, FakeTokenizer(), 1024)())) == 1


# --------------------------------------------------------------------------- #
# subsample
# --------------------------------------------------------------------------- #


class TestSubsample:
    def _ds(self, n):
        from datasets import Dataset
        return Dataset.from_dict({"text": [f"t{i}" for i in range(n)]})

    def test_returns_everything_when_the_limit_exceeds_the_split(self):
        ds = self._ds(10)
        assert tr.subsample(ds, 0, 42) is ds
        assert tr.subsample(ds, -1, 42) is ds
        assert tr.subsample(ds, 10, 42) is ds
        assert tr.subsample(ds, 999, 42) is ds

    def test_is_deterministic_for_a_seed(self):
        ds = self._ds(100)
        a = tr.subsample(ds, 20, 42)["text"]
        b = tr.subsample(ds, 20, 42)["text"]
        assert a == b
        assert len(a) == 20
        assert len(set(a)) == 20

    def test_a_different_seed_picks_a_different_subset(self):
        ds = self._ds(100)
        assert tr.subsample(ds, 20, 1)["text"] != tr.subsample(ds, 20, 2)["text"]

    def test_preserves_file_order(self):
        # Every evaluation point scores the same examples in the same order,
        # so the early-stopping curve compares like with like.
        ds = self._ds(100)
        rows = tr.subsample(ds, 20, 42)["text"]
        assert rows == sorted(rows, key=lambda t: int(t[1:]))


# --------------------------------------------------------------------------- #
# project_eval_minutes
# --------------------------------------------------------------------------- #


class TestProjectEvalMinutes:
    def test_grows_with_the_number_of_evaluations(self):
        assert (tr.project_eval_minutes(2000, 2000, 10)
                > tr.project_eval_minutes(2000, 2000, 1))

    def test_accounts_for_the_final_test_pass_once(self):
        # n_evals multiplies only the val term; the test split is scored once.
        base = tr.project_eval_minutes(2000, 0, 10)
        with_test = tr.project_eval_minutes(2000, 2000, 10)
        assert with_test > base

    def test_defaults_stay_under_the_five_hour_ceiling(self):
        # 500 steps at eval-steps 50 is 10 evaluations.
        projected = tr.project_eval_minutes(2000, 2000, 10)
        assert projected < 300, f"default eval budget is {projected:.0f} min"

    def test_full_splits_would_blow_the_ceiling(self):
        # Documents why the sample defaults exist at all.
        full = tr.project_eval_minutes(3709, 3562, 10)
        assert full > 300


# --------------------------------------------------------------------------- #
# prune_tokenized_cache
# --------------------------------------------------------------------------- #


class TestPruneTokenizedCache:
    def test_removes_directories_this_run_did_not_touch(self, tmp_path, monkeypatch):
        root = tmp_path / "generator"
        fresh = root / "default-fingerprint=aaaa"
        stale = root / "default-fingerprint=bbbb"
        for d in (fresh, stale):
            (d / "0.0.0").mkdir(parents=True)
            (d / "0.0.0" / "generator-train.arrow").write_bytes(b"x")
        monkeypatch.setattr(tr, "TOKENIZED_CACHE_DIR", tmp_path)
        assert tr.prune_tokenized_cache({"aaaa"}) == 1
        assert fresh.is_dir()
        assert not stale.exists()

    def test_is_a_no_op_on_an_empty_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tr, "TOKENIZED_CACHE_DIR", tmp_path)
        assert tr.prune_tokenized_cache(set()) == 0

    def test_tolerates_a_missing_cache_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tr, "TOKENIZED_CACHE_DIR", tmp_path / "absent")
        assert tr.prune_tokenized_cache({"aaaa"}) == 0

"""Tests for tools/check-links.py, the markdown link gate `make lint` runs.

The checker is the only thing standing between a moved page and a 404 in the
rendered docs, so its own failure modes are tested here: a link that resolves
on this machine but not in a clone, a control character left by a partial
rewrite, and an anchor that no longer matches its heading.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    """Import tools/check-links.py, whose hyphen blocks a plain import."""
    path = ROOT / "tools" / "check-links.py"
    spec = importlib.util.spec_from_file_location("check_links", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_links"] = module
    spec.loader.exec_module(module)
    return module


cl = _load_checker()


@pytest.fixture()
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A miniature repository as the checker's ROOT."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "page.md").write_text(
        "# Title\n\n## A Heading Here\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "other.md").write_text("# Other\n", encoding="utf-8")
    monkeypatch.setattr(cl, "ROOT", tmp_path)
    monkeypatch.setattr(cl, "ignored_paths", lambda: None, raising=False)
    monkeypatch.setattr(cl, "repo_paths", lambda: None)
    return tmp_path


def check(repo: Path) -> list[str]:
    """Run the checker over *repo*, returning the error strings."""
    errors: list[str] = []
    for path in cl.md_files():
        cl.check_file(path, {}, errors)
    return errors


def write(repo: Path, text: str, name: str = "docs/page.md") -> None:
    (repo / name).write_text(text, encoding="utf-8")


class TestResolution:
    def test_valid_relative_link_passes(self, repo: Path):
        write(repo, "# T\n\nSee [other](other.md).\n")
        assert check(repo) == []

    def test_missing_file_is_reported(self, repo: Path):
        write(repo, "# T\n\nSee [gone](gone.md).\n")
        assert any("broken link" in e for e in check(repo))

    def test_external_links_are_not_checked(self, repo: Path):
        write(repo, "# T\n\n[up](https://example.invalid/nope) and [m](mailto:a@b.c)\n")
        assert check(repo) == []

    def test_anchor_must_match_a_heading(self, repo: Path):
        write(repo, "# T\n\n[ok](other.md#other) [bad](other.md#missing)\n")
        errors = check(repo)
        assert any("broken anchor" in e and "missing" in e for e in errors)

    def test_anchor_slug_follows_github_rules(self, repo: Path):
        """Punctuation is stripped, spaces hyphenated, per check_links.slugify."""
        write(repo, "# T\n\n[a](other.md#other)\n")
        (repo / "docs" / "other.md").write_text(
            "# Other!\n\n## Spaced Out, Punct: Here\n", encoding="utf-8"
        )
        assert cl.headings_slugs(repo / "docs" / "other.md") == [
            "other", "spaced-out-punct-here",
        ]
        assert check(repo) == []

    def test_fenced_code_is_not_a_link(self, repo: Path):
        write(repo, "# T\n\n```text\n[x](gone.md)\n```\n\n[y](gone.md)\n")
        errors = check(repo)
        assert len(errors) == 1, errors

    def test_target_with_a_space_is_rejected(self, repo: Path):
        """An unescaped space silently breaks the link on GitHub."""
        write(repo, "# T\n\n[x](my file.md)\n")
        assert any("contains a space" in e for e in check(repo))


class TestClonePresence:
    """A link can resolve locally and still 404 for anyone who clones.

    Generated trees are gitignored, so they exist after a build on the
    maintainer's machine and nowhere else.
    """

    IGNORED = (frozenset(), frozenset({"outputs/eval_results.json"}))

    def test_ignored_file_is_reported(self, repo: Path, monkeypatch: pytest.MonkeyPatch):
        (repo / "outputs").mkdir()
        (repo / "outputs" / "eval_results.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(cl, "repo_paths", lambda: self.IGNORED)
        write(repo, "# T\n\n[results](../outputs/eval_results.json)\n")
        assert any("not in the repository" in e for e in check(repo))

    def test_fully_ignored_directory_is_reported(self, repo: Path, monkeypatch: pytest.MonkeyPatch):
        """``outputs/`` is not itself a tracked path, so the entry check misses
        it; the directory has to be judged by what it contains."""
        (repo / "outputs").mkdir()
        (repo / "outputs" / "eval_results.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            cl, "repo_paths",
            lambda: (frozenset(), frozenset({"outputs/eval_results.json"})),
        )
        write(repo, "# T\n\n[outputs](../outputs)\n")
        assert any("not in the repository" in e for e in check(repo))

    def test_mixed_directory_is_allowed(self, repo: Path, monkeypatch: pytest.MonkeyPatch):
        """``data/`` holds tracked scripts and an ignored generated split: it
        does resolve in a clone, so flagging it would be noise."""
        (repo / "data").mkdir()
        (repo / "data" / "build_dataset.py").write_text("", encoding="utf-8")
        (repo / "data" / "processed").mkdir()
        (repo / "data" / "processed" / "dataset.jsonl").write_text("", encoding="utf-8")
        monkeypatch.setattr(
            cl, "repo_paths",
            lambda: (frozenset({"data/build_dataset.py"}),
                     frozenset({"data/processed/dataset.jsonl"})),
        )
        write(repo, "# T\n\n[data](../data)\n")
        assert check(repo) == []

    def test_check_is_skipped_without_git(self, repo: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(cl, "repo_paths", lambda: None)
        (repo / "outputs").mkdir()
        (repo / "outputs" / "x.json").write_text("{}", encoding="utf-8")
        write(repo, "# T\n\n[x](../outputs/x.json)\n")
        assert check(repo) == []


class TestControlCharacters:
    def test_control_character_is_reported(self, repo: Path):
        """A masked link that was never restored leaves U+0001 behind."""
        write(repo, "# T\n\nbroken \x01\x01\x01 label\n")
        assert any("control character U+0001" in e for e in check(repo))

    def test_plain_text_passes(self, repo: Path):
        write(repo, "# T\n\nordinary prose, tabs\tand ünïcode\n")
        assert check(repo) == []


class TestRealRepository:
    """The check must pass on the repository as it stands."""

    def test_repo_has_no_link_problems(self):
        monkey = pytest.MonkeyPatch()
        monkey.setattr(cl, "ROOT", ROOT)
        try:
            errors: list[str] = []
            for path in cl.md_files():
                cl.check_file(path, {}, errors)
        finally:
            monkey.undo()
        assert errors == [], "\n".join(errors)

    def test_control_character_is_reported_beside_valid_links(self, repo: Path):
        """The control check is independent of link resolution: a damaged line
        that also holds a good link must still be flagged."""
        write(repo, "# T\n\n[other](other.md) and a stray \x01 byte\n")
        errors = check(repo)
        assert any("control character" in e for e in errors)
        assert not any("broken link" in e for e in errors)

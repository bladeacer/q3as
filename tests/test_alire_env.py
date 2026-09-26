"""Tests for scripts/alire_env.py.

The module resolves the Alire-managed Ada toolchain for the eval scripts, the
defect validator, and the contract generator, so its search order and its
PATH parsing are load-bearing. The PATH parsing is the fragile part: it
inspects `alr exec -- printenv PATH` output, which interleaves the real value
with alr's own notes, and it has to pick the right line out of that.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import alire_env as ae


@pytest.fixture(autouse=True)
def clear_caches():
    """Both path helpers are lru_cached; each test needs a clean slate."""
    ae._alire_path_entry.cache_clear()
    ae.alire_env_path.cache_clear()
    ae._warn_system_tool.cache_clear()
    yield
    ae._alire_path_entry.cache_clear()
    ae.alire_env_path.cache_clear()
    ae._warn_system_tool.cache_clear()


def _fake_proc(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def _tool(directory: Path, name: str) -> Path:
    """Create an executable stand-in for *name* in *directory*, return its path.

    *directory* is the bin directory itself: it is what goes on PATH (or into
    the Alire prefix), so keeping the two identical avoids a test that passes
    for the wrong reason.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


class TestAlirePathEntry:
    def test_picks_the_path_line_out_of_alr_noise(self, monkeypatch, tmp_path):
        """alr writes notes to stdout too; only the real PATH line matches.

        `alr exec` prepends to the caller's PATH, so the value we want is the
        one that ends with the system PATH. A note never does.
        """
        system = "/usr/bin:/bin"
        monkeypatch.setenv("PATH", system)
        alire = "/opt/alire/bin"
        noisy = "\n".join([
            "warn: could not determine the current crate",
            f"{alire}:{system}",
            "",
        ])
        monkeypatch.setattr(ae.subprocess, "run", lambda *a, **k: _fake_proc(noisy))

        assert ae._alire_path_entry() == f"{alire}:"

    def test_returns_none_when_alr_fails(self, monkeypatch):
        monkeypatch.setattr(
            ae.subprocess, "run", lambda *a, **k: _fake_proc("", returncode=1)
        )
        assert ae._alire_path_entry() is None

    def test_returns_none_when_the_wrapper_is_absent(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ae, "ADA_ENV_SH", tmp_path / "nope.sh")
        assert ae._alire_path_entry() is None

    def test_returns_none_on_timeout(self, monkeypatch):
        def boom(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="alr", timeout=1)

        monkeypatch.setattr(ae.subprocess, "run", boom)
        assert ae._alire_path_entry() is None

    def test_returns_none_when_no_line_extends_the_system_path(self, monkeypatch):
        """Guards the heuristic against a stdout with no PATH value at all."""
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        monkeypatch.setattr(
            ae.subprocess, "run", lambda *a, **k: _fake_proc("some unrelated output\n")
        )
        assert ae._alire_path_entry() is None


class TestAlireEnvPath:
    def test_prepends_the_alire_prefix(self, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setattr(ae, "_alire_path_entry", lambda: "/opt/alire/bin:")
        assert ae.alire_env_path() == "/opt/alire/bin:/usr/bin"

    def test_falls_back_to_the_system_path(self, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        monkeypatch.setattr(ae, "_alire_path_entry", lambda: None)
        assert ae.alire_env_path() == os.environ["PATH"]

    def test_is_cached(self, monkeypatch):
        """The expensive alr call must happen once, not per lookup."""
        calls = []

        def counting():
            calls.append(1)
            return "/opt/alire/bin:"

        monkeypatch.setattr(ae, "_alire_path_entry", counting)
        ae.alire_env_path()
        ae.alire_env_path()
        ae.alire_env_path()
        assert len(calls) == 1


class TestFindTool:
    def test_prefers_the_alire_environment(self, monkeypatch, tmp_path):
        alire = _tool(tmp_path / "alire", "gnatprove")
        system = _tool(tmp_path / "system", "gnatprove")
        monkeypatch.setenv("PATH", str(system.parent))
        monkeypatch.setattr(ae, "_alire_path_entry", lambda: f"{alire.parent}:")
        assert ae.find_tool("gnatprove") == alire

    def test_falls_back_to_system_path_and_warns_once(self, monkeypatch, tmp_path, capsys):
        system = _tool(tmp_path / "system", "gnatprove")
        monkeypatch.setenv("PATH", str(system.parent))
        monkeypatch.setattr(ae, "_alire_path_entry", lambda: None)

        assert ae.find_tool("gnatprove") == system
        ae.find_tool("gnatprove")  # cached warning, so still one message
        err = capsys.readouterr().err
        assert err.count("resolved from the system PATH") == 1

    def test_warns_when_the_prefix_lacks_the_tool(self, monkeypatch, tmp_path, capsys):
        """A system hit is only silent when the Alire prefix supplied it."""
        system = _tool(tmp_path / "system", "gnatprove")
        other = tmp_path / "alire"
        other.mkdir()
        monkeypatch.setenv("PATH", str(system.parent))
        monkeypatch.setattr(ae, "_alire_path_entry", lambda: f"{other}:")

        assert ae.find_tool("gnatprove") == system
        assert "resolved from the system PATH" in capsys.readouterr().err

    def test_warns_when_no_alire_environment_exists(self, monkeypatch, tmp_path, capsys):
        """Regression: with an empty prefix the two searches coincide.

        Keying the warning off "found by the system search" made it
        unreachable in that case, so a distribution gnatprove was silently
        reported as the managed one.
        """
        system = _tool(tmp_path / "system", "gnatprove")
        monkeypatch.setenv("PATH", str(system.parent))
        monkeypatch.setattr(ae, "_alire_path_entry", lambda: None)

        assert ae.find_tool("gnatprove") == system
        assert "resolved from the system PATH" in capsys.readouterr().err

    def test_no_warning_when_the_prefix_supplies_the_tool(
        self, monkeypatch, tmp_path, capsys
    ):
        managed = _tool(tmp_path / "alire", "gnatprove")
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setattr(ae, "_alire_path_entry", lambda: f"{managed.parent}:")

        assert ae.find_tool("gnatprove") == managed
        assert "resolved from the system PATH" not in capsys.readouterr().err

    def test_does_not_warn_for_tools_outside_alire_tools(self, monkeypatch, tmp_path, capsys):
        mine = _tool(tmp_path / "system", "mytool")
        monkeypatch.setenv("PATH", str(mine.parent))
        monkeypatch.setattr(ae, "_alire_path_entry", lambda: None)

        assert ae.find_tool("mytool").name == "mytool"
        assert "resolved from the system PATH" not in capsys.readouterr().err

    def test_raises_with_an_actionable_message(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setattr(ae, "_alire_path_entry", lambda: None)

        with pytest.raises(ae.ToolNotAvailable) as excinfo:
            ae.find_tool("gnatprove")
        message = str(excinfo.value)
        assert "gnatprove" in message
        assert "make prove" in message

    def test_ignores_non_executable_matches(self, monkeypatch, tmp_path):
        candidate = tmp_path / "system" / "gnatprove"
        candidate.parent.mkdir(parents=True)
        candidate.write_text("not executable", encoding="utf-8")
        candidate.chmod(0o644)
        monkeypatch.setenv("PATH", str(candidate.parent))
        monkeypatch.setattr(ae, "_alire_path_entry", lambda: None)

        with pytest.raises(ae.ToolNotAvailable):
            ae.find_tool("gnatprove")

    def test_has_tool_matches_find_tool(self, monkeypatch, tmp_path):
        tool = _tool(tmp_path / "system", "gnat")
        monkeypatch.setenv("PATH", str(tool.parent))
        monkeypatch.setattr(ae, "_alire_path_entry", lambda: None)

        assert ae.has_tool("gnat") is True
        assert ae.has_tool("gnatprove") is False


class TestMain:
    def test_reports_every_tool_and_exits_zero_when_complete(
        self, monkeypatch, tmp_path, capsys
    ):
        for tool in ae.REQUIRED_TOOLS:
            _tool(tmp_path / "alire", tool)
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setattr(ae, "_alire_path_entry", lambda: f"{tmp_path / 'alire'}:")
        monkeypatch.setattr(sys, "argv", ["alire_env.py"])

        assert ae.main() == 0
        out = capsys.readouterr().out
        for tool in ae.REQUIRED_TOOLS:
            assert tool in out

    def test_missing_required_tool_fails(self, monkeypatch, tmp_path, capsys):
        _tool(tmp_path / "alire", "gnat")
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setattr(ae, "_alire_path_entry", lambda: f"{tmp_path / 'alire'}:")
        monkeypatch.setattr(sys, "argv", ["alire_env.py"])

        assert ae.main() == 1
        assert "gnatprove" in capsys.readouterr().err

    def test_optional_tool_never_fails_the_check(self, monkeypatch, tmp_path, capsys):
        for tool in ae.REQUIRED_TOOLS:
            _tool(tmp_path / "alire", tool)
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setattr(ae, "_alire_path_entry", lambda: f"{tmp_path / 'alire'}:")
        monkeypatch.setattr(sys, "argv", ["alire_env.py"])

        assert ae.main() == 0
        out = capsys.readouterr().out
        # gnatdoc is in ALIRE_TOOLS but never a dependency of alire-dev.toml.
        assert "gnatdoc" in out
        assert "missing (optional)" in out

    def test_gnatdoc_is_not_required(self):
        assert "gnatdoc" in ae.ALIRE_TOOLS
        assert "gnatdoc" not in ae.REQUIRED_TOOLS


def test_caches_are_configured():
    """Guard the cache sizes: a stale entry would poison every later run."""
    assert isinstance(ae._alire_path_entry, functools._lru_cache_wrapper)
    assert ae._alire_path_entry.cache_info().maxsize == 1
    assert ae.alire_env_path.cache_info().maxsize == 1

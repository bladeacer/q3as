"""source_paths.py - central resolution of cached source repositories.

All pipeline stages read third-party repositories from the local archive
cache (``data/raw_repos/<owner>/<repo>``, written by ``scripts/fetch_repos.py``).
The old ``../<repo>`` sibling layout is gone; this module is the one place
that knows where sources live now, so no pipeline file hard-codes paths.

Resolution order per repository:

1. ``Q3AS_RAW_REPOS`` environment variable (explicit override, one
   directory per repository, ``;``-separated, same order as SOURCES),
2. the archive cache under ``data/raw_repos/``,
3. the legacy sibling location ``../<dir>`` (kept only so a checkout that
   still has the old layout keeps working until it re-runs setup).

``resolve`` returns the first existing directory. ``resolve_all`` also
returns the missing ones so callers can warn per source.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# fetch_repos lives in scripts/; source_paths may be imported from anywhere.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from fetch_repos import CACHE_DIR, CORE_REPOS, GITHUB_REPO_RE

# repo URL -> directory name used by every pipeline stage. Mostly the repo
# name; kept explicit so a rename upstream does not silently split sources.
_DIR_BY_URL: dict[str, str] = {
    "https://github.com/bladeacer/adacovex": "adacovex",
    "https://github.com/bladeacer/Ada_CRDT": "Ada_CRDT",
    "https://github.com/ViMoBr/Ada-83-TLALOC": "Ada-83-TLALOC",
    "https://github.com/AdaCore/ada-eval": "ada-eval",
    "https://github.com/AdaCore/learn": "learn",
    "https://github.com/AdaCore/training_material": "training_material",
    "https://github.com/agent-sh/ada-spark": "ada-spark",
    "https://github.com/AminBlg/SimpleEnglish": "SimpleEnglish",
    "https://github.com/AdaCore/skills": "skills",
}

SOURCES: list[tuple[str, str]] = [(_url, _DIR_BY_URL[_url]) for _url in CORE_REPOS]

OWNER_BY_NAME: dict[str, str] = {}
for _url, _dir in SOURCES:
    _m = GITHUB_REPO_RE.fullmatch(_url)
    assert _m is not None
    OWNER_BY_NAME[_dir] = _m.group(1)

# Names pipeline stages reference directly (keep in sync with _DIR_BY_URL).
ADACOVEX = "adacovex"
ADA_CRDT = "Ada_CRDT"
ADA_83_TLALOC = "Ada-83-TLALOC"
ADA_EVAL = "ada-eval"
LEARN = "learn"
TRAINING_MATERIAL = "training_material"
ADA_SPARK = "ada-spark"
SIMPLE_ENGLISH = "SimpleEnglish"
SKILLS = "skills"


def root() -> Path:
    """Project root (the directory that contains data/raw_repos)."""
    return Path(__file__).resolve().parents[2]


def cache_root() -> Path:
    """The archive cache directory (data/raw_repos)."""
    return root() / CACHE_DIR


def _cache_dir(dir_name: str) -> Path:
    owner = OWNER_BY_NAME.get(dir_name, "")
    return cache_root() / owner / dir_name


def _sibling_dir(dir_name: str) -> Path:
    return root().parent / dir_name


def _override_dirs() -> list[Path]:
    raw = os.environ.get("Q3AS_RAW_REPOS", "")
    return [Path(p) for p in raw.split(";") if p.strip()]


def resolve(dir_name: str) -> Path | None:
    """First existing directory for *dir_name* (override, cache, sibling)."""
    for candidate in (*_override_dirs(), _cache_dir(dir_name), _sibling_dir(dir_name)):
        if candidate.is_dir():
            return candidate
    return None


def resolve_all(dir_names: list[str]) -> list[tuple[str, Path | None]]:
    """(name, directory-or-None) for every requested source."""
    return [(name, resolve(name)) for name in dir_names]


def require(dir_name: str) -> Path:
    """Resolve *dir_name* or raise FileNotFoundError with setup guidance."""
    resolved = resolve(dir_name)
    if resolved is None:
        raise FileNotFoundError(
            f"Source repository '{dir_name}' not found in the cache. "
            f"Run: python3 scripts/fetch_repos.py"
        )
    return resolved


def default_code_dirs() -> list[Path]:
    """Ada code sources: existing dirs among adacovex, Ada_CRDT, TLALOC, ada-eval."""
    dirs: list[Path] = []
    for name in (ADACOVEX, ADA_CRDT, ADA_83_TLALOC, ADA_EVAL):
        resolved = resolve(name)
        if resolved is not None:
            dirs.append(resolved)
    return dirs


def default_doc_dirs() -> list[Path]:
    """Doc sources: existing dirs among learn and training_material."""
    dirs: list[Path] = []
    for name in (LEARN, TRAINING_MATERIAL):
        resolved = resolve(name)
        if resolved is not None:
            dirs.append(resolved)
    return dirs


def default_guidance_dirs() -> list[Path]:
    """Agent-skill sources: existing dirs among ada-spark, SimpleEnglish, skills."""
    dirs: list[Path] = []
    for name in (ADA_SPARK, SIMPLE_ENGLISH, SKILLS):
        resolved = resolve(name)
        if resolved is not None:
            dirs.append(resolved)
    return dirs


def default_eval_sources() -> list[Path]:
    """Defect-validation sources: adacovex, Ada_CRDT, ada-eval (existing only)."""
    dirs: list[Path] = []
    for name in (ADACOVEX, ADA_CRDT, ADA_EVAL):
        resolved = resolve(name)
        if resolved is not None:
            dirs.append(resolved)
    return dirs

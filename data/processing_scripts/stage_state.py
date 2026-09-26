"""stage_state.py - skip a pipeline stage when nothing it reads has changed.

The dataset stages are ``.PHONY`` make targets, so ``make all`` used to
re-parse every Ada source and rebuild the whole dataset on every run, even
when the cached outputs were already current. That is minutes of work before
training even starts, and it happens on every invocation.

This module makes each stage declare what it reads (input trees with the file
suffixes it actually consumes, input files, the scripts that produce it, and
the parameters that change the output). After a successful run the
fingerprint is written to ``data/processed/.stages/<name>.json``; on the next
run the fingerprint is recomputed and the stage is skipped when it still
matches.

Design notes:

- **Content, not mtime.** The archive cache is re-extracted from tarballs, so
  mtimes change when nothing did. Hashing content keeps a refetched repo from
  forcing a rebuild, and the whole corpus hashes in about a second.
- **Scripts are inputs.** Editing ``build_dataset.py`` or a helper invalidates
  every stage that imports it, so a logic change never silently keeps stale
  output.
- **Output-dependent parameters only.** ``--workers`` is deliberately not part
  of any fingerprint: the stages are documented as producing identical output
  for any worker count, so changing ``DATASET_WORKERS`` must not invalidate.
- **Missing is not fresh.** A stage is never skipped unless every declared
  output exists, is non-empty, and still has the size recorded at write time.
  Deleting or truncating an output therefore always rebuilds.
- **Never fatal.** An unreadable input or a corrupt stamp makes the stage run,
  it does not abort the pipeline.

Usage inside a stage::

    spec = stage_state.StageSpec(
        name="docs_chunks",
        outputs=(args.output,),
        input_trees=((root, (".md", ".rst")) for root in input_dirs),
        params=(("min_chars", str(args.min_chars)),),
        scripts=(Path(__file__).resolve(),),
    )
    if spec.skip_if_fresh(force=args.force):
        return
    ...produce outputs...
    spec.mark_fresh()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("q3as_stage_state")

# Bump when the fingerprint layout changes. Older stamps then never match, so
# every stage rebuilds once instead of trusting a stamp it cannot interpret.
STAMP_VERSION = 2

_MODULE_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _MODULE_DIR.parents[1] / "scripts"
_ROOT = _MODULE_DIR.parents[2]


def _stamp_dir() -> Path:
    """Where stamps live.

    ``Q3AS_STAGE_DIR`` redirects them, which the tests need so they never
    write into the real ``data/processed/.stages``. Stamps are keyed by stage
    name, so a run with a custom ``--output`` deliberately invalidates (and
    overwrites) the stamp of the default run; the next default run rebuilds.
    """
    override = os.environ.get("Q3AS_STAGE_DIR", "").strip()
    if override:
        return Path(override)
    return _ROOT / "data" / "processed" / ".stages"


STAMP_DIR = _stamp_dir()

# Every parser imports build_dataset (the STE rewriter), which in turn imports
# these. A change to any of them changes parser output, so all of them are
# fingerprinted for every stage. Entries that do not exist are recorded as
# absent, so adding one later still invalidates.
SHARED_SCRIPTS: tuple[Path, ...] = (
    _MODULE_DIR / "build_dataset.py",
    _MODULE_DIR / "code_variants.py",
    _MODULE_DIR / "eval_guard.py",
    _MODULE_DIR / "source_paths.py",
    _SCRIPTS_DIR / "fetch_repos.py",
)

_READ_CHUNK = 1 << 20


def _hash_file(hasher: hashlib._Hash, path: Path) -> bool:
    """Feed one file's path, size, and content into *hasher*.

    Returns False when the file cannot be read, which makes the caller treat
    the stage as stale instead of raising.
    """
    try:
        stat = path.stat()
    except OSError as exc:
        logger.warning("stage_state: cannot stat %s (%s); treating stage as stale", path, exc)
        return False
    hasher.update(path.as_posix().encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(str(stat.st_size).encode("ascii"))
    hasher.update(b"\0")
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
                hasher.update(chunk)
    except OSError as exc:
        logger.warning("stage_state: cannot read %s (%s); treating stage as stale", path, exc)
        return False
    return True


def _iter_tree_files(root: Path, suffixes: Sequence[str]) -> Iterator[Path]:
    """Files under *root* whose name ends with one of *suffixes*, sorted."""
    seen: set[Path] = set()
    matches: list[Path] = []
    for suffix in suffixes:
        matches.extend(root.rglob(f"*{suffix}"))
    for path in matches:
        if path in seen:
            continue
        seen.add(path)
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        yield path


def digest_tree(root: Path, suffixes: Sequence[str]) -> dict[str, Any]:
    """Content digest of the files under *root* that end with *suffixes*.

    A missing directory digests to a recorded absence rather than an empty
    digest, so a source repo that appears later invalidates the stamp.
    """
    hasher = hashlib.sha256()
    if not root.is_dir():
        hasher.update(b"\0missing-dir\0")
        return {"path": str(root), "exists": False, "files": 0, "digest": hasher.hexdigest()}
    count = 0
    for path in _iter_tree_files(root, suffixes):
        if not _hash_file(hasher, path):
            return {"path": str(root), "exists": True, "files": -1, "digest": "unreadable"}
        count += 1
    return {
        "path": str(root),
        "exists": True,
        "files": count,
        "suffixes": sorted(suffixes),
        "digest": hasher.hexdigest(),
    }


def digest_file(path: Path) -> dict[str, Any]:
    """Content digest of a single file (absent files are recorded as such)."""
    hasher = hashlib.sha256()
    if not path.is_file():
        hasher.update(b"\0missing-file\0")
        return {"path": str(path), "exists": False, "digest": hasher.hexdigest()}
    ok = _hash_file(hasher, path)
    return {
        "path": str(path),
        "exists": True,
        "size": path.stat().st_size,
        "digest": hasher.hexdigest() if ok else "unreadable",
    }


@dataclass(frozen=True)
class StageSpec:
    """What one pipeline stage reads and writes.

    ``input_trees`` pairs a directory with the file suffixes the stage reads
    from it; ``input_files`` covers single files (the parser JSONL the builder
    merges, the eval methodology manifest). ``scripts`` are the entry point
    plus any stage-local helper. ``params`` holds only the settings that change
    the output.
    """

    name: str
    outputs: tuple[Path, ...]
    input_trees: tuple[tuple[Path, tuple[str, ...]], ...] = ()
    input_files: tuple[Path, ...] = ()
    scripts: tuple[Path, ...] = ()
    params: tuple[tuple[str, str], ...] = ()
    notes: str = field(default="", compare=False)

    @property
    def stamp_path(self) -> Path:
        return STAMP_DIR / f"{self.name}.json"

    def _script_paths(self) -> tuple[Path, ...]:
        # stage_state.py itself is always included: changing the fingerprint
        # layout must invalidate the stamps it wrote.
        unique: dict[str, Path] = {}
        for path in (Path(__file__).resolve(), *SHARED_SCRIPTS, *self.scripts):
            unique.setdefault(str(path), path)
        return tuple(unique.values())

    def fingerprint(self) -> dict[str, Any]:
        """Everything needed to decide whether this stage must run again."""
        return {
            "stage": self.name,
            "stamp_version": STAMP_VERSION,
            "trees": [digest_tree(root, sufs) for root, sufs in self.input_trees],
            "files": [digest_file(path) for path in self.input_files],
            "scripts": [digest_file(path) for path in self._script_paths()],
            "params": {key: str(value) for key, value in sorted(self.params)},
        }

    def output_state(self) -> list[dict[str, Any]]:
        """Existence and size of every declared output."""
        state: list[dict[str, Any]] = []
        for path in self.outputs:
            entry: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
            if entry["exists"]:
                try:
                    entry["size"] = path.stat().st_size
                except OSError:
                    entry["exists"] = False
            state.append(entry)
        return state

    def _read_stamp(self) -> dict[str, Any] | None:
        try:
            stamp = json.loads(self.stamp_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("stage_state: unreadable stamp %s (%s); rebuilding", self.stamp_path, exc)
            return None
        if not isinstance(stamp, dict):
            logger.warning("stage_state: malformed stamp %s; rebuilding", self.stamp_path)
            return None
        return stamp

    def _outputs_intact(self, stamp: dict[str, Any]) -> bool:
        """True when every output still exists, is non-empty, and matches size."""
        recorded = stamp.get("outputs")
        current = self.output_state()
        if not isinstance(recorded, list) or len(recorded) != len(current):
            return False
        for expected, actual in zip(recorded, current, strict=True):
            if not isinstance(expected, dict) or not actual["exists"]:
                return False
            if actual.get("size", 0) <= 0:
                return False
            if expected.get("path") != actual["path"]:
                return False
            if expected.get("size") != actual.get("size"):
                return False
        return True

    def skip_if_fresh(self, force: bool = False) -> bool:
        """True when the stage can be skipped because nothing changed.

        Logs the reason either way at INFO level, so a ``make all`` that skips
        work says why instead of looking hung.
        """
        if force:
            logger.info("stage_state: %s forced; rebuilding", self.name)
            return False
        stamp = self._read_stamp()
        if stamp is None:
            logger.info("stage_state: %s has no stamp; building", self.name)
            return False
        if not self._outputs_intact(stamp):
            logger.info("stage_state: %s output missing or changed; building", self.name)
            return False
        current = self.fingerprint()
        if stamp.get("fingerprint") != current:
            logger.info(
                "stage_state: %s inputs changed since the last build; building", self.name
            )
            return False
        logger.info(
            "stage_state: %s is up to date (%s); skipping (use --force to rebuild)",
            self.name,
            self._summarize(current),
        )
        return True

    @staticmethod
    def _summarize(fingerprint: dict[str, Any]) -> str:
        trees = fingerprint.get("trees", [])
        files = sum(int(t.get("files", 0)) for t in trees if isinstance(t, dict))
        return f"{len(trees)} input trees, {files} files"

    def mark_fresh(self, extra: dict[str, Any] | None = None) -> None:
        """Record the fingerprint that describes the outputs just written."""
        outputs = self.output_state()
        missing = [entry["path"] for entry in outputs if not entry["exists"]]
        if missing:
            logger.warning(
                "stage_state: %s did not produce %s; not recording a stamp",
                self.name, ", ".join(missing),
            )
            return
        payload: dict[str, Any] = {
            "stage": self.name,
            "stamp_version": STAMP_VERSION,
            "fingerprint": self.fingerprint(),
            "outputs": outputs,
        }
        if extra:
            payload["extra"] = extra
        try:
            self.stamp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.stamp_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self.stamp_path)
        except OSError as exc:
            # A missing stamp only costs a rebuild next time; never fail the run.
            logger.warning("stage_state: cannot write stamp %s (%s)", self.stamp_path, exc)
        else:
            logger.info("stage_state: recorded fresh stamp for %s", self.name)


def make_spec(
    name: str,
    outputs: Iterable[Path],
    input_trees: Iterable[tuple[Path, Sequence[str]]] = (),
    input_files: Iterable[Path] = (),
    scripts: Iterable[Path] = (),
    params: Iterable[tuple[str, Any]] = (),
) -> StageSpec:
    """Build a StageSpec, materialising the iterables into tuples."""
    return StageSpec(
        name=name,
        outputs=tuple(outputs),
        input_trees=tuple((Path(root), tuple(sufs)) for root, sufs in input_trees),
        input_files=tuple(Path(path) for path in input_files),
        scripts=tuple(Path(path) for path in scripts),
        params=tuple((str(key), str(value)) for key, value in params),
    )

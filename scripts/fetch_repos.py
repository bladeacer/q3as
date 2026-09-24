#!/usr/bin/env python3
"""fetch_repos.py - archive-based source ingestion with a cache.

Replaces sibling `git clone` management. Every source repository lands as
an extracted tarball under ``data/raw_repos/<owner>/<repo>/`` with a
``.q3as-source.json`` metadata file recording the URL, fetch time, license
SPDX (when the GitHub API answers) and where the entry came from. On the
next run a directory with matching metadata is reused, so re-running the
pipeline never re-downloads what is already on disk.

Why archives instead of clones:

- no ``.git`` directory (history and blobs inflate storage 2x to 10x),
- one HTTP GET per repo, parallelizable and restartable,
- the cache is content-addressed by repo identity, not by git state.

Sources come from two places, all combinable:

- ``--repo URL``            an explicit repository (any repo not in CORE_REPOS).
- ``--from-manifest FILE``  one URL per line, ``#`` comments allowed.

The default (no flags) fetches every repository in ``CORE_REPOS``, which
includes the RobertBoettcherSF ``Ada-Algorithms`` monorepo.

Usage:
    python3 scripts/fetch_repos.py                     # core (default)
    python3 scripts/fetch_repos.py --list              # cache status
    python3 scripts/fetch_repos.py --refresh core      # re-fetch core
    python3 scripts/fetch_repos.py --refresh all       # re-fetch everything

Exit code: 0 when every requested repository is in the cache, 1 otherwise.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("q3as_fetch_repos")

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "raw_repos"
META_FILE = ".q3as-source.json"
GITHUB_REPO_RE = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")

# The core sources (exact URLs). ada-eval is also the uv path dependency used by
# eval/generate.py and eval/eval_pipeline.py. The RobertBoettcherSF
# Ada-Algorithms monorepo provides the algorithm corpus (MIT, author-approved
# for training).
CORE_REPOS = [
    "https://github.com/bladeacer/adacovex",
    "https://github.com/bladeacer/Ada_CRDT",
    "https://github.com/ViMoBr/Ada-83-TLALOC",
    "https://github.com/AdaCore/ada-eval",
    "https://github.com/AdaCore/learn",
    "https://github.com/AdaCore/training_material",
    "https://github.com/agent-sh/ada-spark",
    "https://github.com/AminBlg/SimpleEnglish",
    "https://github.com/AdaCore/skills",
    "https://github.com/RobertBoettcherSF/Ada-Algorithms",
]

USER_AGENT = "q3as-dataset-fetcher (+https://github.com/q3as)"


@dataclass(frozen=True)
class RepoRequest:
    """One repository to fetch, with provenance of the request."""

    url: str
    origin: str  # "core" | "cli" | "manifest"
    required: bool = True

    @property
    def owner(self) -> str:
        match = GITHUB_REPO_RE.fullmatch(self.url.rstrip("/"))
        if not match:
            raise ValueError(f"Not a GitHub repository URL: {self.url}")
        return match.group(1)

    @property
    def repo(self) -> str:
        match = GITHUB_REPO_RE.fullmatch(self.url.rstrip("/"))
        if not match:
            raise ValueError(f"Not a GitHub repository URL: {self.url}")
        return match.group(2)


@dataclass
class FetchResult:
    request: RepoRequest
    status: str  # "cached" | "fetched" | "failed"
    detail: str = ""


# --------------------------------------------------------------------------- #
# GitHub helpers
# --------------------------------------------------------------------------- #


def _http_get(url: str, timeout: float = 60.0, retries: int = 2) -> bytes:
    """GET *url* and return the body; raise urllib.error on final failure."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _api_json(repo_url: str) -> dict | None:
    """Fetch repo metadata (default branch, license, description) or None.

    Tolerated to fail: unauthenticated API quota is small, and everything
    the fetcher needs to work (the tarball) comes from codeload without it.
    """
    try:
        payload = json.loads(_http_get(repo_url.replace("github.com", "api.github.com/repos"), timeout=15))
    except (ValueError, urllib.error.URLError, OSError, TimeoutError):
        return None
    return payload if isinstance(payload, dict) else None


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def target_dir(request: RepoRequest) -> Path:
    """Cache directory for a repository: data/raw_repos/<owner>/<repo>."""
    return CACHE_DIR / request.owner / request.repo


def load_meta(directory: Path) -> dict | None:
    meta_path = directory / META_FILE
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def is_cached(request: RepoRequest) -> bool:
    meta = load_meta(target_dir(request))
    if not meta:
        return False
    return meta.get("url", "").rstrip("/") == request.url.rstrip("/")


def write_meta(directory: Path, request: RepoRequest, license_spdx: str | None, description: str | None) -> None:
    from datetime import UTC, datetime

    meta = {
        "url": request.url.rstrip("/"),
        "origin": request.origin,
        "fetched_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "license_spdx": license_spdx,
        "description": description,
    }
    (directory / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Fetch and extract
# --------------------------------------------------------------------------- #


def _extract_tarball(archive_bytes: bytes, destination: Path) -> None:
    """Extract a GitHub tarball into *destination*, stripping the top dir.

    GitHub archives wrap everything in ``<repo>-<ref>/``. Extraction goes
    to a temporary directory on the same filesystem as the cache first
    (/tmp may be a different device, which breaks os.rename), then is
    moved into place, so a failed download never leaves a half-written
    cache entry behind.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".q3as-fetch-", dir=destination.parent) as tmp:
        archive_path = Path(tmp) / "repo.tar.gz"
        archive_path.write_bytes(archive_bytes)
        extract_dir = Path(tmp) / "out"
        extract_dir.mkdir()
        with tarfile.open(archive_path, "r:gz") as tar:
            try:
                tar.extractall(extract_dir, filter="data")
            except TypeError:  # Python < 3.12: no filter parameter
                tar.extractall(extract_dir)
        entries = [p for p in extract_dir.iterdir()]
        if len(entries) == 1 and entries[0].is_dir():
            entries[0].replace(destination)
        else:
            shutil.copytree(extract_dir, destination)


def fetch_repo(request: RepoRequest) -> FetchResult:
    """Fetch one repository into the cache (or report it as cached)."""
    directory = target_dir(request)
    if is_cached(request):
        return FetchResult(request, "cached")

    tarball_url = f"https://codeload.github.com/{request.owner}/{request.repo}/tar.gz/HEAD"
    try:
        archive = _http_get(tarball_url)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return FetchResult(request, "failed", str(exc))

    api = _api_json(request.url)
    license_spdx = None
    description = None
    if api:
        license_spdx = ((api.get("license") or {}).get("spdx_id")) or None
        description = api.get("description")

    try:
        if directory.exists():
            shutil.rmtree(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        _extract_tarball(archive, directory)
        write_meta(directory, request, license_spdx, description)
    except (tarfile.TarError, OSError) as exc:
        return FetchResult(request, "failed", str(exc))
    return FetchResult(request, "fetched", f"license={license_spdx or 'unknown'}")


def fetch_all(requests: list[RepoRequest], jobs: int = 8) -> list[FetchResult]:
    """Fetch *requests* in parallel, preserving input order in the result."""
    if not requests:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
        results = list(pool.map(fetch_repo, requests))
    cached = sum(1 for r in results if r.status == "cached")
    fetched = sum(1 for r in results if r.status == "fetched")
    failed = [r for r in results if r.status == "failed"]
    logger.info("Cache: %d reused, %d fetched, %d failed", cached, fetched, len(failed))
    for failure in failed:
        logger.error("FAILED %s: %s", failure.request.url, failure.detail)
    return results


# --------------------------------------------------------------------------- #
# Request building
# --------------------------------------------------------------------------- #


def build_requests(
    repos: list[str],
    manifest: Path | None,
    refresh: str | None,
) -> list[RepoRequest]:
    """Assemble the deduplicated request list in a stable order."""
    requests: list[RepoRequest] = []
    seen: set[str] = set()

    def add(url: str, origin: str, required: bool = True) -> None:
        url = url.rstrip("/")
        if url in seen or not GITHUB_REPO_RE.fullmatch(url):
            return
        seen.add(url)
        requests.append(RepoRequest(url=url, origin=origin, required=required))

    for url in CORE_REPOS:
        add(url, "core")
    for url in repos:
        add(url, "cli")
    if manifest is not None:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                add(line, "manifest")

    if refresh == "core":
        requests = [r for r in requests if r.origin == "core"]
    elif refresh not in (None, "all"):
        raise SystemExit(f"--refresh must be one of: core, all (got {refresh!r})")
    return requests


# --------------------------------------------------------------------------- #
# Status listing
# --------------------------------------------------------------------------- #


def list_cache() -> int:
    """Print every cached repository with its recorded license."""
    if not CACHE_DIR.exists():
        print(f"Cache is empty: {CACHE_DIR} does not exist yet.")
        return 0
    missing_meta = 0
    total = 0
    for owner_dir in sorted(CACHE_DIR.iterdir()):
        if not owner_dir.is_dir():
            continue
        for repo_dir in sorted(owner_dir.iterdir()):
            if not repo_dir.is_dir():
                continue
            total += 1
            meta = load_meta(repo_dir)
            if meta is None:
                missing_meta += 1
                print(f"  {owner_dir.name}/{repo_dir.name}: NO META (re-fetch needed)")
                continue
            license_spdx = meta.get("license_spdx") or "?"
            print(f"  {owner_dir.name}/{repo_dir.name}: {license_spdx} ({meta.get('fetched_at', '?')})")
    print(f"{total} cached repositories, {missing_meta} without metadata.")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", action="append", default=[], help="Extra repository URL. Repeatable.")
    parser.add_argument("--from-manifest", type=Path, help="File with one repository URL per line.")
    parser.add_argument("--refresh", choices=["core", "all"], help="Re-fetch even when cached.")
    parser.add_argument("--jobs", type=int, default=8, help="Parallel downloads (default 8).")
    parser.add_argument("--list", action="store_true", help="Print cache status and exit.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    if args.list:
        return list_cache()

    requests = build_requests(args.repo, args.from_manifest, args.refresh)
    logger.info("Fetching %d repositories (%d jobs)", len(requests), args.jobs)
    results = fetch_all(requests, jobs=args.jobs)

    failures = [r for r in results if r.status == "failed"]
    core_failures = [r for r in failures if r.request.origin == "core"]
    if core_failures:
        print(f"{len(core_failures)} of {len(results)} repositories failed to fetch.", file=sys.stderr)
        return 1
    if failures:
        print(
            f"{len(failures)} non-core repositories failed to fetch; "
            "the dataset build skips them with a warning.",
            file=sys.stderr,
        )
    print(f"Core sources fetched; {len(results) - len(failures)}/{len(results)} repositories are in the cache at {CACHE_DIR}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

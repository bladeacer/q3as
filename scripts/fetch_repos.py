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

Because there is no ``.git``, the commit a cache entry came from is recorded in
its metadata file. ``--check`` compares that commit with the upstream head and
reports what moved; ``--update`` re-fetches only those. The dataset stages
fingerprint the cache by content, so a refreshed tree rebuilds them on the next
``make build-dataset`` without anything else having to know a commit exists.

Sources come from two places, all combinable:

- ``--repo URL``            an explicit repository (any repo not in CORE_REPOS).
- ``--from-manifest FILE``  one URL per line, ``#`` comments allowed.

The default (no flags) fetches every repository in ``CORE_REPOS``, which
includes the RobertBoettcherSF ``Ada-Algorithms`` monorepo.

Usage:
    python3 scripts/fetch_repos.py                     # core (default)
    python3 scripts/fetch_repos.py --list              # cache status
    python3 scripts/fetch_repos.py --check             # what moved upstream
    python3 scripts/fetch_repos.py --update            # re-fetch what moved
    python3 scripts/fetch_repos.py --refresh core      # re-fetch core
    python3 scripts/fetch_repos.py --refresh all       # re-fetch everything

Exit code: 0 when every requested repository is in the cache, 1 otherwise.
``--check`` exits 1 when at least one cached repository is behind upstream, so
it can gate a scheduled rebuild. A repository whose head cannot be read (API
quota, network) is reported as unknown and never re-fetched: a failed check
must not destroy a good cache.
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

    @property
    def name(self) -> str:
        """owner/repo, the cache identity used by check and update."""
        return f"{self.owner}/{self.repo}"


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


def head_commit(owner: str, repo: str, branch: str) -> str | None:
    """The upstream head commit of *branch*, or None when it cannot be read.

    ``/git/ref/heads/<branch>`` is the cheap form of the question (one small
    JSON object). None is a real answer here, not an error: the caller must not
    re-fetch a cache entry because a check could not run.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/{branch}"
    try:
        payload = json.loads(_http_get(url, timeout=15, retries=1))
    except (ValueError, urllib.error.URLError, OSError, TimeoutError):
        return None
    if not isinstance(payload, dict):
        return None
    obj = payload.get("object")
    if isinstance(obj, dict) and isinstance(obj.get("sha"), str):
        return obj["sha"]
    return None


def _parse_git_refs(data: bytes) -> tuple[str | None, dict[str, str]] | None:
    """``(head_sha, refs)`` from a git smart-HTTP ref advertisement.

    Parses the pkt-line stream ``GET .../info/refs?service=git-upload-pack``
    answers with. head_sha resolves through the ``symref=HEAD:<branch>``
    capability when present, else through a literal ``HEAD`` entry for a
    detached head. None when the payload is not an advertisement (a dumb
    server, an auth page) so the caller can fall back.
    """

    def pkt_lines(payload: bytes) -> list[str]:
        lines: list[str] = []
        position = 0
        while position + 4 <= len(payload):
            try:
                length = int(payload[position:position + 4], 16)
            except ValueError:
                return lines
            if length == 0:  # flush packet
                position += 4
                continue
            if length < 4 or position + length > len(payload):
                break
            lines.append(payload[position + 4:position + length].decode("utf-8", errors="replace"))
            position += length
        return lines

    lines = pkt_lines(data)
    if not lines or not lines[0].startswith("# service="):
        return None
    refs: dict[str, str] = {}
    symref_head: str | None = None
    for text in lines[1:]:
        if "\x00" in text:
            ref_line, capabilities = text.split("\x00", 1)
        else:
            ref_line, capabilities = text, ""
        parts = ref_line.strip().split(" ", 1)
        if len(parts) != 2:
            continue
        sha, name = parts[0].strip(), parts[1].strip()
        if name == "capabilities^{}":
            continue  # capability listing of an advertisement without refs
        refs[name] = sha
        for capability in capabilities.split(" "):
            if capability.startswith("symref=HEAD:"):
                symref_head = capability.split(":", 1)[1]
    if symref_head and symref_head in refs:
        return refs[symref_head], refs
    if "HEAD" in refs:
        return refs["HEAD"], refs
    # No HEAD entry and no symref: fall back to the branch the server called
    # first, which GitHub orders as the default branch.
    for name, sha in refs.items():
        if name.startswith("refs/heads/"):
            return sha, refs
    return None


def head_commit_git(owner: str, repo: str) -> str | None:
    """HEAD commit via the git smart-HTTP protocol, or None.

    This is the ``git ls-remote`` endpoint, not the REST API: it has no
    hourly quota to exhaust, which matters because a quota-limited API made
    every head look unknown and pushed ``--update`` into re-fetching the
    whole cache forever (the commits were never recorded, so nothing could
    ever compare equal).
    """
    url = f"https://github.com/{owner}/{repo}.git/info/refs?service=git-upload-pack"
    try:
        data = _http_get(url, timeout=30, retries=1)
    except (urllib.error.URLError, OSError, TimeoutError):
        return None
    parsed = _parse_git_refs(data)
    return parsed[0] if parsed else None


def head_commit_resolve(owner: str, repo: str) -> str | None:
    """HEAD commit of a repository: git protocol first, REST API fallback.

    The tarball URL follows ``HEAD``, so resolving HEAD directly is what a
    downloaded or compared cache entry actually contains, and the git
    protocol endpoint keeps working when the API quota is gone.
    """
    sha = head_commit_git(owner, repo)
    if sha:
        return sha
    branch = default_branch(owner, repo)
    if not branch:
        return None
    return head_commit(owner, repo, branch)


def default_branch(owner: str, repo: str) -> str | None:
    """The repository's default branch name, or None when unknown."""
    payload = _api_json(f"https://github.com/{owner}/{repo}")
    if payload is None:
        return None
    branch = payload.get("default_branch")
    return branch if isinstance(branch, str) and branch else None


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


def write_meta(
    directory: Path, request: RepoRequest, license_spdx: str | None,
    description: str | None, commit: str | None = None, branch: str | None = None,
) -> None:
    from datetime import UTC, datetime

    meta = {
        "url": request.url.rstrip("/"),
        "origin": request.origin,
        "fetched_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "license_spdx": license_spdx,
        "description": description,
        # Which upstream commit this tree is. --check and --update compare it
        # with the current head. None on a fetch whose API lookup failed, which
        # is treated as "unknown", never as "changed".
        "default_branch": branch,
        "commit": commit,
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


def fetch_repo(request: RepoRequest, force: bool = False) -> FetchResult:
    """Fetch one repository into the cache (or report it as cached).

    ``force`` re-fetches an entry that is already cached: ``--update`` needs
    this for repositories whose recorded commit moved (or was never
    recorded), and ``--refresh`` for a manual re-fetch. Without it the cache
    short-circuit below would return "cached" for exactly the entries the
    update flags exist to refresh, and ``--update`` would re-fetch nothing
    (which is what happened before this parameter existed).
    """
    directory = target_dir(request)
    if is_cached(request) and not force:
        return FetchResult(request, "cached")

    tarball_url = f"https://codeload.github.com/{request.owner}/{request.repo}/tar.gz/HEAD"
    try:
        archive = _http_get(tarball_url)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return FetchResult(request, "failed", str(exc))

    api = _api_json(request.url)
    license_spdx = None
    description = None
    branch = None
    if api:
        license_spdx = ((api.get("license") or {}).get("spdx_id")) or None
        description = api.get("description")
        branch = api.get("default_branch")
    # Record the commit the tarball is the head of, so a later --check can say
    # whether this cache entry is still current. The tarball URL follows
    # HEAD, so HEAD is what was just downloaded; the git-protocol resolver
    # keeps working when the REST API quota is exhausted.
    commit = head_commit_resolve(request.owner, request.repo)

    try:
        if directory.exists():
            shutil.rmtree(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        _extract_tarball(archive, directory)
        write_meta(directory, request, license_spdx, description, commit, branch)
    except (tarfile.TarError, OSError) as exc:
        return FetchResult(request, "failed", str(exc))
    detail = f"license={license_spdx or 'unknown'}"
    if commit:
        detail += f" commit={commit[:12]}"
    else:
        detail += " commit=unknown"
    return FetchResult(request, "fetched", detail)


def fetch_all(
    requests: list[RepoRequest], jobs: int = 8, force_names: frozenset[str] | None = None,
) -> list[FetchResult]:
    """Fetch *requests* in parallel, preserving input order in the result.

    Repositories whose ``owner/repo`` name is in *force_names* are re-fetched
    even when cached (see fetch_repo).
    """
    if not requests:
        return []
    forced = force_names or frozenset()

    def run(request: RepoRequest) -> FetchResult:
        return fetch_repo(request, force=request.name in forced)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
        results = list(pool.map(run, requests))
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
    """Print every cached repository with its recorded license and commit."""
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
            commit = str(meta.get("commit") or "unknown")[:12]
            print(
                f"  {owner_dir.name}/{repo_dir.name}: {license_spdx} "
                f"({meta.get('fetched_at', '?')}, commit {commit})"
            )
    print(f"{total} cached repositories, {missing_meta} without metadata.")
    return 0


# --------------------------------------------------------------------------- #
# Upstream check: which cached repositories moved
# --------------------------------------------------------------------------- #

# status values a check can report per repository.
CHECK_CURRENT = "current"
CHECK_STALE = "moved"
CHECK_UNTRACKED = "untracked"  # cached before commits were recorded
CHECK_UNKNOWN = "unknown"      # upstream head could not be read
CHECK_ABSENT = "absent"        # in the request list, not in the cache


@dataclass(frozen=True)
class CheckResult:
    """One repository's cache-vs-upstream comparison."""

    name: str
    status: str
    cached_commit: str
    head_commit: str


def _cached_commit(meta: dict | None, request: RepoRequest) -> str | None:
    """The commit recorded for a cache entry, or None when unrecorded."""
    if not meta:
        return None
    recorded = meta.get("commit")
    if isinstance(recorded, str) and recorded:
        return recorded
    # An entry fetched before commits were recorded. Say so rather than
    # treating the absence as a match.
    return None


def check_repo(request: RepoRequest) -> CheckResult:
    """Compare one cached repository against its upstream head."""
    name = f"{request.owner}/{request.repo}"
    meta = load_meta(target_dir(request))
    cached = _cached_commit(meta, request)
    if meta is None:
        return CheckResult(name, CHECK_ABSENT, "", "")
    head = head_commit_resolve(request.owner, request.repo)
    if head is None:
        # A failed check must never look like a change: reporting it as
        # current would hide a real update, reporting it as moved would
        # re-fetch on every network hiccup. Neither is useful, so say unknown.
        return CheckResult(name, CHECK_UNKNOWN, cached or "", "")
    if cached is None:
        return CheckResult(name, CHECK_UNTRACKED, "", head)
    status = CHECK_CURRENT if cached == head else CHECK_STALE
    return CheckResult(name, status, cached, head)


def check_cache(requests: list[RepoRequest], jobs: int = 8) -> list[CheckResult]:
    """Check every request in parallel, preserving input order."""
    if not requests:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
        return list(pool.map(check_repo, requests))


def report_check(results: list[CheckResult]) -> int:
    """Print the check table; return the number of repositories behind."""
    width = max((len(r.name) for r in results), default=10)
    print(f"{'repository':<{width}}  status      cached        upstream")
    for r in results:
        print(
            f"{r.name:<{width}}  {r.status:<11} {r.cached_commit[:12]:<12} {r.head_commit[:12]}"
        )
    behind = [r for r in results if r.status == CHECK_STALE]
    untracked = [r for r in results if r.status == CHECK_UNTRACKED]
    unknown = [r for r in results if r.status == CHECK_UNKNOWN]
    print()
    if behind:
        print(
            f"{len(behind)} repository(ies) moved upstream: "
            + ", ".join(r.name for r in behind)
        )
    elif untracked:
        print(
            f"No cached repository is known to be behind, but "
            f"{len(untracked)} cannot be compared: they were fetched before "
            "commits were recorded."
        )
    else:
        print("Every cached repository is at its upstream commit.")
    if untracked:
        print(
            f"{len(untracked)} cached before commits were recorded (not fetched from git): "
            + ", ".join(r.name for r in untracked)
            + ". Re-fetch with --update to start tracking them."
        )
    if unknown:
        print(
            f"{len(unknown)} upstream head(s) could not be read "
            f"(API quota or network): " + ", ".join(r.name for r in unknown)
            + ". Treated as up to date, not as changed."
        )
    return len(behind) + len(untracked)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", action="append", default=[], help="Extra repository URL. Repeatable.")
    parser.add_argument("--from-manifest", type=Path, help="File with one repository URL per line.")
    parser.add_argument("--refresh", choices=["core", "all"], help="Re-fetch even when cached.")
    parser.add_argument("--check", action="store_true", help="Report which cached repositories moved upstream, then exit.")
    parser.add_argument("--update", action="store_true", help="Re-fetch only the repositories whose commit moved.")
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

    if args.check or args.update:
        checked = check_cache(requests, jobs=args.jobs)
        behind = report_check(checked)
        if args.check:
            return 1 if behind else 0
        stale = {r.name for r in checked if r.status in (CHECK_STALE, CHECK_UNTRACKED)}
        if not stale:
            print("Nothing to re-fetch.")
            return 0
        requests = [r for r in requests if r.name in stale]
        if any(r.status == CHECK_UNTRACKED for r in checked):
            # An untracked entry has no commit to compare, so it is not known
            # to be current. Re-fetching it is the only way to start tracking
            # it; say so, because it is the one case that touches a cache that
            # may be perfectly fine.
            logger.warning(
                "Re-fetching %d cache entr(ies) with no recorded commit; "
                "their content may be unchanged.",
                sum(1 for r in checked if r.status == CHECK_UNTRACKED),
            )
        # fetch_repo's cache short-circuit must not swallow these: they are
        # exactly the entries the check said need a re-fetch.
        force_names = frozenset(stale)
    elif args.refresh:
        # Same for a manual refresh: the filtered request list names what to
        # re-fetch, so every one of them bypasses the cache.
        force_names = frozenset(r.name for r in requests)
    else:
        force_names = frozenset()

    logger.info("Fetching %d repositories (%d jobs)", len(requests), args.jobs)
    results = fetch_all(requests, jobs=args.jobs, force_names=force_names)

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
    refetched = sum(1 for r in results if r.status == "fetched")
    if args.update:
        print(
            f"Re-fetched {refetched} of {len(results)} repositories that moved upstream. "
            "Rebuild the dataset to pick the new sources up ('make update-sources' does both)."
        )
    else:
        print(f"Core sources fetched; {len(results) - len(failures)}/{len(results)} repositories are in the cache at {CACHE_DIR}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

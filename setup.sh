#!/usr/bin/env bash
# setup.sh - One-shot environment bootstrap for q3as.
#
# 1. Fetches the data/guidance repositories into the local archive cache
#    (data/raw_repos/, gitignored) via scripts/fetch_repos.py: HTTP
#    tarballs, no git, cached across runs. Sources stay outside this repo
#    to keep licensing provenance clean.
# 2. Copies .env.dev to .env for the Hugging Face token (never overwrites).
# 3. Registers the vendored Alire index when the installed `alr` is older
#    than the current release, so modern binary crates (gnatprove 16.x,
#    gnatformat 26.x) stay installable on old distro packages (Debian).
# 4. Installs Python dependencies with uv sync.
# 5. Extracts local Python headers for Triton (no sudo), when needed.
#
# Usage:  ./setup.sh          # everything
#         ./setup.sh --repos  # only the sibling repositories
#         ./setup.sh --deps   # only .env, alire index, uv sync, headers

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$ROOT_DIR")"
ONLY="${1:-all}"
# Vendored Alire index: mirrors the crates q3as needs from the current
# community index (branch stable-1.4.0) so an older `alr` (which can only
# read older index branches, e.g. stable-1.2.1 on Debian) still resolves
# them. Refreshed from upstream by setup_alire_index when online.
LOCAL_INDEX_DIR="$ROOT_DIR/q3as-local-index"
LOCAL_INDEX_NAME="q3aslocal"
# Crates mirrored from alire-index stable-1.4.0: crate_name|letter|version
LOCAL_INDEX_CRATES=(
  "gnatformat_bin|gnatformat_bin|26.0.0"
  "gnatprove|gnatprove|16.1.0"
)
INDEX_BRANCH="stable-1.4.0"
# Fallback when the GitHub API is unreachable: latest alire release tag.
FALLBACK_LATEST_ALR="2.0.1"

# Sibling repositories are fetched by scripts/fetch_repos.py (archive
# cache); see the CORE_REPOS list there and the RobertBoettcherSF
# Ada-Algorithms monorepo.

fetch_repos() {
  echo "==> Fetching source repositories into data/raw_repos (archive cache)"
  if python3 "$ROOT_DIR/scripts/fetch_repos.py"; then
    echo "    cache ready: $ROOT_DIR/data/raw_repos"
  else
    echo "Some repositories failed to fetch. The dataset build skips missing" >&2
    echo "sources with a warning, but the full pipeline expects all of them." >&2
    exit 1
  fi
}

setup_env() {
  echo "==> Hugging Face credentials"
  if [ -f "$ROOT_DIR/.env" ]; then
    echo "    .env already exists - keeping it (never overwrite)."
  else
    cp "$ROOT_DIR/.env.dev" "$ROOT_DIR/.env"
    echo "    Created .env from .env.dev."
    echo "    Edit .env and replace your_huggingface_token_here with your token."
  fi
}

sync_deps() {
  echo "==> Installing Python dependencies (uv sync)"
  (cd "$ROOT_DIR" && UV_LINK_MODE=copy uv sync)
}

alr_version() {
  alr version 2>/dev/null | awk '/^alr version:/ {print $3}'
}

latest_alr_release() {
  local tag
  tag="$(curl -sf https://api.github.com/repos/alire-project/alire/releases/latest \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tag_name",""))' 2>/dev/null)"
  # Tags look like v2.0.1.
  echo "${tag#v}"
}

version_lt() {  # version_lt A B -> true when A < B (semver-ish)
  [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" != "$2" ]
}

mirror_index_crate() {  # mirror_index_crate <crate> <dir-name> <version>
  local crate="$1" dir="$2" ver="$3"
  local url="https://raw.githubusercontent.com/alire-project/alire-index/${INDEX_BRANCH}/index/gn/${dir}/${crate}-${ver}.toml"
  local dest="$LOCAL_INDEX_DIR/index/gn/${dir}"
  mkdir -p "$dest"
  if curl -sf -o "$dest/${crate}-${ver}.toml" "$url"; then
    echo "    mirrored  ${crate}-${ver} (from alire-index ${INDEX_BRANCH})"
  else
    if [ -f "$dest/${crate}-${ver}.toml" ]; then
      echo "    vendored  ${crate}-${ver} (offline, using committed copy)"
    else
      echo "    FAILED    ${crate}-${ver} ($url)" >&2
      return 1
    fi
  fi
}

setup_alire_index() {
  if ! command -v alr >/dev/null 2>&1; then
    echo "==> alr not found - skipping Alire index setup (install Alire first)"
    return
  fi
  local installed latest
  installed="$(alr_version)"
  latest="$(latest_alr_release)"
  latest="${latest:-$FALLBACK_LATEST_ALR}"
  if [ -z "$installed" ]; then
    echo "==> WARNING: cannot determine alr version - assuming old alr" >&2
    installed="0.0.0"
  fi
  if ! version_lt "$installed" "$latest"; then
    echo "==> alr $installed is current (latest: $latest) - community index is enough"
    return
  fi

  echo "==> alr $installed is older than latest ($latest) - registering vendored index"
  echo "    (old alr cannot read new index branches; the local mirror carries"
  echo "     the modern binary crates q3as needs, e.g. gnatprove 16.x)"

  # Refresh the mirror from upstream; keep the committed copies offline.
  local failed=0
  local index_toml="$LOCAL_INDEX_DIR/index/index.toml"
  if [ ! -f "$index_toml" ]; then
    mkdir -p "$LOCAL_INDEX_DIR/index"
    echo "version = \"1.2.1\"" > "$index_toml"
  fi
  for entry in "${LOCAL_INDEX_CRATES[@]}"; do
    IFS='|' read -r crate dir ver <<<"$entry"
    mirror_index_crate "$crate" "$dir" "$ver" || failed=1
  done
  if [ "$failed" -ne 0 ]; then
    echo "    WARNING: some crates could not be mirrored; resolve may fail" >&2
  fi

  # Register ahead of the community index so its versions win.
  if alr index --list 2>/dev/null | grep -q "^${LOCAL_INDEX_NAME}"; then
    echo "    index ${LOCAL_INDEX_NAME} already registered"
  else
    alr index --add="file://$LOCAL_INDEX_DIR" --name="$LOCAL_INDEX_NAME" \
      --before=community >/dev/null
    echo "    registered ${LOCAL_INDEX_NAME} ahead of community"
  fi
  alr index --update-all >/dev/null 2>&1 || true
}

# Local Python dev headers (no sudo): Triton needs Python.h to compile its
# CUDA driver shim. Mirrors the Makefile's PY_HDR_ROOT logic.
setup_python_headers() {
  local py_major
  py_major="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  local hdr_root="$HOME/.cache/q3as-python-headers/usr/include"
  if [ -f "$hdr_root/python${py_major}/Python.h" ]; then
    echo "==> Python headers already extracted at $hdr_root"
    return
  fi
  if python3 -c "import sysconfig,os; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')) else 1)" 2>/dev/null; then
    echo "==> System Python headers present - no local extraction needed"
    return
  fi
  echo "==> Python.h not found; extracting python${py_major}-dev headers locally"
  local deb_dir="$HOME/.cache/q3as-python-headers"
  mkdir -p "$deb_dir"
  if command -v apt-get >/dev/null 2>&1; then
    apt-get download "python${py_major}-dev" 2>/dev/null || true
    for deb in python3*-dev*.deb; do
      [ -e "$deb" ] || continue
      dpkg-deb -x "$deb" "$deb_dir"
    done
    # Dependencies of the dev package (python3.13-dev -> python3-dev chain)
    apt-get download "python${py_major}" 2>/dev/null || true
    for deb in python3.*_*_amd64.deb; do
      [ -e "$deb" ] || continue
      dpkg-deb -x "$deb" "$deb_dir"
    done
    rm -f ./*.deb
  fi
  if [ -f "$hdr_root/python${py_major}/Python.h" ]; then
    echo "    Headers extracted. The Makefile picks them up automatically."
  else
    echo "    WARNING: Python.h still not found. Triton may fail to compile." >&2
    echo "    Install python${py_major}-dev manually if training fails." >&2
  fi
}

case "$ONLY" in
  --repos) fetch_repos ;;
  --deps)  setup_env; setup_alire_index; sync_deps; setup_python_headers ;;
  all)     fetch_repos; setup_env; setup_alire_index; sync_deps; setup_python_headers ;;
  *)
    echo "Usage: $0 [--repos|--deps]" >&2
    exit 2
    ;;
esac

echo "Setup complete. Next steps:"
echo "  1. make check-model    (download the base model, needs HF token)"
echo "  2. make build-dataset  (uses the cached sources from data/raw_repos)"
echo "  3. make train"

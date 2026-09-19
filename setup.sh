#!/usr/bin/env bash
# setup.sh - One-shot environment bootstrap for q3as.
#
# 1. Shallow-clones the sibling data/guidance repositories into the parent
#    directory (the model only reads their code/docs as training data, so
#    they stay outside this repo to keep licensing provenance clean).
# 2. Copies .env.dev to .env for the Hugging Face token (never overwrites).
# 3. Installs Python dependencies with uv sync.
# 4. Extracts local Python headers for Triton (no sudo), when needed.
#
# Usage:  ./setup.sh          # everything
#         ./setup.sh --repos  # only the sibling repositories
#         ./setup.sh --deps   # only .env, uv sync, python headers

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$ROOT_DIR")"
ONLY="${1:-all}"

# Sibling repositories: URL|directory. Data sources first, then guidance.
REPOS=(
  "https://github.com/bladeacer/adacovex.git|adacovex"
  "https://github.com/bladeacer/Ada_CRDT.git|Ada_CRDT"
  "https://github.com/ViMoBr/Ada-83-TLALOC.git|Ada-83-TLALOC"
  "https://github.com/AdaCore/ada-eval.git|ada-eval"
  "https://github.com/AdaCore/learn.git|learn"
  "https://github.com/agent-sh/ada-spark.git|ada-spark"
  "https://github.com/AminBlg/SimpleEnglish.git|SimpleEnglish"
  "https://github.com/AdaCore/skills.git|skills"
)

clone_repos() {
  echo "==> Shallow-cloning sibling repositories into $PARENT_DIR"
  local missing=0
  for entry in "${REPOS[@]}"; do
    local url="${entry%%|*}"
    local dir="${entry##*|}"
    local target="$PARENT_DIR/$dir"
    if [ -d "$target/.git" ]; then
      echo "    ok        $dir (already present)"
      continue
    fi
    if git clone --depth 1 "$url" "$target"; then
      echo "    cloned    $dir"
    else
      echo "    FAILED    $dir ($url)" >&2
      missing=1
    fi
  done
  if [ "$missing" -ne 0 ]; then
    echo "Some repositories failed to clone. The dataset build skips missing" >&2
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
  --repos) clone_repos ;;
  --deps)  setup_env; sync_deps; setup_python_headers ;;
  all)     clone_repos; setup_env; sync_deps; setup_python_headers ;;
  *)
    echo "Usage: $0 [--repos|--deps]" >&2
    exit 2
    ;;
esac

echo ""
echo "Setup complete. Next steps:"
echo "  1. make check-model    (download the base model, needs HF token)"
echo "  2. make build-dataset  (uses the sibling repos cloned above)"
echo "  3. make train"

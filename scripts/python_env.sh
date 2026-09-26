#!/usr/bin/env bash
# python_env.sh - CPATH for Triton's CUDA driver shim, resolved from the
# running interpreter (no sudo).
#
# Triton compiles a small C extension at install time and needs Python.h to
# do it. Distributions usually do not ship the headers, so setup.sh unpacks
# the pythonX.Y-dev packages into ~/.cache/q3as-python-headers (gitignored,
# no root). The cache is laid out per interpreter version, so the include
# path depends on which Python actually runs the training.
#
# This script is the single place that resolves it. It is called by the
# Makefile (train target) and by scripts/run_cap_experiment.sh, both of
# which train through `uv run`, so the venv interpreter is the reference; a
# system python3 is the fallback when there is no venv yet.
#
# It prints the CPATH value on stdout, and nothing when the headers are not
# available (the caller then leaves CPATH alone and the compiler uses its
# default search path). A missing header is reported on stderr, because the
# resulting Triton failure is otherwise a confusing compile error deep in a
# wheel build.
#
# Usage:   CPATH="$(bash scripts/python_env.sh)"
#          PY_HDR_ROOT=/somewhere bash scripts/python_env.sh
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Same default as the Makefile and setup.sh, overridable for tests.
PY_HDR_ROOT="${PY_HDR_ROOT:-${HOME}/.cache/q3as-python-headers/usr/include}"

# The interpreter that will import torch: the project venv when it exists.
# Q3AS_PYTHON overrides it (tests use it to fake another interpreter version).
PYTHON="${Q3AS_PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
    PYTHON="${PROJECT_ROOT}/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

if ! command -v "$PYTHON" >/dev/null 2>&1 && [ ! -x "$PYTHON" ]; then
  echo "python_env: no usable Python interpreter ($PYTHON)" >&2
  exit 1
fi

# Include directory name for this interpreter, e.g. python3.13.
PY_TAG="$("$PYTHON" -c 'import sys; print("python%d.%d" % sys.version_info[:2])')" || {
  echo "python_env: cannot determine the Python version from $PYTHON" >&2
  exit 1
}

# The distribution puts versioned headers under the multiarch include dir on
# Debian derivatives; both are on the path for older layouts too.
PY_HDRS="${PY_HDR_ROOT}/${PY_TAG}:${PY_HDR_ROOT}/x86_64-linux-gnu:${PY_HDR_ROOT}"

if [ -f "${PY_HDR_ROOT}/${PY_TAG}/Python.h" ]; then
  printf '%s\n' "$PY_HDRS"
  exit 0
fi

# Headers already provided by the interpreter's own sysconfig need no cache
# and no CPATH; the compiler finds them on its default search path.
if "$PYTHON" -c 'import os, sysconfig; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)' 2>/dev/null; then
  exit 0
fi

echo "python_env: ${PY_TAG} headers not found under ${PY_HDR_ROOT}." >&2
echo "python_env: run './setup.sh --deps' to extract them (no sudo needed)." >&2
echo "python_env: if training fails in Triton's driver build, this is why." >&2
exit 0

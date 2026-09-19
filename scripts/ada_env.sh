#!/usr/bin/env bash
# ada_env.sh - Run a command inside the Alire-managed Ada toolchain environment.
#
# q3as routes every Ada toolchain invocation (gnat, gprbuild, gnatprove,
# gnatdoc, gnatformat, gprclean, gprls) through Alire instead of calling
# system binaries directly: the command runs with the Alire environment on
# PATH, which is what `alr exec` provides.
#
# alr 1.2.1 has no --manifest option and reads alire.toml from the crate
# root, so this wrapper keeps a throwaway workspace at .alire-dev/ holding a
# COPY of alire-dev.toml. The real manifests are never edited or swapped,
# and .alire-dev/ is gitignored.
#
# Usage:   . scripts/ada_env.sh <command> [args...]
#          scripts/ada_env.sh gnatprove --version
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEV_WS="$PROJECT_ROOT/.alire-dev"
DEV_MANIFEST="$PROJECT_ROOT/alire-dev.toml"

if [ ! -f "$DEV_MANIFEST" ]; then
    echo "ada_env: alire-dev.toml not found at $DEV_MANIFEST" >&2
    exit 2
fi

mkdir -p "$DEV_WS"
# Refresh the copy only when the source changes, so `alr` does not see a
# new manifest file (and re-resolve) on every call.
if ! cmp -s "$DEV_MANIFEST" "$DEV_WS/alire.toml"; then
    cp "$DEV_MANIFEST" "$DEV_WS/alire.toml"
fi

cd "$DEV_WS"
exec alr exec -- "$@"

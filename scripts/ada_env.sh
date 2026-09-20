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
# The command runs in the CALLER's working directory (not .alire-dev/), so
# relative paths such as `src/foo.adb` or `-Pmain.gpr` resolve the way the
# caller expects. Alire only needs its workspace to exist somewhere; the
# tools it exports do not care where it lives.
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

# `alr exec` must run inside the dev workspace (alr reads the manifest from
# its cwd), but the user's command must run in THEIR cwd. Run `alr exec`
# from .alire-dev to export the toolchain environment, then re-exec the
# command back in the caller's directory with that environment.
#
# Implementation: printenv PATH from inside `alr exec` gives us the managed
# PATH; prefixing the caller's PATH with it re-creates the environment
# without staying in .alire-dev. Falling back to plain `alr exec --` (in
# .alire-dev) keeps the old behavior when the PATH cannot be captured.
CALLER_DIR="$PWD"
ALIRE_PATH="$(
    cd "$DEV_WS" && alr exec -- printenv PATH 2>/dev/null | tail -n 1
)"

if [ -n "$ALIRE_PATH" ] && [ "$ALIRE_PATH" != "$PATH" ]; then
    cd "$CALLER_DIR"
    PATH="$ALIRE_PATH" exec "$@"
fi

cd "$DEV_WS"
exec alr exec -- "$@"

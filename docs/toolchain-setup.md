# Toolchain Setup

How q3as builds and proves Ada through Alire, and how setup keeps working
on distributions that ship an outdated `alr`.

## The rule: no system Ada tools

q3as resolves every Ada tool through the Alire environment rather than
picking a binary off `PATH`:

- Shell: `scripts/ada_env.sh <cmd>`, a thin wrapper around `alr exec`.
- Python: `from alire_env import find_tool`, then pass
  `env=alire_env_path()` to `subprocess.run`.

The Python side is what the pipeline actually uses: the eval scripts, the
defect validator, and the contract generator all resolve their binaries
through `alire_env`. The shell wrapper exists for interactive use and for
`make ada-env`, which prints the managed `PATH` for debugging.

The dev toolchain (gnatprove, gnatformat) is declared in
[`alire-dev.toml`](../alire-dev.toml); [`alire.toml`](../alire.toml) stays a
clean publishing manifest. Fetch it with `make prove`.

The AST parser has its own, separate toolchain:
[`alire-ast.toml`](../alire-ast.toml) resolves `libadalang`, fetched with `make
ast-deps`. It is kept apart from [`alire-dev.toml`](../alire-dev.toml) on
purpose, because the prover and the dataset parser do not need each other and
the libadalang chain is a large source build.

## The outdated-alr problem, and the vendored index

`alr` can only read index branches at or below its own version. On a
distribution that ships, say, `alr 1.2.1` (Debian), the current community
index (branch `stable-1.4.0`) is invisible, so modern binary crates -
notably `gnatprove` 16.x and `gnatformat_bin` 26.x - fail to resolve even
though the crates exist upstream.

[`setup.sh`](../setup.sh) handles this automatically:

1. It compares the installed `alr` version with the latest release.
2. If `alr` is outdated, it refreshes the **vendored local index**
   ([`q3as-local-index/`](../q3as-local-index), registered under the name
   `q3aslocal`) from the upstream `stable-1.4.0` branch when online; the
   committed copies work offline.
3. It registers the local index **ahead of the community index**
   (`alr index --add ... --before=community`), so `q3aslocal` versions win
   resolution.

The vendored index carries three crates and one patched entry:

| Crate | Version | Why |
|---|---|---|
| `gnatprove` | 16.1.0 | SPARK proof (PROVE eval, defect validation) |
| `gnatformat_bin` | 26.0.0 | Formatting in the eval pipeline |
| `libadalang` | 24.0.0 | Ada AST extraction for the dataset parser |
| `gnatcoll_gmp` | 24.0.0 | patched, see below |

`gnatcoll_gmp` is committed rather than mirrored, because it is edited: the
upstream entry declares `libgmp` as a system external, which makes `alr` shell
out to `sudo apt-get install libgmp-dev`. q3as never installs system packages,
so the dependency edge is removed.
[`scripts/build_libadalang.py`](../scripts/build_libadalang.py) supplies the
headers and library from the user cache instead.

The rest of the `libadalang` chain (`gnatcoll`, `gnatcoll_iconv`, `libgpr2`,
`langkit_support`, `gnat`, ...) resolves from the community index on
`stable-1.2.1`, so only `libadalang` itself is vendored: it is what pins
the version, and pinning it locally keeps the build reproducible whatever
index branch happens to be checked out.

Because `alr 1.2.1` has no `--manifest` option, `make prove` copies
[`alire-dev.toml`](../alire-dev.toml) into the gitignored `.alire-dev/`
workspace and resolves there, and `make ast-deps` does the same with
[`alire-ast.toml`](../alire-ast.toml) in `.alire-ast/`. The real manifests are
never modified by tooling.

If your `alr` is current, the local index is not registered and resolution
uses the community index directly; the vendored entries simply stay
unused.

## The libadalang AST toolchain

[`data/processing_scripts/parse_ada_ast.py`](../data/processing_scripts/parse_ada_ast.py)
prefers libadalang over its structural scanner. The bindings are only half a
library, and the two halves come from different places:

- The pure-Python ctypes wrapper is the `ast` dependency group, installed
  from the official release archive by `uv sync`.
- The shared `libadalang.so` that the wrapper `dlopen`s is not packaged
  anywhere, so `make ast-deps` builds it.

The Alire `libadalang` crate ships a **static** library, so
[`scripts/build_libadalang.py`](../scripts/build_libadalang.py) drives
[`ast.gpr`](../ast.gpr) twice inside the Alire environment: once with
`LIBRARY_TYPE=static-pic` to get position independent archives for the whole
stack, then again with `LIBADALANG_LIBRARY_TYPE=relocatable` to relink
libadalang alone as `libadalang.so`. The second pass reuses the first pass's
objects, so only the link is repeated. The script copies the result next to the
installed wrapper and verifies the import.

Two details the build has to work around:

- `libgpr2` is a C project whose `.build/<mode>` tree is never created for
  us, and gprbuild refuses to start until it exists, so the script runs a
  directory-creating pass first.
- The distribution's static `libgmp.a` is not built with `-fPIC` and the
  linker rejects it inside a shared object, so the build is pointed at the
  shared GMP instead.

The build is optional. Without it the parser logs
`libadalang available: False` and uses its structural scanner, which is
what the dataset has always used. `make ast-deps` is a no-op once the
library is installed; pass `FORCE=1` to rebuild.

## Verifying the toolchain

```bash
make prove                                  # sync the dev toolchain
make ast-deps                               # build the AST parser's libadalang
uv run python scripts/alire_env.py          # report where each tool resolves
```

[`scripts/alire_env.py`](../scripts/alire_env.py) exits non-zero when a tool the
pipeline invokes is missing, so it doubles as the verification step; `gnatdoc`
is reported but never fails it, because [`alire-dev.toml`](../alire-dev.toml)
does not depend on it.

The eval pipeline and the defect validator resolve every tool through
`alire_env`, which searches the Alire prefix first and only then the system
`PATH`. A tool found on the system is still used - that is what covers Alire
"external" installs such as a distribution-provided gnatprove - but it is
reported once on stderr so a mixed environment stays visible. A tool that is
missing entirely raises, and the callers say which command installs it.

Navigation: [project README](../README.md) · [docs index](README.md) · [changelog index](changelogs/index.md) · [results index](results/README.md)

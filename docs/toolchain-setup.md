# Toolchain Setup

How q3as builds and proves Ada through Alire, and how setup keeps working
on distributions that ship an outdated `alr`.

## The rule: no system Ada tools

q3as never calls system `gnat`, `gprbuild`, or `gnatprove` directly. All
Ada tool invocations go through the Alire environment:

- Shell: `scripts/ada_env.sh <cmd>` (used by all Makefile targets).
- Python: `from alire_env import find_tool`, then pass
  `env=alire_env_path()` to `subprocess.run`.

The dev toolchain (gnatprove, gnatformat) is declared in
`alire-dev.toml`; `alire.toml` stays a clean publishing manifest. Fetch it
with `make prove`.

## The outdated-alr problem, and the vendored index

`alr` can only read index branches at or below its own version. On a
distribution that ships, say, `alr 1.2.1` (Debian), the current community
index (branch `stable-1.4.0`) is invisible, so modern binary crates -
notably `gnatprove` 16.x and `gnatformat_bin` 26.x - fail to resolve even
though the crates exist upstream.

`setup.sh` handles this automatically:

1. It compares the installed `alr` version with the latest release.
2. If `alr` is outdated, it refreshes the **vendored local index**
   (`q3as-local-index/`, registered under the name `q3aslocal`) from the
   upstream `stable-1.4.0` branch when online; the committed copies work
   offline.
3. It registers the local index **ahead of the community index**
   (`alr index --add ... --before=community`), so `q3aslocal` versions win
   resolution.

The vendored index currently mirrors:

| Crate | Version | Why |
|---|---|---|
| `gnatprove` | 16.1.0 | SPARK proof (PROVE eval, defect validation) |
| `gnatformat_bin` | 26.0.0 | Formatting in the eval pipeline |

Because `alr 1.2.1` has no `--manifest` option, `make prove` copies
`alire-dev.toml` into the gitignored `.alire-dev/` workspace and resolves
there. The real manifests are never modified by tooling.

If your `alr` is current, the local index is not registered and resolution
uses the community index directly; the vendored entries simply stay
unused.

## Verifying the toolchain

```bash
make prove                                  # sync the dev toolchain
scripts/ada_env.sh gnatprove --version      # runs inside the Alire env
uv run python scripts/alire_env.py          # resolves managed binaries
```

The eval pipeline and the defect validator resolve every tool through this
environment and refuse to run against a missing managed tool.

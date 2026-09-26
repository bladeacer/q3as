# Changelog

Notable changes to q3as, one file per version, newest first. Add a file here
when a release changes behaviour, metrics, or a documented claim; not for
every commit.

The version lives in [`alire.toml`](../../alire.toml) and is mirrored into
[`alire-dev.toml`](../../alire-dev.toml),
[`alire-ast.toml`](../../alire-ast.toml), and
[`pyproject.toml`](../../pyproject.toml); `make bump-version` keeps all four in
step, and two tests in
[`tests/test_reporting.py`](../../tests/test_reporting.py) fail if they ever
disagree. Each release's evaluation summary is written to
`docs/results/result-vX.Y.Z.md` by `make eval-report`, using the same version
string, and is listed in the [results index](../results/README.md).

## All versions

| Version | What changed | Results |
|---|---|---|
| [0.3.0](v0.3.0.md) | Numbers now measure the models; dataset and evaluation cover the whole benchmark | [result-v0.3.0.md](../results/result-v0.3.0.md) |
| [0.2.0](v0.2.0.md) | Ada-Algorithms monorepo migration, 8 GB OOM fixes, AST structural-cap experiments | _withdrawn_ |
| [0.1.0](v0.1.0.md) | First versioned evaluation, 19 samples | [result-v0.1.0.md](../results/result-v0.1.0.md) |

_Version order: newest first. A version without a results file had its run
withdrawn; the reason is stated in its entry._

## How to read these files

- Each entry states what changed and why it mattered, and it links the
  evaluation run behind the numbers it quotes.
- Numbers in a versioned changelog are the numbers that version measured.
  When a later version fixes how those numbers are produced, it says so; do
  not compare a figure from one version against the same figure from another
  unless the later entry says the measurement is unaffected.
- Source-level changes in the repo history are not recorded here. This is the
  release log, not a commit log.

## Related

- [Docs index](../README.md) - the rest of the documentation.
- [Results index](../results/README.md) - per-version eval summaries and the
  cross-version comparison table.
- [Project credits](../../README.md) - upstream sources and their licenses.

Navigation: [project README](../../README.md) · [docs index](../README.md) · [changelog index](index.md) · [results index](../results/README.md)

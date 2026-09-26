# q3as documentation

End-user and developer documentation for the pipeline: what each step does,
where the data comes from, how training is configured, and what the
evaluation numbers mean. Start at the page that matches your question, or
run `make help` for the command list.

## Reference

| Page | Contents |
|---|---|
| [Architecture](architecture.md) | Pipeline overview and module map |
| [Datasets and training](datasets-and-training.md) | Turn kinds, defect families, splits, training configuration |
| [Data provenance](data-provenance.md) | Data sources, licenses, eval-integrity guard |
| [Toolchain setup](toolchain-setup.md) | Alire management, vendored index for outdated `alr` |
| [Evaluation](evaluation.md) | Benchmark, metrics, interpretation, running |

## Versioned artifacts

Both directories hold one file per released version, and each has an index that
links every version it contains. The version string comes from
[`alire.toml`](../alire.toml), so the three directories cannot drift apart.

| Index | Contents |
|---|---|
| [Changelog index](changelogs/index.md) | What changed in each version, one file per release |
| [Results index](results/README.md) | Per-version eval summaries and the cross-version comparison table |

A release adds one file to each directory: `docs/changelogs/vX.Y.Z.md` for
what changed, `docs/results/result-vX.Y.Z.md` for what it measured. Link the
results file from the changelog entry when the run describes the whole
benchmark, and say in prose why it is absent when the run was withdrawn.

## Reading order

1. [Architecture](architecture.md) for the shape of the pipeline.
2. [Data provenance](data-provenance.md) before adding any data source, and
   [datasets and training](datasets-and-training.md) before changing the
   dataset: the first says what may be ingested, the second what the builder
   does with it.
3. [Toolchain setup](toolchain-setup.md) if an Ada tool is missing or the
   installed `alr` is too old for the pinned crates.
4. [Evaluation](evaluation.md) and the [results index](results/README.md)
   before quoting a number, so the metric and its version travel together.

## Keeping these pages current

- Behaviour changes update the page that describes the behaviour, in the same
  change, not in a later cleanup.
- New data sources update the [provenance](data-provenance.md) source table
  and its license notes, then `make check-integrity` must exit 0.
- A release updates [changelogs/](changelogs/index.md) with what changed and
  [`docs/results/`](results) with what it measured.
- Every repo path named in prose is a link, so a reader on GitHub can open
  the file a sentence is about. Only git-tracked paths are linked: generated
  output (`outputs/`, `data/processed/`, `models/`) stays a code span, because
  a link to it would 404.
- `make lint` checks every relative markdown link and anchor on these pages, so
  a moved page fails the build until its links are retargeted. It also rejects
  a link to a path git ignores (it resolves on a built machine and 404s for a
  reader who clones), and it checks that the file tree in `AGENTS.md` is
  current.

Navigation: [project README](../README.md) · [docs index](README.md) · [changelog index](changelogs/index.md) · [results index](results/README.md)

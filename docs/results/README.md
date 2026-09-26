# Evaluation Results

Per-version summaries of every `make eval` / `make eval-pipeline` run. Metrics JSON lives beside each markdown file. Regenerate with `make eval-report` after a run; the version comes from `alire.toml`.

## All versions

| Version | Captured | Build (base → fine-tuned) | Test (base → fine-tuned) | Report |
|---|---|---|---|---|
| v0.2.0 | 2026-09-26 05:51 UTC | 4 (57.1%) → 4 (57.1%) | 2 (28.6%) → 2 (28.6%) | [result-v0.2.0.md](result-v0.2.0.md) |
| v0.1.0 | 2026-09-21 12:52 UTC | 7 (36.8%) → 16 (84.2%) | 4 (21.1%) → 11 (57.9%) | [result-v0.1.0.md](result-v0.1.0.md) |

## Last 2 versions compared

| Metric | v0.2.0 | v0.1.0 |
|---|---|---|
| Fine-tuned build % | 57.1% | 84.2% |
| Fine-tuned test % | 28.6% | 57.9% |
| Base build % | 57.1% | 36.8% |
| Proved samples (FT) | 0 | 0 |
| Test loss (FT) | n/a | n/a |
| Training trend | sparse | sparse |

_Version order: newest first._

Navigation: [project README](../../README.md) · [architecture](../architecture.md) · [evaluation guide](../evaluation.md)

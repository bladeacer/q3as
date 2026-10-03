# Evaluation Results

Per-version summaries of every `make eval` / `make eval-pipeline` run. Metrics JSON lives beside each markdown file. Regenerate with `make eval-report` after a run; the version comes from [`alire.toml`](../../alire.toml).

## All versions

| Version | Captured | Build (base → fine-tuned) | Test (base → fine-tuned) | Report |
|---|---|---|---|---|
| v0.8.0 | 2026-10-03 15:27 UTC | 7 (36.8%) → 15 (78.9%) | 5 (26.3%) → 10 (52.6%) | [result-v0.8.0.md](result-v0.8.0.md) |
| v0.7.0 | 2026-10-03 05:59 UTC | 7 (36.8%) → 10 (52.6%) | 5 (26.3%) → 8 (42.1%) | [result-v0.7.0.md](result-v0.7.0.md) |
| v0.4.1 | 2026-09-28 14:44 UTC | 8 (42.1%) → 16 (84.2%) | 5 (26.3%) → 11 (57.9%) | [result-v0.4.1.md](result-v0.4.1.md) |
| v0.3.0 | 2026-09-26 06:34 UTC | 8 (42.1%) → 15 (78.9%) | 5 (26.3%) → 11 (57.9%) | [result-v0.3.0.md](result-v0.3.0.md) |
| v0.1.0 | 2026-09-21 12:52 UTC | 7 (36.8%) → 16 (84.2%) | 4 (21.1%) → 11 (57.9%) | [result-v0.1.0.md](result-v0.1.0.md) |

## Last 3 versions compared

| Metric | v0.8.0 | v0.7.0 | v0.4.1 |
|---|---|---|---|
| Fine-tuned build % | 78.9% | 52.6% | 84.2% |
| Fine-tuned test % | 52.6% | 42.1% | 57.9% |
| Base build % | 36.8% | 36.8% | 42.1% |
| Proved samples (FT) | 1 | 0 | 0 |
| Test loss (FT) | 0.2118 | 0.2024 | 0.2242 |
| Training trend | healthy | healthy | healthy |

_Version order: newest first._

Navigation: [project README](../../README.md) · [docs index](../README.md) · [changelog index](../changelogs/index.md) · [results index](README.md)

# Results v0.1.0

Eval run captured 2026-09-21 12:52 UTC. Data: [`result-data-v0.1.0.json`](result-data-v0.1.0.json).

## Headline (ada-eval)

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| Samples | 19 | 19 | |
| Build | 7/19 (36.8%) | 16/19 (84.2%) | 47.4 pts |
| Unit tests | 4/19 (21.1%) | 11/19 (57.9%) | 36.8 pts |
| SPARK proved | 0 | 0 | |
| Prove errors | 13 | 5 | |

Fine-tuned model proof blockers (check kinds left unproved):

- `VC_OVERFLOW_CHECK` x8
- `VC_POSTCONDITION` x3
- `UNINITIALIZED` x3
- `VC_RAISE` x2
- `DEPENDS_MISSING` x2

## Per dataset (fine-tuned)

| Dataset | Samples | Build | Test |
|---|---|---|---|
| spark_spark_custom | 2 | 2 | 2 |
| spark_spark_human_eval_silver | 4 | 2 | 0 |
| spark_spark_learn | 13 | 12 | 9 |

## comparison_report.txt excerpt

```text
======================================================================
Q3AS EVALUATION REPORT: BASE vs FINE-TUNED MODEL
======================================================================
--- base_qwen3-8b ---
  Compilation: 7/19 passed (36.8%)
  Unit Tests:  4/19 passed (21.1%)
```

## Training metrics

| Metric | Train | Validation (best) | Test (held out) |
|---|---|---|---|
| Loss | n/a | n/a | n/a |
| Perplexity | n/a | n/a | n/a |

**Trend: sparse**

- Fewer than two evaluation points recorded; no trend to judge.

## Artifacts

- `outputs/eval_results/<model>/<dataset>/*.jsonl` (per-sample results)
- `outputs/generated_solutions/<label>/` (model generations)
- `outputs/q3as/training_summary.json` (loss history; rendered above)

See the [results index](README.md) for the version comparison table.

[← Back to results index](README.md)

Navigation: [project README](../../README.md) · [docs index](../README.md) · [changelog index](../changelogs/index.md) · [results index](README.md)

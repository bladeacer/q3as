# Results v0.3.0

Eval run captured 2026-09-26 06:34 UTC. Data: [`result-data-v0.3.0.json`](result-data-v0.3.0.json).

## Headline (ada-eval)

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| Samples | 19 | 19 | |
| Build | 8/19 (42.1%) | 15/19 (78.9%) | 36.8 pts |
| Unit tests | 5/19 (26.3%) | 11/19 (57.9%) | 31.6 pts |
| SPARK proved | 0 | 0 | |
| Prove errors | 11 | 2 | |

Fine-tuned model proof blockers (check kinds left unproved):

- `VC_OVERFLOW_CHECK` x8
- `VC_POSTCONDITION` x3
- `UNINITIALIZED` x3
- `VC_RAISE` x2
- `VC_DISCRIMINANT_CHECK` x2

## Per dataset (fine-tuned)

| Dataset | Samples | Build | Test |
|---|---|---|---|
| spark_spark_custom | 2 | 2 | 2 |
| spark_spark_human_eval_silver | 4 | 2 | 0 |
| spark_spark_learn | 13 | 11 | 9 |

## baseline_eval aggregates

- **bleu**: `0.6588650442915736`
- **compliance**: `0.2807017543859649`

## comparison_report.txt excerpt

```text
======================================================================
Q3AS EVALUATION REPORT: BASE vs FINE-TUNED MODEL
======================================================================
--- base_qwen3-8b ---
  Compilation: 8/19 passed (42.1%)
  Unit Tests:  5/19 passed (26.3%)
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

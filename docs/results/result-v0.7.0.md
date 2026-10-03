# Results v0.7.0

Eval run captured 2026-10-03 05:59 UTC. Data: [`result-data-v0.7.0.json`](result-data-v0.7.0.json).

## Headline (ada-eval)

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| Samples | 19 | 19 | |
| Build | 7/19 (36.8%) | 10/19 (52.6%) | 15.8 pts |
| Unit tests | 5/19 (26.3%) | 8/19 (42.1%) | 15.8 pts |
| SPARK proved | 0 | 0 | |
| Prove errors | 12 | 9 | |

Fine-tuned model proof blockers (check kinds left unproved):

- `UNINITIALIZED` x3
- `VC_OVERFLOW_CHECK` x2
- `VC_POSTCONDITION` x2
- `VC_RAISE` x2
- `DEPENDS_MISSING` x2

## Per dataset (fine-tuned)

| Dataset | Samples | Build | Test |
|---|---|---|---|
| spark_spark_custom | 2 | 1 | 1 |
| spark_spark_human_eval_silver | 4 | 0 | 0 |
| spark_spark_learn | 13 | 9 | 7 |

## baseline_eval aggregates

- **bleu**: `0.5616810170212344`
- **compliance**: `0.3333333333333333`

## comparison_report.txt excerpt

```text
======================================================================
Q3AS EVALUATION REPORT: BASE vs FINE-TUNED MODEL
======================================================================
--- base_qwen3-8b ---
  Compilation: 7/19 passed (36.8%)
  Unit Tests:  5/19 passed (26.3%)
```

## Training metrics

| Metric | Train | Validation (best) | Test (held out) |
|---|---|---|---|
| Loss | 0.3336 | 0.2176 | 0.2024 |
| Perplexity | 1.3960 | 1.2430 | 1.2240 |

Loss history over 500 training steps.

**Trend: healthy**

- validation loss 0.3221 at step 50 to 0.2176 at step 500; best 0.2176 at step 500
- total validation-loss improvement 32%

Validation loss per evaluation:

| Step | Val loss |
|---|---|
| 50 | 0.3221 |
| 100 | 0.2633 |
| 150 | 0.2422 |
| 200 | 0.2334 |
| 250 | 0.2281 |
| 300 | 0.2244 |
| 350 | 0.2213 |
| 400 | 0.2197 |
| 450 | 0.2179 |
| 500 | 0.2176 |

## Artifacts

- `outputs/eval_results/<model>/<dataset>/*.jsonl` (per-sample results)
- `outputs/generated_solutions/<label>/` (model generations)
- `outputs/q3as/training_summary.json` (loss history; rendered above)

See the [results index](README.md) for the version comparison table.

[← Back to results index](README.md)

Navigation: [project README](../../README.md) · [docs index](../README.md) · [changelog index](../changelogs/index.md) · [results index](README.md)

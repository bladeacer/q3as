# Results v0.4.1

Eval run captured 2026-09-28 14:44 UTC. Data: [`result-data-v0.4.1.json`](result-data-v0.4.1.json).

## Headline (ada-eval)

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| Samples | 19 | 19 | |
| Build | 8/19 (42.1%) | 16/19 (84.2%) | 42.1 pts |
| Unit tests | 5/19 (26.3%) | 11/19 (57.9%) | 31.6 pts |
| SPARK proved | 0 | 0 | |
| Prove errors | 11 | 3 | |

Fine-tuned model proof blockers (check kinds left unproved):

- `VC_OVERFLOW_CHECK` x8
- `UNINITIALIZED` x4
- `VC_POSTCONDITION` x3
- `VC_RAISE` x2
- `DEPENDS_MISSING` x2

## Per dataset (fine-tuned)

| Dataset | Samples | Build | Test |
|---|---|---|---|
| spark_spark_custom | 2 | 2 | 2 |
| spark_spark_human_eval_silver | 4 | 2 | 0 |
| spark_spark_learn | 13 | 12 | 9 |

## baseline_eval aggregates

- **bleu**: `0.8141155632992533`
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
| Loss | 0.3426 | 0.2127 | 0.2242 |
| Perplexity | 1.4090 | 1.2370 | 1.2510 |

Loss history over 500 training steps.

**Trend: healthy**

- validation loss 0.3700 at step 50 to 0.2127 at step 500; best 0.2127 at step 500
- total validation-loss improvement 43%

Validation loss per evaluation:

| Step | Val loss |
|---|---|
| 50 | 0.3700 |
| 100 | 0.2827 |
| 150 | 0.2487 |
| 200 | 0.2319 |
| 250 | 0.2233 |
| 300 | 0.2183 |
| 350 | 0.2155 |
| 400 | 0.2140 |
| 450 | 0.2129 |
| 500 | 0.2127 |

## Artifacts

- `outputs/eval_results/<model>/<dataset>/*.jsonl` (per-sample results)
- `outputs/generated_solutions/<label>/` (model generations)
- `outputs/q3as/training_summary.json` (loss history; rendered above)

See the [results index](README.md) for the version comparison table.

[← Back to results index](README.md)

Navigation: [project README](../../README.md) · [docs index](../README.md) · [changelog index](../changelogs/index.md) · [results index](README.md)

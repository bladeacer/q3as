# Results v0.8.0

Eval run captured 2026-10-03 15:27 UTC. Data: [`result-data-v0.8.0.json`](result-data-v0.8.0.json).

## Headline (ada-eval)

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| Samples | 19 | 19 | |
| Build | 7/19 (36.8%) | 15/19 (78.9%) | 42.1 pts |
| Unit tests | 5/19 (26.3%) | 10/19 (52.6%) | 26.3 pts |
| SPARK proved | 0 | 1 | |
| Prove errors | 12 | 2 | |

Fine-tuned model proof blockers (check kinds left unproved):

- `VC_OVERFLOW_CHECK` x5
- `UNINITIALIZED` x4
- `VC_POSTCONDITION` x2
- `VC_RAISE` x2
- `DEPENDS_MISSING` x2

## Per dataset (fine-tuned)

| Dataset | Samples | Build | Test |
|---|---|---|---|
| spark_spark_custom | 2 | 0 | 0 |
| spark_spark_human_eval_silver | 4 | 3 | 1 |
| spark_spark_learn | 13 | 12 | 9 |

## baseline_eval aggregates

- **bleu**: `0.7092846248081`
- **compliance**: `0.24561403508771926`

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
| Loss | 0.3450 | 0.2436 | 0.2118 |
| Perplexity | 1.4120 | 1.2760 | 1.2360 |

Loss history over 500 training steps.

**Trend: healthy**

- validation loss 0.3555 at step 50 to 0.2436 at step 500; best 0.2436 at step 500
- total validation-loss improvement 31%

Validation loss per evaluation:

| Step | Val loss |
|---|---|
| 50 | 0.3555 |
| 100 | 0.3052 |
| 150 | 0.2811 |
| 200 | 0.2707 |
| 250 | 0.2580 |
| 300 | 0.2499 |
| 350 | 0.2467 |
| 400 | 0.2445 |
| 450 | 0.2438 |
| 500 | 0.2436 |

## Artifacts

- `outputs/eval_results/<model>/<dataset>/*.jsonl` (per-sample results)
- `outputs/generated_solutions/<label>/` (model generations)
- `outputs/q3as/training_summary.json` (loss history; rendered above)

See the [results index](README.md) for the version comparison table.

[← Back to results index](README.md)

Navigation: [project README](../../README.md) · [docs index](../README.md) · [changelog index](../changelogs/index.md) · [results index](README.md)

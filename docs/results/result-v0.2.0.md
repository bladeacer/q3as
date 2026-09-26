# Results v0.2.0

Eval run captured 2026-09-26 05:51 UTC. Data: [`result-data-v0.2.0.json`](result-data-v0.2.0.json).

## Headline (ada-eval)

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| Samples | 7 | 7 | |
| Build | 4/7 (57.1%) | 4/7 (57.1%) | 0.0 pts |
| Unit tests | 2/7 (28.6%) | 2/7 (28.6%) | 0.0 pts |
| SPARK proved | 0 | 0 | |
| Prove errors | 2 | 1 | |

Fine-tuned model proof blockers (check kinds left unproved):

- `VC_OVERFLOW_CHECK` x5
- `SUBPROGRAM_TERMINATION` x1
- `VC_POSTCONDITION` x1
- `VC_FP_OVERFLOW_CHECK` x1

## Per dataset (fine-tuned)

| Dataset | Samples | Build | Test |
|---|---|---|---|
| spark_spark_custom | 2 | 2 | 2 |
| spark_spark_human_eval_silver | 4 | 2 | 0 |
| spark_spark_learn | 1 | 0 | 0 |

## baseline_eval aggregates

- **bleu**: `0.6589372336618904`
- **compliance**: `0.3333333333333333`

## comparison_report.txt excerpt

```text
======================================================================
Q3AS EVALUATION REPORT: BASE vs FINE-TUNED MODEL
======================================================================
--- base_qwen3-8b ---
  Compilation: 4/7 passed (57.1%)
  Unit Tests:  2/7 passed (28.6%)
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

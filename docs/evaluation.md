# Evaluation

How q3as models are scored, what the numbers mean, and how to run the
pipeline.

## Benchmark

Metrics and dataset categories come from the [ada-eval](https://github.com/AdaCore/ada-eval)
framework (cached at `data/raw_repos/AdaCore/ada-eval`). Three sample sets are used:

| Dataset | Contents |
|---|---|
| `spark_learn` | Learning examples with SPARK contracts (13 samples) |
| `spark_custom` | Custom SPARK verification challenges (2 samples) |
| `spark_human_eval_silver` | HumanEval-style silver-standard tasks (4 samples) |

Samples live in the cached ada-eval's `data/base/expanded/<dataset>/<sample>/`
with a `base/` project (what the model may edit), a `solution/` project (the
reference answer - `make eval` scores generations against it, and it must never
appear in training data; see [Data provenance](data-provenance.md)), and a
`tests/` project.

These samples are the *entire* ada-eval corpus; there is no surplus data
there for training.

## Metrics

Each model response is scored on three dimensions:

1. **BUILD** - `gprbuild` compiles the generated project. Scored per
   sample (`compiled: true/false`).
2. **TEST** - the sample's unit tests are built and run; the result is
   recorded as pass/fail per sample.
3. **PROVE** - `gnatprove -P main.gpr -k <unit>` checks proof obligations
   for the target subprogram. Results are `proved`, `unproved` (with the
   specific check kinds left over, e.g. `VC_OVERFLOW_CHECK`), or `error`.

`make eval-pipeline` runs BUILD, TEST, and PROVE, in that order, and the
fine-tuned and base runs are compared side by side to measure the
fine-tuning effect. There is no RUN or BASE eval kind in q3as: the three
kinds above are the whole set, and `--evals` selects among them.

`make eval` scores the generations in `outputs/generated_solutions/` against
the canonical solutions directly (BLEU-4, exact match, file-set match,
standard compliance), so it measures the models rather than the corpus. It
needs `make generate` to have run first and exits non-zero when there is
nothing to score.

## Reading the results

[`eval/eval_pipeline.py`](../eval/eval_pipeline.py) writes per-sample JSONL
under `outputs/eval_results/<model>/<dataset>/` and a human-readable
`outputs/comparison_report.txt`. When interpreting:

- **n is small** (19 samples per model). Treat percentages as directional;
  a 10-point swing is within noise.
- **BUILD/TEST gains** are the primary fine-tuning signal; PROVE
  `unproved` with specific check kinds (overflow, postcondition) is a
  data-quality signal: the model writes near-correct code but not yet
  provable contracts.
- **All-error PROVE rows** (result `error` instead of `unproved`) mean the
  harness never really ran gnatprove - a pipeline bug, not a model result.
  [`eval/ada_eval_common.py`](../eval/ada_eval_common.py) classifies
  `proved_incorrectly` and `subprogram_not_found` as unproved rather than as
  errors, so an incorrect proof cannot inflate the proved rate.

## Running

```bash
make generate        # batch generation with base and fine-tuned models
                     # defaults: 12,000 prompt chars, 512 new tokens
make eval            # BLEU + compliance metrics -> outputs/eval_results.json
make eval-pipeline   # ada-eval BUILD/TEST/PROVE -> outputs/eval_results/
```

The pipeline resolves gnatprove/gprbuild through the Alire environment
(see [Toolchain setup](toolchain-setup.md)) and seeds generation (seed 42)
for reproducible sampling.

Navigation: [project README](../README.md) · [docs index](README.md) · [changelog index](changelogs/index.md) · [results index](results/README.md)

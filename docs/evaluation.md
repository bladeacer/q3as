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

Samples live in the cached ada-eval's
`data/base/expanded/<dataset>/<sample>/` with a
`base/` project (what the model may edit), a `solution/` project (the
reference answer, used only for `canonical_evaluation` sanity checks - it
must never appear in training data; see
[Data provenance](data-provenance.md)), and a `tests/` project.

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

RUN and BASE are compared to measure fine-tuning effect; a canonical
solutions pass (13/13 across all three) doubles as a standing toolchain
sanity check.

## Reading the results

`eval/eval_pipeline.py` writes per-sample JSONL under
`outputs/eval_results/<model>/<dataset>/` and a human-readable
`outputs/comparison_report.txt`. When interpreting:

- **n is small** (19 samples per model). Treat percentages as directional;
  a 10-point swing is within noise.
- **BUILD/TEST gains** are the primary fine-tuning signal; PROVE
  `unproved` with specific check kinds (overflow, postcondition) is a
  data-quality signal: the model writes near-correct code but not yet
  provable contracts.
- **All-error PROVE rows** (result `error` instead of `unproved`) mean the
  harness never really ran gnatprove - a pipeline bug, not a model result.
  The canonical sanity check distinguishes the two.

## Running

```bash
make generate        # batch generation with base and fine-tuned models
make eval            # BLEU + compliance metrics -> outputs/eval_results.json
make eval-pipeline   # ada-eval BUILD/TEST/PROVE -> outputs/eval_results/
```

The pipeline resolves gnatprove/gprbuild through the Alire environment
(see [Toolchain setup](toolchain-setup.md)) and seeds generation (seed 42)
for reproducible sampling.

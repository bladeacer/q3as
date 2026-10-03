# Evaluation analysis: v0.8.0

What the v0.8.0 run measured, what changed between v0.6.0 and v0.8.0, and how
much of the headline the model earned. The numbers themselves are in
[`result-v0.8.0.md`](results/result-v0.8.0.md) and
[`result-data-v0.8.0.json`](results/result-data-v0.8.0.json); this page
decomposes them. The next version's scope is [`plan.md`](../plan.md).

Run identity: captured 2026-10-03 15:27 UTC, 19 samples per model, 500 training
steps (0.06 of one epoch), adapter trained on the 80,718-turn dataset recorded
in `data/processed/dataset_metadata.json`.

## Headline

| Metric (fine-tuned) | v0.4.1 | v0.6.0 | v0.7.0 | v0.8.0 |
|---|---|---|---|---|
| Build | 16/19 (84.2%) | no run | 10/19 (52.6%) | 15/19 (78.9%) |
| Unit tests | 11/19 (57.9%) | no run | 8/19 (42.1%) | 10/19 (52.6%) |
| SPARK proved | 0 | no run | 0 | **1** |
| Prove errors | 3 | no run | 9 | **2** |
| Unproved samples | 16 | no run | 10 | 16 |
| Discharged checks (all samples) | 31 | no run | 23 | 25 |
| BLEU-4 | 0.8141 | no run | 0.5617 | 0.7093 |
| Exact match | 8/19 | no run | 4/19 | 6/19 |
| Standard compliance | 0.2807 | no run | 0.3333 | 0.2456 |
| Reply format `File:` entries | not recorded | no run | 4/19 | **19/19** |
| Replies that changed nothing | not recorded | no run | not recorded | 13/19 |
| Test loss | 0.2242 | no run | 0.2024 | 0.2118 |

Base model on the same 19 samples: build 7, test 5, proved 0, prove errors 12,
identical on v0.7.0 and v0.8.0. On v0.4.1 the base model was build 8, test 5,
prove errors 11.

**Verdict.** v0.6.0 shipped no evaluation run, so the measured comparison
across this stretch is v0.4.1 to v0.8.0. On the dimension the benchmark scores
with a compiler and a prover, v0.8.0 is level with v0.4.1 (15 against 16
builds, 10 against 11 tests, on 19 samples, where a 10-point swing is noise),
it proves one sample where v0.4.1 proved none, and it posts the lowest prove
error count of any run. The v0.7.0 regression is gone: no sample failed BUILD
because a package body was written into a package spec. On the reference
metrics v0.8.0 is still below v0.4.1 (BLEU 0.709 against 0.814, exact match 6
against 8, compliance 0.246 against 0.281). That is the honest summary: the
proof dimension improved, the similarity metrics did not.

## What changed from v0.6.0 to v0.8.0

No new data source was added in this stretch. Every turn comes from the cached
repositories [`docs/data-provenance.md`](data-provenance.md) already lists.

### Dataset

| Turn kind | v0.6.0 | v0.7.0 | v0.8.0 |
|---|---|---|---|
| `contract_synth` | 708 | 708 | 708 |
| `spark_verified` | 41 | 568 | 568 |
| `lab_pair` | 129 | 252 | 252 |
| `assurance_qa` | 0 | 5 | 5 |
| `toolchain_qa` | 17 | 17 | 17 |
| Total turns | 79,173 | 80,262 | 80,718 |

v0.7.0 added the prover-verified SPARK turns (568 units from all 16 SPARK2
trees that gnatprove discharged), the lab explanation twins, and the assurance
QA turns. v0.8.0 added 456 turns from two extraction bug fixes, not from a new
kind: the ada-spark guidance globs now match the repository's real layout (2
documents loaded before, 5 of 6 now), and `pair_files` keys a body on
`(directory, stem)`, which fixes the 138 turns that had paired a
specification with a foreign body.

### Harness and measurement

| Change | Effect on the numbers |
|---|---|
| A body-only fallback block routes to the sibling `.adb` | Recovered the 6 samples v0.7.0 lost to a routing mistake |
| Reply shape, changed-file count and routing recorded per sample in `generation_meta.json` | First run where format compliance and no-op replies are visible |
| `make eval` refuses a benchmark packed without solution files | Removes a hollow BLEU 0.0 report that read as a model result |
| `make all` runs `eval-pipeline` before `eval` | Removes a published summary that showed the previous run's tallies |
| Answer chunking starts on a line, after a heading, at the anchor | 7 toolchain QA answers no longer open mid-sentence |
| AdaDoc and skill-doc paths matched to the real repository layout | 824 records now carry the proof workflow text |

### Training

Both runs used 500 steps with the same hyper-parameters. Loss is not
comparable across the two datasets. Held-out test loss is 0.2024 for v0.7.0
and 0.2118 for v0.8.0, both below v0.4.1's 0.2242, and both runs are diagnosed
`healthy` by the trend check in
[`scripts/gen_eval_report.py`](../scripts/gen_eval_report.py).

## How much of the headline the model earned

`generation_meta.json` records `changed_files` per sample: the number of
overlay entries whose content differs from the base project. Splitting the
tallies on that field:

| Model | Group | Samples | Build | Test | Proved |
|---|---|---|---|---|---|
| Fine-tuned | changed nothing | 13 | 12 | 8 | 0 |
| Fine-tuned | edited | 6 | 3 | 2 | 1 |
| Base | changed nothing | 2 | 2 | 1 | 0 |
| Base | edited | 17 | 5 | 4 | 0 |

So 15/19 builds is 12 inherited plus 3 earned, and 10/19 tests is 8 inherited
plus 2 earned. The single proved sample is fully earned. On the samples where
it actually edits, the fine-tune is currently no better than the base model
(3 builds in 6 against 5 in 17). Its measured advantage over the base model
comes entirely from the no-op group, where the base project is left in place.

### Why the no-op group scores well

18 of the 19 ada-eval base trees already contain a body for the target
subprogram; the base project compiles for 12 of the 13 samples the fine-tune
declined to touch. The benchmark's canonical solutions mostly add contracts or
adjust a body, so a model that reproduces the specification it was shown keeps
a compiling project. Echoing is a scoring strategy here, not a solution.

`HumanEval_2_truncate_number` is the single exception: its base tree has the
specification and no body at all. The model wrote the body into
`src/placeholder.adb`, kept the given Pre and Post, and gnatprove discharged
6 of 6 checks (`VC_POSTCONDITION`, `VC_FP_OVERFLOW_CHECK`, `UNINITIALIZED`,
`SUBPROGRAM_TERMINATION`, `VC_ASSERT`) with `pragma_assume_count: 0`. It builds,
its tests pass, and it is the first sample this project has proved. At 500 steps
that is the most informative single number in the run.

### The four BUILD failures, itemised

| Sample | Changed | Cause |
|---|---|---|
| `HumanEval_0_has_close_elements_spec` | no | Base tree does not build; PROVE `error` |
| `char_count_1` | yes | Harness: `main.gpr` text written into `src/string_utils.ads` |
| `char_count_2` | yes | Same reply, byte-identical, same cause |
| `search_array_3` | yes | Model deleted `Not_Found : exception;` from the spec, so the body that raises it no longer compiles |

The `char_count` pair is a harness fault, not a model fault.
[`eval/generate.py`](../eval/generate.py) drops a `File:` entry whose path is
outside the target directory, and then falls through to the single-block
fallback, which writes the fenced block at the target path. The model named
`main.gpr`, the entry was dropped, and the project file's text was written over
the specification. Both samples then fail BUILD and report
`subprogram_not_found`. Discarding such a reply instead of relocating its block
turns both into no-ops, which is worth 2 builds and 2 tests.

### Why the similarity metrics fell

BLEU-4 is computed on the primary (target) file only
([`eval/baseline_eval.py`](../eval/baseline_eval.py)), and for 8 of the 19
samples the base file and the canonical file are identical on that file. All 6
exact matches in this run are no-op replies, so a model that echoes collects
BLEU 1.0 on those samples. That is most of why BLEU recovered from 0.5617 to
0.7093. The two samples where the base file was already the canonical text and
the model still edited (`char_count_1` at BLEU 0.023 and `show_uninitialized`
at 0.656) show the same mechanism from the other side: the edit moved the file
away from the reference. Compliance moved the other way (0.3333 to 0.2456)
because the base files carry Ada 83 constructs that the check counts as
non-compliant, and the model now reproduces them.

### Verdict per metric

| Metric | v0.4.1 to v0.8.0 | Reading |
|---|---|---|
| Build | 16 to 15 | Parity. Includes 2 samples lost to a harness fault |
| Unit tests | 11 to 10 | Parity, same cause |
| SPARK proved | 0 to 1 | Improvement, and the one unambiguously earned result |
| Prove errors | 3 to 2 | Improvement; fewer projects where gnatprove never ran |
| Discharged checks | 31 to 25 | Slight fall, with more of them concentrated in one proved sample |
| Reply format | unrecorded to 19/19 | Improvement, the harness contract is now obeyed |
| BLEU-4 | 0.814 to 0.709 | Regression, and partly an artifact of no-op replies |
| Exact match | 8 to 6 | Regression; v0.8.0's 6 are all echoes, and v0.4.1 matched 2 samples v0.8.0 did not (`char_count_1`, `show_uninitialized`) |
| Compliance | 0.281 to 0.246 | Regression, follows the no-op replies |
| Test loss | 0.2242 to 0.2118 | Improvement, and not comparable across datasets |

## What moves each number next

- **Build and tests**: fixing the dropped-entry fallback, worth 2 and 2.
- **Proved samples**: the blocker histogram is `VC_OVERFLOW_CHECK` x5,
  `UNINITIALIZED` x4, `VC_POSTCONDITION` x2, `VC_RAISE` x2, `DEPENDS_MISSING`
  x2. Contract-synthesis families aimed at those check kinds, and error
  diagnosis turns from the Ada 83 suite's inline `-- ERROR:` markers, are the
  two data moves that target them.
- **Edit quality**: only 94 of the 72,464 train turns (0.13%) have the shape of
  the evaluation prompt, a project listing in and updated files out, counted
  over the whole split. Separately, 5,274 turns (7.3%) are at least 95%
  covered by the 8-gram shingles of their own user turn, but those sit in the
  question kinds, where quoting the code under discussion is the correct
  answer. The corpus barely demonstrates the task the benchmark poses, and
  nothing in it demonstrates the decision to act.
- **Unit tests**: nothing in the corpus trains the test dimension, and the
  cache holds 1,858 `tests.adb` harnesses to build turns from.
- **Read the numbers honestly**: every future report needs the split by
  changed and unchanged samples, so a run cannot reach a build count by
  declining to act.

[`plan.md`](../plan.md) carries that work as the v0.9.0 scope.

## Reproducing this analysis

```bash
make generate eval-pipeline eval eval-report
```

The per-sample inputs are `outputs/eval_results/<model>/<dataset>/*.jsonl`
(build, test and prove results) and
`outputs/generated_solutions/<label>/generation_meta.json` (reply shape,
changed files, routing). The canonical solutions the reference metrics score
against are packed by `make eval-data` into ada-eval's
`data/base/compacted/*.jsonl`.

Navigation: [project README](../README.md) · [docs index](README.md) · [changelog index](changelogs/index.md) · [results index](results/README.md)

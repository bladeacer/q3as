# plan.md - v0.10.0 scope

State of the 2026-10-04 session: v0.9.0 is measured, but its result report was
not generated in this checkout, so the link to `docs/results/result-v0.9.0.md`
is deliberately not written. The harness no longer
writes a reply into the wrong file, and every number states the split between
the samples the model changed and the samples it echoed. No new data source is
in scope and none is needed: every turn still comes from the cached
repositories [`docs/data-provenance.md`](docs/data-provenance.md) lists.

The one-line summary of the audit below: the corpus contains **no
demonstration of the reply format the benchmark scores on** (0 turns, measured,
where the previous scope estimated 94), nothing in it trains the dimension the
benchmark's unit tests measure, and a run is not yet an artefact that can be
re-measured. The first two are data, the third is bookkeeping, and all three
have to move before a build count means anything.

## What the audit found

Findings 1 to 4 are what the v0.9.0 numbers and the harness taught; findings 5
to 7 are the state of the extraction track. Every volume below was counted in
the cache on 2026-10-04, not estimated.

### 1. The corpus has zero write-shaped turns

Counted over `data/processed/dataset_train.jsonl` (72,511 turns): an assistant
message that carries a `File: <path>` line followed by a fenced block occurs
**0 times**. Six user turns contain the string `File:`, and all six have it
inside an Ada comment (`--  File: tests.adb`), not as a project listing. The
previous scope said 94 of 72,464; that came from a looser match over exactly
these comment lines.

Consequences, both load-bearing:

- Reply-format compliance (19/19 `file_blocks` on the v0.8.0 run) is
  prompt-following, not training. The model was never shown a turn in which
  acting on a project is the answer.
- The no-op reply mode is the default when the corpus contains no example of
  editing a project: reproducing the specification it was shown is the only
  behaviour with training signal behind it. 13 of 19 replies in the v0.8.0 run
  changed nothing, and the 6 that did edit scored 3 builds and 2 tests.

### 2. Most of the score is the base project, and that is now measured everywhere

18 of the 19 ada-eval base trees already carry a body for the target
subprogram and already compile, so an echo collects BUILD and TEST for the base
project. v0.9.0 reports the split in `make eval`, in
`outputs/eval_results.json`, and in every `docs/results/result-vX.Y.Z.md`
("Samples the model changed"), and the cross-version table carries
"Build on edited samples (FT)" from this version on. The number to move is now
stated in one place and cannot be read as a total.

### 3. A filtered evaluation run still tallies the previous run

`eval/eval_pipeline.py` writes `outputs/eval_results/<model>/<dataset>/` and
never clears it. `make eval-pipeline --dataset spark_learn` therefore leaves the
last run's `spark_custom` results in place, and every tally in the project
(`aggregate_eval_results`, `collect_ada_eval_metrics`, `make eval`) reads the
whole tree, so a partial run reports a mixture of two runs. Nothing warns about
it: the numbers look complete.

### 4. A run is not an artefact that can be re-measured

Three separate gaps, all of which cost this project a claim:

- the adapter is written to `outputs/q3as/` and never kept per version. The
  v0.8.0 checkpoint was not on the machine that produced v0.9.0, so the v0.9.0
  run is a **new adapter**, not a re-score of the v0.8.0 one, and no difference
  against v0.8.0 can be attributed to the harness alone.
- `generation_meta.json` is overwritten per run, so the per-sample split of an
  older version cannot be recovered (v0.8.0's is only readable because it was
  copied into its analysis page by hand).
- generation is seeded (42) but CUDA kernels are not bitwise deterministic, so
  even the same adapter would not reproduce a run digit for digit. A harness
  fix therefore has to be argued from per-sample evidence, which is what the
  v0.9.0 changelog does for the two `char_count` samples.

### 5. Nothing trains the unit-test dimension

Zero turns teach writing a test harness, and the benchmark scores unit tests.
The cache has the material: **1,828** `tests.adb` files (1,809 in
`Ada-Algorithms`, 19 in ada-eval), of which **747** name exactly one unit under
test that is present in the same directory and are therefore pairable without
guessing. The assertion idiom in those files is a **local** helper, not
`Ada.Assertions`: 62,237 `Check (` call sites, 4,949 `Assert (`, 338 `Expect (`
and 21 `Assert_Equal (`.

### 6. Volume corrections to the v0.9.0 inventory

Counted again on 2026-10-04; where the earlier figure was wrong, the corrected
one is given.

| Item | v0.9.0 scope said | Measured now |
|---|---|---|
| Ada-83 suite `.ada` files | 4,142 | 4,142 |
| `btests/` files | 1,515 | 1,515 `.ada` of 1,592 files |
| `-- ERROR:` markers | 12,091 | 12,195 (12,048 inside `.ada`) |
| files carrying `-- OBJECTIVE:` | 1,403 | 1,397 `.ada` (1,975 including the harness's own `.txt`) |
| ctests/atests/ltests self-checking | 2,181 files | 2,998 files, `FAILED` in 2,576, `REPORT.TEST` in 5 |
| `tests.adb` pairable with a unit under test | implied 1,858 | **747** |
| tests harnesses using `Ada.Assertions.Assert` | 707 | 7 name it explicitly; 1,151 mention `Ada.Assertions`, 565 `use Ada.Assertions` |
| tests harnesses inside `SPARK2` trees | 836 | 852 |
| unparsed documentation sections | ~11,000 | 14,704 (Ada-Algorithms 10,904, adacovex 1,989, Ada_CRDT 1,118, skills 598, ada-spark 95) |
| AdaDoc comment blocks above declarations | 721 declarations, ~20k words | **9,423 declarations, ~142k words** (6,271 / ~97k in Ada-Algorithms) |
| `.gpr` files | 2,254 (4 read) | 2,515 in the cache, 2,260 in Ada-Algorithms (`-gnat2022` 1,023, `-gnata` 850) |
| `training_material` Ada no builder reaches | 363 files | 683 Ada files total: 63 in `courses/ada_essentials/examples`, 69 in mini-projects, 28 in quizzes, 136 in `labs/` trees |
| TLALOC `.finc` / `.sub` / `.dcl` | 169 / 112 / 44 | 169 / 112 / 44 (`.FINC`, `.SUB`, `.DCL`) |
| SimpleEnglish raw model outputs | 430 | 435 `.txt` under `evals/results` |

Two of those corrections change a conclusion, not just a number:

- the AdaDoc comment blocks are **13 times larger** than scoped, which moves
  that item from "low volume, parser risk" to a real one;
- the `training_material` labs have **no solution side**. The 17 `*.lab.rst`
  instructions sit beside starter `source/` trees and nothing else, so there is
  no gold answer to train against. Item 8 of the old scope rested on a
  `labs/source/` versus `labs/solution/` twin that does not exist in the cache.
  Its value drops to nothing until a solutions tree is added.

### 7. The extraction track is still unfinished

The v0.7.0 changelog said it was finished. Findings 1 and 5 say otherwise:
the corpus barely demonstrates the task the benchmark poses, and nothing in it
demonstrates the dimension the benchmark scores for unit tests. The corrected
volumes are above; ordered by value per unit of work they become the deferred
list at the end of this file.

## To build in v0.10.0

### 1. Write-shaped turns for the evaluation prompt (highest value)

Finding 1. Scope: a `file_block_pair` turn kind, built from the pairs the
builder already finds (2,009 `code_pair` units and 46,603 AST-derived units are
available as content). The user turn is the project tree in
`File: <path>` + fenced-block form, the assistant turn the same Ada emitted as
one entry per changed file.

The framing must be **one implementation**, not a copy: `build_user_prompt` in
[`eval/generate.py`](eval/generate.py) already inserts
`data/processing_scripts/` on `sys.path`, so the shared function belongs there
and both the builder and the eval prompt call it. A second copy of that prompt
text is how the v0.6.0 acceptance criterion ("format compliance rises without
harness repair") would be met by luck rather than by construction.

Content comes from the existing pair, contract and variant builders; only the
framing is new. Targets, in order: at least one turn per `code_pair` unit, and
one turn per sampled AST unit that has both a spec and a body. The v0.9.0
criterion for this item is unchanged and is now measurable: the no-op count and
the build/test counts on the changed samples, both reported by `make eval`.

### 2. Unit-test turns from the harnesses already in the cache

Finding 5. Scope: a `gnattest_pair` turn kind pairing a unit's `.ads` (and its
body when present) with the `tests.adb` beside it: 747 pairable harnesses in
`Ada-Algorithms` by the measurement above, each naming exactly one unit.

The previous scope said the only new code would be the turn builder, because
`parse_ada_ast.pair_subprograms` "can find the unit under test from the
`with <Unit>;` clause". It cannot: nothing in
[`parse_ada_ast.py`](data/processing_scripts/parse_ada_ast.py) reads context
clauses, on either the structural or the libadalang path. So a small
`with`/`use` reader is part of this item. It does not need the AST: the clause
is in the harness file's own text and the unit it names is checked against the
same directory, which is also what makes the 747 unambiguous.

Shape: the user turn is the unit under test plus a request to write its test
harness; the assistant turn is the `tests.adb`. That is the shape the
benchmark's TEST dimension scores, and the shape
`AdaCore/skills`' `test_skeletons.md` and `test-separate-drivers.md` document,
which are already loaded as guidance and never trained against.

Caveat to check first: `Ada-Algorithms` is the repository the eval guard fires
on (827 contaminated groups dropped in the v0.9.0 build), so the drop count in
`dataset_metadata.json` will move and `make check-integrity` must still exit 0.

### 3. Make a run an artefact that can be re-measured (small)

Findings 3 and 4, and the bookkeeping the last two versions needed:

- [`eval/eval_pipeline.py`](eval/eval_pipeline.py) clears the per-model
  `outputs/eval_results/` tree at the start of a run (or writes a run id and
  the tallies filter on it), so a filtered run cannot report a mixture of two.
- [`eval/generate.py`](eval/generate.py) records a fingerprint of the adapter it
  loaded (sha256 over the adapter weights plus the training config) in
  `generation_meta.json`, and `scripts/gen_eval_report.py` copies that sidecar
  to `docs/results/meta-vX.Y.Z.json`. A versioned split then survives the next
  run, which is what v0.8.0 had to do by hand.
- [`eval/baseline_eval.py`](eval/baseline_eval.py) reports BLEU, exact match and
  compliance split by changed and unchanged samples as well, because all six of
  the v0.8.0 exact matches were echoes and the metric currently rewards them.

This is the smallest item here and the one that makes the other two arguable
after the fact.

## Deferred, with measured volumes

Ordered by value per unit of work. None of it ships without a result file to
compare against v0.9.0.

1. **`ada83_diag`**: the Ada 83 conformance suite as error-diagnosis turns.
   4,142 `.ada` files, 12,195 inline `-- ERROR:` markers, 1,397 `.ada` files with
   an `-- OBJECTIVE:` header, and 2,576 of the 2,998 `ctests`/`atests`/`ltests`
   files that self-check with `FAILED`. Largest volume in the cache and it
   matches how the eval reports failures. Hard constraint: `btests/` is
   *intentionally illegal* Ada, so `.ada` needs a dedicated extension constant
   and a dedicated builder; it must never enter `code_pair` or reach
   `inject_defect`, or the 17 defect families will "repair" already-broken code
   into wrong targets.
2. **AdaDoc comment blocks above declarations**: 9,423 declarations and ~142k
   words that `parse_ada_ast` discards because it starts at the declaration
   keyword. Bigger than scoped (finding 6) and it gives the write side of the
   `gnatdoc` guidance that is already loaded. The one item here with parser
   risk: both the regex path and the libadalang path need it, and `parent.text`
   excludes the comment as well.
3. **The doc trees no extractor reads**: 14,704 sections across
   `Ada-Algorithms` (10,904), `adacovex` (1,989), `Ada_CRDT` (1,118),
   `AdaCore/skills` (598) and `ada-spark` (95). `default_doc_dirs()` is
   `learn` + `training_material` only, so adding the rest needs no parser
   change. Low risk, low reward: it is prose and `doc_section` is already the
   largest prose kind at 5,379.
4. **Contract-documentation turns from the generated AdaDoc pages**: 111 files
   (`adacovex/docs/api-docs/` 75, `Ada_CRDT/docs/api-docs/` 36).
   `_SPARK_ASSURANCE_DOC_DIRS` lists `docs/proof`, `docs/compliance`,
   `docs/usage` and `docs/archive`, not `docs/api-docs`, so none of it is read.
   Two products: "document this declaration" turns with a real target, and a
   much richer source for `assurance_qa` than the 5 turns it produces today.
5. **Build-configuration turns from `.gpr` files**: 2,515 files,
   `discover_files` collapses them to the first readable file per input
   directory, used only as trailing context.

## Not worth it

- `SimpleEnglish`'s 435 raw model outputs on STE tasks: other models'
  generations, not Ada.
- TLALOC's 169 `.FINC` / 112 `.SUB` / 44 `.DCL`: TLALOC's own register DSL.
- `Ada-Algorithms/*/SPARK2/*/.spark`, `.gnat-json`, `.cswi`, `.sarif`: prover
  provenance for units already proved; useful only as a filter.
- The 279 `spark_verified` candidates `gen_verified_spark.py` drops for
  `with`-ing a sibling: real SPARK, but proving them needs a per-directory
  synthetic project, and 567 already-proved units have not yet been shown to
  raise the proved count.
- `training_material`'s `labs/` starters: instructions without solutions
  (finding 6), so there is no answer to train on.

## Rules for this track

- Every new turn kind gets a unit test per branch, and a count in
  [`docs/datasets-and-training.md`](docs/datasets-and-training.md).
- Every new source directory updates the
  [`docs/data-provenance.md`](docs/data-provenance.md) source table, then
  `make check-integrity` and `make validate-defects` must pass.
- No new turn kind may lower reply-format compliance, the BUILD/TEST numbers,
  or the build count on the samples the model changed. That last one is the bar
  this version exists to clear: a run that reaches 16/19 build by leaving the
  base project in place is not the win v0.4.1's 16/19 was.
- Numbers come from `make eval-report` only. A claim without a
  `docs/results/result-vX.Y.Z.md` is prose, not a result.
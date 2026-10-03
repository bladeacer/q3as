# plan.md - v0.9.0 scope

State of the 2026-10-03 session: v0.8.0 is measured
([`result-v0.8.0.md`](docs/results/result-v0.8.0.md)), and the v0.7.0
regression is recovered. The decomposed reading is in
[`docs/eval-analysis-v0.8.0.md`](docs/eval-analysis-v0.8.0.md). No new data
source is in scope here, and none is needed: every turn still comes from the
cached repositories [`docs/data-provenance.md`](docs/data-provenance.md)
lists.

The one-line summary: v0.8.0 scored 15/19 builds, 10/19 tests and its first
proved sample, and 13 of 19 fine-tuned replies changed nothing, so most of
that score is the ada-eval base project left in place. v0.9.0 closes the last
harness fault that costs samples, and makes every future number state the
split between samples the model changed and samples it echoed.

## Measured state

| Metric (fine-tuned) | v0.4.1 | v0.7.0 | v0.8.0 |
|---|---|---|---|
| Build | 16/19 | 10/19 | 15/19 |
| Unit tests | 11/19 | 8/19 | 10/19 |
| SPARK proved | 0 | 0 | 1 |
| Prove errors | 3 | 9 | 2 |
| Reply format `File:` | not recorded | 4/19 | 19/19 |
| Replies that changed nothing | not recorded | not recorded | 13/19 |
| Test loss | 0.2242 | 0.2024 | 0.2118 |

| Group | Samples | Build | Test | Proved |
|---|---|---|---|---|
| Changed nothing | 13 | 12 | 8 | 0 |
| Edited | 6 | 3 | 2 | 1 |

Two facts drive this version. First, `char_count_1` and `char_count_2` fail
BUILD because the harness wrote the project's `main.gpr` text into
`src/string_utils.ads`. Second, 18 of 19 ada-eval base trees already contain a
body for the target subprogram, so a reply that echoes its input keeps a
compiling project and collects the score. `HumanEval_2_truncate_number` is the
only sample with no body in its base tree, and it is the one the model
implemented, tested and proved (6 of 6 checks, no `pragma Assume`).

## To build in v0.9.0

### 1. A dropped `File:` entry must never be written into the target

[`eval/generate.py`](eval/generate.py) `parse_generated_files` keeps only the
entries next to the prompt's target directory, which is right: small models
echo `main.gpr` and `main.adc`, and their reconstructed copies are corrupt. The
fault is what happens next. When every entry was dropped, the function fell
through to the single-block fallback, which writes the fenced block at the
target path, so the one block the reply did contain (the project file) landed
in the specification.

Measured on the stored v0.8.0 generations: two samples name only
`main.gpr`, both replies byte-identical, both fail BUILD, and both report
`subprogram_not_found` on the prove side. The base model has no such reply, so
its numbers do not move.

Rule to implement: the fallback chain fires only when the reply carries no
`File:` header at all.

1. Collect the `_FILE_BLOCK_RE` matches and note whether any header matched,
   regardless of whether the path survived `_safe_source_path`.
2. Entries next to the target directory are the overlay, unchanged.
3. Otherwise, if any header matched, return an empty overlay and log a warning
   naming the dropped paths and the target. A reply that addressed files but
   named none of them usably must leave the project alone.
4. Otherwise (no header at all) run the existing chain: fenced block at the
   target, `package body` opening routed to the sibling `.adb`, raw text at the
   target.

Deliberately unchanged: the v0.8.0 body routing, an explicit `File:` entry
winning over routing, and `reply_format` (a reply that names `main.gpr` still
counts as `file_blocks`, which is accurate, the model did use the contract).

Expected effect when the stored run is re-scored: build 15 to 17, tests 10 to
12, no-op replies 13 to 15, edited samples 6 to 4 (3 build, 2 tests, 1 proved).
`HumanEval_2_truncate_number` is unaffected, so the proved sample stays.

Tests in [`tests/test_generate.py`](tests/test_generate.py):

- `test_absolute_path_rejected` and `test_non_source_suffix_rejected` assert
  today's behaviour ("the reply still has a fenced block, so it lands at the
  target path"). Both change to expect an empty overlay, and their names stay.
- new `test_all_entries_dropped_leaves_the_project_alone`: the `char_count`
  shape, one `File: main.gpr` entry against a `src/` target.
- new `test_headerless_reply_still_falls_back`: pins the three fallback steps,
  so the fix cannot silently remove the routing repair.
- new sidecar test: an all-dropped reply records `changed_files: 0`.

### 2. Every report states the changed and unchanged split

One implementation, in [`eval/ada_eval_common.py`](eval/ada_eval_common.py),
which already owns the build/test/prove tally and its result classification.

- new `split_tally_by_changed_files(eval_results_dir, meta_path)` returning
  `{"edited": <stats>, "unchanged": <stats>}` from the same `empty_stats`
  buckets, joining per-sample results to `generation_meta.json` on the sample
  name. It returns nothing usable when the sidecar is absent, so older runs
  render as not recorded rather than as zero.
- [`scripts/gen_eval_report.py`](scripts/gen_eval_report.py):
  `collect_ada_eval_metrics` stores `by_change` per model; `render_markdown`
  gains a "Samples the model changed" table (both models, samples, build, test,
  proved); `render_index` gains "No-op replies (FT)" and "Build on edited
  samples (FT)" in the last-three comparison. Missing data renders `n/a`
  through the existing `_fmt_pct(None)` path.
- [`eval/baseline_eval.py`](eval/baseline_eval.py): extend the `Reply format`
  line, for example `Build 15/19 (3 of 4 changed, 12 of 15 unchanged)`.

Data available for the index row: the no-op count is in the stored
`reply_shapes.untouched` for v0.8.0 (13) and absent for v0.4.1 and v0.7.0,
whose sidecars were overwritten. Build-on-edited cannot be recovered for any
earlier version, so that row starts at v0.9.0. Open decision: keep it in the
cross-version table with `n/a` for older versions, or show it only in the
per-version report.

Tests: [`tests/test_eval_scoring.py`](tests/test_eval_scoring.py) for the new
helper (both buckets, missing sidecar, unknown sample names, a sample that is
in the results but not the sidecar), and
[`tests/test_reporting.py`](tests/test_reporting.py) for the rendered section
and its `n/a` path.

### 3. Documentation reconciliation

Stale at the time of writing, and what fixes it:

| Place | Stale content | Fix |
|---|---|---|
| `docs/changelogs/v0.8.0.md` | Opens with "No model has been trained for this version yet" | Replace with the measured section, the grouped table, and a link to the analysis page |
| `docs/changelogs/index.md` | v0.8.0 row says "not run" | Link `result-v0.8.0.md` |
| `plan.md` | Scoped for v0.8.1, quotes the v0.7.0 regression as current state, says `untouched 9/19` where the run recorded 13 | This file |
| `docs/evaluation.md` | No mention that an echo keeps the base project and scores on it | New bullet under "Reading the results" plus a link to the analysis page |
| `docs/README.md` | Analysis page not indexed | Row in the reference table and in the reading order |
| `AGENTS.md` | `eval/generate.py` bullet describes routing but not the fallback hazard | One sentence naming the hazard and pointing here |

`docs/results/README.md` is generated by `render_index`, and
`tests/test_reporting.py` fails when it differs from that output, so it is
never hand-edited.

### 4. Publishing

`make bump-version PART=minor` moves 0.8.0 to 0.9.0 across
[`alire.toml`](alire.toml), `alire-dev.toml`, `alire-ast.toml` and
`pyproject.toml`; two tests in `tests/test_reporting.py` fail if those ever
disagree. The dataset does not change in this version, so publishing is a
re-score of the current adapter:

```bash
make bump-version PART=minor
make generate eval-pipeline eval eval-report
make test lint check-integrity validate-defects agents-tree
```

`docs/changelogs/v0.9.0.md` states that this is a harness re-score of the
v0.8.0 adapter, in the same terms v0.8.0 used for the v0.7.0 adapter, and
`result-v0.8.0.md` stays as it was measured. The alternative is to retrain
first, which costs about 2.6 hours and makes the v0.9.0 numbers cover a new
adapter as well; the harness fix is worth 2 builds and 2 tests either way.

## Scoped, not built: extraction

The v0.7.0 changelog said the extraction track was finished. Two measured
results say otherwise: the corpus barely demonstrates the task the benchmark
poses, and nothing in it demonstrates the dimension the benchmark scores for
unit tests. Every volume below was counted in the cache, not estimated.
Ordered by value per unit of work against the v0.8.0 numbers.

### 1. Write-shaped turns for the evaluation prompt (highest value)

Only 94 of 72,464 train turns (0.13%) have the shape of the evaluation prompt:
a project listing in, updated files out. Separately, 5,274 turns (7.3%) are at
least 95% covered by the 8-gram shingles of their own user turn, and those sit
in the question kinds, where quoting the code under discussion is correct. So
the no-op reply mode is not taught directly by restatement; it is the default
when the corpus has no demonstration of acting, and the fix is the eval prompt's
own shape.

Scope: a `file_block_pair` turn kind, built from the pairs the builder already
finds. The user turn is the project tree in `File: <path>` + fenced-block form
(exactly what `build_user_prompt` sends); the assistant turn is the same Ada
the existing builders already extract, emitted as one entry per changed file.
Reuse the pair, contract and variant builders for content and add only the
framing. The v0.6.0 acceptance criterion ("reply-format compliance rises
without harness repair") is met by other means: format compliance is already
19/19. The criterion for this version is build and test counts on the samples
the model changed, currently 3 of 6 and 2 of 6.

### 2. Unit-test turns from the harnesses already in the cache (1,858 files)

Nothing in the corpus trains the unit-test dimension, and the benchmark scores
it. `Ada-Algorithms` alone has 1,809 `tests.adb`/`tests.ads` files (836 inside
`SPARK2` trees, 707 using `Ada.Assertions.Assert`), each a spec-under-test plus
input aggregate plus `Assert (...)` pair:

```ada
Result : constant Input_Array := Sort (Input);
   for I in Index loop
      if I < Index'Last then Assert (Result (I) <= Result (I + 1)); end if;
```

Scope: a `gnattest_pair` turn kind pairing a unit's `.ads` with the
`tests.adb` beside it. `parse_ada_ast.pair_subprograms` already pairs
`(package, kind, name)` correctly and can find the unit under test from the
`with <Unit>;` clause, so the new code is the turn builder. The shape is the
one `AdaCore/skills` documents in `test_skeletons.md` and
`test-separate-drivers.md`, which are already loaded as guidance and never
trained against. Caveat to check first: `Ada-Algorithms` is the one repo the
eval guard already fires on, so expect the drop count in
`dataset_metadata.json` to move.

### 3. Ada 83 conformance suite as error-diagnosis turns (4,142 files)

`Ada-83-TLALOC/acvc83_11/` holds the ACVC83 suite. `btests/` alone is 1,515
files with **12,091** `-- ERROR:` markers stating the violated rule inline
(`RAISE;  -- ERROR: NON-SPECIFIC RAISE OUTSIDE HANDLER.`) and 1,403 files
carry `-- OBJECTIVE:` headers stating what an RM rule requires. The v0.8.0
blocker histogram is `VC_OVERFLOW_CHECK` x5 and `UNINITIALIZED` x4 first, so
diagnosis of a violated rule is the matching skill.

Scope: an `ada83_diag` turn kind. User = "this unit must be rejected by the
compiler", assistant = the violated rule and why, from the inline markers;
plus `objective_to_test` turns built from the `ctests`/`atests`/`ltests`
bodies (2,181 files self-check with `REPORT.TEST`/`FAILED`). Largest volume in
the whole cache and it matches how the eval reports failures.

Hard constraint: `btests/` is *intentionally illegal* Ada. It must never enter
`code_pair` or reach `inject_defect`, or the 17 defect families will "repair"
already-broken code into wrong targets. `.ada` is not in
`ADA_SPEC_EXTENSIONS` today, which is why all 4,142 files are invisible to
every extractor; add a dedicated extension constant and a dedicated builder
rather than widening the code pool.

### 4. The doc trees no extractor reads (~11,000 sections)

`default_doc_dirs()` is `learn` + `training_material` only. Adding the rest
needs no parser change:

| Repo | files | sections | `doc_section` turns |
|---|---|---|---|
| `Ada-Algorithms` | 1,851 | 11,414 | ~8,266 |
| `adacovex` | 216 | 2,053 | ~1,575 |
| `Ada_CRDT` | 95 | 1,106 | ~763 |
| `AdaCore/skills` | 71 | 662 | ~519 |
| `ada-spark` | 9 | 94 | ~83 |

`Ada-Algorithms/misc/Ada/*/README.md` (1,130 files) carries algorithm
overviews and component tables. Low risk, low reward: it is prose, and
`doc_section` is already the largest-prose kind in the corpus at 5,379.

### 5. Contract-documentation turns from the generated AdaDoc pages (111 files)

`adacovex/docs/api-docs/` (75 files) and `Ada_CRDT/docs/api-docs/` (36) are
generated API pages: package description, per-parameter table, `**Returns:**`
semantic line. `_SPARK_ASSURANCE_DOC_DIRS` lists `docs/proof`,
`docs/compliance`, `docs/usage`, `docs/archive` and not `docs/api-docs`, so
none of it is read. Two products: "document this declaration" turns with a
real target, and a much richer source for `assurance_qa` than the 5 turns it
produces today (`adacovex-spark-levels.md` spells out all five SPARK levels
with their `gnatprove` commands).

### 6. AdaDoc comment blocks above declarations (721 declarations, ~20k words)

`parse_ada_ast` starts at the declaration keyword, so the comment above it is
discarded. 673 of those declarations are in `Ada-Algorithms` (18,488 words),
21 in `Ada_CRDT`, 20 in `adacovex`. Gives the write side of the `gnatdoc`
guidance that is already loaded. Requires touching both the regex path and the
libadalang path (`parent.text` also excludes the comment), so it is the one
item here with parser risk.

### 7. Build-configuration turns from `.gpr` files (2,254 files, 4 read)

`discover_files` collapses `.gpr` to the first readable file per input
directory, used only as trailing context. `Ada-Algorithms` has 2,260, with
`-gnat2022` in 1,026 and `-gnata` in 885. Turn kind: "which standard does
this project target, which warnings does it enable", answered from the file,
plus the reverse (write the `.gpr` for a SPARK project at level 2).

### 8. `training_material` Ada that no builder reaches (363 files)

Only the `prompt/`+`answer/` lab twins are read. Unread: 63 files in
`courses/ada_essentials/examples`, 54 in `mini_projects`, 28 in `quiz`, and
208 in `labs/` trees with no prompt/answer pair (`gnatsas`, `gnatdas`,
`gnat_project_facility`). The `labs/source/` vs `labs/solution/` layout is the
natural twin of the prompt/answer rule that already exists.

### Not worth it

- `SimpleEnglish`'s 430 raw model outputs on STE tasks: other models'
  generations, not Ada.
- `TLALOC`'s 169 `.finc` / 112 `.sub` / 44 `.dcl`: TLALOC's own register DSL.
- `Ada-Algorithms/*/SPARK2/*/.spark`, `.gnat-json`, `.cswi`, `.sarif`:
  prover provenance for units already proved; useful only as a filter.
- The 279 `spark_verified` candidates `gen_verified_spark.py` drops for
  `with`-ing a sibling: real SPARK, but proving them needs a per-directory
  synthetic project, and 568 already-proved units have not yet been shown to
  raise the proved count.

## Rules for this track

- Every new turn kind gets a unit test per branch, and a count in
  `docs/datasets-and-training.md`.
- Every new source directory updates the
  [`docs/data-provenance.md`](docs/data-provenance.md) source table, then
  `make check-integrity` and `make validate-defects` must pass.
- No new turn kind may lower reply-format compliance, the BUILD/TEST numbers,
  or the build count on the samples the model changed. That last one is the
  bar this version exists to clear: a run that reaches 16/19 build by leaving
  the base project in place is not the win v0.4.1's 16/19 was.
- Numbers come from `make eval-report` only. A claim without a
  `docs/results/result-vX.Y.Z.md` is prose, not a result.

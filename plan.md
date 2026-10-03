# plan.md - v0.8.0 scope

State of the 2026-10-03 session: the v0.7.0 run is measured
([`result-v0.7.0.md`](docs/results/result-v0.7.0.md)) and it went backwards.
The harness and two extraction bugs found while measuring it are fixed here,
the test suite grew from 406 to 561 cases, and this file scopes the
extraction work that is deliberately **not** in this change.

The one-line summary: v0.7.0 bought prover-shaped Ada and lost the multi-file
reply shape the harness parses, so six of nineteen fine-tuned samples failed
BUILD for a routing mistake. v0.8.0 makes the harness state the truth about
that, and adds the training signal the model is actually missing (its own
`File:` contract, and unit tests).

## Done in this change

### The regression: a body written into a spec

Every ada-eval sample targets a spec (`.ads`) file, and the fine-tune often
replies with the unit's *body*. Written to the spec it becomes
`package body ... end Foo;` inside a package spec, which GNAT rejects, so the
sample fails BUILD for a routing mistake rather than for wrong Ada. Six of
nineteen fine-tuned replies did this on the v0.7.0 run and all six failed;
the base model emitted the requested `File: <path>` format on all nineteen.

`eval/generate.py` now routes a body-only fallback block to the sibling
`.adb`, and only when the base tree already carries that mirror file, so the
repair never invents a file the project did not have. An explicit
`File:` entry still wins: when the model names the file, its choice is
authoritative.

A harness repair is only defensible if it cannot flatter the model, so the
reply shape is now recorded per sample
(`outputs/generated_solutions/<label>/generation_meta.json`) and reported by
`make eval` next to the scores: `Reply format: file_blocks=4,
fenced_block=15 (untouched 9/19)`. The "untouched" count matters as much as
the format count, because a reply that only echoes the base file leaves the
base project in place and is then scored on the base tree's build result.

### Two pipeline gaps that produced wrong numbers quietly

- **`make eval` scored against a hollow benchmark.** ada-eval's compacted
  JSONL is derived, not version controlled, so a re-fetched cache has none
  (`make eval` exited 1, loudly). Worse, ada-eval's packer is git-aware: the
  cache sits inside this repository where `data/raw_repos/` is gitignored, so
  `git ls-files` returns nothing and the pack *succeeds* while writing
  records with no `canonical_solution`. `make eval` then reported BLEU 0.0
  with every standard `Unknown`, which looks like a model result.
  `scripts/pack_eval_data.py` (new, wired as `make eval-data`) packs with
  `GIT_CEILING_DIRECTORIES` set, passes `force=True`, and **verifies** that
  every record carries solution files. `load_reference_index` drops hollow
  records and errors out rather than scoring them.
- **`make all` reported the previous run's tallies.** `baseline_eval` prints
  BUILD/TEST/PROVE but reads them from `outputs/eval_results/`, which only
  `eval_pipeline.py` writes; `all` ran `eval` first, so the summary showed
  stale numbers, and on a clean tree `print_stats_block` printed nothing at
  all, which reads as "nothing failed". The order is now
  `generate eval-pipeline eval eval-report`, the block prints
  `no results (run make eval-pipeline first)` instead of nothing, and
  `tests/test_makefile.py` fails if the order changes again.

### Two extraction bugs

- **ada-spark guidance was silently dropped.** `_SKILL_DOC_PATTERNS` matched
  the repo root (`SKILL.md`, `agent-knowledge/*.md`) while the repo keeps its
  skill under `skills/ada-spark/`, the layout the other two repos already
  used. Two files were loaded; seven were missed, including
  `references/spark-proof.md` and `references/toolchain.md`, which is where
  the proof guidance lives. One-line glob fix, now 2 -> 6 documents.
- **138 `code_pair` turns paired a spec with an unrelated body.**
  `pair_files` indexed bodies by stem alone, so `sorting/sort.ads` could pair
  with `search/sort.adb`. The key is now `(directory, stem)`.
  `extract_package_name` had a related bug: its regex required the word
  "package" twice, so `package body Foo is` never matched and every body
  file fell back to the filename stem.

### Tests: 406 -> 561

| File | Before | After | Covers |
|---|---|---|---|
| `tests/test_eval_pipeline.py` | 0 | 37 | the BUILD/TEST/PROVE driver: per-dataset output path, failure isolation, exit codes, report rendering |
| `tests/test_pack_eval_data.py` | 0 | 15 | the benchmark pack: ceiling variable, `force`, hollow-pack detection |
| `tests/test_makefile.py` | 0 | 16 | `all` ordering, `eval` -> `eval-data`, lint coverage, help completeness |
| `tests/test_generate.py` | 39 | 73 | body routing, reply classification, the overlay, echo skip, the sidecar |
| `tests/test_eval_scoring.py` | 34 | 61 | hollow-reference refusal, shape reporting, `print_stats_block`, helpers |
| `tests/test_build_dataset.py` | 62 | 81 | directory-aware pairing, package-name extraction, skill globs |
| `tests/test_defect_families.py` | 41 | 48 | aggregate-position declarations the `typo` and `scoping` families must decline |

`make lint` now type-checks `eval/generate.py` and `eval/eval_pipeline.py`
too (they were the only unscanned Python in the repo); `MYPYPATH` gained
`scripts` so `alire_env` and friends resolve.

## Scoped, not built: extraction for v0.8.1

The v0.7.0 changelog said the extraction track was finished. The measured
result says otherwise, and this is the inventory that says so. Every volume
below was counted in the cache, not estimated.

Ordered by value per unit of work.

### 1. `File:`-headed answers in the curriculum (highest value)

The format regression is a *data* problem wearing a harness costume. The
model answers with one fenced block because that is what every training
answer looks like. The eval prompt states the `File: <path>` contract; the
corpus never demonstrates it.

Scope: a `file_block_pair` turn kind. The user turn shows a project tree in
the `File: <path>` + fenced-block shape (exactly what
`build_user_prompt` sends) and asks for the updated files; the assistant turn
is the same Ada the existing builders already extract, emitted as one entry
per changed file. Reuse the pair/contract/variant builders for content and
only add the framing. Cheapest fix that addresses the measured regression
directly. Acceptance: reply-format compliance on the benchmark rises without
harness repair.

### 2. Unit-test turns from the harnesses already in the cache (1,858 files)

Nothing in the corpus trains the unit-test dimension, and the benchmark
scores it. `Ada-Algorithms` alone has 1,809 `tests.adb`/`tests.ads` files
(836 inside `SPARK2` trees, 707 using `Ada.Assertions.Assert`), each a
spec-under-test plus input aggregate plus `Assert (...)` pair:

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
carry `-- OBJECTIVE:` headers stating what an RM rule requires.

Scope: a `ada83_diag` turn kind. User = "this unit must be rejected by the
compiler", assistant = the violated rule and why, from the inline markers;
plus `objective_to_test` turns built from the `ctests`/`atests`/`ltests`
bodies (2,181 files self-check with `REPORT.TEST`/`FAILED`). Largest volume
in the whole cache and it matches how the eval reports failures.

Hard constraint: `btests/` is *intentionally illegal* Ada. It must never
enter `code_pair` or reach `inject_defect`, or the 17 defect families will
"repair" already-broken code into wrong targets. `.ada` is not in
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
guidance that is already loaded. Requires touching both the regex path and
the libadalang path (`parent.text` also excludes the comment), so it is the
one item here with parser risk.

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
`gnat_project_facility`). The `labs/source/` vs `labs/solution/` layout is
the natural twin of the prompt/answer rule that already exists.

### Not worth it

- `SimpleEnglish`'s 430 raw model outputs on STE tasks: other models'
  generations, not Ada.
- `TLALOC`'s 169 `.finc` / 112 `.sub` / 44 `.dcl`: TLALOC's own register DSL.
- `Ada-Algorithms/*/SPARK2/*/.spark`, `.gnat-json`, `.cswi`, `.sarif`:
  prover provenance for units already proved; useful only as a filter.
- The 279 `spark_verified` candidates `gen_verified_spark.py` drops for
  `with`-ing a sibling: real SPARK, but proving them needs a per-directory
  synthetic project, and 568 already-proved units are not yet shown to help.

## Rules for this track

- Every new turn kind gets a unit test per branch, and a count in
  `docs/datasets-and-training.md`.
- Every new source directory updates the
  [`docs/data-provenance.md`](docs/data-provenance.md) source table, then
  `make check-integrity` and `make validate-defects` must pass.
- No new turn kind may lower reply-format compliance or the BUILD/TEST
  numbers; that is the bar this version exists to clear.
- Numbers come from `make eval-report` only. A claim without a
  `docs/results/result-vX.Y.Z.md` is prose, not a result.

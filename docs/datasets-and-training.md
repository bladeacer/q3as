# Datasets and Training

How the q3as dataset is built, what each turn teaches, and how training is
configured for reproducibility and early stopping.

## Training turn kinds

| Kind | What it teaches |
|---|---|
| `code_pair` | Spec/body completion from real Ada trees |
| `defect_pair` | Correct code next to a deliberately broken variant, with diagnosis, the GNAT-verified compiler message, and the fix |
| `doc_qa` | Ada code blocks from AdaCore course material, with a completion answer and an STE-compliant explanation answer |
| `doc_section` | Heading-chunked sections of course material ([`parse_docs.py`](../data/processing_scripts/parse_docs.py)) |
| `ast_qa` | AST-derived turns ([`parse_ada_ast.py`](../data/processing_scripts/parse_ada_ast.py)): body-from-spec completion, contract reading, **contract writing** (bare spec in, `Pre`/`Post`/`Global`/`Depends` declaration out), and constrained types. The parser emits these as `ast_impl`, `ast_contract`, `ast_contract_write` and `ast_type`; the builder records them under the single dataset kind `ast_qa`, which is the name that appears in `dataset_metadata.json` |
| `toolchain_qa` | Question/answer turns from the AdaCore agent skills |
| `contract_synth` | gnatprove-verified synthetic contract turns ([`scripts/gen_contract_mutations.py`](../scripts/gen_contract_mutations.py)): contract writing, why-weakened-contracts-fail, and fix turns |
| `ast_defect` / `variant` | Defect and renamed-variant turns derived from ingested parser records (same split group as their source record) |

### How AST units are extracted

[`parse_ada_ast.py`](../data/processing_scripts/parse_ada_ast.py) has two
interchangeable backends behind one record shape. It uses **libadalang** when
the shared library is present (`make ast-deps`, see
[toolchain setup](toolchain-setup.md)) and its **structural scanner** otherwise,
so the dataset builds either way.

The difference is precision, not reach. libadalang parses the unit, so it
resolves the enclosing package through the real tree, separates
declarations from bodies, knows whether a spec is syntactically valid
(recorded as `valid`), and reads aspect clauses off the declaration rather
than by pattern matching. The scanner is regex-based and approximates the
same fields.

Both backends emit the same keys, and the turn kinds above do not change between
them, so switching backends changes how many units are found but not what a turn
looks like. Two behaviours are deliberate and shared: anonymous access types
declare an unnamed dereference function, which has no name to train on and is
skipped by both; and the aspect clause is parsed by the same `_extract_aspects`
helper in both, so the `Pre`/`Post` maps match.
[`tests/test_parsers.py`](../tests/test_parsers.py) pins the libadalang path
when it is installed and asserts the two backends agree on subprogram names.

## Natural-language variety, code exactness

Explanations, questions, and diagnoses vary: the same question is phrased
several ways (chosen content-deterministically), and doc sections are
rephrased into Simplified Technical English. Ada code itself is not
paraphrased: the language has one strict syntax, so variety comes from
*which* real code is shown, not from rewriting it. The one deliberate
exception is the defect families below, where wrong code is generated on
purpose.

## Defect families (incorrect-example generation)

[`build_dataset.py`](../data/processing_scripts/build_dataset.py) injects
wrong-code variants next to correct code. Each family's claimed compiler message
is GNAT-verified by
[`scripts/validate_defects.py`](../scripts/validate_defects.py) (`make
validate-defects`); any family that "compiles clean" fails the check.

## Parser outputs and provenance

The parser modules write standalone JSONL files under `data/processed/`
(`docs_chunks`, `ada_ast_units`, `contract_mutations` from `make
parse-data` and `make gen-contracts`). `make build-dataset`
merges all of them via `--extra-turns`. The build metadata
(`dataset_metadata.json`) records per-file ingestion counts under
`extra_turns_files` (`records` / `ingested` / `defects` / `variants`) and
per-input-dir pair counts under `records_by_input_dir`, so a missing or
empty parser output is visible instead of vanishing silently. Records
dropped by the eval guard or by dedup do not appear in any count. The
per-kind tally is taken at ingestion, *before* those two passes, so the
kinds sum to more than the split totals: the difference is exactly
`deduped_duplicates` plus `eval_guard.dropped_records`, both recorded in
`dataset_metadata.json`.
Empty-assistant records ("provide the body" questions with no answer,
from spec-only or impl-only sources) are dropped at build time and at
training-load time; their count is recorded under
`empty_assistant_dropped` and per file under `empty_assistant`.

### Rebuild skipping

`parse-data`, `gen-contracts`, and `build-dataset` are `.PHONY` targets, so
make runs them on every `make all`. Each stage therefore declares what it
reads in [`stage_state.py`](../data/processing_scripts/stage_state.py): the
input trees (with the file suffixes the stage actually consumes), single input
files, the scripts that produce it, and the parameters that change the
output. After a successful run the fingerprint is written to
`data/processed/.stages/<stage>.json`, and the next run recomputes it and
skips the work when it still matches.

Four properties make the "fresh" verdict safe:

- **Content, not timestamps.** Trees are hashed by content. The archive
  cache is re-extracted from tarballs, so mtimes change when nothing did;
  content hashing keeps a refetched repo from forcing a rebuild. The whole
  corpus (about 10,000 files) hashes in roughly half a second.
- **Scripts are inputs.** Editing `build_dataset.py`, `code_variants.py`,
  `eval_guard.py`, `source_paths.py`, or a parser invalidates every stage
  that imports it, so a logic change never keeps stale output.
- **Missing is not fresh.** A stage is skipped only when every declared
  output exists, is non-empty, and still has the size recorded when it was
  written. Deleting or truncating `dataset.jsonl` always rebuilds.
- **`--workers` is excluded.** The stages produce identical output for any
  worker count, so changing `DATASET_WORKERS` must not invalidate a build.
  A test asserts no stage puts it in its fingerprint.

`FORCE=1` (or `--force` on the scripts) rebuilds regardless, which is what to
use after a toolchain change: `gnatprove` is a binary, not a hashed source
input, so upgrading the prover does not invalidate `contract_mutations`.
`make clean` removes the stamps along with the dataset.

Effect on the full pipeline, measured on 16 cores over the whole corpus
(6,514 Ada files, 4,337 pairs, 73,080 turns):

| Step | Before | After |
|---|---|---|
| `make parse-data build-dataset`, nothing changed | 11 min 30 s | 2 s |
| `parse_ada_ast` extract phase | 7 min 38 s (1 worker) | 1 min 08 s (8) / 51 s (16) |
| `build_dataset` pair phase | 24 s (1 worker) | 5 s (8) |
| `DATASET_WORKERS` default | 1 | 0 (one per core) |

Worker memory is about 90 MiB each and the builder's parent peaks near
1.7 GiB holding every record, so cores run out before RAM does. The phases
left after parallelising are serial by nature: the eval guard (about 84 s),
ingesting the parser JSONL (30 s), and dedup (13 s).

[`progress.py`](../data/processing_scripts/progress.py) keeps the wait
legible. Long stages log a named phase per step (`[extract] starting
(files=6514, workers=8)` then `[extract] done in 1m08s`) and, inside long
loops, an item count with a rate and an ETA at a bounded cadence. The build
no longer prints nothing for minutes, which used to look like a hang.

| Family | Example defect |
|---|---|
| `syntax` | Misspelled keyword (`procedur`) |
| `context` | Missing `with Ada.Text_IO;` |
| `visibility` | `use` clause removed |
| `mismatch` | Spec/body contract or profile differs |
| `contract` | Broken `Pre`/`Post` aspect |
| `typo` | Wrong identifier at a use site (`Coutn := ...`) |
| `hallucinated` | Call to a nonexistent predefined unit (`Ada.Strings.Tensor`) |
| `ordering` | Declaration used after `begin` |
| `scoping` | Inner block shadows / leaves scope |
| `type` | Numeric value assigned to a Boolean |
| `lang_confusion` | `=` used where `:=` is required |
| `wrong_ref` | Component name corrupted in a predefined-unit reference (`Ada.Text_IO.Foo_Line`) |
| `nonexistent_call` | Locally declared procedure renamed at its call site |
| `bad_typing` | String literal assigned to a numeric object |
| `arity` | Extra argument added to a call |
| `stray_aspect` | `with Pre => True;` inside a package declarative part |
| `old_misuse` | `'Old` read inside a precondition |

The families also apply to AST-extracted units, so contract turns, impl
turns, and doc-QA code all have broken counterparts where applicable.

## Splits and leakage control

[`build_dataset.py`](../data/processing_scripts/build_dataset.py) writes
`dataset.jsonl` plus `dataset_{train,val,test}.jsonl` (~90/5/5). Three
mechanisms keep the splits honest:

1. **Group-aware split** - every turn derived from one source unit (a code
   pair and its defect pairs, the doc sections of one file) stays in one
   split.
2. **Dedup before split** - two duplicate families are removed:
   verbatim copies generated from different groups, and AST-derived
    records whose assistant code has the same alpha-renamed shape beyond
    a small cap (`AST_STRUCTURAL_CAP`, currently 10): the Ada-Algorithms
   corpus is templated, and hundreds of its records are the same
   algorithm under different identifier spellings. Deliberately kept:
   the dataset's intended variety - variant turns (renamed/reordered
   code with their own wording), the plain vs STE paraphrase pairs, and
   identical user prompts with different answers. Dedup reports its
   breakdown (`exact`, `ast_structural_capped`) in the build metadata;
   duplicates cannot straddle splits because the first occurrence wins.

   The cap value is evidence-based, not a guess. Two rounds of a controlled
   experiment
   ([`scripts/build_cap_variants.sh`](../scripts/build_cap_variants.sh) +
   [`scripts/run_cap_experiment.sh`](../scripts/run_cap_experiment.sh)) trained
   identical 40-step QLoRA probes (same seed, LR, batch 1x8, 10-step eval
   schedule) on datasets that differ only in the cap; a probe takes 14 to 18
   minutes per cap on the reference GPU. Exact repro of both rounds: `STEPS=40
   ACC=8 EVAL_STEPS=10`.

   Round 1 screened all caps on a 40-example probe carved from the
   then-current cap-2 build's val/test splits:

   | cap | val loss (step 40) | test loss | test ppl |
   |----:|-------------------:|----------:|---------:|
   | 1   | 1.310              | 1.333     | 3.79     |
   | 2   | 1.050              | 1.070     | 2.92     |
   | 3   | 1.117              | 1.128     | 3.09     |
   | 5   | 1.052              | 1.065     | 2.90     |
   | 10  | 1.038              | 1.041     | 2.83     |
   | inf | 1.169              | 1.187     | 3.28     |

   Caps 1, 3, and inf lost clearly (too little variety at cap 1,
   unbounded duplication at inf), and caps 2, 5, and 10 finished within a
   few percent of each other. The round-1 probe was later found flawed:
   split slices depend on the post-dedup group list, so the same source
   content moves between train and val across builds with different caps,
   and 20 to 36 of the 40 probe records sat inside the arms' training
   data (verified afterwards by content hash). That flatters whichever
   arm memorizes more of the probe, so the near-tie was not trustworthy.

   Round 2 re-scored caps 2, 5, and 10 on a leak-free 121-record probe (55 val +
   66 test, every record content-checked absent from all three arm train sets;
   built with [`scripts/make_probe_splits.py`](../scripts/make_probe_splits.py),
   run with `OUT=outputs/capexp_big DATA_ROOT=outputs/capexp/datasets` plus the
   probe overrides):

   | cap | val loss (step 40) | test loss | test ppl |
   |----:|-------------------:|----------:|---------:|
   | 2   | 1.057              | 0.945     | 2.57     |
   | 5   | 1.052              | 0.935     | 2.55     |
   | **10** | **0.972**       | **0.846** | **2.33** |

   Cap 10 wins by 8 to 10 percent on both splits, an order larger than the
   round-1 spread, with the same ranking (5 above 2, 10 on top). The default
   `AST_STRUCTURAL_CAP` is 10. Results CSVs: `outputs/capexp/results.csv` (round
   1) and `outputs/capexp_big/ results.csv` (round 2), regenerable and not
   committed; see
   [`scripts/collect_cap_results.py`](../scripts/collect_cap_results.py).
   Per-cap summaries live in `outputs/capexp{,_big}/run_cap*/`.
3. **Eval guard** - ada-eval benchmark content is dropped before splitting
   (see [Data provenance](data-provenance.md)).

`make check-integrity` fails if any split file contains eval content.

## Training configuration

[`training/train_unsloth.py`](../training/train_unsloth.py):

- **Seed 42** everywhere: `SFTConfig(seed=...)`, LoRA `random_state`,
  dataset shuffling. Content-derived randomness in the builder is
  deterministic too, so dataset bytes are identical for any worker count.
- **8 GB host profile**: the training window is 1024 tokens. Dataset
  tokenization uses one process, the data loader uses no workers, and
  checkpoints omit optimizer state. The `make` pipeline also uses one
  dataset worker and saves the LoRA adapter without a merged 16-bit export.
- **Splits**: trains on `dataset_train.jsonl` by default; val loss is
  computed every 50 steps (`--eval-steps`); a fallback seeded carve-out
  protects custom single-file datasets.
- **Early stopping**: patience 10 evals with threshold 0.001 - training
  stops when val loss shows no meaningful gain for 10 consecutive evals.
  Validation and test losses are computed by a chunked callback that
  avoids materializing logits on GPU (8 GB VRAM constraint), so
  Trainer-managed evaluation stays off. The exported adapter is the
  final state at stop time, not a reloaded best checkpoint.
- **Test metrics**: after training, test-split loss and perplexity are
  computed and written to `training_summary.json`.
- **Loss histories**: `training_summary.json` carries the full train-loss
  log (`train_loss_history`), every eval point (`eval_loss_history`), the
  final train loss, and the test metrics, so the report can judge the run
  without re-parsing trainer state.

Reproducibility caveat: GPU kernels (FlashAttention, cuBLAS) are not
bit-deterministic; seed 42 gives functional, not bitwise, reproducibility.

## Result reporting: training metrics and trend analysis

`make eval-report`
([`scripts/gen_eval_report.py`](../scripts/gen_eval_report.py)) extends each
`docs/results/result-vX.Y.Z.md` with a **Training metrics** section built from
`outputs/q3as/training_summary.json`:

- the traditional loss table: train / validation (best) / test loss and
  perplexity,
- a **trend verdict** computed from the curves, with the numbers it rests
  on:
  - `healthy` - validation loss improves overall and its best value sits
    in the second half of the eval steps,
  - `plateau` - the best value came early, or improvement stopped while
    train loss had also stopped falling (early stopping did its job),
  - `overfit` - validation loss rises more than 5 percent past its best
    while train loss keeps falling,
  - `unstable` - a validation-loss step jumps upward by more than 25
    percent,
  - `sparse` - fewer than two evaluation points, no trend to judge,
- the per-step validation-loss table, so a reader can audit the verdict.

The versioned JSON (`result-data-vX.Y.Z.json`) carries the same data
(`training.train_loss`, `training.val_loss`, `training.test_loss`,
`training.trend`, plus the raw histories); the results-index comparison
table adds the fine-tuned test loss and trend verdict per version.

## Validation loop

```bash
make test               # unit tests (builders, parsers, guard, injectors)
make lint               # ruff + mypy, markdown link check, AGENTS tree check
make validate-defects   # GNAT-compiles defect pairs, checks claimed messages
make check-integrity    # fails if any split contains eval content
```

Navigation: [project README](../README.md) · [docs index](README.md) · [changelog index](changelogs/index.md) · [results index](results/README.md)

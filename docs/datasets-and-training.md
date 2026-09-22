# Datasets and Training

How the q3as dataset is built, what each turn teaches, and how training is
configured for reproducibility and early stopping.

## Training turn kinds

| Kind | What it teaches |
|---|---|
| `code_pair` | Spec/body completion from real Ada trees |
| `defect_pair` | Correct code next to a deliberately broken variant, with diagnosis, the GNAT-verified compiler message, and the fix |
| `doc_qa` | Ada code blocks from AdaCore course material, with a completion answer and an STE-compliant explanation answer |
| `doc_section` | Heading-chunked sections of course material (`parse_docs.py`) |
| `ast_impl` / `ast_contract` / `ast_contract_write` / `ast_type` | AST-derived turns (`parse_ada_ast.py`): body-from-spec completion, contract reading, **contract writing** (bare spec in, `Pre`/`Post`/`Global`/`Depends` declaration out), and constrained types |
| `toolchain_qa` | Question/answer turns from the AdaCore agent skills |
| `contract_synth` | gnatprove-verified synthetic contract turns (`scripts/gen_contract_mutations.py`): contract writing, why-weakened-contracts-fail, and fix turns |
| `ast_defect` / `variant` | Defect and renamed-variant turns derived from ingested parser records (same split group as their source record) |

## Natural-language variety, code exactness

Explanations, questions, and diagnoses vary: the same question is phrased
several ways (chosen content-deterministically), and doc sections are
rephrased into Simplified Technical English. Ada code itself is not
paraphrased: the language has one strict syntax, so variety comes from
*which* real code is shown, not from rewriting it. The one deliberate
exception is the defect families below, where wrong code is generated on
purpose.

## Defect families (incorrect-example generation)

`build_dataset.py` injects wrong-code variants next to correct code. Each
family's claimed compiler message is GNAT-verified by
`scripts/validate_defects.py` (`make validate-defects`); any family that
"compiles clean" fails the check.

## Parser outputs and provenance

The parser modules write standalone JSONL files under `data/processed/`
(`docs_chunks`, `ada_ast_units`, `hub_ast_units` from `make parse-data`,
`contract_mutations` from `make gen-contracts`). `make build-dataset`
merges all of them via `--extra-turns`. The build metadata
(`dataset_metadata.json`) records per-file ingestion counts under
`extra_turns_files` (`records` / `ingested` / `defects` / `variants`) and
per-input-dir pair counts under `records_by_input_dir`, so a missing or
empty parser output is visible instead of vanishing silently. Records
dropped by the eval guard or by dedup do not appear in any count.
Empty-assistant records ("provide the body" questions with no answer,
from spec-only or impl-only sources) are dropped at build time and at
training-load time; their count is recorded under
`empty_assistant_dropped` and per file under `empty_assistant`.

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

`build_dataset.py` writes `dataset.jsonl` plus
`dataset_{train,val,test}.jsonl` (~90/5/5). Three mechanisms keep the
splits honest:

1. **Group-aware split** - every turn derived from one source unit (a code
   pair and its defect pairs, the doc sections of one file) stays in one
   split.
2. **Dedup before split** - two duplicate families are removed:
   verbatim copies generated from different groups, and AST-derived
   records whose assistant code has the same alpha-renamed shape beyond
   a small cap (`AST_STRUCTURAL_CAP`, currently 2): the algorithm-hub
   corpus is templated, and hundreds of its records are the same
   algorithm under different identifier spellings. Deliberately kept:
   the dataset's intended variety - variant turns (renamed/reordered
   code with their own wording), the plain vs STE paraphrase pairs, and
   identical user prompts with different answers. Dedup reports its
   breakdown (`exact`, `ast_structural_capped`) in the build metadata;
   duplicates cannot straddle splits because the first occurrence wins.

   The cap value is evidence-based, not a guess. A controlled experiment
   (`scripts/build_cap_variants.sh` + `scripts/run_cap_experiment.sh`)
   trained identical 40-step QLoRA probes (same seed, LR, batch
   (1x8, acc 8), and a fixed 40-example val/test) on datasets that differ
   only in the cap. Exact repro: `STEPS=40 ACC=8 EVAL_STEPS=10`; a probe
   takes 14 to 18 minutes per cap on the reference GPU.

   | cap | val loss (step 40) | test loss | test ppl |
   |----:|-------------------:|----------:|---------:|
   | 1   | 1.310              | 1.333     | 3.79     |
   | **2** | **1.050**        | 1.070     | 2.92     |
   | 3   | 1.117              | 1.128     | 3.09     |
   | 5   | 1.052              | 1.065     | 2.90     |
   | 10  | 1.038              | 1.041     | 2.83     |
   | inf | 1.169              | 1.187     | 3.28     |

   Caps 1, 3, and inf lose clearly: too little variety at cap 1, unbounded
   duplication at inf. Caps 2, 5, and 10 finish within a few percent of
   each other on the 40-example probe; cap 10 edges out cap 2 (test loss
   1.041 vs 1.070, ~18% more records), so cap 2 stays the default for
   now: the probe cannot separate them with confidence, and the smaller
   dataset carries less templated repetition. Switching to cap 10 is the
   standing candidate for the next iteration; it changes the dataset, so
   it needs a rebuild plus `make check-integrity` and a retrain. Results
   CSV: `outputs/capexp/results.csv` (regenerable, not committed; see
   `scripts/collect_cap_results.py`), per-cap summaries in
   `outputs/capexp/run_cap*/`.
3. **Eval guard** - ada-eval benchmark content is dropped before splitting
   (see [Data provenance](data-provenance.md)).

`make check-integrity` fails if any split file contains eval content.

## Training configuration

`training/train_unsloth.py`:

- **Seed 42** everywhere: `SFTConfig(seed=...)`, LoRA `random_state`,
  dataset shuffling. Content-derived randomness in the builder is
  deterministic too, so dataset bytes are identical for any worker count.
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

`make eval-report` (`scripts/gen_eval_report.py`) extends each
`docs/results/result-vX.Y.Z.md` with a **Training metrics** section built
from `outputs/q3as/training_summary.json`:

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
make lint               # ruff + mypy over all project sources
make validate-defects   # GNAT-compiles defect pairs, checks claimed messages
make check-integrity    # fails if any split contains eval content
```

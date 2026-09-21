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

The families also apply to AST-extracted units, so contract turns, impl
turns, and doc-QA code all have broken counterparts where applicable.

## Splits and leakage control

`build_dataset.py` writes `dataset.jsonl` plus
`dataset_{train,val,test}.jsonl` (~90/5/5). Three mechanisms keep the
splits honest:

1. **Group-aware split** - every turn derived from one source unit (a code
   pair and its defect pairs, the doc sections of one file) stays in one
   split.
2. **Dedup before split** - identical records generated from different
   groups (the same spec reached through two source paths) are dropped, so
   duplicates cannot straddle splits.
3. **Eval guard** - ada-eval benchmark content is dropped before splitting
   (see [Data provenance](data-provenance.md)).

`make check-integrity` fails if any split file contains eval content.

## Training configuration

`training/train_unsloth.py`:

- **Seed 42** everywhere: `SFTConfig(seed=...)`, LoRA `random_state`,
  dataset shuffling. Content-derived randomness in the builder is
  deterministic too, so dataset bytes are identical for any worker count.
- **Splits**: trains on `dataset_train.jsonl` by default; val loss is
  computed every 50 steps; a fallback seeded carve-out protects custom
  single-file datasets.
- **Early stopping**: patience 10 evals with threshold 0.001 - training
  stops when val loss shows no meaningful gain for 10 consecutive evals;
  the best checkpoint is restored.
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

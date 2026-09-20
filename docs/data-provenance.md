# Data Provenance and Eval Integrity

Where q3as training data comes from, what licenses apply, and how the
pipeline guarantees that evaluation content never leaks into training.

## Data sources

| Source | Local path | Used for | License |
|---|---|---|---|
| [adacovex](https://github.com/bladeacer/adacovex) | `../adacovex` | Ada/SPARK source with contract specs (code pairs, defect pairs, AST turns) | Apache-2.0 |
| [Ada_CRDT](https://github.com/bladeacer/Ada_CRDT) | `../Ada_CRDT` | Spec/body pairs for diversity | MIT |
| [Ada-83-TLALOC](https://github.com/ViMoBr/Ada-83-TLALOC) | `../Ada-83-TLALOC` | Ada 83-era source (legacy patterns). Training use explicitly permitted by the author ([forum post](https://forum.ada-lang.io/t/fine-tuning-8b-ai-model-on-ada-spark/4746/3)) | GPL-3.0-or-later w/ GCC runtime exception; tests CC-BY-SA-4.0 |
| [ada-eval](https://github.com/AdaCore/ada-eval) | `../ada-eval` | **Eval-proper only** (see below). Sample code walked by parsers, guarded | Apache-2.0 |
| [AdaCore/learn](https://github.com/AdaCore/learn) | `../learn` | Course material: doc-QA and heading-chunked doc sections | CC-BY-4.0 |
| [agent-sh/ada-spark](https://github.com/agent-sh/ada-spark) | `../ada-spark` | Current-toolchain guidance in system prompts | MIT |
| [AminBlg/SimpleEnglish](https://github.com/AminBlg/SimpleEnglish) | `../SimpleEnglish` | STE writing rules and word map (paraphrased, no spec text) | MIT |
| [AdaCore/skills](https://github.com/AdaCore/skills) | `../skills` | Toolchain QA (gnatprove, alire, gnatdoc, gnattest, gnatfuzz) | Apache-2.0 |

Licensing summary: Apache-2.0 and MIT code is redistributable with
attribution; CC-BY-4.0 course material is used with attribution; the
GPL-licensed Ada-83-TLALOC code is used for model training only (weights are
not source-code redistribution) and is covered by the author's explicit
permission.

## ada-eval is eval-proper

Everything under `../ada-eval/data/base/{expanded,compacted}` are the same
19 samples q3as is scored on (`spark_learn`, `spark_custom`,
`spark_human_eval_silver`), and each record's `canonical_solution` is the
literal answer key. `data/generated` and `data/evaluated` hold completions
for those very prompts. Training on any of it would let the model memorize
the benchmark.

The sample-authoring recipe in the ada-eval README ("Adding a new Sample")
is documentation we follow when extending the benchmark, not data.

Parser targets do include `../ada-eval` for AST and contract turns; every
record derived from that tree is subject to the guard below.

## The eval guard

`data/processing_scripts/eval_guard.py` blocks eval content from reaching
any training split:

1. **Blocklist.** It hashes all eval samples' base projects, canonical
   solutions, tests, prompts, and any `data/generated`/`data/evaluated`
   completions in two forms:
   - *normalized text*: case-folded (Ada is case-insensitive), comments
     stripped, whitespace collapsed. Catches verbatim and reformatted
     copies.
   - *structural*: user identifiers alpha-renamed by first occurrence.
     Numbers and logic stay significant, so a changed bound changes the
     hash. Catches renamed copies.
2. **Detection.** Every chat record's code fences are extracted with the
   same parser the corpus uses and compared against the blocklist. Eval
   prompt text is also caught as a substring of any message.
3. **Enforcement.** `build_dataset` runs guard, then dedup, then split.
   The whole group of a matching record is dropped (sibling turns are
   paraphrases of the same eval content), and the count is recorded in
   `dataset_metadata.json` under `splits.eval_guard`.

This catches more than copy-paste: it has caught sibling-repo code that is
structurally identical to an eval sample after renaming, and `../learn` doc
examples that are verbatim eval subprograms (eval samples were authored
from the same corpus we mine). Detection is content-based, so it works no
matter which repo a turn claims as its source.

Known limit: hash-based blocking identifies content up to renaming and
reformatting. A semantically equivalent but restructured algorithm is not
caught; the split discipline (val/test are never trained on) is the second
layer of defense.

## Rules for contributors

1. Never train on ada-eval `canonical_solution`, `base/`, `tests/`,
   `prompt.md`, or compacted records.
2. After any change to data sources or the dataset, run
   `make check-integrity` (must exit 0) alongside `make validate-defects`.
3. **When adding a new data source:** update the table above, add it to
   `setup.sh` and the `Makefile` source list, and re-run
   `make check-integrity` before building the dataset. AGENTS.md reminds
   agents of this obligation.
4. Adding new eval samples to ada-eval automatically grows the blocklist;
   rebuild the dataset afterward.
5. If the guard reports `degraded` (ada-eval missing), builds proceed but
   `make check-integrity` exits 2: integrity is unverified, not proven.

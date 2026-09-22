# Data Provenance and Eval Integrity

Where q3as training data comes from, what licenses apply, and how the
pipeline guarantees that evaluation content never leaks into training.

## Data sources

All sources are fetched into a local archive cache by
`scripts/fetch_repos.py` (`make fetch-sources`). The cache lives at
`data/raw_repos/<owner>/<repo>/` (gitignored), holds HTTP tarballs (no git
metadata), and is content-addressed by repository identity: a re-run
re-fetches nothing already on disk. Per-repository metadata
(`.q3as-source.json`) records the URL, fetch time, and the license SPDX
when the GitHub API answers. The legacy `../<repo>` sibling layout still
resolves as a fallback, but nothing creates it anymore.

| Source | Cache path | Used for | License |
|---|---|---|---|
| [adacovex](https://github.com/bladeacer/adacovex) | `data/raw_repos/bladeacer/adacovex` | Ada/SPARK source with contract specs (code pairs, defect pairs, AST turns) | Apache-2.0 |
| [Ada_CRDT](https://github.com/bladeacer/Ada_CRDT) | `data/raw_repos/bladeacer/Ada_CRDT` | Spec/body pairs for diversity | MIT |
| [Ada-83-TLALOC](https://github.com/ViMoBr/Ada-83-TLALOC) | `data/raw_repos/ViMoBr/Ada-83-TLALOC` | Ada 83-era source (legacy patterns). Training use explicitly permitted by the author ([forum post](https://forum.ada-lang.io/t/fine-tuning-8b-ai-model-on-ada-spark/4746/3)) | GPL-3.0-or-later w/ GCC runtime exception; tests CC-BY-SA-4.0 |
| [ada-eval](https://github.com/AdaCore/ada-eval) | `data/raw_repos/AdaCore/ada-eval` | **Eval-proper only** (see below). Sample code walked by parsers, guarded; also the uv path dependency for eval tooling | Apache-2.0 |
| [AdaCore/learn](https://github.com/AdaCore/learn) | `data/raw_repos/AdaCore/learn` | Course material: doc-QA and heading-chunked doc sections | CC-BY-4.0 |
| [AdaCore/training_material](https://github.com/AdaCore/training_material) | `data/raw_repos/AdaCore/training_material` | AdaCore training courses (RST): doc-QA and heading-chunked doc sections. Description follows the repo README: collection of Ada/SPARK teaching courses in ReStructured Text | CC-BY-4.0 |
| [agent-sh/ada-spark](https://github.com/agent-sh/ada-spark) | `data/raw_repos/agent-sh/ada-spark` | Current-toolchain guidance in system prompts | MIT |
| [AminBlg/SimpleEnglish](https://github.com/AminBlg/SimpleEnglish) | `data/raw_repos/AminBlg/SimpleEnglish` | STE writing rules and word map (paraphrased, no spec text) | MIT |
| [AdaCore/skills](https://github.com/AdaCore/skills) | `data/raw_repos/AdaCore/skills` | Toolchain QA (gnatprove, alire, gnatdoc, gnattest, gnatfuzz) | Apache-2.0 |
| [Sternenfisch algorithm hub](https://github.com/RobertBoettcherSF/RobertBoettcherSF.github.io) (and the ~1400 repositories its README links) | `data/raw_repos/RobertBoettcherSF/<repo>` | Ada/SPARK algorithm implementations: distributed systems, graph algorithms, image processing, compression, SPARK-verified sheets, parsers. Fetcher parses the hub README and caches every linked repository owned by the hub owner. Per-repo descriptions follow each repository's own README. All repos MIT; the author approved training use (LLM-usage disclosure and license in the hub README) | MIT |

Licensing summary: Apache-2.0 and MIT code is redistributable with
attribution; CC-BY-4.0 course material is used with attribution; the
GPL-licensed Ada-83-TLALOC code is used for model training only (weights are
not source-code redistribution) and is covered by the author's explicit
permission. The Sternenfisch hub repositories are MIT with the author's
green light for training use.

## ada-eval is eval-proper

Everything under the cached ada-eval's `data/base/{expanded,compacted}`
directories are the same 19 samples q3as is scored on (`spark_learn`,
`spark_custom`, `spark_human_eval_silver`), and each record's
`canonical_solution` is the literal answer key. `data/generated` and
`data/evaluated` hold completions for those very prompts. Training on any
of it would let the model memorize the benchmark.

The sample-authoring recipe in the ada-eval README ("Adding a new Sample")
is documentation we follow when extending the benchmark, not data.

Parser targets do include the cached ada-eval tree for AST and contract
turns; every record derived from that tree is subject to the guard below.

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
structurally identical to an eval sample after renaming, and `learn` doc
examples that are verbatim eval subprograms (eval samples were authored
from the same corpus we mine). Detection is content-based, so it works no
matter which repo a turn claims as its source. It also fires on the new
hub sources: the first rebuild after adding them dropped 261 records in 53
groups whose code matched eval content structurally.

Known limit: hash-based blocking identifies content up to renaming and
reformatting. A semantically equivalent but restructured algorithm is not
caught; the split discipline (val/test are never trained on) is the second
layer of defense.

## Rules for contributors

1. Never train on ada-eval `canonical_solution`, `base/`, `tests/`,
   `prompt.md`, or compacted records.
2. After any change to data sources or the dataset, run
   `make check-integrity` (must exit 0) alongside `make validate-defects`.
3. **When adding a new data source:** add its URL to `CORE_REPOS` in
   `scripts/fetch_repos.py` (or list it under `hub:`-style discovery),
   update the table above and the `Makefile` source list, and re-run
   `make check-integrity` before building the dataset. AGENTS.md reminds
   agents of this obligation.
4. Adding new eval samples to ada-eval automatically grows the blocklist;
   rebuild the dataset afterward.
5. If the guard reports `degraded` (ada-eval missing), builds proceed but
   `make check-integrity` exits 2: integrity is unverified, not proven.

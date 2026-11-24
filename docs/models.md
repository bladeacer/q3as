# Base models

Which base models q3as can train, where the official weights live, and what
each model needs to stay inside the 8 GB VRAM budget. The machine-readable
version of this table is
[`training/model_registry.py`](../training/model_registry.py); `make model-table`
prints it from that file, so the registry is the single source of truth.

## Support table

| Key | Model | Official | Training base | Unsloth | 4-bit base | Packing | Tested |
|---|---|---|---|---|---|---|---|
| `qwen3-8b` | Qwen 3 8B | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | [unsloth/Qwen3-8B-unsloth-bnb-4bit](https://huggingface.co/unsloth/Qwen3-8B-unsloth-bnb-4bit) | native | Yes | Yes | No |
| `qwen3.5-4b` | Qwen 3.5 4B | [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | [unsloth/Qwen3.5-4B-Base](https://huggingface.co/unsloth/Qwen3.5-4B-Base) | partial | No | Yes | No |
| `spark-x2.6-4b` | Spark-X2.6-4B | [XHToken/Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B) | [XHToken/Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B) | generic | No | Yes | No |
| `k2-horizon-7b` | K2 Horizon 7B | [IFM/K2-Horizon-7B](https://huggingface.co/IFM/K2-Horizon-7B) | [IFM/K2-Horizon-7B](https://huggingface.co/IFM/K2-Horizon-7B) | generic | No | Yes | No |
| `lfm2.5-2.6b` | LFM 2.5 2.6B | [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) | [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) | unlikely | No | No | No |

**Tested = No for every row.** The "Unsloth" column records what the loader is
expected to do, not the result of a completed QLoRA run. No candidate other
than the old BF16 Qwen3-8B configuration has been run on the reference machine
yet, and the old configuration is not the default here: it used the original
16 GB weights, while the default now uses Unsloth's 4-bit base. Flip a row to
`Yes` only after a real `make train MODEL=<key>` run that reaches the summary
file on the target hardware.

## What the columns mean

- **Official** - the model authors' repository (source weights and license).
- **Training base** - what `make download` actually fetches. For `qwen3-8b`
  this is deliberately *not* the official repo: it is Unsloth's pre-quantized
  NF4 copy, about 7.5 GB on disk, which keeps QLoRA training near 5-6 GB of
  VRAM. Loading `Qwen/Qwen3-8B` in BF16 on the 8 GB card is not supported.
- **Unsloth** - `native` means Unsloth publishes an optimized 4-bit training
  base and `FastLanguageModel` applies its fused kernels. `partial` means the
  family is supported but the exact 4-bit base or architecture needs checking.
  `generic` means the HF loader can read it but no Unsloth-specific quant or
  kernel applies. `unlikely` means non-standard layers defeat the fused MHA
  path, so the HF + PEFT backend is used.
- **4-bit base** - whether a pre-quantized 4-bit checkpoint exists. When it
  does not, `load_in_4bit=True` quantizes the BF16 weights at load time, which
  needs more host RAM during the load but the same VRAM during training.
- **Packing** - whether short records are concatenated into one 1024-token row
  during training. Off for LFM 2.5 until document boundaries are checked for
  its convolution layers.

## Per-model notes

### qwen3-8b

First-class Unsloth support. `unsloth/Qwen3-8B-unsloth-bnb-4bit` is the only
base this pipeline had been tuned for before this table. It ships a
safetensors index and two shards, a config with `quantization_config` and
`unsloth_fixed: true`, and the Qwen3 chat template. The local cache path stays
`models/qwen3-8b` so generation labels (`base_qwen3-8b`) keep their shape, but
the bytes in it are now the 4-bit quant.

### qwen3.5-4b

Use `Qwen/Qwen3.5-4B-Base` for text work: the post-trained
`Qwen/Qwen3.5-4B` is an image-text model with a vision tower. Unsloth mirrors
the Base repo in 16-bit, but no `unsloth/Qwen3.5-4B-unsloth-bnb-4bit` repo
existed when this table was written. The architecture mixes linear attention
and full attention, so both the LoRA module names and `FastLanguageModel`
compatibility need a first-run check. If loading fails, use
`--backend hf --no-packing` for the probe, then decide what to change.

### spark-x2.6-4b

`XHToken/Spark-X2.6-4B` ships `configuration_spark.py` and
`modeling_spark.py` and declares an `auto_map`. It needs
`trust_remote_code=True`, which the registry sets. There is no Unsloth quant.
Check the projection names in `modeling_spark.py` against
`STANDARD_TARGET_MODULES` before committing to a long run.

### k2-horizon-7b

`IFM/K2-Horizon-7B` is a custom-code dense decoder (`K2HorizonForCausalLM`)
with 9.0B actual parameters despite the 7B name and a native 512K context.
On an 8 GB card keep `max_seq_length` at 1024 (2048 is the absolute ceiling),
`optim="adamw_8bit"`, and LoRA rank at or below 16. The registry sets rank 16
for this reason. The 512K context is not usable here.

### lfm2.5-2.6b

`LiquidAI/LFM2.5-2.6B` alternates convolution and full-attention blocks.
Unsloth's fused multi-head-attention kernels do not apply to the convolution
layers, so the registry selects the HF + PEFT backend and LFM2-specific LoRA
targets (`q_proj`, `k_proj`, `v_proj`, `out_proj`, `in_proj`, `w1`, `w2`, `w3`).
Packing is off because flattened document boundaries across a convolution
layer are not verified. The license is LFM 1.0, not Apache-2.0; check it before
redistributing an adapter. Transformers 5.2 or newer is required to read the
checkpoint.

## Using a model

`MODEL` selects a registry key for the whole pipeline. The download step is a
no-op once the cache is complete, and `make all` only fetches the selected
model:

```bash
make model-table                       # authoritative table from the registry
make download                          # default: Unsloth's Qwen3-8B 4-bit base
make download MODEL=qwen3.5-4b         # one other model
make download-models                   # all five, skipping complete caches
make check-models                      # check all caches, fetch only what is missing
make train MODEL=k2-horizon-7b
make generate MODEL=k2-horizon-7b      # uses models/k2-horizon-7b as the base
```

The same key works for direct script runs:

```bash
uv run python training/download_model.py --model qwen3.5-4b
uv run python training/train_unsloth.py --model qwen3.5-4b --no-packing
export Q3AS_MODEL=qwen3.5-4b   # default key when --model / MODEL is omitted
```

Useful overrides for a probe: `--backend hf` (force transformers + PEFT),
`--packing-strategy wrapped` or `bfd`, `--no-offload-embedding`, and
`--trust-remote-code` / `--no-trust-remote-code`. `--model-name <id>` bypasses
the registry's repo and `--cache-dir <path>` its cache path; the downloader
then derives a `models/<name>` path so a custom repo cannot overwrite a
registered cache.

## Adding a model

1. Add a `ModelSpec` to `MODELS` in
   [`training/model_registry.py`](../training/model_registry.py) with the
   official repo, the repo training should load, a cache directory, and the
   8 GB settings. Leave `tested=False`.
2. Add a case to
   [`tests/test_model_registry.py`](../tests/test_model_registry.py) so the
   URL and unique-key invariants are checked.
3. Run `make model-table` and paste its output into the table above.
4. Fetch and probe with `make download MODEL=<key>` then a short
   `make train MODEL=<key> TRAIN_FLAGS="--max-steps 20 --skip-merged-save"`.
   Record what actually happened in a changelog entry, and only then set
   `tested=True`.

Never add a model to the dataset sources: these repos are base weights, not
training data. The eval-integrity rules in
[data provenance](data-provenance.md) still apply.

Navigation: [project README](../README.md) · [docs index](README.md) · [changelog index](changelogs/index.md) · [results index](results/README.md)

# silica-mlx bench report

Generated: 2026-04-26T15:58:56

Scenarios: total=2 Runs: total=6 ok=6 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-4b-wikitext-ppl-fp16 |  | 3 | 3 | 0 | 0 |  |  |  | 9346.5 | 1.856 ± 0.320 | 511 |  |
| qwen3.5-4b-wikitext-ppl-block-tq-b64-b4 | block_tq_b64_b4 | 3 | 3 | 0 | 0 |  |  |  | 9355.1 | 1.945 ± 0.135 | 511 |  |

## Scenario details

### `qwen3.5-4b-wikitext-ppl-fp16` (seed=42)

- repo: `Qwen/Qwen3.5-4B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-3-C5-step2-E fp16 baseline PPL row on Qwen3.5-4B (hybrid-DeltaNet, the model vqbench REPORT used for its BlockTQ lossless claim). Same chunk_size=256 / max_tokens=512 wikitext-2 workload as the Qwen3.5-0.8B and Qwen3-0.6B fp16 rows so the codec rows below are directly comparable across model sizes.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": null,
  "codec_quality_path": "prefix_store_pre_norm",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": null,
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1114.5384311676025,
  "ppl": 8.855978965984526,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3.5-4b-wikitext-ppl-fp16` (seed=43)

- repo: `Qwen/Qwen3.5-4B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-3-C5-step2-E fp16 baseline PPL row on Qwen3.5-4B (hybrid-DeltaNet, the model vqbench REPORT used for its BlockTQ lossless claim). Same chunk_size=256 / max_tokens=512 wikitext-2 workload as the Qwen3.5-0.8B and Qwen3-0.6B fp16 rows so the codec rows below are directly comparable across model sizes.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": null,
  "codec_quality_path": "prefix_store_pre_norm",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": null,
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1114.5384311676025,
  "ppl": 8.855978965984526,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3.5-4b-wikitext-ppl-fp16` (seed=44)

- repo: `Qwen/Qwen3.5-4B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-3-C5-step2-E fp16 baseline PPL row on Qwen3.5-4B (hybrid-DeltaNet, the model vqbench REPORT used for its BlockTQ lossless claim). Same chunk_size=256 / max_tokens=512 wikitext-2 workload as the Qwen3.5-0.8B and Qwen3-0.6B fp16 rows so the codec rows below are directly comparable across model sizes.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": null,
  "codec_quality_path": "prefix_store_pre_norm",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": null,
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1114.5384311676025,
  "ppl": 8.855978965984526,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3.5-4b-wikitext-ppl-block-tq-b64-b4` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3.5-4B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-3-C5-step2-E BlockTurboQuantMSE B=64 4-bit K+V PPL row on Qwen3.5-4B. Drives the C5-step2-W hybrid-aware codec oracle (heterogeneous cache + recurrent snapshot/restore across chunks) on the same model vqbench REPORT used for its lossless claim. Cross-method comparison: vqbench's pre-RoPE projection-patch path vs silica's post-RoPE production-store path on identical inputs.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b4",
  "codec_quality_path": "prefix_store_pre_norm",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1115.96653175354,
  "ppl": 8.880763541210271,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3.5-4b-wikitext-ppl-block-tq-b64-b4` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3.5-4B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-3-C5-step2-E BlockTurboQuantMSE B=64 4-bit K+V PPL row on Qwen3.5-4B. Drives the C5-step2-W hybrid-aware codec oracle (heterogeneous cache + recurrent snapshot/restore across chunks) on the same model vqbench REPORT used for its lossless claim. Cross-method comparison: vqbench's pre-RoPE projection-patch path vs silica's post-RoPE production-store path on identical inputs.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b4",
  "codec_quality_path": "prefix_store_pre_norm",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1113.57151222229,
  "ppl": 8.839237444571916,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3.5-4b-wikitext-ppl-block-tq-b64-b4` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3.5-4B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-3-C5-step2-E BlockTurboQuantMSE B=64 4-bit K+V PPL row on Qwen3.5-4B. Drives the C5-step2-W hybrid-aware codec oracle (heterogeneous cache + recurrent snapshot/restore across chunks) on the same model vqbench REPORT used for its lossless claim. Cross-method comparison: vqbench's pre-RoPE projection-patch path vs silica's post-RoPE production-store path on identical inputs.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b4",
  "codec_quality_path": "prefix_store_pre_norm",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1114.343698501587,
  "ppl": 8.852604758862839,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

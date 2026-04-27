# silica-mlx bench report

Generated: 2026-04-24T12:47:23

Scenarios: total=28 Runs: total=924 ok=360 skipped=0 failed=564

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma4-31b-b1-parity | block_tq_b32_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-b1-parity | block_tq_b32_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-b1-parity | block_tq_b64_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-b1-parity | block_tq_b64_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-b1-parity | ext_rabitq_b2 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-b1-parity | ext_rabitq_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-b1-parity | ext_rabitq_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-b1-parity | fp16 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-b1-parity | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-b1-parity | tq_mse_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-b1-parity | tq_mse_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-bgt1-parity | block_tq_b32_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-bgt1-parity | block_tq_b32_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-bgt1-parity | block_tq_b64_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-bgt1-parity | block_tq_b64_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-bgt1-parity | ext_rabitq_b2 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-bgt1-parity | ext_rabitq_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-bgt1-parity | ext_rabitq_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-bgt1-parity | fp16 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-bgt1-parity | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-bgt1-parity | tq_mse_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-bgt1-parity | tq_mse_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-smoke | block_tq_b32_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-smoke | block_tq_b32_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-smoke | block_tq_b64_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-smoke | block_tq_b64_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-smoke | ext_rabitq_b2 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-smoke | ext_rabitq_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-smoke | ext_rabitq_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-smoke | fp16 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-smoke | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-smoke | tq_mse_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-31b-smoke | tq_mse_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-moe-smoke | block_tq_b32_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-moe-smoke | block_tq_b32_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-moe-smoke | block_tq_b64_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-moe-smoke | block_tq_b64_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-moe-smoke | ext_rabitq_b2 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-moe-smoke | ext_rabitq_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-moe-smoke | ext_rabitq_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-moe-smoke | fp16 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-moe-smoke | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-moe-smoke | tq_mse_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| gemma4-moe-smoke | tq_mse_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-admission-headroom-prefix-heavy | block_tq_b32_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-admission-headroom-prefix-heavy | block_tq_b32_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-admission-headroom-prefix-heavy | block_tq_b64_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-admission-headroom-prefix-heavy | block_tq_b64_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-admission-headroom-prefix-heavy | ext_rabitq_b2 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-admission-headroom-prefix-heavy | ext_rabitq_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-admission-headroom-prefix-heavy | ext_rabitq_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-admission-headroom-prefix-heavy | fp16 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-admission-headroom-prefix-heavy | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-admission-headroom-prefix-heavy | tq_mse_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-admission-headroom-prefix-heavy | tq_mse_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-b1-parity | block_tq_b32_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-b1-parity | block_tq_b32_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-b1-parity | block_tq_b64_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-b1-parity | block_tq_b64_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-b1-parity | ext_rabitq_b2 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-b1-parity | ext_rabitq_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-b1-parity | ext_rabitq_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-b1-parity | fp16 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-b1-parity | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-b1-parity | tq_mse_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-b1-parity | tq_mse_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-bgt1-parity | block_tq_b32_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-bgt1-parity | block_tq_b32_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-bgt1-parity | block_tq_b64_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-bgt1-parity | block_tq_b64_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-bgt1-parity | ext_rabitq_b2 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-bgt1-parity | ext_rabitq_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-bgt1-parity | ext_rabitq_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-bgt1-parity | fp16 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-bgt1-parity | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-bgt1-parity | tq_mse_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-bgt1-parity | tq_mse_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-compression-block-tq-b64-b4 | block_tq_b32_b3 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.737 ± 0.066 | 6 |  |
| qwen3-0.6b-compression-block-tq-b64-b4 | block_tq_b32_b4 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.742 ± 0.011 | 5 |  |
| qwen3-0.6b-compression-block-tq-b64-b4 | block_tq_b64_b3 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.724 ± 0.017 | 6 |  |
| qwen3-0.6b-compression-block-tq-b64-b4 | block_tq_b64_b4 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.808 ± 0.059 | 6 |  |
| qwen3-0.6b-compression-block-tq-b64-b4 | ext_rabitq_b2 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.802 ± 0.017 | 6 |  |
| qwen3-0.6b-compression-block-tq-b64-b4 | ext_rabitq_b3 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.754 ± 0.051 | 6 |  |
| qwen3-0.6b-compression-block-tq-b64-b4 | ext_rabitq_b4 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.766 ± 0.054 | 5 ± 1 |  |
| qwen3-0.6b-compression-block-tq-b64-b4 | fp16 | 3 | 3 | 0 | 0 |  |  |  | 1366.2 | 0.675 ± 0.102 | 5 |  |
| qwen3-0.6b-compression-block-tq-b64-b4 | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-compression-block-tq-b64-b4 | tq_mse_b3 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.733 ± 0.013 | 6 |  |
| qwen3-0.6b-compression-block-tq-b64-b4 | tq_mse_b4 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.714 ± 0.015 | 6 |  |
| qwen3-0.6b-compression-ext-rabitq-b4 | block_tq_b32_b3 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.787 ± 0.037 | 6 |  |
| qwen3-0.6b-compression-ext-rabitq-b4 | block_tq_b32_b4 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.803 ± 0.061 | 5 |  |
| qwen3-0.6b-compression-ext-rabitq-b4 | block_tq_b64_b3 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.823 ± 0.059 | 6 |  |
| qwen3-0.6b-compression-ext-rabitq-b4 | block_tq_b64_b4 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.766 ± 0.083 | 6 |  |
| qwen3-0.6b-compression-ext-rabitq-b4 | ext_rabitq_b2 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.691 ± 0.018 | 6 |  |
| qwen3-0.6b-compression-ext-rabitq-b4 | ext_rabitq_b3 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.758 ± 0.046 | 6 |  |
| qwen3-0.6b-compression-ext-rabitq-b4 | ext_rabitq_b4 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.785 ± 0.059 | 5 ± 1 |  |
| qwen3-0.6b-compression-ext-rabitq-b4 | fp16 | 3 | 3 | 0 | 0 |  |  |  | 1366.2 | 0.668 ± 0.062 | 5 |  |
| qwen3-0.6b-compression-ext-rabitq-b4 | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-compression-ext-rabitq-b4 | tq_mse_b3 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.731 ± 0.089 | 6 |  |
| qwen3-0.6b-compression-ext-rabitq-b4 | tq_mse_b4 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.725 ± 0.019 | 6 |  |
| qwen3-0.6b-compression-fp16 | block_tq_b32_b3 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.772 ± 0.051 | 6 |  |
| qwen3-0.6b-compression-fp16 | block_tq_b32_b4 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.793 ± 0.068 | 5 |  |
| qwen3-0.6b-compression-fp16 | block_tq_b64_b3 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.774 ± 0.037 | 6 |  |
| qwen3-0.6b-compression-fp16 | block_tq_b64_b4 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.771 ± 0.025 | 6 |  |
| qwen3-0.6b-compression-fp16 | ext_rabitq_b2 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.724 ± 0.011 | 6 |  |
| qwen3-0.6b-compression-fp16 | ext_rabitq_b3 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.765 ± 0.057 | 6 |  |
| qwen3-0.6b-compression-fp16 | ext_rabitq_b4 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.773 ± 0.060 | 5 ± 1 |  |
| qwen3-0.6b-compression-fp16 | fp16 | 3 | 3 | 0 | 0 |  |  |  | 1366.2 | 0.654 ± 0.011 | 5 |  |
| qwen3-0.6b-compression-fp16 | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-compression-fp16 | tq_mse_b3 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.739 ± 0.021 | 6 |  |
| qwen3-0.6b-compression-fp16 | tq_mse_b4 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.745 ± 0.008 | 6 |  |
| qwen3-0.6b-compression-tq-mse-b4 | block_tq_b32_b3 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.725 ± 0.017 | 6 |  |
| qwen3-0.6b-compression-tq-mse-b4 | block_tq_b32_b4 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.734 ± 0.009 | 5 |  |
| qwen3-0.6b-compression-tq-mse-b4 | block_tq_b64_b3 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.746 ± 0.028 | 6 |  |
| qwen3-0.6b-compression-tq-mse-b4 | block_tq_b64_b4 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.754 ± 0.009 | 6 |  |
| qwen3-0.6b-compression-tq-mse-b4 | ext_rabitq_b2 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.730 ± 0.071 | 6 |  |
| qwen3-0.6b-compression-tq-mse-b4 | ext_rabitq_b3 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.790 ± 0.068 | 6 |  |
| qwen3-0.6b-compression-tq-mse-b4 | ext_rabitq_b4 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.729 ± 0.022 | 5 ± 1 |  |
| qwen3-0.6b-compression-tq-mse-b4 | fp16 | 3 | 3 | 0 | 0 |  |  |  | 1366.2 | 0.701 ± 0.079 | 5 |  |
| qwen3-0.6b-compression-tq-mse-b4 | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-compression-tq-mse-b4 | tq_mse_b3 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.714 ± 0.017 | 6 |  |
| qwen3-0.6b-compression-tq-mse-b4 | tq_mse_b4 | 3 | 3 | 0 | 0 |  |  |  | 1366.3 | 0.802 ± 0.073 | 6 |  |
| qwen3-0.6b-concurrent-shared-prefix | block_tq_b32_b3 | 3 | 3 | 0 | 0 |  |  |  | 1368.1 | 0.512 ± 0.031 | 32 |  |
| qwen3-0.6b-concurrent-shared-prefix | block_tq_b32_b4 | 3 | 3 | 0 | 0 |  |  |  | 1368.1 ± 0.0 | 0.579 ± 0.069 | 32 |  |
| qwen3-0.6b-concurrent-shared-prefix | block_tq_b64_b3 | 3 | 3 | 0 | 0 |  |  |  | 1368.1 ± 0.0 | 0.551 ± 0.014 | 32 |  |
| qwen3-0.6b-concurrent-shared-prefix | block_tq_b64_b4 | 3 | 3 | 0 | 0 |  |  |  | 1368.0 ± 0.0 | 0.579 ± 0.060 | 32 |  |
| qwen3-0.6b-concurrent-shared-prefix | ext_rabitq_b2 | 3 | 3 | 0 | 0 |  |  |  | 1368.1 | 0.510 ± 0.034 | 32 |  |
| qwen3-0.6b-concurrent-shared-prefix | ext_rabitq_b3 | 3 | 3 | 0 | 0 |  |  |  | 1368.1 ± 0.0 | 0.513 ± 0.004 | 32 |  |
| qwen3-0.6b-concurrent-shared-prefix | ext_rabitq_b4 | 3 | 3 | 0 | 0 |  |  |  | 1368.1 ± 0.0 | 0.520 ± 0.029 | 32 |  |
| qwen3-0.6b-concurrent-shared-prefix | fp16 | 3 | 3 | 0 | 0 |  |  |  | 1368.0 ± 0.0 | 0.550 ± 0.056 | 32 |  |
| qwen3-0.6b-concurrent-shared-prefix | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-concurrent-shared-prefix | tq_mse_b3 | 3 | 3 | 0 | 0 |  |  |  | 1368.1 ± 0.0 | 0.576 ± 0.061 | 32 |  |
| qwen3-0.6b-concurrent-shared-prefix | tq_mse_b4 | 3 | 3 | 0 | 0 |  |  |  | 1368.1 | 0.570 ± 0.090 | 32 |  |
| qwen3-0.6b-long-in-short-out | block_tq_b32_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-long-in-short-out | block_tq_b32_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-long-in-short-out | block_tq_b64_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-long-in-short-out | block_tq_b64_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-long-in-short-out | ext_rabitq_b2 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-long-in-short-out | ext_rabitq_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-long-in-short-out | ext_rabitq_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-long-in-short-out | fp16 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-long-in-short-out | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-long-in-short-out | tq_mse_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-long-in-short-out | tq_mse_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4 | block_tq_b32_b3 | 3 | 3 | 0 | 0 | 235.4 ± 5.9 | 151.2 ± 1.3 |  | 1366.3 | 0.774 ± 0.015 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4 | block_tq_b32_b4 | 3 | 3 | 0 | 0 | 252.0 ± 22.7 | 150.7 ± 12.7 |  | 1366.3 | 0.833 ± 0.133 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4 | block_tq_b64_b3 | 3 | 3 | 0 | 0 | 232.2 ± 1.2 | 152.7 ± 1.9 |  | 1366.3 | 0.758 ± 0.019 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4 | block_tq_b64_b4 | 3 | 3 | 0 | 0 | 231.9 ± 10.9 | 154.1 ± 5.4 |  | 1366.3 | 0.766 ± 0.026 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4 | ext_rabitq_b2 | 3 | 3 | 0 | 0 | 219.4 ± 5.2 | 154.8 ± 1.7 |  | 1366.3 | 0.747 ± 0.024 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4 | ext_rabitq_b3 | 3 | 3 | 0 | 0 | 232.6 ± 3.8 | 149.0 ± 2.4 |  | 1366.3 | 0.803 ± 0.038 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4 | ext_rabitq_b4 | 3 | 3 | 0 | 0 | 240.5 ± 10.7 | 156.3 ± 0.5 |  | 1366.3 | 0.751 ± 0.043 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4 | fp16 | 3 | 3 | 0 | 0 | 139.8 ± 2.0 | 150.8 ± 1.9 |  | 1366.2 | 0.677 ± 0.023 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4 | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4 | tq_mse_b3 | 3 | 3 | 0 | 0 | 208.0 ± 3.2 | 155.9 ± 3.4 |  | 1366.3 | 0.721 ± 0.012 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4 | tq_mse_b4 | 3 | 3 | 0 | 0 | 216.6 ± 5.6 | 152.5 ± 5.5 |  | 1366.3 | 0.733 ± 0.013 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4 | block_tq_b32_b3 | 3 | 3 | 0 | 0 | 237.2 ± 9.2 | 153.8 ± 1.6 |  | 1366.3 | 0.765 ± 0.035 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4 | block_tq_b32_b4 | 3 | 3 | 0 | 0 | 237.5 ± 6.6 | 155.3 ± 2.6 |  | 1366.3 | 0.762 ± 0.015 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4 | block_tq_b64_b3 | 3 | 3 | 0 | 0 | 226.4 ± 3.8 | 148.3 ± 12.2 |  | 1366.3 | 0.759 ± 0.006 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4 | block_tq_b64_b4 | 3 | 3 | 0 | 0 | 229.3 ± 1.6 | 156.9 ± 1.2 |  | 1366.3 | 0.762 ± 0.032 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4 | ext_rabitq_b2 | 3 | 3 | 0 | 0 | 215.6 ± 7.3 | 152.6 ± 2.5 |  | 1366.3 | 0.738 ± 0.009 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4 | ext_rabitq_b3 | 3 | 3 | 0 | 0 | 233.4 ± 6.4 | 148.9 ± 11.7 |  | 1366.3 | 0.756 ± 0.014 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4 | ext_rabitq_b4 | 3 | 3 | 0 | 0 | 235.0 ± 13.4 | 152.9 ± 5.5 |  | 1366.3 | 0.769 ± 0.041 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4 | fp16 | 3 | 3 | 0 | 0 | 140.2 ± 6.6 | 158.5 ± 3.4 |  | 1366.2 | 0.696 ± 0.043 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4 | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4 | tq_mse_b3 | 3 | 3 | 0 | 0 | 209.7 ± 7.4 | 154.6 ± 2.8 |  | 1366.3 | 0.767 ± 0.016 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4 | tq_mse_b4 | 3 | 3 | 0 | 0 | 220.2 ± 1.5 | 155.7 ± 0.9 |  | 1366.3 | 0.755 ± 0.007 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-fp16 | block_tq_b32_b3 | 3 | 3 | 0 | 0 | 226.9 ± 3.9 | 154.9 ± 3.3 |  | 1366.3 | 0.769 ± 0.015 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-fp16 | block_tq_b32_b4 | 3 | 3 | 0 | 0 | 237.2 ± 1.7 | 156.1 ± 0.5 |  | 1366.3 | 0.778 ± 0.042 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-fp16 | block_tq_b64_b3 | 3 | 3 | 0 | 0 | 226.5 ± 2.8 | 152.7 ± 5.8 |  | 1366.3 | 0.804 ± 0.079 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-fp16 | block_tq_b64_b4 | 3 | 3 | 0 | 0 | 235.8 ± 8.0 | 155.6 ± 8.3 |  | 1366.3 | 0.780 ± 0.040 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-fp16 | ext_rabitq_b2 | 3 | 3 | 0 | 0 | 206.3 ± 7.1 | 160.0 ± 4.4 |  | 1366.3 | 0.787 ± 0.042 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-fp16 | ext_rabitq_b3 | 3 | 3 | 0 | 0 | 221.5 ± 2.7 | 149.8 ± 6.6 |  | 1366.3 | 0.823 ± 0.109 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-fp16 | ext_rabitq_b4 | 3 | 3 | 0 | 0 | 229.6 ± 6.7 | 158.9 ± 3.5 |  | 1366.3 | 0.783 ± 0.075 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-fp16 | fp16 | 3 | 3 | 0 | 0 | 138.9 ± 8.7 | 150.6 ± 8.3 |  | 1366.2 | 0.690 ± 0.032 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-fp16 | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-prefix-hit-decode-fp16 | tq_mse_b3 | 3 | 3 | 0 | 0 | 223.4 ± 9.4 | 140.4 ± 3.1 |  | 1366.3 | 0.780 ± 0.043 | 32 |  |
| qwen3-0.6b-prefix-hit-decode-fp16 | tq_mse_b4 | 3 | 3 | 0 | 0 | 227.1 ± 0.7 | 140.1 ± 2.1 |  | 1366.3 | 0.821 ± 0.045 | 32 |  |
| qwen3-0.6b-short-in-long-out | block_tq_b32_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-short-in-long-out | block_tq_b32_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-short-in-long-out | block_tq_b64_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-short-in-long-out | block_tq_b64_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-short-in-long-out | ext_rabitq_b2 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-short-in-long-out | ext_rabitq_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-short-in-long-out | ext_rabitq_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-short-in-long-out | fp16 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-short-in-long-out | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-short-in-long-out | tq_mse_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-short-in-long-out | tq_mse_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-smoke | block_tq_b32_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-smoke | block_tq_b32_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-smoke | block_tq_b64_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-smoke | block_tq_b64_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-smoke | ext_rabitq_b2 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-smoke | ext_rabitq_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-smoke | ext_rabitq_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-smoke | fp16 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-smoke | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-smoke | tq_mse_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-smoke | tq_mse_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-teacher-forced-argmax | block_tq_b32_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-teacher-forced-argmax | block_tq_b32_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-teacher-forced-argmax | block_tq_b64_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-teacher-forced-argmax | block_tq_b64_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-teacher-forced-argmax | ext_rabitq_b2 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-teacher-forced-argmax | ext_rabitq_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-teacher-forced-argmax | ext_rabitq_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-teacher-forced-argmax | fp16 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-teacher-forced-argmax | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-teacher-forced-argmax | tq_mse_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-teacher-forced-argmax | tq_mse_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-ttft-under-concurrency | block_tq_b32_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-ttft-under-concurrency | block_tq_b32_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-ttft-under-concurrency | block_tq_b64_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-ttft-under-concurrency | block_tq_b64_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-ttft-under-concurrency | ext_rabitq_b2 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-ttft-under-concurrency | ext_rabitq_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-ttft-under-concurrency | ext_rabitq_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-ttft-under-concurrency | fp16 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-ttft-under-concurrency | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-ttft-under-concurrency | tq_mse_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-ttft-under-concurrency | tq_mse_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 | block_tq_b32_b3 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 ± 0.0 | 1.508 ± 0.067 | 511 |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 | block_tq_b32_b4 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.424 ± 0.090 | 511 |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 | block_tq_b64_b3 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.546 ± 0.118 | 511 |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 | block_tq_b64_b4 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.499 ± 0.098 | 511 |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 | ext_rabitq_b2 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.392 ± 0.052 | 511 |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 | ext_rabitq_b3 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.421 ± 0.138 | 511 |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 | ext_rabitq_b4 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.530 ± 0.051 | 511 |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 | fp16 | 3 | 3 | 0 | 0 |  |  |  | 1936.4 | 1.044 ± 0.039 | 511 |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 | tq_mse_b3 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.372 ± 0.092 | 511 |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 | tq_mse_b4 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.374 ± 0.074 | 511 |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned | block_tq_b32_b3 | 3 | 3 | 0 | 0 |  |  |  | 1909.0 | 1.592 ± 0.012 | 511 |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned | block_tq_b32_b4 | 3 | 3 | 0 | 0 |  |  |  | 1952.0 | 1.663 ± 0.013 | 511 |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned | block_tq_b64_b3 | 3 | 3 | 0 | 0 |  |  |  | 1908.6 | 1.579 ± 0.039 | 511 |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned | block_tq_b64_b4 | 3 | 3 | 0 | 0 |  |  |  | 1951.6 | 1.681 ± 0.054 | 511 |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned | ext_rabitq_b2 | 3 | 3 | 0 | 0 |  |  |  | 1770.4 ± 0.0 | 1.018 ± 0.027 | 511 |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned | ext_rabitq_b3 | 3 | 3 | 0 | 0 |  |  |  | 1769.6 | 0.992 ± 0.024 | 511 |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned | ext_rabitq_b4 | 3 | 3 | 0 | 0 |  |  |  | 1765.9 | 1.030 ± 0.109 | 511 |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned | fp16 | 3 | 3 | 0 | 0 |  |  |  | 1939.2 | 0.968 ± 0.092 | 511 |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned | tq_mse_b3 | 3 | 3 | 0 | 0 |  |  |  | 1908.4 | 1.393 ± 0.005 | 511 |  |
| qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned | tq_mse_b4 | 3 | 3 | 0 | 0 |  |  |  | 1951.3 | 1.428 ± 0.081 | 511 |  |
| qwen3-0.6b-wikitext-ppl-ext-rabitq-b4 | block_tq_b32_b3 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.402 ± 0.043 | 511 |  |
| qwen3-0.6b-wikitext-ppl-ext-rabitq-b4 | block_tq_b32_b4 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.388 ± 0.073 | 511 |  |
| qwen3-0.6b-wikitext-ppl-ext-rabitq-b4 | block_tq_b64_b3 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.431 ± 0.034 | 511 |  |
| qwen3-0.6b-wikitext-ppl-ext-rabitq-b4 | block_tq_b64_b4 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.389 ± 0.099 | 511 |  |
| qwen3-0.6b-wikitext-ppl-ext-rabitq-b4 | ext_rabitq_b2 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.491 ± 0.141 | 511 |  |
| qwen3-0.6b-wikitext-ppl-ext-rabitq-b4 | ext_rabitq_b3 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.493 ± 0.129 | 511 |  |
| qwen3-0.6b-wikitext-ppl-ext-rabitq-b4 | ext_rabitq_b4 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.493 ± 0.114 | 511 |  |
| qwen3-0.6b-wikitext-ppl-ext-rabitq-b4 | fp16 | 3 | 3 | 0 | 0 |  |  |  | 1936.4 | 0.935 ± 0.026 | 511 |  |
| qwen3-0.6b-wikitext-ppl-ext-rabitq-b4 | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-wikitext-ppl-ext-rabitq-b4 | tq_mse_b3 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.328 ± 0.064 | 511 |  |
| qwen3-0.6b-wikitext-ppl-ext-rabitq-b4 | tq_mse_b4 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.338 ± 0.088 | 511 |  |
| qwen3-0.6b-wikitext-ppl-fp16 | block_tq_b32_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-wikitext-ppl-fp16 | block_tq_b32_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-wikitext-ppl-fp16 | block_tq_b64_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-wikitext-ppl-fp16 | block_tq_b64_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-wikitext-ppl-fp16 | ext_rabitq_b2 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-wikitext-ppl-fp16 | ext_rabitq_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-wikitext-ppl-fp16 | ext_rabitq_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-wikitext-ppl-fp16 | fp16 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-wikitext-ppl-fp16 | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-wikitext-ppl-fp16 | tq_mse_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-wikitext-ppl-fp16 | tq_mse_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-wikitext-ppl-tq-mse-b4 | block_tq_b32_b3 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.381 ± 0.052 | 511 |  |
| qwen3-0.6b-wikitext-ppl-tq-mse-b4 | block_tq_b32_b4 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.357 ± 0.012 | 511 |  |
| qwen3-0.6b-wikitext-ppl-tq-mse-b4 | block_tq_b64_b3 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.323 ± 0.063 | 511 |  |
| qwen3-0.6b-wikitext-ppl-tq-mse-b4 | block_tq_b64_b4 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.348 ± 0.042 | 511 |  |
| qwen3-0.6b-wikitext-ppl-tq-mse-b4 | ext_rabitq_b2 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.286 ± 0.060 | 511 |  |
| qwen3-0.6b-wikitext-ppl-tq-mse-b4 | ext_rabitq_b3 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.338 ± 0.042 | 511 |  |
| qwen3-0.6b-wikitext-ppl-tq-mse-b4 | ext_rabitq_b4 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.375 ± 0.021 | 511 |  |
| qwen3-0.6b-wikitext-ppl-tq-mse-b4 | fp16 | 3 | 3 | 0 | 0 |  |  |  | 1936.4 | 0.952 ± 0.021 | 511 |  |
| qwen3-0.6b-wikitext-ppl-tq-mse-b4 | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3-0.6b-wikitext-ppl-tq-mse-b4 | tq_mse_b3 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.321 ± 0.036 | 511 |  |
| qwen3-0.6b-wikitext-ppl-tq-mse-b4 | tq_mse_b4 | 3 | 3 | 0 | 0 |  |  |  | 1936.5 | 1.354 ± 0.033 | 511 |  |
| qwen3.5-0.8b-b1-parity | block_tq_b32_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-0.8b-b1-parity | block_tq_b32_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-0.8b-b1-parity | block_tq_b64_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-0.8b-b1-parity | block_tq_b64_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-0.8b-b1-parity | ext_rabitq_b2 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-0.8b-b1-parity | ext_rabitq_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-0.8b-b1-parity | ext_rabitq_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-0.8b-b1-parity | fp16 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-0.8b-b1-parity | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-0.8b-b1-parity | tq_mse_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-0.8b-b1-parity | tq_mse_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-27b-smoke | block_tq_b32_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-27b-smoke | block_tq_b32_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-27b-smoke | block_tq_b64_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-27b-smoke | block_tq_b64_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-27b-smoke | ext_rabitq_b2 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-27b-smoke | ext_rabitq_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-27b-smoke | ext_rabitq_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-27b-smoke | fp16 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-27b-smoke | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-27b-smoke | tq_mse_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-27b-smoke | tq_mse_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-moe-smoke | block_tq_b32_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-moe-smoke | block_tq_b32_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-moe-smoke | block_tq_b64_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-moe-smoke | block_tq_b64_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-moe-smoke | ext_rabitq_b2 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-moe-smoke | ext_rabitq_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-moe-smoke | ext_rabitq_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-moe-smoke | fp16 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-moe-smoke | rabitq_b1 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-moe-smoke | tq_mse_b3 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |
| qwen3.5-moe-smoke | tq_mse_b4 | 3 | 0 | 0 | 3 |  |  |  |  |  |  |  |

## Scenario details

### `gemma4-31b-b1-parity` (codec=block_tq_b32_b3, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 42
}
```

### `gemma4-31b-b1-parity` (codec=block_tq_b32_b3, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 43
}
```

### `gemma4-31b-b1-parity` (codec=block_tq_b32_b3, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 44
}
```

### `gemma4-31b-b1-parity` (codec=block_tq_b32_b4, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 42
}
```

### `gemma4-31b-b1-parity` (codec=block_tq_b32_b4, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 43
}
```

### `gemma4-31b-b1-parity` (codec=block_tq_b32_b4, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 44
}
```

### `gemma4-31b-b1-parity` (codec=block_tq_b64_b3, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 42
}
```

### `gemma4-31b-b1-parity` (codec=block_tq_b64_b3, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 43
}
```

### `gemma4-31b-b1-parity` (codec=block_tq_b64_b3, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 44
}
```

### `gemma4-31b-b1-parity` (codec=block_tq_b64_b4, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 42
}
```

### `gemma4-31b-b1-parity` (codec=block_tq_b64_b4, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 43
}
```

### `gemma4-31b-b1-parity` (codec=block_tq_b64_b4, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 44
}
```

### `gemma4-31b-b1-parity` (codec=ext_rabitq_b2, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 42
}
```

### `gemma4-31b-b1-parity` (codec=ext_rabitq_b2, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 43
}
```

### `gemma4-31b-b1-parity` (codec=ext_rabitq_b2, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 44
}
```

### `gemma4-31b-b1-parity` (codec=ext_rabitq_b3, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 42
}
```

### `gemma4-31b-b1-parity` (codec=ext_rabitq_b3, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 43
}
```

### `gemma4-31b-b1-parity` (codec=ext_rabitq_b3, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 44
}
```

### `gemma4-31b-b1-parity` (codec=ext_rabitq_b4, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 42
}
```

### `gemma4-31b-b1-parity` (codec=ext_rabitq_b4, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 43
}
```

### `gemma4-31b-b1-parity` (codec=ext_rabitq_b4, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 44
}
```

### `gemma4-31b-b1-parity` (codec=fp16, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 42
}
```

### `gemma4-31b-b1-parity` (codec=fp16, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 43
}
```

### `gemma4-31b-b1-parity` (codec=fp16, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 44
}
```

### `gemma4-31b-b1-parity` (codec=rabitq_b1, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `gemma4-31b-b1-parity` (codec=rabitq_b1, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `gemma4-31b-b1-parity` (codec=rabitq_b1, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `gemma4-31b-b1-parity` (codec=tq_mse_b3, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 42
}
```

### `gemma4-31b-b1-parity` (codec=tq_mse_b3, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 43
}
```

### `gemma4-31b-b1-parity` (codec=tq_mse_b3, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 44
}
```

### `gemma4-31b-b1-parity` (codec=tq_mse_b4, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 42
}
```

### `gemma4-31b-b1-parity` (codec=tq_mse_b4, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 43
}
```

### `gemma4-31b-b1-parity` (codec=tq_mse_b4, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `b1_parity_vs_single`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of tests/test_p3_gemma4_batched_parity.py: the B=1 batched path must emit the same token stream as the single-request Engine.generate path. The sliding-window BatchRotatingKVCache + full-attention BatchKVCache hybrid cache list produced by Gemma4Adapter.make_batch_cache is the specific thing being exercised at B=1. Dual-gated on SILICA_REAL_GEMMA4_31B because every parity run pays the cost of both a single-request and a batched forward.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 44
}
```

### `gemma4-31b-bgt1-parity` (codec=block_tq_b32_b3, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 42
}
```

### `gemma4-31b-bgt1-parity` (codec=block_tq_b32_b3, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 43
}
```

### `gemma4-31b-bgt1-parity` (codec=block_tq_b32_b3, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 44
}
```

### `gemma4-31b-bgt1-parity` (codec=block_tq_b32_b4, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 42
}
```

### `gemma4-31b-bgt1-parity` (codec=block_tq_b32_b4, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 43
}
```

### `gemma4-31b-bgt1-parity` (codec=block_tq_b32_b4, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 44
}
```

### `gemma4-31b-bgt1-parity` (codec=block_tq_b64_b3, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 42
}
```

### `gemma4-31b-bgt1-parity` (codec=block_tq_b64_b3, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 43
}
```

### `gemma4-31b-bgt1-parity` (codec=block_tq_b64_b3, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 44
}
```

### `gemma4-31b-bgt1-parity` (codec=block_tq_b64_b4, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 42
}
```

### `gemma4-31b-bgt1-parity` (codec=block_tq_b64_b4, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 43
}
```

### `gemma4-31b-bgt1-parity` (codec=block_tq_b64_b4, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 44
}
```

### `gemma4-31b-bgt1-parity` (codec=ext_rabitq_b2, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 42
}
```

### `gemma4-31b-bgt1-parity` (codec=ext_rabitq_b2, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 43
}
```

### `gemma4-31b-bgt1-parity` (codec=ext_rabitq_b2, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 44
}
```

### `gemma4-31b-bgt1-parity` (codec=ext_rabitq_b3, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 42
}
```

### `gemma4-31b-bgt1-parity` (codec=ext_rabitq_b3, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 43
}
```

### `gemma4-31b-bgt1-parity` (codec=ext_rabitq_b3, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 44
}
```

### `gemma4-31b-bgt1-parity` (codec=ext_rabitq_b4, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 42
}
```

### `gemma4-31b-bgt1-parity` (codec=ext_rabitq_b4, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 43
}
```

### `gemma4-31b-bgt1-parity` (codec=ext_rabitq_b4, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 44
}
```

### `gemma4-31b-bgt1-parity` (codec=fp16, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 42
}
```

### `gemma4-31b-bgt1-parity` (codec=fp16, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 43
}
```

### `gemma4-31b-bgt1-parity` (codec=fp16, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 44
}
```

### `gemma4-31b-bgt1-parity` (codec=rabitq_b1, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `gemma4-31b-bgt1-parity` (codec=rabitq_b1, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `gemma4-31b-bgt1-parity` (codec=rabitq_b1, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `gemma4-31b-bgt1-parity` (codec=tq_mse_b3, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 42
}
```

### `gemma4-31b-bgt1-parity` (codec=tq_mse_b3, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 43
}
```

### `gemma4-31b-bgt1-parity` (codec=tq_mse_b3, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 44
}
```

### `gemma4-31b-bgt1-parity` (codec=tq_mse_b4, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 42
}
```

### `gemma4-31b-bgt1-parity` (codec=tq_mse_b4, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 43
}
```

### `gemma4-31b-bgt1-parity` (codec=tq_mse_b4, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `bgt1_direct_batched_reference`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Gemma4-31B-4bit B>1 parity against a direct mlx-lm batched reference. Mirrors the B>1 half of tests/test_p3_gemma4_batched_parity.py — the exact-vs-single-request drift observed there (first mismatch at token 2 on fp16 SDPA 4-bit) is precisely why the oracle compares against the direct reference instead of against Engine.generate. Runner overrides stop_token_ids=() on both sides so the reference (max_tokens unconditional) and Silica's batched stream stay length-aligned. Different-tokenized-length prompts exercise the non-trivial left_padding branch of make_batch_cache. Dual-gated on SILICA_REAL_GEMMA4_31B.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 44
}
```

### `gemma4-31b-smoke` (codec=block_tq_b32_b3, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 42
}
```

### `gemma4-31b-smoke` (codec=block_tq_b32_b3, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 43
}
```

### `gemma4-31b-smoke` (codec=block_tq_b32_b3, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 44
}
```

### `gemma4-31b-smoke` (codec=block_tq_b32_b4, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 42
}
```

### `gemma4-31b-smoke` (codec=block_tq_b32_b4, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 43
}
```

### `gemma4-31b-smoke` (codec=block_tq_b32_b4, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 44
}
```

### `gemma4-31b-smoke` (codec=block_tq_b64_b3, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 42
}
```

### `gemma4-31b-smoke` (codec=block_tq_b64_b3, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 43
}
```

### `gemma4-31b-smoke` (codec=block_tq_b64_b3, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 44
}
```

### `gemma4-31b-smoke` (codec=block_tq_b64_b4, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 42
}
```

### `gemma4-31b-smoke` (codec=block_tq_b64_b4, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 43
}
```

### `gemma4-31b-smoke` (codec=block_tq_b64_b4, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 44
}
```

### `gemma4-31b-smoke` (codec=ext_rabitq_b2, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 42
}
```

### `gemma4-31b-smoke` (codec=ext_rabitq_b2, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 43
}
```

### `gemma4-31b-smoke` (codec=ext_rabitq_b2, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 44
}
```

### `gemma4-31b-smoke` (codec=ext_rabitq_b3, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 42
}
```

### `gemma4-31b-smoke` (codec=ext_rabitq_b3, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 43
}
```

### `gemma4-31b-smoke` (codec=ext_rabitq_b3, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 44
}
```

### `gemma4-31b-smoke` (codec=ext_rabitq_b4, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 42
}
```

### `gemma4-31b-smoke` (codec=ext_rabitq_b4, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 43
}
```

### `gemma4-31b-smoke` (codec=ext_rabitq_b4, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 44
}
```

### `gemma4-31b-smoke` (codec=fp16, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 42
}
```

### `gemma4-31b-smoke` (codec=fp16, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 43
}
```

### `gemma4-31b-smoke` (codec=fp16, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 44
}
```

### `gemma4-31b-smoke` (codec=rabitq_b1, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `gemma4-31b-smoke` (codec=rabitq_b1, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `gemma4-31b-smoke` (codec=rabitq_b1, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `gemma4-31b-smoke` (codec=tq_mse_b3, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 42
}
```

### `gemma4-31b-smoke` (codec=tq_mse_b3, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 43
}
```

### `gemma4-31b-smoke` (codec=tq_mse_b3, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 44
}
```

### `gemma4-31b-smoke` (codec=tq_mse_b4, seed=42)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 42
}
```

### `gemma4-31b-smoke` (codec=tq_mse_b4, seed=43)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 43
}
```

### `gemma4-31b-smoke` (codec=tq_mse_b4, seed=44)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Dense Gemma4-31B-4bit single-request smoke. Mirrors the end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py ('model loads + forward runs on the real weights'). Pytest side additionally pins adapter capability flags + KV layout details that the bench SMOKE oracle does not check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk and the 50 sliding + 10 full layer forward is heavier than Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 44
}
```

### `gemma4-moe-smoke` (codec=block_tq_b32_b3, seed=42)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 42
}
```

### `gemma4-moe-smoke` (codec=block_tq_b32_b3, seed=43)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 43
}
```

### `gemma4-moe-smoke` (codec=block_tq_b32_b3, seed=44)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 44
}
```

### `gemma4-moe-smoke` (codec=block_tq_b32_b4, seed=42)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 42
}
```

### `gemma4-moe-smoke` (codec=block_tq_b32_b4, seed=43)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 43
}
```

### `gemma4-moe-smoke` (codec=block_tq_b32_b4, seed=44)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 44
}
```

### `gemma4-moe-smoke` (codec=block_tq_b64_b3, seed=42)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 42
}
```

### `gemma4-moe-smoke` (codec=block_tq_b64_b3, seed=43)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 43
}
```

### `gemma4-moe-smoke` (codec=block_tq_b64_b3, seed=44)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 44
}
```

### `gemma4-moe-smoke` (codec=block_tq_b64_b4, seed=42)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 42
}
```

### `gemma4-moe-smoke` (codec=block_tq_b64_b4, seed=43)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 43
}
```

### `gemma4-moe-smoke` (codec=block_tq_b64_b4, seed=44)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 44
}
```

### `gemma4-moe-smoke` (codec=ext_rabitq_b2, seed=42)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 42
}
```

### `gemma4-moe-smoke` (codec=ext_rabitq_b2, seed=43)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 43
}
```

### `gemma4-moe-smoke` (codec=ext_rabitq_b2, seed=44)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 44
}
```

### `gemma4-moe-smoke` (codec=ext_rabitq_b3, seed=42)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 42
}
```

### `gemma4-moe-smoke` (codec=ext_rabitq_b3, seed=43)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 43
}
```

### `gemma4-moe-smoke` (codec=ext_rabitq_b3, seed=44)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 44
}
```

### `gemma4-moe-smoke` (codec=ext_rabitq_b4, seed=42)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 42
}
```

### `gemma4-moe-smoke` (codec=ext_rabitq_b4, seed=43)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 43
}
```

### `gemma4-moe-smoke` (codec=ext_rabitq_b4, seed=44)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 44
}
```

### `gemma4-moe-smoke` (codec=fp16, seed=42)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 42
}
```

### `gemma4-moe-smoke` (codec=fp16, seed=43)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 43
}
```

### `gemma4-moe-smoke` (codec=fp16, seed=44)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 44
}
```

### `gemma4-moe-smoke` (codec=rabitq_b1, seed=42)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `gemma4-moe-smoke` (codec=rabitq_b1, seed=43)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `gemma4-moe-smoke` (codec=rabitq_b1, seed=44)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `gemma4-moe-smoke` (codec=tq_mse_b3, seed=42)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 42
}
```

### `gemma4-moe-smoke` (codec=tq_mse_b3, seed=43)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 43
}
```

### `gemma4-moe-smoke` (codec=tq_mse_b3, seed=44)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 44
}
```

### `gemma4-moe-smoke` (codec=tq_mse_b4, seed=42)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 42
}
```

### `gemma4-moe-smoke` (codec=tq_mse_b4, seed=43)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 43
}
```

### `gemma4-moe-smoke` (codec=tq_mse_b4, seed=44)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not crash through the always-on dense MLP + experts additive forward path on the real weights'). Dual-gated: the 4-bit checkpoint is ~16 GB on disk and the always-on dense MLP summed with the SwitchGLU experts branch makes per-token forward more expensive than dense Gemma4-31B.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 44
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 42
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 43
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 44
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 42
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 43
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 44
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 42
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 43
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 44
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 42
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 43
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 44
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 42
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 43
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 44
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 42
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 43
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 44
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 42
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 43
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 44
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 42
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 43
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 44
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 42
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 43
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 44
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 42
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 43
}
```

### `qwen3-0.6b-admission-headroom-prefix-heavy` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `admission_headroom`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'admission-headroom-prefix-heavy': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.3 step 2 admission-headroom row demonstrating §4.7 mode (B) vs mode (C) and pinning §7(c) acceptance n_block > n_fp16. Runner warms a prefix cache under IdentityCodec until store.resident_bytes() >= cap_bytes * warmup_ratio, replays the identical block recipe under block_tq_b64_b4, then runs a consecutive-AdmitDecision count against each. Compressed residency frees more headroom, so the compressed arm strictly admits more trial requests than the fp16 baseline. Hard gate on the inequality is the oracle's acceptance check; metadata surfaces the admission delta and the residency ratio for the bench report. Cache-only gate (weight load via engine_factory is paid but the engine itself is discarded — v1 accepts this cost rather than inventing a metadata-only adapter loader).

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 44
}
```

### `qwen3-0.6b-b1-parity` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 42
}
```

### `qwen3-0.6b-b1-parity` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 43
}
```

### `qwen3-0.6b-b1-parity` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 44
}
```

### `qwen3-0.6b-b1-parity` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 42
}
```

### `qwen3-0.6b-b1-parity` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 43
}
```

### `qwen3-0.6b-b1-parity` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 44
}
```

### `qwen3-0.6b-b1-parity` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 42
}
```

### `qwen3-0.6b-b1-parity` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 43
}
```

### `qwen3-0.6b-b1-parity` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 44
}
```

### `qwen3-0.6b-b1-parity` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 42
}
```

### `qwen3-0.6b-b1-parity` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 43
}
```

### `qwen3-0.6b-b1-parity` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 44
}
```

### `qwen3-0.6b-b1-parity` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 42
}
```

### `qwen3-0.6b-b1-parity` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 43
}
```

### `qwen3-0.6b-b1-parity` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 44
}
```

### `qwen3-0.6b-b1-parity` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 42
}
```

### `qwen3-0.6b-b1-parity` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 43
}
```

### `qwen3-0.6b-b1-parity` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 44
}
```

### `qwen3-0.6b-b1-parity` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 42
}
```

### `qwen3-0.6b-b1-parity` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 43
}
```

### `qwen3-0.6b-b1-parity` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 44
}
```

### `qwen3-0.6b-b1-parity` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 42
}
```

### `qwen3-0.6b-b1-parity` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 43
}
```

### `qwen3-0.6b-b1-parity` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 44
}
```

### `qwen3-0.6b-b1-parity` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-b1-parity` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-b1-parity` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-b1-parity` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 42
}
```

### `qwen3-0.6b-b1-parity` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 43
}
```

### `qwen3-0.6b-b1-parity` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 44
}
```

### `qwen3-0.6b-b1-parity` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 42
}
```

### `qwen3-0.6b-b1-parity` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 43
}
```

### `qwen3-0.6b-b1-parity` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B=1 parity regression: for the same prompt, the B=1 batched path through Engine.generate_batch(..., max_batch_size=1) must emit the same token stream as the single-request Engine.generate path. Runner drives both executions with the shared _build_sampling_params helper so divergence cannot be blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B weights so the scheduler B=1 claim rides every dev run.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 44
}
```

### `qwen3-0.6b-bgt1-parity` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 42
}
```

### `qwen3-0.6b-bgt1-parity` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 43
}
```

### `qwen3-0.6b-bgt1-parity` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 44
}
```

### `qwen3-0.6b-bgt1-parity` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 42
}
```

### `qwen3-0.6b-bgt1-parity` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 43
}
```

### `qwen3-0.6b-bgt1-parity` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 44
}
```

### `qwen3-0.6b-bgt1-parity` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 42
}
```

### `qwen3-0.6b-bgt1-parity` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 43
}
```

### `qwen3-0.6b-bgt1-parity` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 44
}
```

### `qwen3-0.6b-bgt1-parity` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 42
}
```

### `qwen3-0.6b-bgt1-parity` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 43
}
```

### `qwen3-0.6b-bgt1-parity` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 44
}
```

### `qwen3-0.6b-bgt1-parity` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 42
}
```

### `qwen3-0.6b-bgt1-parity` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 43
}
```

### `qwen3-0.6b-bgt1-parity` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 44
}
```

### `qwen3-0.6b-bgt1-parity` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 42
}
```

### `qwen3-0.6b-bgt1-parity` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 43
}
```

### `qwen3-0.6b-bgt1-parity` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 44
}
```

### `qwen3-0.6b-bgt1-parity` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 42
}
```

### `qwen3-0.6b-bgt1-parity` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 43
}
```

### `qwen3-0.6b-bgt1-parity` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 44
}
```

### `qwen3-0.6b-bgt1-parity` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 42
}
```

### `qwen3-0.6b-bgt1-parity` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 43
}
```

### `qwen3-0.6b-bgt1-parity` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 44
}
```

### `qwen3-0.6b-bgt1-parity` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-bgt1-parity` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-bgt1-parity` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-bgt1-parity` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 42
}
```

### `qwen3-0.6b-bgt1-parity` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 43
}
```

### `qwen3-0.6b-bgt1-parity` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 44
}
```

### `qwen3-0.6b-bgt1-parity` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 42
}
```

### `qwen3-0.6b-bgt1-parity` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 43
}
```

### `qwen3-0.6b-bgt1-parity` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `bgt1_direct_batched_reference`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=8`, `prompts=2`
- status: **failed**
- reason: `codec_override_invalid:Workload 'bgt1-different-length-prompts': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

B>1 scheduler glue regression: Silica's generate_batch at max_batch_size=2 on two prompts whose tokenized lengths differ must emit the same per-row tokens as a direct mlx-lm batched forward driven with adapter.make_batch_cache(left_padding). Runner overrides stop_token_ids=() on both sides so the reference (which runs unconditionally for max_tokens) and Silica's stream stay length-aligned regardless of EOS. Prompts 'Hello' (1 token) and 'The capital of Japan is' (5 tokens) force left_padding=[4, 0] on the Qwen3 tokenizer, exercising the non-trivial left-pad branch of make_batch_cache. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 44
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b3",
  "kv_codec": "block_tq_b32_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2408448,
  "resident_bytes_per_block": 401408,
  "seed": 42
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b3",
  "kv_codec": "block_tq_b32_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2408448,
  "resident_bytes_per_block": 401408,
  "seed": 43
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b3",
  "kv_codec": "block_tq_b32_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2408448,
  "resident_bytes_per_block": 401408,
  "seed": 44
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b4",
  "kv_codec": "block_tq_b32_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2580480,
  "resident_bytes_per_block": 516096,
  "seed": 42
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b4",
  "kv_codec": "block_tq_b32_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2580480,
  "resident_bytes_per_block": 516096,
  "seed": 43
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b4",
  "kv_codec": "block_tq_b32_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2580480,
  "resident_bytes_per_block": 516096,
  "seed": 44
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b3",
  "kv_codec": "block_tq_b64_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2236416,
  "resident_bytes_per_block": 372736,
  "seed": 42
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b3",
  "kv_codec": "block_tq_b64_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2236416,
  "resident_bytes_per_block": 372736,
  "seed": 43
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b3",
  "kv_codec": "block_tq_b64_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2236416,
  "resident_bytes_per_block": 372736,
  "seed": 44
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b4",
  "kv_codec": "block_tq_b64_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2924544,
  "resident_bytes_per_block": 487424,
  "seed": 42
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b4",
  "kv_codec": "block_tq_b64_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2924544,
  "resident_bytes_per_block": 487424,
  "seed": 43
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b4",
  "kv_codec": "block_tq_b64_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2924544,
  "resident_bytes_per_block": 487424,
  "seed": 44
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b2",
  "kv_codec": "ext_rabitq_b2",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 1634304,
  "resident_bytes_per_block": 272384,
  "seed": 42
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b2",
  "kv_codec": "ext_rabitq_b2",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 1634304,
  "resident_bytes_per_block": 272384,
  "seed": 43
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b2",
  "kv_codec": "ext_rabitq_b2",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 1634304,
  "resident_bytes_per_block": 272384,
  "seed": 44
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b3",
  "kv_codec": "ext_rabitq_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2322432,
  "resident_bytes_per_block": 387072,
  "seed": 42
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b3",
  "kv_codec": "ext_rabitq_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2322432,
  "resident_bytes_per_block": 387072,
  "seed": 43
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b3",
  "kv_codec": "ext_rabitq_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2322432,
  "resident_bytes_per_block": 387072,
  "seed": 44
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b4",
  "kv_codec": "ext_rabitq_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2508800,
  "resident_bytes_per_block": 501760,
  "seed": 42
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b4",
  "kv_codec": "ext_rabitq_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2508800,
  "resident_bytes_per_block": 501760,
  "seed": 43
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b4",
  "kv_codec": "ext_rabitq_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 3010560,
  "resident_bytes_per_block": 501760,
  "seed": 44
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "fp16",
  "kv_codec": "fp16",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 9175040,
  "resident_bytes_per_block": 1835008,
  "seed": 42
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "fp16",
  "kv_codec": "fp16",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 9175040,
  "resident_bytes_per_block": 1835008,
  "seed": 43
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "fp16",
  "kv_codec": "fp16",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 9175040,
  "resident_bytes_per_block": 1835008,
  "seed": 44
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'compression-block-tq-b64-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'compression-block-tq-b64-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'compression-block-tq-b64-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b3",
  "kv_codec": "tq_mse_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2150400,
  "resident_bytes_per_block": 358400,
  "seed": 42
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b3",
  "kv_codec": "tq_mse_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2150400,
  "resident_bytes_per_block": 358400,
  "seed": 43
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b3",
  "kv_codec": "tq_mse_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2150400,
  "resident_bytes_per_block": 358400,
  "seed": 44
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b4",
  "kv_codec": "tq_mse_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2838528,
  "resident_bytes_per_block": 473088,
  "seed": 42
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b4",
  "kv_codec": "tq_mse_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2838528,
  "resident_bytes_per_block": 473088,
  "seed": 43
}
```

### `qwen3-0.6b-compression-block-tq-b64-b4` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression row. Production recommendation per vqbench REPORT §3.1 (3.76x total-KV compression on Qwen3.5-4B). The resident_bytes number this row surfaces is the headline compression observable of P-5.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b4",
  "kv_codec": "tq_mse_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2838528,
  "resident_bytes_per_block": 473088,
  "seed": 44
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b3",
  "kv_codec": "block_tq_b32_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2408448,
  "resident_bytes_per_block": 401408,
  "seed": 42
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b3",
  "kv_codec": "block_tq_b32_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2408448,
  "resident_bytes_per_block": 401408,
  "seed": 43
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b3",
  "kv_codec": "block_tq_b32_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2408448,
  "resident_bytes_per_block": 401408,
  "seed": 44
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b4",
  "kv_codec": "block_tq_b32_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2580480,
  "resident_bytes_per_block": 516096,
  "seed": 42
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b4",
  "kv_codec": "block_tq_b32_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2580480,
  "resident_bytes_per_block": 516096,
  "seed": 43
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b4",
  "kv_codec": "block_tq_b32_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2580480,
  "resident_bytes_per_block": 516096,
  "seed": 44
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b3",
  "kv_codec": "block_tq_b64_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2236416,
  "resident_bytes_per_block": 372736,
  "seed": 42
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b3",
  "kv_codec": "block_tq_b64_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2236416,
  "resident_bytes_per_block": 372736,
  "seed": 43
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b3",
  "kv_codec": "block_tq_b64_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2236416,
  "resident_bytes_per_block": 372736,
  "seed": 44
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b4",
  "kv_codec": "block_tq_b64_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2924544,
  "resident_bytes_per_block": 487424,
  "seed": 42
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b4",
  "kv_codec": "block_tq_b64_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2924544,
  "resident_bytes_per_block": 487424,
  "seed": 43
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b4",
  "kv_codec": "block_tq_b64_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2924544,
  "resident_bytes_per_block": 487424,
  "seed": 44
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b2",
  "kv_codec": "ext_rabitq_b2",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 1634304,
  "resident_bytes_per_block": 272384,
  "seed": 42
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b2",
  "kv_codec": "ext_rabitq_b2",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 1634304,
  "resident_bytes_per_block": 272384,
  "seed": 43
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b2",
  "kv_codec": "ext_rabitq_b2",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 1634304,
  "resident_bytes_per_block": 272384,
  "seed": 44
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b3",
  "kv_codec": "ext_rabitq_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2322432,
  "resident_bytes_per_block": 387072,
  "seed": 42
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b3",
  "kv_codec": "ext_rabitq_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2322432,
  "resident_bytes_per_block": 387072,
  "seed": 43
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b3",
  "kv_codec": "ext_rabitq_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2322432,
  "resident_bytes_per_block": 387072,
  "seed": 44
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b4",
  "kv_codec": "ext_rabitq_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2508800,
  "resident_bytes_per_block": 501760,
  "seed": 42
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b4",
  "kv_codec": "ext_rabitq_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2508800,
  "resident_bytes_per_block": 501760,
  "seed": 43
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b4",
  "kv_codec": "ext_rabitq_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 3010560,
  "resident_bytes_per_block": 501760,
  "seed": 44
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "fp16",
  "kv_codec": "fp16",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 9175040,
  "resident_bytes_per_block": 1835008,
  "seed": 42
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "fp16",
  "kv_codec": "fp16",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 9175040,
  "resident_bytes_per_block": 1835008,
  "seed": 43
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "fp16",
  "kv_codec": "fp16",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 9175040,
  "resident_bytes_per_block": 1835008,
  "seed": 44
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'compression-ext-rabitq-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'compression-ext-rabitq-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'compression-ext-rabitq-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b3",
  "kv_codec": "tq_mse_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2150400,
  "resident_bytes_per_block": 358400,
  "seed": 42
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b3",
  "kv_codec": "tq_mse_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2150400,
  "resident_bytes_per_block": 358400,
  "seed": 43
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b3",
  "kv_codec": "tq_mse_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2150400,
  "resident_bytes_per_block": 358400,
  "seed": 44
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b4",
  "kv_codec": "tq_mse_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2838528,
  "resident_bytes_per_block": 473088,
  "seed": 42
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b4",
  "kv_codec": "tq_mse_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2838528,
  "resident_bytes_per_block": 473088,
  "seed": 43
}
```

### `qwen3-0.6b-compression-ext-rabitq-b4` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 ExtRaBitQ 4-bit K+V compression row. effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Reports raw resident_bytes; compared against the fp16 baseline row at the bench report layer.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b4",
  "kv_codec": "tq_mse_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2838528,
  "resident_bytes_per_block": 473088,
  "seed": 44
}
```

### `qwen3-0.6b-compression-fp16` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b3",
  "kv_codec": "block_tq_b32_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2408448,
  "resident_bytes_per_block": 401408,
  "seed": 42
}
```

### `qwen3-0.6b-compression-fp16` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b3",
  "kv_codec": "block_tq_b32_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2408448,
  "resident_bytes_per_block": 401408,
  "seed": 43
}
```

### `qwen3-0.6b-compression-fp16` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b3",
  "kv_codec": "block_tq_b32_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2408448,
  "resident_bytes_per_block": 401408,
  "seed": 44
}
```

### `qwen3-0.6b-compression-fp16` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b4",
  "kv_codec": "block_tq_b32_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2580480,
  "resident_bytes_per_block": 516096,
  "seed": 42
}
```

### `qwen3-0.6b-compression-fp16` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b4",
  "kv_codec": "block_tq_b32_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2580480,
  "resident_bytes_per_block": 516096,
  "seed": 43
}
```

### `qwen3-0.6b-compression-fp16` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b4",
  "kv_codec": "block_tq_b32_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2580480,
  "resident_bytes_per_block": 516096,
  "seed": 44
}
```

### `qwen3-0.6b-compression-fp16` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b3",
  "kv_codec": "block_tq_b64_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2236416,
  "resident_bytes_per_block": 372736,
  "seed": 42
}
```

### `qwen3-0.6b-compression-fp16` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b3",
  "kv_codec": "block_tq_b64_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2236416,
  "resident_bytes_per_block": 372736,
  "seed": 43
}
```

### `qwen3-0.6b-compression-fp16` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b3",
  "kv_codec": "block_tq_b64_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2236416,
  "resident_bytes_per_block": 372736,
  "seed": 44
}
```

### `qwen3-0.6b-compression-fp16` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b4",
  "kv_codec": "block_tq_b64_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2924544,
  "resident_bytes_per_block": 487424,
  "seed": 42
}
```

### `qwen3-0.6b-compression-fp16` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b4",
  "kv_codec": "block_tq_b64_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2924544,
  "resident_bytes_per_block": 487424,
  "seed": 43
}
```

### `qwen3-0.6b-compression-fp16` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b4",
  "kv_codec": "block_tq_b64_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2924544,
  "resident_bytes_per_block": 487424,
  "seed": 44
}
```

### `qwen3-0.6b-compression-fp16` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b2",
  "kv_codec": "ext_rabitq_b2",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 1634304,
  "resident_bytes_per_block": 272384,
  "seed": 42
}
```

### `qwen3-0.6b-compression-fp16` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b2",
  "kv_codec": "ext_rabitq_b2",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 1634304,
  "resident_bytes_per_block": 272384,
  "seed": 43
}
```

### `qwen3-0.6b-compression-fp16` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b2",
  "kv_codec": "ext_rabitq_b2",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 1634304,
  "resident_bytes_per_block": 272384,
  "seed": 44
}
```

### `qwen3-0.6b-compression-fp16` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b3",
  "kv_codec": "ext_rabitq_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2322432,
  "resident_bytes_per_block": 387072,
  "seed": 42
}
```

### `qwen3-0.6b-compression-fp16` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b3",
  "kv_codec": "ext_rabitq_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2322432,
  "resident_bytes_per_block": 387072,
  "seed": 43
}
```

### `qwen3-0.6b-compression-fp16` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b3",
  "kv_codec": "ext_rabitq_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2322432,
  "resident_bytes_per_block": 387072,
  "seed": 44
}
```

### `qwen3-0.6b-compression-fp16` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b4",
  "kv_codec": "ext_rabitq_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2508800,
  "resident_bytes_per_block": 501760,
  "seed": 42
}
```

### `qwen3-0.6b-compression-fp16` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b4",
  "kv_codec": "ext_rabitq_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2508800,
  "resident_bytes_per_block": 501760,
  "seed": 43
}
```

### `qwen3-0.6b-compression-fp16` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b4",
  "kv_codec": "ext_rabitq_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 3010560,
  "resident_bytes_per_block": 501760,
  "seed": 44
}
```

### `qwen3-0.6b-compression-fp16` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "fp16",
  "kv_codec": "fp16",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 9175040,
  "resident_bytes_per_block": 1835008,
  "seed": 42
}
```

### `qwen3-0.6b-compression-fp16` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "fp16",
  "kv_codec": "fp16",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 9175040,
  "resident_bytes_per_block": 1835008,
  "seed": 43
}
```

### `qwen3-0.6b-compression-fp16` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "fp16",
  "kv_codec": "fp16",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 9175040,
  "resident_bytes_per_block": 1835008,
  "seed": 44
}
```

### `qwen3-0.6b-compression-fp16` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'compression-fp16': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-compression-fp16` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'compression-fp16': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-compression-fp16` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'compression-fp16': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-compression-fp16` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b3",
  "kv_codec": "tq_mse_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2150400,
  "resident_bytes_per_block": 358400,
  "seed": 42
}
```

### `qwen3-0.6b-compression-fp16` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b3",
  "kv_codec": "tq_mse_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2150400,
  "resident_bytes_per_block": 358400,
  "seed": 43
}
```

### `qwen3-0.6b-compression-fp16` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b3",
  "kv_codec": "tq_mse_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2150400,
  "resident_bytes_per_block": 358400,
  "seed": 44
}
```

### `qwen3-0.6b-compression-fp16` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b4",
  "kv_codec": "tq_mse_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2838528,
  "resident_bytes_per_block": 473088,
  "seed": 42
}
```

### `qwen3-0.6b-compression-fp16` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b4",
  "kv_codec": "tq_mse_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2838528,
  "resident_bytes_per_block": 473088,
  "seed": 43
}
```

### `qwen3-0.6b-compression-fp16` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 IdentityCodec baseline compression row. Same shared-prefix 2-prompt workload as the A.3c decode-speed rows; after the event stream drains, the runner reads prefix_cache.store.resident_bytes() and live-block counts. Under IdentityCodec the resident_bytes equals the uncompressed fp16 K/V sum, giving every subsequent compression row a directly-comparable baseline. Explicitly uses kv_codec='fp16' (not None) so the store is SyntheticPrefixBlockStore with IdentityCodec; the pass-through path would leave resident_bytes_per_block as None and break cross-row compression comparisons. Cache-only gate.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b4",
  "kv_codec": "tq_mse_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2838528,
  "resident_bytes_per_block": 473088,
  "seed": 44
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b3",
  "kv_codec": "block_tq_b32_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2408448,
  "resident_bytes_per_block": 401408,
  "seed": 42
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b3",
  "kv_codec": "block_tq_b32_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2408448,
  "resident_bytes_per_block": 401408,
  "seed": 43
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b3",
  "kv_codec": "block_tq_b32_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2408448,
  "resident_bytes_per_block": 401408,
  "seed": 44
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b4",
  "kv_codec": "block_tq_b32_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2580480,
  "resident_bytes_per_block": 516096,
  "seed": 42
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b4",
  "kv_codec": "block_tq_b32_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2580480,
  "resident_bytes_per_block": 516096,
  "seed": 43
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b32_b4",
  "kv_codec": "block_tq_b32_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2580480,
  "resident_bytes_per_block": 516096,
  "seed": 44
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b3",
  "kv_codec": "block_tq_b64_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2236416,
  "resident_bytes_per_block": 372736,
  "seed": 42
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b3",
  "kv_codec": "block_tq_b64_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2236416,
  "resident_bytes_per_block": 372736,
  "seed": 43
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b3",
  "kv_codec": "block_tq_b64_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2236416,
  "resident_bytes_per_block": 372736,
  "seed": 44
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b4",
  "kv_codec": "block_tq_b64_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2924544,
  "resident_bytes_per_block": 487424,
  "seed": 42
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b4",
  "kv_codec": "block_tq_b64_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2924544,
  "resident_bytes_per_block": 487424,
  "seed": 43
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "block_tq_b64_b4",
  "kv_codec": "block_tq_b64_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2924544,
  "resident_bytes_per_block": 487424,
  "seed": 44
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b2",
  "kv_codec": "ext_rabitq_b2",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 1634304,
  "resident_bytes_per_block": 272384,
  "seed": 42
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b2",
  "kv_codec": "ext_rabitq_b2",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 1634304,
  "resident_bytes_per_block": 272384,
  "seed": 43
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b2",
  "kv_codec": "ext_rabitq_b2",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 1634304,
  "resident_bytes_per_block": 272384,
  "seed": 44
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b3",
  "kv_codec": "ext_rabitq_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2322432,
  "resident_bytes_per_block": 387072,
  "seed": 42
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b3",
  "kv_codec": "ext_rabitq_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2322432,
  "resident_bytes_per_block": 387072,
  "seed": 43
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b3",
  "kv_codec": "ext_rabitq_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2322432,
  "resident_bytes_per_block": 387072,
  "seed": 44
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b4",
  "kv_codec": "ext_rabitq_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2508800,
  "resident_bytes_per_block": 501760,
  "seed": 42
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b4",
  "kv_codec": "ext_rabitq_b4",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 2508800,
  "resident_bytes_per_block": 501760,
  "seed": 43
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "ext_rabitq_b4",
  "kv_codec": "ext_rabitq_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 3010560,
  "resident_bytes_per_block": 501760,
  "seed": 44
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "fp16",
  "kv_codec": "fp16",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 9175040,
  "resident_bytes_per_block": 1835008,
  "seed": 42
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "fp16",
  "kv_codec": "fp16",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 9175040,
  "resident_bytes_per_block": 1835008,
  "seed": 43
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "fp16",
  "kv_codec": "fp16",
  "live_blocks": 5,
  "prefix_cache_hits": 1,
  "resident_bytes": 9175040,
  "resident_bytes_per_block": 1835008,
  "seed": 44
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'compression-tq-mse-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'compression-tq-mse-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'compression-tq-mse-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b3",
  "kv_codec": "tq_mse_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2150400,
  "resident_bytes_per_block": 358400,
  "seed": 42
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b3",
  "kv_codec": "tq_mse_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2150400,
  "resident_bytes_per_block": 358400,
  "seed": 43
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b3",
  "kv_codec": "tq_mse_b3",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2150400,
  "resident_bytes_per_block": 358400,
  "seed": 44
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b4",
  "kv_codec": "tq_mse_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2838528,
  "resident_bytes_per_block": 473088,
  "seed": 42
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b4",
  "kv_codec": "tq_mse_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2838528,
  "resident_bytes_per_block": 473088,
  "seed": 43
}
```

### `qwen3-0.6b-compression-tq-mse-b4` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `storage`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-C.3 TurboQuantMSE 4-bit K+V compression row. Scalar 4-bit quantization; effective bits/coord = 4 + 2/head_dim (one fp16 scale per vector). Reports raw resident_bytes; cross-row compression ratio vs the fp16 row is a downstream derivation.

Metadata:

```
{
  "block_size": 16,
  "codec_id": "tq_mse_b4",
  "kv_codec": "tq_mse_b4",
  "live_blocks": 6,
  "prefix_cache_hits": 1,
  "resident_bytes": 2838528,
  "resident_bytes_per_block": 473088,
  "seed": 44
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 15.059,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 15.063,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 15.064,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 15.064,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 42,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 20.712,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.713,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.714,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.714,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 43,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 20.643,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.644,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.644,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.645,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 44,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 20.651,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.652,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.653,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.653,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 42,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 15.654,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 15.655,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 15.656,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 15.656,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 43,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 21.51,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 21.512,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 21.512,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 21.512,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 44,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 20.829,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.831,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.831,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.831,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 42,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 15.817,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 15.818,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 15.818,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 15.819,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 43,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 16.614,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 16.615,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 16.615,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 16.616,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 44,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 25.192,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 25.193,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 25.193,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 25.194,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 42,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 21.532,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 21.534,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 21.534,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 21.535,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 43,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 18.054,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 18.055,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 18.055,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 18.055,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 44,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 20.464,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.465,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.465,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.466,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 42,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 20.708,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.709,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.709,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.71,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 43,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 20.668,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.669,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.669,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.669,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 44,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 12.683,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 12.684,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 12.684,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 12.685,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 42,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 20.86,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.861,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.862,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.862,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 43,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 20.772,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.774,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.774,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.774,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 44,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 20.751,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.753,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.753,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 20.753,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 42,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 12.973,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 12.974,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 12.975,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 12.975,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 43,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 26.608,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 26.609,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 26.609,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 26.609,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 44,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 21.293,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 21.294,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 21.294,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 21.295,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 42,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 17.501,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 17.503,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 17.503,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 17.503,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 43,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 21.548,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 21.549,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 21.549,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 21.55,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 44,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **failed**
- reason: `ValueError: workload 'concurrent-shared-prefix': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **failed**
- reason: `ValueError: workload 'concurrent-shared-prefix': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **failed**
- reason: `ValueError: workload 'concurrent-shared-prefix': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 23.379,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 23.38,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 23.381,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 23.381,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 42,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 27.736,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 27.738,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 27.739,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 27.739,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 43,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 22.266,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 22.268,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 22.268,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 22.269,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 44,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 21.243,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 21.244,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 21.244,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 21.245,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 42,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 25.346,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 25.348,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 25.348,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 25.349,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 43,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-concurrent-shared-prefix` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=8`, `prompts=4`
- status: **ok**

Four prompts share a three-token prefix 'The capital of' (tokenized as three ids on Qwen3). Runs at max_batch_size=4 with prefix_cache=True so the radix prefix cache reuses the shared block across rows. SMOKE oracle validates every row emits valid tokens; the JSONL row's per-row metadata makes the reader aware how many tokens each of the four rows produced. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "max_token_id": 24081,
  "rows": [
    {
      "first_token_ms_offset": 15.33,
      "row": 0,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 15.332,
      "row": 1,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 15.332,
      "row": 2,
      "token_count": 8
    },
    {
      "first_token_ms_offset": 15.333,
      "row": 3,
      "token_count": 8
    }
  ],
  "seed": 44,
  "total_tokens": 32,
  "vocab_size": 151643
}
```

### `qwen3-0.6b-long-in-short-out` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 42
}
```

### `qwen3-0.6b-long-in-short-out` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 43
}
```

### `qwen3-0.6b-long-in-short-out` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 44
}
```

### `qwen3-0.6b-long-in-short-out` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 42
}
```

### `qwen3-0.6b-long-in-short-out` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 43
}
```

### `qwen3-0.6b-long-in-short-out` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 44
}
```

### `qwen3-0.6b-long-in-short-out` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 42
}
```

### `qwen3-0.6b-long-in-short-out` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 43
}
```

### `qwen3-0.6b-long-in-short-out` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 44
}
```

### `qwen3-0.6b-long-in-short-out` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 42
}
```

### `qwen3-0.6b-long-in-short-out` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 43
}
```

### `qwen3-0.6b-long-in-short-out` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 44
}
```

### `qwen3-0.6b-long-in-short-out` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 42
}
```

### `qwen3-0.6b-long-in-short-out` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 43
}
```

### `qwen3-0.6b-long-in-short-out` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 44
}
```

### `qwen3-0.6b-long-in-short-out` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 42
}
```

### `qwen3-0.6b-long-in-short-out` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 43
}
```

### `qwen3-0.6b-long-in-short-out` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 44
}
```

### `qwen3-0.6b-long-in-short-out` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 42
}
```

### `qwen3-0.6b-long-in-short-out` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 43
}
```

### `qwen3-0.6b-long-in-short-out` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 44
}
```

### `qwen3-0.6b-long-in-short-out` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 42
}
```

### `qwen3-0.6b-long-in-short-out` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 43
}
```

### `qwen3-0.6b-long-in-short-out` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 44
}
```

### `qwen3-0.6b-long-in-short-out` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-long-in-short-out` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-long-in-short-out` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-long-in-short-out` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 42
}
```

### `qwen3-0.6b-long-in-short-out` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 43
}
```

### `qwen3-0.6b-long-in-short-out` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 44
}
```

### `qwen3-0.6b-long-in-short-out` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 42
}
```

### `qwen3-0.6b-long-in-short-out` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 43
}
```

### `qwen3-0.6b-long-in-short-out` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'long-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Prefill-dominated throughput shape. Long synthetic prompt ('The quick brown fox jumps over the lazy dog. ' x 30 = ~301 tokens on Qwen3) + max_tokens=4 means prefill dominates wall time. SMOKE oracle — the value is in the prefill_tok_s metric. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 144.19789632278307,
  "row0_first_token_ms": 26.62120806053281,
  "row0_tokens": 16,
  "row1_decode_tok_s": 152.06266614743018,
  "row1_first_token_ms": 241.07650003861636,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 150.27450203272215,
  "row0_first_token_ms": 25.78975004144013,
  "row0_tokens": 16,
  "row1_decode_tok_s": 151.88553271198478,
  "row1_first_token_ms": 235.8671249821782,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 153.32863748657718,
  "row0_first_token_ms": 16.52079203631729,
  "row0_tokens": 16,
  "row1_decode_tok_s": 149.71280146293515,
  "row1_first_token_ms": 229.332874994725,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 130.9771030224731,
  "row0_first_token_ms": 22.012208006344736,
  "row0_tokens": 16,
  "row1_decode_tok_s": 136.67711392687133,
  "row1_first_token_ms": 277.7789169922471,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 151.83825562905378,
  "row0_first_token_ms": 25.87304194457829,
  "row0_tokens": 16,
  "row1_decode_tok_s": 161.32732898450473,
  "row1_first_token_ms": 243.51929198019207,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 158.71357349324936,
  "row0_first_token_ms": 25.72233392857015,
  "row0_tokens": 16,
  "row1_decode_tok_s": 153.9969256601305,
  "row1_first_token_ms": 234.84275001101196,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 156.2899550003527,
  "row0_first_token_ms": 25.865582982078195,
  "row0_tokens": 16,
  "row1_decode_tok_s": 153.34614150517658,
  "row1_first_token_ms": 231.14237491972744,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 155.6060985992986,
  "row0_first_token_ms": 25.47345799393952,
  "row0_tokens": 16,
  "row1_decode_tok_s": 150.5011688890478,
  "row1_first_token_ms": 233.44370804261416,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 153.83215124751877,
  "row0_first_token_ms": 25.8645829744637,
  "row0_tokens": 16,
  "row1_decode_tok_s": 154.1640337000047,
  "row1_first_token_ms": 231.9357079686597,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 153.56481519045877,
  "row0_first_token_ms": 16.73133298754692,
  "row0_tokens": 16,
  "row1_decode_tok_s": 152.91175956613165,
  "row1_first_token_ms": 230.27441697195172,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 146.62093726035266,
  "row0_first_token_ms": 26.242917054332793,
  "row0_tokens": 16,
  "row1_decode_tok_s": 160.0980500855821,
  "row1_first_token_ms": 243.55170805938542,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 158.69013512240906,
  "row0_first_token_ms": 16.780999954789877,
  "row0_tokens": 16,
  "row1_decode_tok_s": 149.4252104508628,
  "row1_first_token_ms": 222.0082499552518,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 148.5540741742954,
  "row0_first_token_ms": 25.626666960306466,
  "row0_tokens": 16,
  "row1_decode_tok_s": 152.90974402482527,
  "row1_first_token_ms": 223.72908296529204,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 152.00603577917354,
  "row0_first_token_ms": 25.754540925845504,
  "row0_tokens": 16,
  "row1_decode_tok_s": 155.26731479356462,
  "row1_first_token_ms": 220.7383329514414,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 156.12910728628708,
  "row0_first_token_ms": 25.743625010363758,
  "row0_tokens": 16,
  "row1_decode_tok_s": 156.3094299999002,
  "row1_first_token_ms": 213.69262493681163,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 154.73528658331742,
  "row0_first_token_ms": 25.79533401876688,
  "row0_tokens": 16,
  "row1_decode_tok_s": 149.90449740359304,
  "row1_first_token_ms": 231.85575008392334,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 151.4482236824055,
  "row0_first_token_ms": 25.159000069834292,
  "row0_tokens": 16,
  "row1_decode_tok_s": 150.94244704238517,
  "row1_first_token_ms": 229.2161660734564,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 148.42551403766845,
  "row0_first_token_ms": 26.019167038612068,
  "row0_tokens": 16,
  "row1_decode_tok_s": 146.2985243200003,
  "row1_first_token_ms": 236.79574998095632,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 162.69987734899152,
  "row0_first_token_ms": 25.507666054181755,
  "row0_tokens": 16,
  "row1_decode_tok_s": 155.72672085222703,
  "row1_first_token_ms": 230.7622500229627,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 154.35327790728664,
  "row0_first_token_ms": 25.4012499935925,
  "row0_tokens": 16,
  "row1_decode_tok_s": 156.44358596173964,
  "row1_first_token_ms": 238.84133307728916,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 151.7835847068476,
  "row0_first_token_ms": 35.00420902855694,
  "row0_tokens": 16,
  "row1_decode_tok_s": 156.66757626961243,
  "row1_first_token_ms": 252.0024999976158,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 148.3148350827022,
  "row0_first_token_ms": 26.0624170769006,
  "row0_tokens": 16,
  "row1_decode_tok_s": 152.79999571658684,
  "row1_first_token_ms": 141.62012503948063,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 150.76734249840928,
  "row0_first_token_ms": 25.71283304132521,
  "row0_tokens": 16,
  "row1_decode_tok_s": 150.2743754931648,
  "row1_first_token_ms": 140.06333309225738,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 149.8716095103853,
  "row0_first_token_ms": 22.772499942220747,
  "row0_tokens": 16,
  "row1_decode_tok_s": 149.18118171059288,
  "row1_first_token_ms": 137.69016694277525,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'prefix-hit-decode-block-tq-b64-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'prefix-hit-decode-block-tq-b64-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'prefix-hit-decode-block-tq-b64-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 162.96494864762727,
  "row0_first_token_ms": 30.473625054582953,
  "row0_tokens": 16,
  "row1_decode_tok_s": 159.7050081639628,
  "row1_first_token_ms": 210.51079197786748,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 159.2627197063561,
  "row0_first_token_ms": 25.688249967060983,
  "row0_tokens": 16,
  "row1_decode_tok_s": 153.11447538517228,
  "row1_first_token_ms": 209.03795794583857,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 153.4007017993276,
  "row0_first_token_ms": 16.59562496934086,
  "row0_tokens": 16,
  "row1_decode_tok_s": 154.88127697170245,
  "row1_first_token_ms": 204.39091697335243,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 157.34306885792006,
  "row0_first_token_ms": 25.39899991825223,
  "row0_tokens": 16,
  "row1_decode_tok_s": 146.1947336895205,
  "row1_first_token_ms": 218.0776249151677,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 155.85016161802966,
  "row0_first_token_ms": 25.5915840389207,
  "row0_tokens": 16,
  "row1_decode_tok_s": 156.23175990772165,
  "row1_first_token_ms": 221.2514589773491,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c compression row. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 num_bits=4, the opening §5.2 / vqbench REPORT §3.1 production recommendation (strictly lossless at std=0% across three seeds, 3.76× total-KV compression). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks — the codec overhead this scenario measures. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 152.502954753932,
  "row0_first_token_ms": 16.44333405420184,
  "row0_tokens": 16,
  "row1_decode_tok_s": 154.97875759696078,
  "row1_first_token_ms": 210.33654198981822,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 133.11896214513584,
  "row0_first_token_ms": 25.301874964497983,
  "row0_tokens": 16,
  "row1_decode_tok_s": 153.95694922995614,
  "row1_first_token_ms": 247.8240829659626,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 152.78618439812868,
  "row0_first_token_ms": 25.817499961704016,
  "row0_tokens": 16,
  "row1_decode_tok_s": 155.2634978147626,
  "row1_first_token_ms": 232.3694999795407,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 151.64471423335792,
  "row0_first_token_ms": 25.575166917406023,
  "row0_tokens": 16,
  "row1_decode_tok_s": 152.0705022486477,
  "row1_first_token_ms": 231.44508292898536,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 159.29238660265094,
  "row0_first_token_ms": 25.581665919162333,
  "row0_tokens": 16,
  "row1_decode_tok_s": 155.418805709404,
  "row1_first_token_ms": 233.00279094837606,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 155.04416816017093,
  "row0_first_token_ms": 25.568499928340316,
  "row0_tokens": 16,
  "row1_decode_tok_s": 157.86067215123455,
  "row1_first_token_ms": 245.14691601507366,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 157.97427770275436,
  "row0_first_token_ms": 25.4870830103755,
  "row0_tokens": 16,
  "row1_decode_tok_s": 152.57851269574974,
  "row1_first_token_ms": 234.38304103910923,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 153.59442921816233,
  "row0_first_token_ms": 25.567500037141144,
  "row0_tokens": 16,
  "row1_decode_tok_s": 153.7891069251632,
  "row1_first_token_ms": 229.0358750615269,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 153.07678050118142,
  "row0_first_token_ms": 16.694291960448027,
  "row0_tokens": 16,
  "row1_decode_tok_s": 134.2742472476537,
  "row1_first_token_ms": 228.04962494410574,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 152.5499999662576,
  "row0_first_token_ms": 16.266250051558018,
  "row0_tokens": 16,
  "row1_decode_tok_s": 156.75585109629708,
  "row1_first_token_ms": 221.99579200241715,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 154.17710789373072,
  "row0_first_token_ms": 16.84091694187373,
  "row0_tokens": 16,
  "row1_decode_tok_s": 158.16507437046553,
  "row1_first_token_ms": 227.4673340143636,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 155.4079382277339,
  "row0_first_token_ms": 16.15729194600135,
  "row0_tokens": 16,
  "row1_decode_tok_s": 155.92380410723055,
  "row1_first_token_ms": 229.61520799435675,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 153.3588783759688,
  "row0_first_token_ms": 16.60829095635563,
  "row0_tokens": 16,
  "row1_decode_tok_s": 156.59527139567854,
  "row1_first_token_ms": 230.68337503354996,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 155.80530567309285,
  "row0_first_token_ms": 25.69087501615286,
  "row0_tokens": 16,
  "row1_decode_tok_s": 155.20465952966165,
  "row1_first_token_ms": 223.98666699882597,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 150.82532831026032,
  "row0_first_token_ms": 25.646583060733974,
  "row0_tokens": 16,
  "row1_decode_tok_s": 152.50250350714342,
  "row1_first_token_ms": 212.06804201938212,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 155.57859409737057,
  "row0_first_token_ms": 27.022792026400566,
  "row0_tokens": 16,
  "row1_decode_tok_s": 150.1369999545146,
  "row1_first_token_ms": 210.7477500103414,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 155.00324589614053,
  "row0_first_token_ms": 31.78191604092717,
  "row0_tokens": 16,
  "row1_decode_tok_s": 159.44922578309877,
  "row1_first_token_ms": 228.77591603901237,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 155.4439711107647,
  "row0_first_token_ms": 25.812749983742833,
  "row0_tokens": 16,
  "row1_decode_tok_s": 136.29646820835495,
  "row1_first_token_ms": 240.69483298808336,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 152.17180980771187,
  "row0_first_token_ms": 25.83812503144145,
  "row0_tokens": 16,
  "row1_decode_tok_s": 150.90935405033034,
  "row1_first_token_ms": 230.6130409706384,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 157.54461470038095,
  "row0_first_token_ms": 16.449457965791225,
  "row0_tokens": 16,
  "row1_decode_tok_s": 157.95694864987675,
  "row1_first_token_ms": 225.30133300460875,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 157.40471119907033,
  "row0_first_token_ms": 25.419916957616806,
  "row0_tokens": 16,
  "row1_decode_tok_s": 147.06681415402295,
  "row1_first_token_ms": 229.38404197338969,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 154.11598805190468,
  "row0_first_token_ms": 25.652291951701045,
  "row0_tokens": 16,
  "row1_decode_tok_s": 153.78832002199252,
  "row1_first_token_ms": 250.33445900771767,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 153.18510116234458,
  "row0_first_token_ms": 27.799875009804964,
  "row0_tokens": 16,
  "row1_decode_tok_s": 154.5960765023952,
  "row1_first_token_ms": 139.61979106534272,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 154.400677165185,
  "row0_first_token_ms": 36.31491702981293,
  "row0_tokens": 16,
  "row1_decode_tok_s": 160.53755174427272,
  "row1_first_token_ms": 147.06550003029406,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 158.64076591000503,
  "row0_first_token_ms": 25.734417024068534,
  "row0_tokens": 16,
  "row1_decode_tok_s": 160.283096760979,
  "row1_first_token_ms": 134.00220905896276,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'prefix-hit-decode-ext-rabitq-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'prefix-hit-decode-ext-rabitq-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'prefix-hit-decode-ext-rabitq-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 154.8110108657828,
  "row0_first_token_ms": 31.305584008805454,
  "row0_tokens": 16,
  "row1_decode_tok_s": 157.27089618437114,
  "row1_first_token_ms": 218.29320897813886,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 161.36059265856386,
  "row0_first_token_ms": 25.498708011582494,
  "row0_tokens": 16,
  "row1_decode_tok_s": 154.83837728145,
  "row1_first_token_ms": 205.96062496770173,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 154.5830663560873,
  "row0_first_token_ms": 16.81666704826057,
  "row0_tokens": 16,
  "row1_decode_tok_s": 151.76899349351652,
  "row1_first_token_ms": 204.90474998950958,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 156.24755858147557,
  "row0_first_token_ms": 25.367834023199975,
  "row0_tokens": 16,
  "row1_decode_tok_s": 156.61557164967604,
  "row1_first_token_ms": 221.38637502212077,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 154.51923809743096,
  "row0_first_token_ms": 25.91887500602752,
  "row0_tokens": 16,
  "row1_decode_tok_s": 155.73797085227523,
  "row1_first_token_ms": 220.63137497752905,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit decode-speed acceptance gate. Identical workload shape to qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec='ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, effective bits/coord = 4 + 48/head_dim = 4.375 at head_dim=128). Row 1's seeded-admission path exercises ``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × num_layers × num_hit_blocks with ExtRaBitQ's integer-grid codebook lookup + per-vector scale multiply + re-normalization + inverse rotation; the codec overhead this scenario measures. The B.3 acceptance test compares this row's decode_tok_s against the paired fp16 baseline to gate ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). rabitq_b1 is deliberately excluded from the gate — it is K-only (``v_supported=False``) so the symmetric kv_codec= shorthand cannot install it, and its hypercube MSE is worse than BlockTQ at matching bit budget. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 153.1303674310865,
  "row0_first_token_ms": 19.05720797367394,
  "row0_tokens": 16,
  "row1_decode_tok_s": 154.81127592211553,
  "row1_first_token_ms": 218.44470803625882,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 156.39913635897085,
  "row0_first_token_ms": 25.6185419857502,
  "row0_tokens": 16,
  "row1_decode_tok_s": 155.60724153478716,
  "row1_first_token_ms": 229.6499169897288,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 152.57573319370385,
  "row0_first_token_ms": 16.28912496380508,
  "row0_tokens": 16,
  "row1_decode_tok_s": 157.7857402485282,
  "row1_first_token_ms": 222.47820801567286,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 158.06784947447076,
  "row0_first_token_ms": 26.015000068582594,
  "row0_tokens": 16,
  "row1_decode_tok_s": 151.25459336610334,
  "row1_first_token_ms": 228.6762910662219,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 154.1326825647611,
  "row0_first_token_ms": 25.785582954995334,
  "row0_tokens": 16,
  "row1_decode_tok_s": 155.61383431892935,
  "row1_first_token_ms": 239.17512490879744,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 157.0123687584575,
  "row0_first_token_ms": 25.58241703081876,
  "row0_tokens": 16,
  "row1_decode_tok_s": 155.96372858121762,
  "row1_first_token_ms": 235.8679169556126,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 156.46881187555277,
  "row0_first_token_ms": 25.727792060934007,
  "row0_tokens": 16,
  "row1_decode_tok_s": 156.63615212748763,
  "row1_first_token_ms": 236.68841703329235,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 160.29515692331125,
  "row0_first_token_ms": 25.590208009816706,
  "row0_tokens": 16,
  "row1_decode_tok_s": 156.70174182250506,
  "row1_first_token_ms": 223.83912501391023,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 155.73985626954746,
  "row0_first_token_ms": 25.74449998792261,
  "row0_tokens": 16,
  "row1_decode_tok_s": 155.31902892005203,
  "row1_first_token_ms": 229.38899998553097,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 157.37443362910082,
  "row0_first_token_ms": 25.871749967336655,
  "row0_tokens": 16,
  "row1_decode_tok_s": 145.97888881532802,
  "row1_first_token_ms": 226.19349998421967,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 155.41216345689034,
  "row0_first_token_ms": 25.81520809326321,
  "row0_tokens": 16,
  "row1_decode_tok_s": 148.30389725444337,
  "row1_first_token_ms": 242.32558300718665,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 152.71255676843631,
  "row0_first_token_ms": 25.92883398756385,
  "row0_tokens": 16,
  "row1_decode_tok_s": 153.7284923477114,
  "row1_first_token_ms": 238.3517089765519,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 162.8507566972059,
  "row0_first_token_ms": 25.645249988883734,
  "row0_tokens": 16,
  "row1_decode_tok_s": 164.6363040204576,
  "row1_first_token_ms": 226.8652500351891,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 163.94980520858894,
  "row0_first_token_ms": 25.735124945640564,
  "row0_tokens": 16,
  "row1_decode_tok_s": 165.02315888096695,
  "row1_first_token_ms": 198.46316694747657,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 153.82649713405087,
  "row0_first_token_ms": 25.700250058434904,
  "row0_tokens": 16,
  "row1_decode_tok_s": 158.14659275425734,
  "row1_first_token_ms": 212.36241701990366,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 158.34818274745425,
  "row0_first_token_ms": 25.415000040084124,
  "row0_tokens": 16,
  "row1_decode_tok_s": 156.77530656792925,
  "row1_first_token_ms": 208.0246249679476,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 159.82185194175466,
  "row0_first_token_ms": 25.395417003892362,
  "row0_tokens": 16,
  "row1_decode_tok_s": 142.54309464018883,
  "row1_first_token_ms": 218.4423329308629,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 154.83864261757125,
  "row0_first_token_ms": 25.412374990992248,
  "row0_tokens": 16,
  "row1_decode_tok_s": 155.42336955256167,
  "row1_first_token_ms": 223.7397920107469,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 156.31038153002768,
  "row0_first_token_ms": 25.9584590094164,
  "row0_tokens": 16,
  "row1_decode_tok_s": 151.37651665680454,
  "row1_first_token_ms": 222.31804195325822,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 162.28065674314342,
  "row0_first_token_ms": 25.8480419870466,
  "row0_tokens": 16,
  "row1_decode_tok_s": 162.94348223902898,
  "row1_first_token_ms": 226.85824998188764,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 161.4196318412474,
  "row0_first_token_ms": 25.933208060450852,
  "row0_tokens": 16,
  "row1_decode_tok_s": 156.559926774937,
  "row1_first_token_ms": 224.72487506456673,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 154.52149197873786,
  "row0_first_token_ms": 25.572832906618714,
  "row0_tokens": 16,
  "row1_decode_tok_s": 157.16989417064818,
  "row1_first_token_ms": 237.2024579672143,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "fp16",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 158.08499301132025,
  "row0_first_token_ms": 25.86354094091803,
  "row0_tokens": 16,
  "row1_decode_tok_s": 155.9904221659959,
  "row1_first_token_ms": 134.6424159128219,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "fp16",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 154.87807877156902,
  "row0_first_token_ms": 22.355458000674844,
  "row0_tokens": 16,
  "row1_decode_tok_s": 154.67445540729287,
  "row1_first_token_ms": 133.27124994248152,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "fp16",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 141.50932311374544,
  "row0_first_token_ms": 27.812624932266772,
  "row0_tokens": 16,
  "row1_decode_tok_s": 141.04604439995586,
  "row1_first_token_ms": 148.8962909206748,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'prefix-hit-decode-fp16': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'prefix-hit-decode-fp16': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **failed**
- reason: `ValueError: workload 'prefix-hit-decode-fp16': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 144.06569354411522,
  "row0_first_token_ms": 36.13720799330622,
  "row0_tokens": 16,
  "row1_decode_tok_s": 137.07116281205953,
  "row1_first_token_ms": 234.16324995923787,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 141.812122580836,
  "row0_first_token_ms": 17.805457930080593,
  "row0_tokens": 16,
  "row1_decode_tok_s": 143.19045889847663,
  "row1_first_token_ms": 217.50745794270188,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 142.58572080610247,
  "row0_first_token_ms": 18.26554210856557,
  "row0_tokens": 16,
  "row1_decode_tok_s": 140.89550833651063,
  "row1_first_token_ms": 218.39037502650172,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 141.54225965392874,
  "row0_first_token_ms": 19.373749964870512,
  "row0_tokens": 16,
  "row1_decode_tok_s": 137.76429218733222,
  "row1_first_token_ms": 227.78862505219877,
  "row1_tokens": 16,
  "seed": 42
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 140.05035631782974,
  "row0_first_token_ms": 18.992291996255517,
  "row0_tokens": 16,
  "row1_decode_tok_s": 140.50173176436437,
  "row1_first_token_ms": 226.3499170076102,
  "row1_tokens": 16,
  "seed": 43
}
```

### `qwen3-0.6b-prefix-hit-decode-fp16` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `decode_tok_s_with_prefix_hit`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=16`, `prompts=2`
- status: **ok**

P-5-A.3c baseline row. Two identical prompts at max_batch_size=1 prefix_cache=True: row 0 admits into the initial cohort and runs miss-path prefill + decode; row 1 enters the waiting queue and is admitted mid-run through _admit_single_hit_row. Under kv_codec='fp16' the store installs IdentityCodec, so row 1's seeded-admission + decode loop reconstructs fp16 K/V by reference — no compression, honest baseline. The oracle reports row 1's steady-state decode tok/s (inter-token interval, excluding first-token latency). Cache-only gate; the A.3c acceptance test compares this row's decode_tok_s against the paired block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. Opening §7(d) referred to Qwen3.5-0.8B as the target; that checkpoint is hybrid-DeltaNet and cannot host RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, no recurrent state).

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "prefix_cache_hits": 1,
  "row0_decode_tok_s": 141.48162606350988,
  "row0_first_token_ms": 18.00670800730586,
  "row0_tokens": 16,
  "row1_decode_tok_s": 141.99273296707443,
  "row1_first_token_ms": 227.11370803881437,
  "row1_tokens": 16,
  "seed": 44
}
```

### `qwen3-0.6b-short-in-long-out` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 42
}
```

### `qwen3-0.6b-short-in-long-out` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 43
}
```

### `qwen3-0.6b-short-in-long-out` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 44
}
```

### `qwen3-0.6b-short-in-long-out` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 42
}
```

### `qwen3-0.6b-short-in-long-out` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 43
}
```

### `qwen3-0.6b-short-in-long-out` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 44
}
```

### `qwen3-0.6b-short-in-long-out` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 42
}
```

### `qwen3-0.6b-short-in-long-out` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 43
}
```

### `qwen3-0.6b-short-in-long-out` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 44
}
```

### `qwen3-0.6b-short-in-long-out` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 42
}
```

### `qwen3-0.6b-short-in-long-out` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 43
}
```

### `qwen3-0.6b-short-in-long-out` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 44
}
```

### `qwen3-0.6b-short-in-long-out` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 42
}
```

### `qwen3-0.6b-short-in-long-out` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 43
}
```

### `qwen3-0.6b-short-in-long-out` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 44
}
```

### `qwen3-0.6b-short-in-long-out` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 42
}
```

### `qwen3-0.6b-short-in-long-out` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 43
}
```

### `qwen3-0.6b-short-in-long-out` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 44
}
```

### `qwen3-0.6b-short-in-long-out` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 42
}
```

### `qwen3-0.6b-short-in-long-out` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 43
}
```

### `qwen3-0.6b-short-in-long-out` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 44
}
```

### `qwen3-0.6b-short-in-long-out` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 42
}
```

### `qwen3-0.6b-short-in-long-out` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 43
}
```

### `qwen3-0.6b-short-in-long-out` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 44
}
```

### `qwen3-0.6b-short-in-long-out` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-short-in-long-out` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-short-in-long-out` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-short-in-long-out` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 42
}
```

### `qwen3-0.6b-short-in-long-out` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 43
}
```

### `qwen3-0.6b-short-in-long-out` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 44
}
```

### `qwen3-0.6b-short-in-long-out` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 42
}
```

### `qwen3-0.6b-short-in-long-out` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 43
}
```

### `qwen3-0.6b-short-in-long-out` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=64`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-long-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Decode-dominated throughput shape. One-token prompt 'Hello' + max_tokens=64 means decode (64 steps) vastly exceeds prefill (1 token). SMOKE oracle — the point of the row is the decode_tok_s metric, not output correctness. Cache-only gate (reuses the qwen3-0.6b-smoke weights) so it rides every dev run.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 44
}
```

### `qwen3-0.6b-smoke` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 42
}
```

### `qwen3-0.6b-smoke` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 43
}
```

### `qwen3-0.6b-smoke` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 44
}
```

### `qwen3-0.6b-smoke` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 42
}
```

### `qwen3-0.6b-smoke` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 43
}
```

### `qwen3-0.6b-smoke` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 44
}
```

### `qwen3-0.6b-smoke` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 42
}
```

### `qwen3-0.6b-smoke` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 43
}
```

### `qwen3-0.6b-smoke` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 44
}
```

### `qwen3-0.6b-smoke` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 42
}
```

### `qwen3-0.6b-smoke` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 43
}
```

### `qwen3-0.6b-smoke` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 44
}
```

### `qwen3-0.6b-smoke` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 42
}
```

### `qwen3-0.6b-smoke` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 43
}
```

### `qwen3-0.6b-smoke` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 44
}
```

### `qwen3-0.6b-smoke` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 42
}
```

### `qwen3-0.6b-smoke` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 43
}
```

### `qwen3-0.6b-smoke` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 44
}
```

### `qwen3-0.6b-smoke` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 42
}
```

### `qwen3-0.6b-smoke` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 43
}
```

### `qwen3-0.6b-smoke` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 44
}
```

### `qwen3-0.6b-smoke` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 42
}
```

### `qwen3-0.6b-smoke` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 43
}
```

### `qwen3-0.6b-smoke` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 44
}
```

### `qwen3-0.6b-smoke` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-smoke` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-smoke` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-smoke` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 42
}
```

### `qwen3-0.6b-smoke` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 43
}
```

### `qwen3-0.6b-smoke` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 44
}
```

### `qwen3-0.6b-smoke` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 42
}
```

### `qwen3-0.6b-smoke` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 43
}
```

### `qwen3-0.6b-smoke` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded tokens from prompt 'Hello'. Cache-only gate so it runs on any dev box that has pulled the P-2 parity test weights (see tests/test_p2_batched_parity.py). Exercises the full runner path (load, generate, oracle, metrics, JSONL) without claiming correctness beyond 'did not crash and emitted valid token ids'.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 44
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 42
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 43
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 44
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 42
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 43
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 44
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 42
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 43
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 44
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 42
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 43
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 44
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 42
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 43
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 44
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 42
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 43
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 44
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 42
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 43
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 44
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 42
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 43
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 44
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 42
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 43
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 44
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 42
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 43
}
```

### `qwen3-0.6b-teacher-forced-argmax` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `teacher_forced_argmax`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=1`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'teacher-forced-argmax': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Teacher-forced next-token argmax parity (PLAN §P-3 exit criterion). Silica drives adapter.prefill + decode_step with teacher-forced target tokens; the reference is a single mlx-lm forward over prompt + target[:-1] with positional logits sliced to match. Oracle passes when silica's per-position argmax matches the reference at >= min_agreement_rate (0.98 per PLAN; 100% expected on cached 0.6B because both paths drive the same mlx-lm model). Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 44
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 42
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 43
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 44
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 42
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 43
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 44
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 42
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 43
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 44
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 42
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 43
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 44
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 42
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 43
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 44
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 42
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 43
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 44
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 42
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 43
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 44
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 42
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 43
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 44
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 42
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 43
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 44
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 42
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 43
}
```

### `qwen3-0.6b-ttft-under-concurrency` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `smoke`
- gate: `(cache-only)`
- workload: `max_batch_size=4`, `max_tokens=4`, `prompts=4`
- status: **failed**
- reason: `codec_override_invalid:Workload 'ttft-under-concurrency': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

One long (~301 tokens on Qwen3) prompt admitted alongside three one-character prompts at max_batch_size=4, prefix_cache=False, max_tokens=4. PLAN §P-4 specifies this shape to resolve Q-010 (chunked-prefill promotion): with Silica's current non-chunked prefill, the long prompt's prefill blocks the short prompts, inflating their TTFT. The metrics this row collects are the input to the Q-010 decision — the oracle itself is SMOKE because 'did not crash + all rows emitted valid tokens' is the correctness floor. Cache-only gate.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 44
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2209.2860202789307,
  "ppl": 75.44893036498772,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2551.6434421539307,
  "ppl": 147.44148624486556,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2255.0379734039307,
  "ppl": 82.51583078348384,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1580.8882541656494,
  "ppl": 22.058869879214225,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1598.5643405914307,
  "ppl": 22.835262687125994,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1715.063425064087,
  "ppl": 28.682537928893264,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2346.3011569976807,
  "ppl": 98.65090353122767,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2532.6593112945557,
  "ppl": 142.0643966917217,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2327.836679458618,
  "ppl": 95.14988493678995,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1660.6739597320557,
  "ppl": 25.786502425114588,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1636.5769748687744,
  "ppl": 24.598726346419596,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1643.1718845367432,
  "ppl": 24.918252244862614,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b2",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b2",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 3864.4185886383057,
  "ppl": 1924.5799044750288,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b2",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b2",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 3269.8673191070557,
  "ppl": 601.2179844210533,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b2",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b2",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 3757.4063816070557,
  "ppl": 1560.943549101275,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2956.3167819976807,
  "ppl": 325.497811383089,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2919.1109714508057,
  "ppl": 302.64059077330995,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2684.0135593414307,
  "ppl": 191.03806834004726,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1769.513864517212,
  "ppl": 31.90762903951316,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1823.3533420562744,
  "ppl": 35.45293878004074,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2006.3073825836182,
  "ppl": 50.71580203742226,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "fp16",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "fp16",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1522.3972263336182,
  "ppl": 19.67307465653248,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "fp16",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "fp16",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1522.3972263336182,
  "ppl": 19.67307465653248,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "fp16",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "fp16",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1522.3972263336182,
  "ppl": 19.67307465653248,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `ValueError: workload 'wikitext-ppl-block-tq-b64-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `ValueError: workload 'wikitext-ppl-block-tq-b64-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `ValueError: workload 'wikitext-ppl-block-tq-b64-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2164.565683364868,
  "ppl": 69.12667368197864,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2503.1563816070557,
  "ppl": 134.09450037787852,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2367.0990085601807,
  "ppl": 102.74885276241311,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1687.910348892212,
  "ppl": 27.198215586028898,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1689.4229831695557,
  "ppl": 27.278845536921125,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. Production recommendation per vqbench REPORT §3.1 (strictly lossless at std=0% across three seeds on Qwen3.5-4B; 3.76x total-KV compression). silica mirrors the configuration on Qwen3-0.6B; the row emits a raw PPL number at step 3a. ΔPPL against the fp16 baseline is a downstream computation (bench report / C.6 vqbench cross-check), not wired into the per-row runner in 3a. C.6 step 1 wires this row through --vqbench-xcheck so a side-by-side vqbench PPL lands in metadata.vqbench_*.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1652.4350681304932,
  "ppl": 25.374078404608635,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b3",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1575.584119796753,
  "ppl": 21.83108501810531,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b3",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1599.462200164795,
  "ppl": 22.875420969285482,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b3",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1534.5563926696777,
  "ppl": 20.14680627311678,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b4",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1541.6840705871582,
  "ppl": 20.42979280947669,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b4",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1540.761344909668,
  "ppl": 20.392935500864212,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b4",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1528.7344989776611,
  "ppl": 19.918573534490193,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b3",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1579.8783493041992,
  "ppl": 22.01531731422077,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b3",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1593.5773620605469,
  "ppl": 22.61349149100017,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b3",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1563.6384449005127,
  "ppl": 21.326657537916063,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b4",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1544.7753143310547,
  "ppl": 20.55375538345001,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b4",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1526.9075813293457,
  "ppl": 19.847488169961025,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b4",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1534.646348953247,
  "ppl": 20.15035322291844,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b2",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b2",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2276.60711479187,
  "ppl": 86.07334944400162,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b2",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b2",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2388.1551456451416,
  "ppl": 107.07113596889556,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b2",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b2",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2557.662389755249,
  "ppl": 149.18843271984767,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b3",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1613.344913482666,
  "ppl": 23.505413362076272,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b3",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1622.053554534912,
  "ppl": 23.90943380310049,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b3",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1588.2922286987305,
  "ppl": 22.380811674037783,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b4",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1548.7733783721924,
  "ppl": 20.715198709962984,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b4",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1536.575922012329,
  "ppl": 20.2265862581983,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b4",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1549.3867816925049,
  "ppl": 20.740080121534646,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "fp16",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "fp16",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1522.3972263336182,
  "ppl": 19.67307465653248,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "fp16",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "fp16",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1522.3972263336182,
  "ppl": 19.67307465653248,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "fp16",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "fp16",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1522.3972263336182,
  "ppl": 19.67307465653248,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `RuntimeError: scenario 'qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned': codec_quality_path='vqbench_aligned' requires a symmetric codec (k_supported and v_supported); got 'rabitq_b1' which is k_supported=True, v_supported=False. An asymmetric codec on this path would silently wrap V with a K-only configuration.`

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `RuntimeError: scenario 'qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned': codec_quality_path='vqbench_aligned' requires a symmetric codec (k_supported and v_supported); got 'rabitq_b1' which is k_supported=True, v_supported=False. An asymmetric codec on this path would silently wrap V with a K-only configuration.`

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `RuntimeError: scenario 'qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned': codec_quality_path='vqbench_aligned' requires a symmetric codec (k_supported and v_supported); got 'rabitq_b1' which is k_supported=True, v_supported=False. An asymmetric codec on this path would silently wrap V with a K-only configuration.`

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b3",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1594.8120651245117,
  "ppl": 22.668197374348907,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b3",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1600.4263896942139,
  "ppl": 22.918624615236837,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b3",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1556.8714218139648,
  "ppl": 21.04609666162314,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b4",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1547.9828205108643,
  "ppl": 20.683175418782973,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b4",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1538.764654159546,
  "ppl": 20.3134072478059,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-D.2a diagnostic PPL row. Identical workload shape to qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec in vqbench's pre-RoPE projection-patch semantic instead of the production prefix-cache store. Exists because the D.2 investigation found the post-RoPE store path and the pre-RoPE projection-patch path give ΔPPL numbers that differ by ~20x at the same codec Frobenius; the C.6 vqbench cross-check binds against this row so the comparison is apples-to-apples with vqbench's own harness (both sides inject noise in pre-RoPE space). The post-RoPE store row is preserved as a separate observable of the production path's quality cost. See plans/P5_D2_INVESTIGATION/ for the probe scripts that established this split.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b4",
  "codec_quality_path": "vqbench_aligned",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1537.876272201538,
  "ppl": 20.278122733323954,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2209.2860202789307,
  "ppl": 75.44893036498772,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2551.6434421539307,
  "ppl": 147.44148624486556,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2255.0379734039307,
  "ppl": 82.51583078348384,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1580.8882541656494,
  "ppl": 22.058869879214225,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1598.5643405914307,
  "ppl": 22.835262687125994,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1715.063425064087,
  "ppl": 28.682537928893264,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2346.3011569976807,
  "ppl": 98.65090353122767,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2532.6593112945557,
  "ppl": 142.0643966917217,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2327.836679458618,
  "ppl": 95.14988493678995,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1660.6739597320557,
  "ppl": 25.786502425114588,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1636.5769748687744,
  "ppl": 24.598726346419596,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1643.1718845367432,
  "ppl": 24.918252244862614,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b2",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b2",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 3864.4185886383057,
  "ppl": 1924.5799044750288,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b2",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b2",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 3269.8673191070557,
  "ppl": 601.2179844210533,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b2",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b2",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 3757.4063816070557,
  "ppl": 1560.943549101275,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2956.3167819976807,
  "ppl": 325.497811383089,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2919.1109714508057,
  "ppl": 302.64059077330995,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2684.0135593414307,
  "ppl": 191.03806834004726,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1769.513864517212,
  "ppl": 31.90762903951316,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1823.3533420562744,
  "ppl": 35.45293878004074,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2006.3073825836182,
  "ppl": 50.71580203742226,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "fp16",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "fp16",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1522.3972263336182,
  "ppl": 19.67307465653248,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "fp16",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "fp16",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1522.3972263336182,
  "ppl": 19.67307465653248,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "fp16",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "fp16",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1522.3972263336182,
  "ppl": 19.67307465653248,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `ValueError: workload 'wikitext-ppl-ext-rabitq-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `ValueError: workload 'wikitext-ppl-ext-rabitq-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `ValueError: workload 'wikitext-ppl-ext-rabitq-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2164.565683364868,
  "ppl": 69.12667368197864,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2503.1563816070557,
  "ppl": 134.09450037787852,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2367.0990085601807,
  "ppl": 102.74885276241311,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1687.910348892212,
  "ppl": 27.198215586028898,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1689.4229831695557,
  "ppl": 27.278845536921125,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm of the PPL bench; effective bits/coord = 4 + 48/head_dim (head_dim=128 -> 4.375). Emits a raw PPL number at step 3a; ΔPPL derivation against fp16 is downstream. rabitq_b1 is deliberately excluded: K-only codec (``v_supported=False``) that cannot install via the symmetric ``kv_codec=`` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1652.4350681304932,
  "ppl": 25.374078404608635,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 42
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 43
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 44
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 42
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 43
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 44
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 42
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 43
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 44
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 42
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 43
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 44
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 42
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 43
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 44
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 42
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 43
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 44
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 42
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 43
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 44
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 42
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 43
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 44
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 42
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 43
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 44
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 42
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 43
}
```

### `qwen3-0.6b-wikitext-ppl-fp16` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `codec_override_invalid:Workload 'wikitext-ppl-fp16': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on WikiText-2 raw test split (chunk_size=256, max_tokens=512 matching vqbench REPORT headline). Cache-agnostic fp16 baseline — drives the adapter's own BatchKVCache across chunks (shared cache, chunk-invariant). Cache-only gate plus WikiText cache file presence at ~/.cache/silica/wikitext2-test.txt (populate once via scripts/prepare_wikitext2_cache.py).

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 44
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2209.2860202789307,
  "ppl": 75.44893036498772,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2551.6434421539307,
  "ppl": 147.44148624486556,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2255.0379734039307,
  "ppl": 82.51583078348384,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1580.8882541656494,
  "ppl": 22.058869879214225,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1598.5643405914307,
  "ppl": 22.835262687125994,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b32_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b32_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1715.063425064087,
  "ppl": 28.682537928893264,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2346.3011569976807,
  "ppl": 98.65090353122767,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2532.6593112945557,
  "ppl": 142.0643966917217,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2327.836679458618,
  "ppl": 95.14988493678995,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1660.6739597320557,
  "ppl": 25.786502425114588,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1636.5769748687744,
  "ppl": 24.598726346419596,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "block_tq_b64_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "block_tq_b64_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1643.1718845367432,
  "ppl": 24.918252244862614,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b2",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b2",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 3864.4185886383057,
  "ppl": 1924.5799044750288,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b2",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b2",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 3269.8673191070557,
  "ppl": 601.2179844210533,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b2",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b2",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 3757.4063816070557,
  "ppl": 1560.943549101275,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2956.3167819976807,
  "ppl": 325.497811383089,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2919.1109714508057,
  "ppl": 302.64059077330995,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2684.0135593414307,
  "ppl": 191.03806834004726,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1769.513864517212,
  "ppl": 31.90762903951316,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1823.3533420562744,
  "ppl": 35.45293878004074,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "ext_rabitq_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "ext_rabitq_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2006.3073825836182,
  "ppl": 50.71580203742226,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "fp16",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "fp16",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1522.3972263336182,
  "ppl": 19.67307465653248,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "fp16",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "fp16",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1522.3972263336182,
  "ppl": 19.67307465653248,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "fp16",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "fp16",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1522.3972263336182,
  "ppl": 19.67307465653248,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `ValueError: workload 'wikitext-ppl-tq-mse-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `ValueError: workload 'wikitext-ppl-tq-mse-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **failed**
- reason: `ValueError: workload 'wikitext-ppl-tq-mse-b4': kv_codec='rabitq_b1' is not symmetric (k_supported=True, v_supported=False); shorthand installs one codec on both sides. Asymmetric K/V codecs need a split-id field (P-5-C scope) rather than the kv_codec shorthand.`

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2164.565683364868,
  "ppl": 69.12667368197864,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2503.1563816070557,
  "ppl": 134.09450037787852,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b3",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b3",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 2367.0990085601807,
  "ppl": 102.74885276241311,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1687.910348892212,
  "ppl": 27.198215586028898,
  "seed": 42,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1689.4229831695557,
  "ppl": 27.278845536921125,
  "seed": 43,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3-0.6b-wikitext-ppl-tq-mse-b4` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `ppl`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=0`, `prompts=0`
- status: **ok**

P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every chunk's prior prefix through ``tq_mse_b4`` encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 baseline is a downstream derivation (bench report / C.6 vqbench cross-check) — step 3a does not propagate the fp16 PPL between rows at runtime.

Metadata:

```
{
  "chunk_size": 256,
  "codec_id": "tq_mse_b4",
  "codec_quality_path": "prefix_store_post_rope",
  "delta_ppl": null,
  "delta_ppl_pct": null,
  "kv_codec": "tq_mse_b4",
  "max_tokens": 512,
  "n_tokens": 511,
  "nll_sum": 1652.4350681304932,
  "ppl": 25.374078404608635,
  "seed": 44,
  "wikitext_path": "/Users/xinyu/.cache/silica/wikitext2-test.txt"
}
```

### `qwen3.5-0.8b-b1-parity` (codec=block_tq_b32_b3, seed=42)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 42
}
```

### `qwen3.5-0.8b-b1-parity` (codec=block_tq_b32_b3, seed=43)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 43
}
```

### `qwen3.5-0.8b-b1-parity` (codec=block_tq_b32_b3, seed=44)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 44
}
```

### `qwen3.5-0.8b-b1-parity` (codec=block_tq_b32_b4, seed=42)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 42
}
```

### `qwen3.5-0.8b-b1-parity` (codec=block_tq_b32_b4, seed=43)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 43
}
```

### `qwen3.5-0.8b-b1-parity` (codec=block_tq_b32_b4, seed=44)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 44
}
```

### `qwen3.5-0.8b-b1-parity` (codec=block_tq_b64_b3, seed=42)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 42
}
```

### `qwen3.5-0.8b-b1-parity` (codec=block_tq_b64_b3, seed=43)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 43
}
```

### `qwen3.5-0.8b-b1-parity` (codec=block_tq_b64_b3, seed=44)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 44
}
```

### `qwen3.5-0.8b-b1-parity` (codec=block_tq_b64_b4, seed=42)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 42
}
```

### `qwen3.5-0.8b-b1-parity` (codec=block_tq_b64_b4, seed=43)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 43
}
```

### `qwen3.5-0.8b-b1-parity` (codec=block_tq_b64_b4, seed=44)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 44
}
```

### `qwen3.5-0.8b-b1-parity` (codec=ext_rabitq_b2, seed=42)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 42
}
```

### `qwen3.5-0.8b-b1-parity` (codec=ext_rabitq_b2, seed=43)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 43
}
```

### `qwen3.5-0.8b-b1-parity` (codec=ext_rabitq_b2, seed=44)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 44
}
```

### `qwen3.5-0.8b-b1-parity` (codec=ext_rabitq_b3, seed=42)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 42
}
```

### `qwen3.5-0.8b-b1-parity` (codec=ext_rabitq_b3, seed=43)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 43
}
```

### `qwen3.5-0.8b-b1-parity` (codec=ext_rabitq_b3, seed=44)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 44
}
```

### `qwen3.5-0.8b-b1-parity` (codec=ext_rabitq_b4, seed=42)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 42
}
```

### `qwen3.5-0.8b-b1-parity` (codec=ext_rabitq_b4, seed=43)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 43
}
```

### `qwen3.5-0.8b-b1-parity` (codec=ext_rabitq_b4, seed=44)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 44
}
```

### `qwen3.5-0.8b-b1-parity` (codec=fp16, seed=42)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 42
}
```

### `qwen3.5-0.8b-b1-parity` (codec=fp16, seed=43)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 43
}
```

### `qwen3.5-0.8b-b1-parity` (codec=fp16, seed=44)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 44
}
```

### `qwen3.5-0.8b-b1-parity` (codec=rabitq_b1, seed=42)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3.5-0.8b-b1-parity` (codec=rabitq_b1, seed=43)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3.5-0.8b-b1-parity` (codec=rabitq_b1, seed=44)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3.5-0.8b-b1-parity` (codec=tq_mse_b3, seed=42)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 42
}
```

### `qwen3.5-0.8b-b1-parity` (codec=tq_mse_b3, seed=43)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 43
}
```

### `qwen3.5-0.8b-b1-parity` (codec=tq_mse_b3, seed=44)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 44
}
```

### `qwen3.5-0.8b-b1-parity` (codec=tq_mse_b4, seed=42)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 42
}
```

### `qwen3.5-0.8b-b1-parity` (codec=tq_mse_b4, seed=43)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 43
}
```

### `qwen3.5-0.8b-b1-parity` (codec=tq_mse_b4, seed=44)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `b1_parity_vs_single`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Cache-only B=1 parity row for the Qwen3.5 hybrid DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the cache-only spirit of qwen3-0.6b-b1-parity but on the hybrid scheduler path wired in P-3-C3c/d. Pytest side still holds the stronger B>1 batched-vs-single-request parity claim (tests/test_p3_hybrid_batched_parity.py); this bench row covers the B=1 slice so regressions on the hybrid scheduler surface in the unified harness.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 44
}
```

### `qwen3.5-27b-smoke` (codec=block_tq_b32_b3, seed=42)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 42
}
```

### `qwen3.5-27b-smoke` (codec=block_tq_b32_b3, seed=43)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 43
}
```

### `qwen3.5-27b-smoke` (codec=block_tq_b32_b3, seed=44)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 44
}
```

### `qwen3.5-27b-smoke` (codec=block_tq_b32_b4, seed=42)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 42
}
```

### `qwen3.5-27b-smoke` (codec=block_tq_b32_b4, seed=43)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 43
}
```

### `qwen3.5-27b-smoke` (codec=block_tq_b32_b4, seed=44)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 44
}
```

### `qwen3.5-27b-smoke` (codec=block_tq_b64_b3, seed=42)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 42
}
```

### `qwen3.5-27b-smoke` (codec=block_tq_b64_b3, seed=43)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 43
}
```

### `qwen3.5-27b-smoke` (codec=block_tq_b64_b3, seed=44)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 44
}
```

### `qwen3.5-27b-smoke` (codec=block_tq_b64_b4, seed=42)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 42
}
```

### `qwen3.5-27b-smoke` (codec=block_tq_b64_b4, seed=43)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 43
}
```

### `qwen3.5-27b-smoke` (codec=block_tq_b64_b4, seed=44)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 44
}
```

### `qwen3.5-27b-smoke` (codec=ext_rabitq_b2, seed=42)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 42
}
```

### `qwen3.5-27b-smoke` (codec=ext_rabitq_b2, seed=43)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 43
}
```

### `qwen3.5-27b-smoke` (codec=ext_rabitq_b2, seed=44)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 44
}
```

### `qwen3.5-27b-smoke` (codec=ext_rabitq_b3, seed=42)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 42
}
```

### `qwen3.5-27b-smoke` (codec=ext_rabitq_b3, seed=43)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 43
}
```

### `qwen3.5-27b-smoke` (codec=ext_rabitq_b3, seed=44)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 44
}
```

### `qwen3.5-27b-smoke` (codec=ext_rabitq_b4, seed=42)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 42
}
```

### `qwen3.5-27b-smoke` (codec=ext_rabitq_b4, seed=43)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 43
}
```

### `qwen3.5-27b-smoke` (codec=ext_rabitq_b4, seed=44)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 44
}
```

### `qwen3.5-27b-smoke` (codec=fp16, seed=42)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 42
}
```

### `qwen3.5-27b-smoke` (codec=fp16, seed=43)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 43
}
```

### `qwen3.5-27b-smoke` (codec=fp16, seed=44)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 44
}
```

### `qwen3.5-27b-smoke` (codec=rabitq_b1, seed=42)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3.5-27b-smoke` (codec=rabitq_b1, seed=43)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3.5-27b-smoke` (codec=rabitq_b1, seed=44)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3.5-27b-smoke` (codec=tq_mse_b3, seed=42)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 42
}
```

### `qwen3.5-27b-smoke` (codec=tq_mse_b3, seed=43)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 43
}
```

### `qwen3.5-27b-smoke` (codec=tq_mse_b3, seed=44)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 44
}
```

### `qwen3.5-27b-smoke` (codec=tq_mse_b4, seed=42)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 42
}
```

### `qwen3.5-27b-smoke` (codec=tq_mse_b4, seed=43)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 43
}
```

### `qwen3.5-27b-smoke` (codec=tq_mse_b4, seed=44)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

First end-to-end bench smoke for Qwen3.5-27B-4bit. Before P-4.2d-ii the 27B checkpoint only had the metadata probe scripts/probe_qwen3_5_27b_load.py — no pytest-side functional test exercised the forward path. This bench row fills the gap with a four-token greedy generation on prompt 'Hello'. Dual-gated: the 4-bit checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB, so opt-in via SILICA_REAL_QWEN3_5_27B=1 is mandatory.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 44
}
```

### `qwen3.5-moe-smoke` (codec=block_tq_b32_b3, seed=42)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 42
}
```

### `qwen3.5-moe-smoke` (codec=block_tq_b32_b3, seed=43)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 43
}
```

### `qwen3.5-moe-smoke` (codec=block_tq_b32_b3, seed=44)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "block_tq_b32_b3",
  "seed": 44
}
```

### `qwen3.5-moe-smoke` (codec=block_tq_b32_b4, seed=42)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 42
}
```

### `qwen3.5-moe-smoke` (codec=block_tq_b32_b4, seed=43)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 43
}
```

### `qwen3.5-moe-smoke` (codec=block_tq_b32_b4, seed=44)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b32_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "block_tq_b32_b4",
  "seed": 44
}
```

### `qwen3.5-moe-smoke` (codec=block_tq_b64_b3, seed=42)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 42
}
```

### `qwen3.5-moe-smoke` (codec=block_tq_b64_b3, seed=43)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 43
}
```

### `qwen3.5-moe-smoke` (codec=block_tq_b64_b3, seed=44)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "block_tq_b64_b3",
  "seed": 44
}
```

### `qwen3.5-moe-smoke` (codec=block_tq_b64_b4, seed=42)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 42
}
```

### `qwen3.5-moe-smoke` (codec=block_tq_b64_b4, seed=43)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 43
}
```

### `qwen3.5-moe-smoke` (codec=block_tq_b64_b4, seed=44)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='block_tq_b64_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "block_tq_b64_b4",
  "seed": 44
}
```

### `qwen3.5-moe-smoke` (codec=ext_rabitq_b2, seed=42)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 42
}
```

### `qwen3.5-moe-smoke` (codec=ext_rabitq_b2, seed=43)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 43
}
```

### `qwen3.5-moe-smoke` (codec=ext_rabitq_b2, seed=44)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b2' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "ext_rabitq_b2",
  "seed": 44
}
```

### `qwen3.5-moe-smoke` (codec=ext_rabitq_b3, seed=42)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 42
}
```

### `qwen3.5-moe-smoke` (codec=ext_rabitq_b3, seed=43)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 43
}
```

### `qwen3.5-moe-smoke` (codec=ext_rabitq_b3, seed=44)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "ext_rabitq_b3",
  "seed": 44
}
```

### `qwen3.5-moe-smoke` (codec=ext_rabitq_b4, seed=42)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 42
}
```

### `qwen3.5-moe-smoke` (codec=ext_rabitq_b4, seed=43)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 43
}
```

### `qwen3.5-moe-smoke` (codec=ext_rabitq_b4, seed=44)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='ext_rabitq_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "ext_rabitq_b4",
  "seed": 44
}
```

### `qwen3.5-moe-smoke` (codec=fp16, seed=42)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 42
}
```

### `qwen3.5-moe-smoke` (codec=fp16, seed=43)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 43
}
```

### `qwen3.5-moe-smoke` (codec=fp16, seed=44)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='fp16' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "fp16",
  "seed": 44
}
```

### `qwen3.5-moe-smoke` (codec=rabitq_b1, seed=42)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 42
}
```

### `qwen3.5-moe-smoke` (codec=rabitq_b1, seed=43)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 43
}
```

### `qwen3.5-moe-smoke` (codec=rabitq_b1, seed=44)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='rabitq_b1' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "rabitq_b1",
  "seed": 44
}
```

### `qwen3.5-moe-smoke` (codec=tq_mse_b3, seed=42)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 42
}
```

### `qwen3.5-moe-smoke` (codec=tq_mse_b3, seed=43)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 43
}
```

### `qwen3.5-moe-smoke` (codec=tq_mse_b3, seed=44)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b3' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "tq_mse_b3",
  "seed": 44
}
```

### `qwen3.5-moe-smoke` (codec=tq_mse_b4, seed=42)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 42
}
```

### `qwen3.5-moe-smoke` (codec=tq_mse_b4, seed=43)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 43
}
```

### `qwen3.5-moe-smoke` (codec=tq_mse_b4, seed=44)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `smoke`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=1`
- status: **failed**
- reason: `codec_override_invalid:Workload 'short-in-short-out': kv_codec='tq_mse_b4' requires prefix_cache=True; codecs install on the prefix cache's store, so they are meaningless when no prefix cache exists`

Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end claim ('does not crash on the real MoE weights') and adds the bench metrics + JSONL row. Dual-gated: the 4-bit checkpoint is ~20 GB on disk and the forward peaks at ~30+ GB device memory, so a cache-only gate would risk surprise activations on dev boxes with other 0.6B / 4B weights already cached. No token parity claim — mlx-lm silently drops attn_output_gate on this family (E-open-5), so a hypothetical HF-parity attempt would fail on attention before reaching MoE.

Metadata:

```
{
  "codec_id": "tq_mse_b4",
  "seed": 44
}
```

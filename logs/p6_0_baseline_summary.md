# P-6.0 Baseline Summary

Run on M5 Pro 48 GB, MLX-native silica @ 2730dee, 2026-04-27.

| scenario | B | bytes/step (GB) | ceiling tok/s | measured | util % | TTFT (ms) | peak (GB) | wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.6B (cache-only) | 1 | 0.3 | 1023.3 | **161.16** | 15.7% | 39 | 1.44 | 2.2 |
| 0.6B B=2 (cache-only) | 2 | 0.3 | 1023.3 | **208.72** | 20.4% | 54 | 1.72 | 3.0 |
| 0.8B hybrid (cache-only) | 1 | 0.4 | 767.5 | **123.28** | 16.1% | 43 | 1.77 | 3.1 |
| **27B dense (P-6 primary)** | 1 | 13.5 | 22.7 | **16.05** | 70.6% | 1920 | 15.36 | 29.6 |
| 31B dense (Gemma4) | 1 | 15.5 | 19.8 | **13.63** | 68.8% | 950 | 17.51 | 32.2 |
| MoE 35B-A3B B=1 | 1 | 1.5 | 204.7 | **76.01** | 37.1% | 2732 | 19.40 | 12.5 |
| **MoE 35B-A3B B=2 (P-6 stretch)** | 2 | 1.5 | 204.7 | **120.93** | 59.1% | 2056 | 19.68 | 12.5 |
| MoE 26B-A4B (Gemma4) | 1 | 2.0 | 153.5 | **68.62** | 44.7% | 454 | 14.38 | 8.5 |

## §6 acceptance gates — baseline assessment

| gate | required | baseline | gap |
| --- | --- | --- | --- |
| (1) dense Qwen3.5-27B-4bit ≥60 tok/s | 60 | 16.05 | **-43.95** (26.8% of gate) |
| (2) MoE Qwen3.5-35B-A3B-4bit ≥100 tok/s aggregate | 100 | 120.93 | **+20.93** (120.9% of gate) |

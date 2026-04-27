# P-6.0 Baseline Run Log

| Field | Value |
|---|---|
| Date | 2026-04-27T13:32:34-07:00 |
| Host | XinYus-MacBook-Pro.local |
| Platform | Darwin 25.4.0 arm64 |
| Branch | sonnet |
| HEAD | 2730dee |

Logs grouped by step; each scenario's run produces one `.log` (full stderr+stdout)
in this directory plus `plans/P6_0_BASELINE/<scenario>.jsonl` + `.md`.

## Run plan
- Step 1: cache-only sanity — qwen3-0.6b b1/b2, qwen3.5-0.8b b1
- Step 2: dense 27B baseline — qwen3.5-27b b1
- Step 3: MoE 35B-A3B — qwen3.5-moe b1, qwen3.5-moe b2 (primary stretch)
- Step 4: dense Gemma4-31B — gemma4-31b b1
- Step 5: Gemma4-MoE — gemma4-moe-26b-a4b b1

OOM-risk Qwen3.5-MoE B=4 explicitly NOT run (skipped per P3 acceptance plan).

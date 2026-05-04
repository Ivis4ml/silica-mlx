# P-6 Autoresearch Kernel Cycle 25 — B52 Reverify Miss

## Hypothesis

Before opening the 40 GB cliff probe, the cycle-14 running-best line
should reproduce at B=52 with the documented stack:

```bash
SILICA_USE_FA_DECODE_V10=1
SILICA_USE_BF16_DELTANET_STATE=1
```

Pass threshold: three B=52 reproductions should land near the recorded
`206.2 +/- 0.5 tok/s` strict-envelope line. If they do not, the next
work item is attribution, not another performance lever.

## Result

| Runtime path | Runs | Mean | Std | Peak GB | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `uv run python` | 185.5 / 187.1 / 183.3 | **185.3** | 1.9 | 35.0 avg | invalid for P-6 perf |
| conda `python` | 205.5 / 201.3 / 193.6 | **200.1** | 6.0 | 35.52 | reverify miss |

The conda path matched the old regime on the first pass (205.5), then
drifted down. The `uv` path is consistently about 10% slower despite
resolving the same pinned MLX versions.

## Attribution

The decisive local finding: the Qwen3.5 attention path currently produces
bf16 activations. The v10 shadow path required `queries.dtype == mx.float16`,
so the custom FA-decode kernel was not firing. Instrumenting a 2-token
27B decode reported:

```text
fa_v10_calls 0
```

The bf16 DeltaNet state was active, as shown by the B=52 peak memory
staying at the 35.5 GB bf16-state level. The missing piece was FA-decode
dtype eligibility.

## Decision

**diagnostic / reverify miss.** Do not run the 40 GB cliff probe until the
B=52 line is stable again. Cycle 26 opens the smallest kernel repair:
bf16-capable v8/v10 FA-decode kernels plus shadow eligibility.

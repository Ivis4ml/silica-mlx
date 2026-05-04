# P-6 Autoresearch Kernel Cycle 24 — MLX Pin Hardening

## Hypothesis

The final autoresearch report says the safe runtime is `mlx==0.31.1`,
`mlx-lm==0.31.2`, and `mlx-metal==0.31.1`, because the 0.31.2/0.31.3
stack introduced a deterministic greedy-token drift in
`tests/test_p2_preload_parity.py`. If `pyproject.toml` and `uv.lock`
still resolve to the newer stack, future `uv run` measurements can be
invalid even when the conda environment happens to be pinned.

Lever family: reproducibility / measurement validity.

## Pass / Fail Threshold

Pass if default `uv run` resolves to the known-good MLX stack and
`tests/test_p2_preload_parity.py` passes 3/3. Fail if `uv run` keeps
installing 0.31.2/0.31.3, or if the preload-parity gate fails.

## Change

- Pinned `mlx==0.31.1`, `mlx-lm==0.31.2`, and Darwin-only
  `mlx-metal==0.31.1` in `pyproject.toml`.
- Regenerated `uv.lock`.
- Repaired local `.venv` packaging metadata by removing a stale malformed
  `packaging-26.2.dist-info` directory left by prior `uv` syncs; the lock
  now uses `packaging==26.1`.

## Commands

```bash
uv lock
uv run python - <<'PY'
import importlib.metadata as md
for pkg in ('packaging','mlx','mlx-lm','mlx-metal'):
    print(pkg, md.version(pkg))
PY
uv run pytest tests/test_p2_preload_parity.py -q
python -m pytest tests/test_p2_preload_parity.py -q
```

## Result

Default `uv run` now reports:

```text
packaging 26.1
mlx 0.31.1
mlx-lm 0.31.2
mlx-metal 0.31.1
```

`uv run pytest tests/test_p2_preload_parity.py -q`: **3 passed**.
`python -m pytest tests/test_p2_preload_parity.py -q`: **3 passed**.

## Interpretation

This is not a throughput keep. It is a measurement-foundation keep: the
default project lock now matches the final-report pin, so the mandatory
determinism gate no longer depends on using an out-of-band conda Python.
Future dense-27B throughput measurements can be compared against the
cycle-14 running-best line without silently switching to the known-bad
MLX stack.

## Decision

**keep** for reproducibility. Continue performance work from the TODO
list only after this pin stays green.

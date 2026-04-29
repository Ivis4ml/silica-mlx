"""silica.bench.microbench — bench-adjacent measurement harnesses
that do not fit the ``Scenario`` schema.

P-6.0.5 sub-unit 7 lands the first member: ``target_verify``,
which **primes the prefix KV untimed** (via a fresh per-rep
``mlx_cache.make_prompt_cache`` plus an untimed prefix forward),
then **times only the candidate-slice forward** of length
``verify_k`` on dense Qwen3.5-27B-4bit. The marginal column
``verify_k_marginal_ms = forward_ms_p50[k] - forward_ms_p50[k=1]``
is computed by looking up k=1 (not by run order), so reordering
``--verify-ks`` does not corrupt the column. The output feeds
Decision Gate 1 (D-021 step 4) ROI estimation for Track
C.4 / C.5 block-diffusion drafters.

Microbench harnesses are not registered in
``BUILTIN_SCENARIOS``. They have their own CLI entry points and
their own JSONL / Markdown artefacts, separate from the
``scripts/bench.py`` runner. See ``plans/P6_0_5_OPENING.md`` §3.7.
"""

from __future__ import annotations

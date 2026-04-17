"""P-1 acceptance: greedy token-for-token parity with the mlx-lm reference.

PLAN.md §7 P-1 Acceptance: "Greedy decoding is token-for-token identical to the
mlx-lm reference implementation (fixed seed, same model)."

Loads Qwen3.5-0.8B once, then generates against a fixed prompt + fixed
``max_tokens`` under greedy (argmax) through two independent cache instances:

  A) ``mlx_lm.generate.generate_step(prompt, model, ...)`` with mlx-lm's own
     internal cache — the reference.
  B) ``silica.engine.Engine.generate(prompt_text, ...)`` wiring
     ``Qwen3_5Adapter`` + ``SimpleKVCache`` + P-0 ``Sampler`` — the candidate.

Both paths use the same ``model`` object (MLX forward is cache-scoped and
does not mutate the module) so any divergence isolates to Silica's
orchestration. Exit 0 on PASS, 1 on MISMATCH.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure the repo root is importable when run as `python scripts/...` from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402
from mlx_lm.generate import generate_step  # noqa: E402
from mlx_lm.utils import load  # noqa: E402

from silica.core.sampling import SamplingParams  # noqa: E402
from silica.engine import Engine  # noqa: E402
from silica.kvcache.simple import SimpleKVCache  # noqa: E402
from silica.models.qwen3_5 import Qwen3_5Adapter  # noqa: E402

REPO = "Qwen/Qwen3.5-0.8B"
PROMPT = "The capital of France is"
MAX_TOKENS = 32


def _mlx_lm_reference(
    model: Any, tokenizer: Any, prompt: str, max_tokens: int
) -> list[int]:
    prompt_ids = list(tokenizer.encode(prompt))
    prompt_arr = mx.array(prompt_ids, dtype=mx.int32)
    out: list[int] = []
    for i, (tok, _) in enumerate(
        generate_step(prompt_arr, model, max_tokens=max_tokens)
    ):
        out.append(int(tok))
        if i >= max_tokens - 1:
            break
    return out


def _silica_candidate(
    model: Any, tokenizer: Any, prompt: str, max_tokens: int
) -> list[int]:
    kv = SimpleKVCache.from_model(model)
    adapter = Qwen3_5Adapter(model, tokenizer, kv_manager=kv)
    engine = Engine(adapter, kv)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    return list(engine.generate(prompt, params))


def main() -> int:
    print(f"P-1 acceptance — mlx-lm parity ({REPO})")
    print(f"  prompt: {PROMPT!r}")
    print(f"  max_tokens: {MAX_TOKENS}")
    # mlx-lm load() returns Union[2tuple, 3tuple] without @overload.
    model, tokenizer = load(REPO)  # type: ignore[misc]

    print("Running mlx-lm reference...")
    ref = _mlx_lm_reference(model, tokenizer, PROMPT, MAX_TOKENS)

    print("Running Silica candidate...")
    cand = _silica_candidate(model, tokenizer, PROMPT, MAX_TOKENS)

    print(f"  reference:  {ref}")
    print(f"  candidate:  {cand}")
    print(f"  decoded ref: {tokenizer.decode(ref)!r}")

    if ref == cand:
        print("RESULT: PASS — token-for-token match (P-1 acceptance satisfied).")
        return 0

    # Locate the first divergence for debug.
    for i, (a, b) in enumerate(zip(ref, cand)):
        if a != b:
            print(f"  first mismatch at index {i}: ref={a} candidate={b}")
            break
    else:
        print(f"  length mismatch: ref={len(ref)} candidate={len(cand)}")
    print("RESULT: FAIL — Silica diverges from mlx-lm reference.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Gate-0 probe — mlx-lm BatchKVCache batched injection smoke test.

P-2 analogue of Gate A (D-010 external cache injection). Required by the
P-2 opening proposal (``docs/P2_OPENING.md``) before any unit work begins:
the entire P-2 scheduler, paged cache, and prefix bookkeeping are built on
top of ``mlx_lm.models.cache.BatchKVCache`` as the physical storage layer.
If its batched semantics drift, every P-2 deliverable drifts with it.

Three layers of verification:

  (1) **Direct invariant test** on ``BatchKVCache`` in isolation — asserts
      ``_idx`` (shared scalar insert cursor) and ``offset`` (per-row
      effective sequence length) stay in lockstep under prefill + decode.
      This is the targeted assertion for the 2026-04 upstream offset-bug
      class flagged by external review; running green on the installed
      mlx-lm version is the per-version regression guard.

  (2) **Model-forward integration test** mirroring the Gate A pattern — a
      toy 1-layer duck-typed model receives a BatchKVCache through
      ``model(tokens, cache=[bkv])`` and the cache is actually written.

  (3) **Custom-kernel negative check** — Silica's runtime does not call
      ``mlx.core.fast.metal_kernel`` anywhere; the 2026-03 lazy-graph
      use-after-free bug class (also flagged by external review) is
      therefore not on our path. Asserted by grepping the silica package.

Run: ``python scripts/probe_batch_kvcache.py``
Exit 0 on PASS, 1 on FAIL (with the first failing assertion surfaced).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure repo root is importable when run as `python scripts/...` from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402
import mlx_lm  # noqa: E402
from mlx_lm.models.cache import BatchKVCache  # noqa: E402


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def _zeros_kv(
    B: int, n_kv_heads: int, T: int, head_dim: int
) -> tuple[mx.array, mx.array]:
    shape = (B, n_kv_heads, T, head_dim)
    return mx.zeros(shape, dtype=mx.float16), mx.zeros(shape, dtype=mx.float16)


# --- (1) Direct invariant test ---


def _check_invariants_after_prefill_and_decode() -> tuple[bool, str]:
    """Prefill (T=4 over left-padded B=3) → decode (T=1) → verify state."""
    left_padding = [1, 3, 0]
    B, n_kv_heads, head_dim = len(left_padding), 1, 8
    cache = BatchKVCache(left_padding=left_padding)

    # Initial invariants.
    if int(cache._idx) != 0:
        return False, f"pre-prefill _idx expected 0, got {int(cache._idx)}"
    initial_offset = cache.offset.tolist()
    if initial_offset != [-1, -3, 0]:
        return False, f"pre-prefill offset expected [-1,-3,0], got {initial_offset}"

    # Prefill step: T=4 tokens across the batch.
    k, v = _zeros_kv(B, n_kv_heads, T=4, head_dim=head_dim)
    k_out, v_out = cache.update_and_fetch(k, v)
    if tuple(k_out.shape) != (B, n_kv_heads, 4, head_dim):
        return False, f"post-prefill k_out shape unexpected: {tuple(k_out.shape)}"
    if tuple(v_out.shape) != (B, n_kv_heads, 4, head_dim):
        return False, f"post-prefill v_out shape unexpected: {tuple(v_out.shape)}"
    if cache._idx != 4:
        return False, f"post-prefill _idx expected 4, got {cache._idx}"
    # offset[i] == _idx - left_padding[i]
    offset_after_prefill = cache.offset.tolist()
    expected_after_prefill = [4 - lp for lp in left_padding]
    if offset_after_prefill != expected_after_prefill:
        return (
            False,
            f"offset drift after prefill: got {offset_after_prefill}, "
            f"expected {expected_after_prefill}",
        )

    # Decode step: T=1 token across the batch.
    k1, v1 = _zeros_kv(B, n_kv_heads, T=1, head_dim=head_dim)
    k_out, v_out = cache.update_and_fetch(k1, v1)
    if tuple(k_out.shape) != (B, n_kv_heads, 5, head_dim):
        return False, f"post-decode k_out shape unexpected: {tuple(k_out.shape)}"
    if cache._idx != 5:
        return False, f"post-decode _idx expected 5, got {cache._idx}"
    offset_after_decode = cache.offset.tolist()
    expected_after_decode = [5 - lp for lp in left_padding]
    if offset_after_decode != expected_after_decode:
        return (
            False,
            f"offset drift after decode: got {offset_after_decode}, "
            f"expected {expected_after_decode}",
        )

    # Memory accounting surface (D-012).
    if int(cache.nbytes) <= 0:
        return False, f"nbytes expected > 0 after writes, got {int(cache.nbytes)}"
    if cache.empty():
        return False, "empty() expected False after writes"

    return True, ""


# --- (2) Model-forward integration test ---


class _TracedBatchKVCache(BatchKVCache):
    """BatchKVCache that counts update_and_fetch invocations."""

    def __init__(self, left_padding: list[int]) -> None:
        super().__init__(left_padding)
        self.calls = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        self.calls += 1
        return super().update_and_fetch(keys, values)  # type: ignore[no-any-return]


class _BatchedToyModel:
    """Minimal duck-typed model that exercises the batched cache contract."""

    VOCAB = 8

    def __init__(self, n_kv_heads: int = 1, head_dim: int = 8) -> None:
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

    def __call__(
        self, tokens: mx.array, cache: list[Any] | None = None
    ) -> mx.array:
        B, T = tokens.shape
        k = mx.zeros(
            (B, self.n_kv_heads, T, self.head_dim), dtype=mx.float16
        )
        v = mx.zeros(
            (B, self.n_kv_heads, T, self.head_dim), dtype=mx.float16
        )
        if cache is not None and cache[0] is not None:
            cache[0].update_and_fetch(k, v)
        return mx.zeros((B, T, self.VOCAB), dtype=mx.float16)


def _check_model_forward_drives_batched_cache() -> tuple[bool, str]:
    left_padding = [0, 2, 1]
    B = len(left_padding)
    traced = _TracedBatchKVCache(left_padding=left_padding)
    model = _BatchedToyModel()

    tokens = mx.zeros((B, 3), dtype=mx.int32)
    logits = model(tokens, cache=[traced])

    if traced.calls != 1:
        return False, f"update_and_fetch expected 1 call, got {traced.calls}"
    if int(traced._idx) != 3:
        return False, f"_idx after forward expected 3, got {int(traced._idx)}"
    if tuple(logits.shape) != (B, 3, model.VOCAB):
        return False, f"logits shape unexpected: {tuple(logits.shape)}"
    return True, ""


# --- (3) Custom-kernel negative check ---


def _check_silica_uses_no_metal_kernel() -> tuple[bool, str]:
    """Walk the silica package for any reference to mlx.core.fast.metal_kernel."""
    silica_root = Path(__file__).resolve().parent.parent / "silica"
    for py in silica_root.rglob("*.py"):
        text = py.read_text()
        if "metal_kernel" in text or "mx.fast.metal_kernel" in text:
            return False, f"unexpected metal_kernel reference in {py.name}"
    return True, ""


def main() -> int:
    print(f"Gate-0 probe — mlx-lm {mlx_lm.__version__}")
    print(
        "Claim: BatchKVCache batched invariants hold; model forward drives it; "
        "Silica uses no custom Metal kernels."
    )

    print("(1) Direct invariant test (prefill + decode across B=3):")
    ok, msg = _check_invariants_after_prefill_and_decode()
    if not ok:
        return _fail(f"(1) {msg}")
    print("    PASS (offset and _idx stay in lockstep; 2026-04 bug class clear)")

    print("(2) Model-forward integration (toy B=3 model, traced cache):")
    ok, msg = _check_model_forward_drives_batched_cache()
    if not ok:
        return _fail(f"(2) {msg}")
    print("    PASS (batched model(tokens, cache=[bkv]) path green)")

    print("(3) Silica does not use mlx.core.fast.metal_kernel:")
    ok, msg = _check_silica_uses_no_metal_kernel()
    if not ok:
        return _fail(f"(3) {msg}")
    print("    PASS (2026-03 custom-kernel lazy-graph bug class not on our path)")

    print()
    print(
        "RESULT: PASS — P-2 Gate-0 cleared. Unit work may begin with "
        "task #12 (RequestState)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

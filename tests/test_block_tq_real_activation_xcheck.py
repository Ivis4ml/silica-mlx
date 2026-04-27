"""(a-real) — real Qwen3.5-0.8B K / V activation BlockTQ cross-check.

Closes the real-activation half of P-5 §7(a) via a standalone test
gated on the HF cache. Design doc: plans/P5_A_REAL_OPENING.md.

Full sweep form (step 3):

- Loads Qwen/Qwen3.5-0.8B once via ``adapter_for_repo``.
- Captures pre-RoPE K and V projection outputs on every GLOBAL
  layer (``layer.is_linear == False``) during a prefill pass on a
  checked-in deterministic 128-ish-token prompt.
- Loops over ``(vq_block_size, num_bits) ∈ {32, 64} × {3, 4}``,
  ``seed ∈ {42, 43, 44}``, every GLOBAL layer, and ``side ∈ {"K", "V"}``.
- Produces
  ``plans/P5_ACCEPTANCE_SWEEP/real_activation_xcheck.jsonl`` with
  one row per ``(layer_idx, side, vq_block_size, num_bits, seed)``.
- Asserts per-row gates (§2.5):
  - ``|silica_frob - numpy_frob|`` tolerance: ``< 1e-3`` for the
    production-recommended ``(B=64, num_bits=4)`` cell, ``< 5e-3``
    otherwise. Reuses the synthetic (a-algo) envelope.
  - ``silica_frob <= 2 * baseline_frob`` when the ``IdentityCodec``
    baseline is non-degenerate; degenerate tolerant when
    ``baseline_frob ~= 0`` per §2.4.

Expected outcomes per skeleton run (2026-04-24):

- 6 GLOBAL layers at indices ``[3, 7, 11, 15, 19, 23]`` on
  Qwen3.5-0.8B (24 hidden layers, ``full_attention_interval=4``).
- ``n_kv_heads=2``, ``head_dim=256``, dtype ``bfloat16``.
- Baseline Frobenius uniformly ``0`` (``IdentityCodec`` +
  ``RawFp16Payload`` is lossless); ratio gate degenerates across
  all rows and the silica-vs-numpy gap carries the close weight.
- Skeleton observation on layer 3, ``K``, ``(B=64, bits=4, seed=42)``:
  ``|silica - numpy| ~ 2.3e-5``, ~43× headroom against the
  ``< 1e-3`` production tolerance. V is expected similar; full
  sweep lands the raw data.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from silica.kvcache.codec import IdentityCodec
from silica.kvcache.manager import KVHandle
from silica.vq import BlockTurboQuantMSE
from silica.vq._calibration import haar_rotation, lloyd_max_codebook
from tests.test_block_tq_vqbench_xcheck import _numpy_block_tq_round_trip

_REPO = "Qwen/Qwen3.5-0.8B"


def _hf_cache_has_repo(repo: str) -> bool:
    """Mirror the HF cache sentinel used by ``tests/test_engine_admission_reorder.py``."""
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    cache_dir = Path(hf_home) / "hub" / f"models--{repo.replace('/', '--')}"
    return cache_dir.exists()


_SKIP = not _hf_cache_has_repo(_REPO)
_SKIP_REASON = (
    f"Qwen3.5-0.8B ({_REPO}) not present in the local HF cache. Run "
    f"qwen3.5-0.8b-b1-parity once to populate, or export HF_HOME to a "
    f"cache with the weights."
)


# §2.7 — hand-written deterministic English passage. ~128 tokens under
# the Qwen3.5 tokenizer; §2.7 pins the 96..160 token envelope as a
# tokenizer-drift guard in the fixture below.
_PROMPT = (
    "The study of prime numbers dates back to classical antiquity. "
    "Ancient Greek mathematicians recognized that the integers greater "
    "than one split into two categories: those with exactly two positive "
    "divisors, called primes, and those with more, called composites. "
    "The prime counting function describes how many primes are less than "
    "or equal to a given integer. Approximate formulas for this function "
    "were conjectured independently in the late eighteenth century. "
    "Analytic number theory eventually proved these conjectures using "
    "complex analysis and properties of the Riemann zeta function. "
    "Modern cryptography depends heavily on the difficulty of factoring "
    "the product of two large primes, a computational problem for which "
    "no efficient classical algorithm is known to exist today."
)


_EVIDENCE_DIR = (
    Path(__file__).resolve().parent.parent / "docs" / "P5_ACCEPTANCE_SWEEP"
)
_EVIDENCE_JSONL = _EVIDENCE_DIR / "real_activation_xcheck.jsonl"


_TOLERANCE = 5e-3
_PRODUCTION_TOLERANCE = 1e-3
_BASELINE_DEGENERATE_EPS = 1e-8


def _relative_frobenius_np(decoded: np.ndarray, original: np.ndarray) -> float:
    """``||decoded - original||_F / ||original||_F`` on float32 NumPy inputs."""
    d = decoded.astype(np.float32)
    o = original.astype(np.float32)
    num = float(np.linalg.norm(d - o))
    den = float(np.linalg.norm(o))
    return num / den if den > 0 else float("nan")


def _silica_round_trip(
    tensor: mx.array,
    *,
    n_tokens: int,
    n_kv_heads: int,
    head_dim: int,
    vq_block_size: int,
    num_bits: int,
    seed: int,
) -> np.ndarray:
    """Run silica MLX BlockTurboQuantMSE encode→decode and return fp32 NumPy."""
    codec = BlockTurboQuantMSE(
        block_size=n_tokens,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dtype=tensor.dtype,
        vq_block_size=vq_block_size,
        num_bits=num_bits,
        seed=seed,
    )
    decoded = codec.decode_tensor(codec.encode_tensor(tensor))
    mx.eval(decoded)
    return np.asarray(decoded.astype(mx.float32))


def _numpy_reference_round_trip(
    tensor_np_f64: np.ndarray,
    *,
    n_kv_heads: int,
    n_tokens: int,
    head_dim: int,
    vq_block_size: int,
    num_bits: int,
    seed: int,
) -> np.ndarray:
    """Run the (a-algo) NumPy BlockTQ reference on the same tensor and
    return a reshape-back-to-(1, n_kv_heads, n_tokens, head_dim) fp32 array."""
    reference_input = tensor_np_f64.reshape(n_kv_heads * n_tokens, head_dim)
    rotation = haar_rotation(head_dim, seed).copy()
    sigma = 1.0 / math.sqrt(vq_block_size)
    centroids_frozen, boundaries_frozen = lloyd_max_codebook(num_bits, sigma)
    decoded = _numpy_block_tq_round_trip(
        reference_input,
        rotation=rotation,
        centroids=centroids_frozen.copy(),
        boundaries=boundaries_frozen.copy(),
        vq_block_size=vq_block_size,
    )
    return decoded.reshape(1, n_kv_heads, n_tokens, head_dim).astype(np.float32)


def _identity_round_trip(
    tensor: mx.array,
    *,
    n_tokens: int,
    n_kv_heads: int,
    head_dim: int,
) -> np.ndarray:
    """Run ``IdentityCodec`` round trip to measure the dtype-roundtrip baseline."""
    codec = IdentityCodec(
        block_size=n_tokens,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dtype=tensor.dtype,
    )
    decoded = codec.decode_tensor(codec.encode_tensor(tensor))
    mx.eval(decoded)
    return np.asarray(decoded.astype(mx.float32))


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestARealFullSweep:
    """Full (a-real) sweep: every GLOBAL layer × K/V × (B, bits) × seed."""

    @pytest.fixture(scope="class")
    def captured(self):
        """Load Qwen3.5-0.8B, capture pre-RoPE K / V on every GLOBAL layer
        during a prefill pass on the checked-in prompt.

        Captures by temporarily replacing ``self_attn.k_proj`` /
        ``self_attn.v_proj`` on each non-linear layer with a wrapper
        that records the projection output and returns it unchanged.
        Restored in ``finally`` so the adapter is clean on exit.
        """
        from silica.models.factory import adapter_for_repo

        adapter, kv = adapter_for_repo(_REPO)
        # Adapter exposes the loaded mlx-lm model via _model (internal);
        # no public accessor on the ModelAdapter Protocol, matching the
        # pattern the P-3 adapter test suite already uses.
        model = adapter._model
        tokenizer = adapter.tokenizer()

        raw_tokens = tokenizer.encode(_PROMPT)
        n_tokens = len(raw_tokens)
        assert 96 <= n_tokens <= 160, (
            f"Checked-in prompt tokenized to {n_tokens} tokens; expected "
            f"96..160 for the ~128-token design target. Update _PROMPT or "
            f"widen the envelope if the tokenizer / model config changed."
        )
        tokens_mx = mx.array(raw_tokens, dtype=mx.int32)

        originals: dict[int, tuple[object, object]] = {}
        captured_kv: dict[int, dict[str, mx.array | None]] = {}
        global_indices: list[int] = []

        for idx, layer in enumerate(model.layers):
            if getattr(layer, "is_linear", False):
                continue
            global_indices.append(idx)
            attn = layer.self_attn
            originals[idx] = (attn.k_proj, attn.v_proj)
            captured_kv[idx] = {"k": None, "v": None}

            def _make_capture(store: dict, key: str, orig):
                def capture(x):
                    out = orig(x)
                    store[key] = out
                    return out

                return capture

            attn.k_proj = _make_capture(captured_kv[idx], "k", originals[idx][0])
            attn.v_proj = _make_capture(captured_kv[idx], "v", originals[idx][1])

        try:
            req_id = "a-real-full-sweep"
            kv.reserve_for_prefill(req_id, raw_tokens)
            _logits, _delta = adapter.prefill(tokens_mx, KVHandle(req_id=req_id))
            mx.eval(_logits)
        finally:
            for idx, (k_orig, v_orig) in originals.items():
                attn = model.layers[idx].self_attn
                attn.k_proj = k_orig
                attn.v_proj = v_orig

        first_idx = global_indices[0]
        first_k = captured_kv[first_idx]["k"]
        assert first_k is not None and first_k.ndim == 3
        B, T, combined = int(first_k.shape[0]), int(first_k.shape[1]), int(first_k.shape[2])
        assert B == 1 and T == n_tokens

        first_attn = model.layers[first_idx].self_attn
        n_kv_heads = int(getattr(first_attn, "num_key_value_heads", 0) or 0)
        assert n_kv_heads > 0
        assert combined % n_kv_heads == 0
        head_dim = combined // n_kv_heads

        # Reshape every captured K/V into (1, n_kv_heads, T, head_dim) upfront.
        reshaped: dict[int, dict[str, mx.array]] = {}
        for idx, kv_dict in captured_kv.items():
            reshaped[idx] = {}
            for side in ("k", "v"):
                t = kv_dict[side]
                assert t is not None, f"layer {idx} {side} capture missing"
                reshaped[idx][side] = t.reshape(
                    1, n_tokens, n_kv_heads, head_dim
                ).transpose(0, 2, 1, 3)
                mx.eval(reshaped[idx][side])

        return {
            "reshaped": reshaped,
            "global_indices": global_indices,
            "n_tokens": n_tokens,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "dtype": first_k.dtype,
        }

    def test_shape_report(self, captured):
        """Pure introspection cell — prints layer / shape summary that
        the full sweep rows carry in their metadata."""
        n = len(captured["global_indices"])
        print(
            f"\n[a-real] GLOBAL layers: {n}, indices={captured['global_indices']}, "
            f"n_tokens={captured['n_tokens']}, n_kv_heads={captured['n_kv_heads']}, "
            f"head_dim={captured['head_dim']}, dtype={captured['dtype']}"
        )
        assert n > 0

    def test_full_sweep(self, captured):
        """Full (a-real) sweep. Produces the JSONL evidence file and
        asserts per-row gates."""
        n_tokens = captured["n_tokens"]
        n_kv_heads = captured["n_kv_heads"]
        head_dim = captured["head_dim"]
        global_indices = captured["global_indices"]
        reshaped = captured["reshaped"]

        cells = [
            (32, 3),
            (32, 4),
            (64, 3),
            (64, 4),
        ]
        seeds = [42, 43, 44]
        sides = ("K", "V")

        # Skip any (B, bits) cells incompatible with head_dim (none
        # expected on 0.8B since head_dim=256 >= 64, but the check is
        # cheap and future-proofs against a smaller-head-dim target).
        active_cells = [
            (b, bits) for (b, bits) in cells if head_dim >= b
        ]
        skipped_cells = [cell for cell in cells if cell not in active_cells]

        rows: list[dict] = []
        production_gap_violations: list[tuple] = []
        regular_gap_violations: list[tuple] = []

        for vq_block_size, num_bits in active_cells:
            is_production_cell = (vq_block_size == 64 and num_bits == 4)
            tol = _PRODUCTION_TOLERANCE if is_production_cell else _TOLERANCE
            for seed in seeds:
                for layer_idx in global_indices:
                    for side in sides:
                        key = side.lower()
                        tensor = reshaped[layer_idx][key]
                        tensor_np_f64 = (
                            np.asarray(tensor.astype(mx.float32)).astype(np.float64)
                        )

                        silica_decoded = _silica_round_trip(
                            tensor,
                            n_tokens=n_tokens,
                            n_kv_heads=n_kv_heads,
                            head_dim=head_dim,
                            vq_block_size=vq_block_size,
                            num_bits=num_bits,
                            seed=seed,
                        )
                        numpy_decoded = _numpy_reference_round_trip(
                            tensor_np_f64,
                            n_kv_heads=n_kv_heads,
                            n_tokens=n_tokens,
                            head_dim=head_dim,
                            vq_block_size=vq_block_size,
                            num_bits=num_bits,
                            seed=seed,
                        )
                        identity_decoded = _identity_round_trip(
                            tensor,
                            n_tokens=n_tokens,
                            n_kv_heads=n_kv_heads,
                            head_dim=head_dim,
                        )

                        tensor_np_f32 = tensor_np_f64.astype(np.float32)
                        silica_frob = _relative_frobenius_np(
                            silica_decoded, tensor_np_f32
                        )
                        numpy_frob = _relative_frobenius_np(
                            numpy_decoded, tensor_np_f32
                        )
                        baseline_frob = _relative_frobenius_np(
                            identity_decoded, tensor_np_f32
                        )
                        gap_abs = abs(silica_frob - numpy_frob)
                        baseline_degenerate = baseline_frob < _BASELINE_DEGENERATE_EPS
                        silica_vs_numpy_pass = gap_abs < tol
                        if not silica_vs_numpy_pass:
                            bucket = (
                                production_gap_violations
                                if is_production_cell
                                else regular_gap_violations
                            )
                            bucket.append(
                                (layer_idx, side, vq_block_size, num_bits, seed,
                                 silica_frob, numpy_frob, gap_abs)
                            )
                        rows.append({
                            "layer_idx": layer_idx,
                            "side": side,
                            "vq_block_size": vq_block_size,
                            "num_bits": num_bits,
                            "seed": seed,
                            "silica_frob": silica_frob,
                            "numpy_frob": numpy_frob,
                            "baseline_frob": baseline_frob,
                            "silica_vs_numpy_abs_gap": gap_abs,
                            "gate_pass_silica_vs_numpy": silica_vs_numpy_pass,
                            "gate_tolerance": tol,
                            "is_production_cell": is_production_cell,
                            "baseline_degenerate": baseline_degenerate,
                        })

        # Write JSONL evidence — one row per (layer, side, B, bits, seed).
        _EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        metadata = {
            "_record": "metadata",
            "repo": _REPO,
            "n_global_layers": len(global_indices),
            "global_layer_indices": global_indices,
            "n_tokens": n_tokens,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "dtype": str(captured["dtype"]),
            "cells": [
                {"vq_block_size": b, "num_bits": bits} for (b, bits) in active_cells
            ],
            "skipped_cells": [
                {"vq_block_size": b, "num_bits": bits} for (b, bits) in skipped_cells
            ],
            "seeds": seeds,
            "sides": list(sides),
            "tolerance_production": _PRODUCTION_TOLERANCE,
            "tolerance_regular": _TOLERANCE,
            "baseline_degenerate_eps": _BASELINE_DEGENERATE_EPS,
        }
        with _EVIDENCE_JSONL.open("w") as fp:
            fp.write(json.dumps(metadata) + "\n")
            for r in rows:
                fp.write(json.dumps(r) + "\n")

        # Summary print for -s runs.
        n_rows = len(rows)
        n_baseline_degenerate = sum(1 for r in rows if r["baseline_degenerate"])
        worst_gap = max((r["silica_vs_numpy_abs_gap"] for r in rows), default=0.0)
        worst_prod_gap = max(
            (r["silica_vs_numpy_abs_gap"] for r in rows if r["is_production_cell"]),
            default=0.0,
        )
        worst_k_gap = max(
            (r["silica_vs_numpy_abs_gap"] for r in rows if r["side"] == "K"),
            default=0.0,
        )
        worst_v_gap = max(
            (r["silica_vs_numpy_abs_gap"] for r in rows if r["side"] == "V"),
            default=0.0,
        )
        print(
            f"\n[a-real sweep] rows={n_rows} baseline_degenerate={n_baseline_degenerate}/{n_rows} "
            f"worst_gap={worst_gap:.2e} worst_prod_gap={worst_prod_gap:.2e} "
            f"worst_K_gap={worst_k_gap:.2e} worst_V_gap={worst_v_gap:.2e}"
        )

        # Gate assertions.
        assert not production_gap_violations, (
            f"(a-real) production-recommended cell failures "
            f"(tolerance {_PRODUCTION_TOLERANCE}): {production_gap_violations}"
        )
        assert not regular_gap_violations, (
            f"(a-real) non-production cell failures "
            f"(tolerance {_TOLERANCE}): {regular_gap_violations}"
        )
        # Per §2.4 / §2.5: baseline degeneracy is expected; assert that
        # every row reports the degenerate flag so future IdentityCodec
        # changes that break dtype preservation surface loudly.
        assert n_baseline_degenerate == n_rows, (
            f"expected baseline degenerate on every row (IdentityCodec + "
            f"RawFp16Payload is lossless); got "
            f"{n_baseline_degenerate}/{n_rows} degenerate"
        )

"""P-3-C5.2 numeric-drift probe.

Question this probe answers: under any Qwen3.5 hybrid-DeltaNet
target plus the ``forward_batched`` runner, does the
**single-step decode** path produce a recurrent state byte-equal
to the **batched-prefill** path when both consume the same input
token sequence?

Concretely, with prefix length ``T`` and decode steps ``k``:

- **Path A (single-step, simulates the original no-preempt path)**:
  ``forward_batched(prefix, cache_a)`` consumes ``T`` tokens, then
  ``k - 1`` ``forward_batched`` calls each consuming one decoded
  token. Cache_a ends having consumed ``T + (k - 1)`` tokens.
- **Path B (batched re-prefill, simulates today's replay path)**:
  ``forward_batched(prefix + decoded_a[:k - 1], cache_b)`` — one
  batched call over the same ``T + (k - 1)`` tokens path A
  consumed.

Both paths process the identical input sequence; only the
batching boundary differs. Snapshot per-DeltaNet-layer
``(conv_state, recurrent_state)`` from each cache and compare.

The answer decides C5.2's acceptance shape:

- **State byte-exact**: today's replay-by-re-prefill produces the
  same recurrent state as the original single-step path; C5.2's
  snapshot then only locks call-order and edge cases (snapshot
  taken before filter, restore happens before extend).
- **State differs but greedy tokens match**: there is a numeric
  drift below the argmax threshold; C5.2 still cannot rest on
  "fixes observable token drift" but it ratifies bit-exact
  recurrent state across preempt/replay.
- **Greedy tokens diverge**: drift IS observable. C5.2 binds a
  bit-exact token oracle as a hard gate.

Output: stdout summary plus a JSON evidence blob to the path
given by ``--out``.

Skip-gated on the requested repo's HF cache (``--repo`` flag,
default Qwen3.5-0.8B); probe exits with status 2 when the
requested repo is absent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx

_REPO = "Qwen/Qwen3.5-0.8B"


def _hf_cache_has_repo(repo: str) -> bool:
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    cache_dir = Path(hf_home) / "hub" / f"models--{repo.replace('/', '--')}"
    return cache_dir.exists()


# Fixed prompt — content does not matter for the numeric question.
# Tokenizes to T=61 tokens under the Qwen3.5 tokenizer (recorded as
# ``T_prefix_tokens`` in every committed JSON); a different repo with
# a different tokenizer will produce a different T but the
# experiment shape is unchanged.
_PROMPT = (
    "The recurrent associative memory accumulates over every "
    "processed token, so the cache state grows without resetting "
    "until the request completes. Snapshot semantics under preempt "
    "should preserve byte-equal recurrent state across replay; the "
    "open question is whether today's re-prefill path already "
    "satisfies that property under bf16 single-step decode."
)


def _greedy_argmax(logits: mx.array) -> int:
    """Greedy decode: argmax over the vocab dim of a (1, V) tensor."""
    return int(mx.argmax(logits[0]))


def _decode_step_one(model: Any, last_tok: int, cache_list: list[Any]) -> int:
    from silica.mlx.runner import forward_batched

    tok_2d = mx.array([[last_tok]], dtype=mx.int32)
    logits = forward_batched(model, tok_2d, cache_list)
    mx.eval(logits)
    return _greedy_argmax(logits)


def _capture_layer_states(adapter: Any, cache_list: list[Any]) -> list[dict[str, Any]]:
    """Walk DeltaNet layers, return per-layer (conv, recurrent) raw arrays
    and metadata. Uses adapter.snapshot_recurrent_state then unpacks for
    detailed comparison."""
    snap = adapter.snapshot_recurrent_state(cache_list, row_idx=0)
    out: list[dict[str, Any]] = []
    for entry in snap.entries:
        out.append({
            "layer_idx": entry.layer_idx,
            "conv_state": entry.conv_state,
            "recurrent_state": entry.recurrent_state,
        })
    return out


def _array_diff_stats(a: mx.array | None, b: mx.array | None) -> dict[str, Any]:
    if a is None and b is None:
        # Both unallocated (lazy slots before any forward); treat as
        # bit-exact for the statistic so the both-None case does not
        # silently fall into the "drift" bucket via a missing key.
        return {"both_none": True, "bit_exact": True}
    if (a is None) != (b is None):
        return {"shape_mismatch": True, "a_none": a is None, "b_none": b is None}
    assert a is not None and b is not None
    if a.shape != b.shape:
        return {"shape_mismatch": True, "a_shape": list(a.shape), "b_shape": list(b.shape)}
    bit_exact = bool(mx.array_equal(a, b))
    a32 = a.astype(mx.float32)
    b32 = b.astype(mx.float32)
    diff = a32 - b32
    abs_diff = mx.abs(diff)
    max_abs = float(mx.max(abs_diff))
    mean_abs = float(mx.mean(abs_diff))
    a_abs = mx.abs(a32)
    a_norm = float(mx.sqrt(mx.sum(a32 * a32)))
    diff_norm = float(mx.sqrt(mx.sum(diff * diff)))
    return {
        "bit_exact": bit_exact,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rel_frobenius": (diff_norm / a_norm) if a_norm > 0 else float("nan"),
        "a_abs_max": float(mx.max(a_abs)),
        "shape": list(a.shape),
        "dtype": str(a.dtype),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=_REPO,
        help="HF model repo id (default: Qwen/Qwen3.5-0.8B)",
    )
    parser.add_argument(
        "--decode-k",
        type=int,
        default=8,
        help="number of single-step decode tokens before snapshot",
    )
    parser.add_argument(
        "--continue-m",
        type=int,
        default=8,
        help="continue both paths by M decode steps and compare token streams",
    )
    parser.add_argument(
        "--out",
        default="docs/P3_C5_DRIFT_EXPERIMENT/drift_qwen3_5_0_8b.json",
    )
    args = parser.parse_args()
    repo = args.repo

    if not _hf_cache_has_repo(repo):
        print(f"error: {repo} not in HF cache", file=sys.stderr)
        return 2

    from silica.mlx.runner import forward_batched
    from silica.models.factory import adapter_for_repo
    from silica.models.qwen3_5 import Qwen3_5Adapter

    adapter, _kv = adapter_for_repo(repo)
    assert isinstance(adapter, Qwen3_5Adapter)
    model = adapter._model
    tokenizer = adapter.tokenizer()
    prefix = tokenizer.encode(_PROMPT)
    T = len(prefix)
    k = args.decode_k
    M = args.continue_m

    # ===== Path A: prefill(T) -> decode_step × k =====
    cache_a = adapter.make_batch_cache([0])
    prefix_2d = mx.array([prefix], dtype=mx.int32)
    logits_a = forward_batched(model, prefix_2d, cache_a)
    mx.eval(logits_a)
    last_tok_a = _greedy_argmax(logits_a)
    decoded_a: list[int] = [last_tok_a]
    for _ in range(k - 1):
        next_tok = _decode_step_one(model, decoded_a[-1], cache_a)
        decoded_a.append(next_tok)
    state_a = _capture_layer_states(adapter, cache_a)

    # ===== Path B: batched re-prefill of the same input sequence =====
    # Path A's cache_a has consumed T + (k - 1) tokens — the prefix
    # (positions 0..T-1) plus decoded_a[0..k-2] as decode-step
    # inputs. decoded_a[k-1] was sampled at position T + (k - 1) - 1
    # but has NOT been fed back as input, so it is not part of the
    # consumed sequence. Path B prefills exactly that T + (k - 1)
    # sequence in one batched forward; the resulting state_b is what
    # we compare against state_a.
    composite = list(prefix) + decoded_a[: k - 1]
    cache_b = adapter.make_batch_cache([0])
    composite_2d = mx.array([composite], dtype=mx.int32)
    logits_b = forward_batched(model, composite_2d, cache_b)
    mx.eval(logits_b)
    state_b = _capture_layer_states(adapter, cache_b)

    # The next-token argmax from cache_b should match decoded_a[k-1]
    # (the last token path A sampled), if state evolved equivalently.
    next_tok_b = _greedy_argmax(logits_b)

    # ===== Compare per-layer states =====
    assert len(state_a) == len(state_b)
    layer_compare: list[dict[str, Any]] = []
    bit_exact_count = {"conv": 0, "recurrent": 0}
    drift_count = {"conv": 0, "recurrent": 0}
    worst_rel: dict[str, float] = {"conv": 0.0, "recurrent": 0.0}
    for ea, eb in zip(state_a, state_b):
        assert ea["layer_idx"] == eb["layer_idx"]
        conv_diff = _array_diff_stats(ea["conv_state"], eb["conv_state"])
        recur_diff = _array_diff_stats(ea["recurrent_state"], eb["recurrent_state"])
        layer_compare.append({
            "layer_idx": ea["layer_idx"],
            "conv_diff": conv_diff,
            "recurrent_diff": recur_diff,
        })
        if conv_diff.get("bit_exact"):
            bit_exact_count["conv"] += 1
        else:
            drift_count["conv"] += 1
            worst_rel["conv"] = max(
                worst_rel["conv"],
                conv_diff.get("rel_frobenius", 0.0) or 0.0,
            )
        if recur_diff.get("bit_exact"):
            bit_exact_count["recurrent"] += 1
        else:
            drift_count["recurrent"] += 1
            worst_rel["recurrent"] = max(
                worst_rel["recurrent"],
                recur_diff.get("rel_frobenius", 0.0) or 0.0,
            )

    # ===== Continue both paths by M more decode-steps and compare token streams =====
    # Path A: keep decoding from where it left off (its kth sample is decoded_a[-1]).
    continue_a: list[int] = []
    cur_a = decoded_a[-1]
    for _ in range(M):
        nxt = _decode_step_one(model, cur_a, cache_a)
        continue_a.append(nxt)
        cur_a = nxt
    # Path B: similar — but path B's "last sampled" is next_tok_b, which should
    # equal decoded_a[-1] modulo any drift across the prefill-vs-decode split.
    continue_b: list[int] = []
    cur_b = next_tok_b
    for _ in range(M):
        nxt = _decode_step_one(model, cur_b, cache_b)
        continue_b.append(nxt)
        cur_b = nxt

    tokens_match = (decoded_a[-1] == next_tok_b) and (continue_a == continue_b)
    first_divergence: int | None = None
    if decoded_a[-1] != next_tok_b:
        first_divergence = 0
    else:
        for i, (ta, tb) in enumerate(zip(continue_a, continue_b), start=1):
            if ta != tb:
                first_divergence = i
                break

    summary = {
        "repo": repo,
        "T_prefix_tokens": T,
        "k_decode_steps": k,
        "M_continue_steps": M,
        "n_layers_compared": len(layer_compare),
        "bit_exact_conv": bit_exact_count["conv"],
        "bit_exact_recurrent": bit_exact_count["recurrent"],
        "drift_conv": drift_count["conv"],
        "drift_recurrent": drift_count["recurrent"],
        "worst_rel_frobenius_conv": worst_rel["conv"],
        "worst_rel_frobenius_recurrent": worst_rel["recurrent"],
        "path_A_kth_token": int(decoded_a[-1]),
        "path_B_next_token_after_composite_prefill": int(next_tok_b),
        "tokens_match_through_M": tokens_match,
        "first_divergence_step": first_divergence,
        "continue_A": continue_a,
        "continue_B": continue_b,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "summary": summary,
        "per_layer": layer_compare,
    }, indent=2) + "\n")

    # stdout summary
    print(f"[C5.2 drift] {repo}: T={T} k={k} M={M}")
    print(
        f"[C5.2 drift] state bit-exact: conv {bit_exact_count['conv']}/{len(layer_compare)}, "
        f"recurrent {bit_exact_count['recurrent']}/{len(layer_compare)}"
    )
    if drift_count["conv"] or drift_count["recurrent"]:
        print(
            f"[C5.2 drift] worst rel-frobenius: conv {worst_rel['conv']:.3e}, "
            f"recurrent {worst_rel['recurrent']:.3e}"
        )
    print(
        f"[C5.2 drift] greedy tokens match through M={M}: {tokens_match} "
        f"(first divergence: {first_divergence})"
    )
    print(f"[C5.2 drift] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

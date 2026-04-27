"""P-3-C5.0 state-inventory probe.

Loads one Qwen3.5 repo via ``adapter_for_repo``, runs a short
prefill, then walks the live ``cache_list`` to capture the
runtime shape / dtype / nbytes of every cache slot. Emits one
JSON blob to stdout so the inventory README can consolidate
multiple repo runs.

Usage:

    uv run python plans/P3_C5_STATE_INVENTORY/probe.py \
        --repo Qwen/Qwen3.5-0.8B \
        --out plans/P3_C5_STATE_INVENTORY/inventory_qwen3_5_0_8b.json

Design contract: ``plans/P3_C5_OPENING.md`` §6.1 (C5.0 state inventory).
Runs HF-cache-gated — fails loudly if the repo is not cached rather
than downloading; the inventory is a bounded offline measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Short fixed prompts — enough tokens to populate cache slots;
# content does not matter for the shape / dtype measurement. Two
# different-length prompts drive the batched path so left_padding
# is non-trivial on row 0 (= shorter row).
_PROMPT = (
    "The recurrent associative memory accumulates over every "
    "processed token, so the cache state grows without resetting "
    "until the request completes."
)
_PROMPT_SHORT = (
    "State accumulates across every token of the sequence."
)


def _hf_cache_has_repo(repo: str) -> bool:
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    cache_dir = Path(hf_home) / "hub" / f"models--{repo.replace('/', '--')}"
    return cache_dir.exists()


def _describe_tensor(t: Any) -> dict[str, Any] | None:
    """Return (shape, dtype, nbytes) for anything that looks like an
    ``mx.array``. Returns ``None`` for ``None`` / placeholders."""
    if t is None:
        return None
    shape = tuple(int(s) for s in getattr(t, "shape", ()))
    dtype = str(getattr(t, "dtype", None))
    nbytes = int(getattr(t, "nbytes", 0))
    return {"shape": list(shape), "dtype": dtype, "nbytes": nbytes}


def _inventory_layer(layer: Any, layer_cache: Any) -> dict[str, Any]:
    """Inventory one layer's cache — DeltaNet (ArraysCache) or full-attention (KVCache)."""
    is_linear = bool(getattr(layer, "is_linear", False))
    entry: dict[str, Any] = {
        "is_linear": is_linear,
        "kind": "HYBRID_DELTANET" if is_linear else "GLOBAL",
        "cache_class": type(layer_cache).__name__,
        "cache_nbytes_total": int(getattr(layer_cache, "nbytes", 0)),
    }

    if is_linear:
        # ArraysCache(size=2) — slot 0 conv state, slot 1 recurrent state.
        slots = []
        inner = getattr(layer_cache, "cache", None)
        if inner is not None:
            for idx, slot in enumerate(list(inner)):
                slots.append({
                    "slot_idx": idx,
                    "tensor": _describe_tensor(slot),
                })
        entry["slots"] = slots
        lengths = getattr(layer_cache, "lengths", None)
        entry["lengths"] = (
            _describe_tensor(lengths) if lengths is not None else None
        )
    else:
        # KVCache / BatchKVCache — slots are keys / values.
        k = getattr(layer_cache, "keys", None)
        v = getattr(layer_cache, "values", None)
        entry["keys"] = _describe_tensor(k)
        entry["values"] = _describe_tensor(v)
        offset = getattr(layer_cache, "offset", None)
        # Single-request ``KVCache.offset`` is a Python int;
        # batched ``BatchKVCache.offset`` is a per-row ``mx.array``.
        if offset is None:
            entry["offset"] = None
        elif hasattr(offset, "shape"):
            entry["offset"] = {
                "shape": [int(s) for s in offset.shape],
                "dtype": str(offset.dtype),
                "values": [int(v) for v in offset.tolist()],
            }
        else:
            entry["offset"] = int(offset)
        left_padding = getattr(layer_cache, "left_padding", None)
        if left_padding is not None and hasattr(left_padding, "shape"):
            entry["left_padding"] = {
                "shape": [int(s) for s in left_padding.shape],
                "dtype": str(left_padding.dtype),
                "values": [int(v) for v in left_padding.tolist()],
            }

    return entry


def _summarise_layers(layers_inventory: list[dict[str, Any]]) -> dict[str, Any]:
    n_global = sum(1 for e in layers_inventory if e["kind"] == "GLOBAL")
    n_deltanet = sum(
        1 for e in layers_inventory if e["kind"] == "HYBRID_DELTANET"
    )
    total_recurrent_nbytes = 0
    total_conv_nbytes = 0
    total_kv_nbytes = 0
    for e in layers_inventory:
        if e["kind"] == "HYBRID_DELTANET":
            for slot in e.get("slots", []):
                t = slot["tensor"]
                if t is None:
                    continue
                if slot["slot_idx"] == 0:
                    total_conv_nbytes += t["nbytes"]
                elif slot["slot_idx"] == 1:
                    total_recurrent_nbytes += t["nbytes"]
        else:
            for key in ("keys", "values"):
                t = e.get(key)
                if t is not None:
                    total_kv_nbytes += t["nbytes"]
    return {
        "n_global_layers": n_global,
        "n_deltanet_layers": n_deltanet,
        "global_layer_indices": [
            e["layer_idx"] for e in layers_inventory if e["kind"] == "GLOBAL"
        ],
        "deltanet_layer_indices": [
            e["layer_idx"]
            for e in layers_inventory
            if e["kind"] == "HYBRID_DELTANET"
        ],
        "total_conv_state_nbytes": total_conv_nbytes,
        "total_recurrent_state_nbytes": total_recurrent_nbytes,
        "total_kv_nbytes": total_kv_nbytes,
    }


def _capture_single_request(
    adapter: Any, kv: Any, mx: Any, KVHandle: Any
) -> dict[str, Any]:
    """KVManager.cache_list(req_id) single-request path — what
    ``adapter.prefill`` writes into today."""
    tokenizer = adapter.tokenizer()
    raw_tokens = tokenizer.encode(_PROMPT)
    tokens_mx = mx.array(raw_tokens, dtype=mx.int32)
    req_id = "c5-inv-single"
    kv.reserve_for_prefill(req_id, raw_tokens)
    logits, delta = adapter.prefill(tokens_mx, KVHandle(req_id=req_id))
    mx.eval(logits)

    model = adapter._model
    cache_list = kv.cache_list(req_id)
    assert len(cache_list) == len(model.layers)

    layers_inventory: list[dict[str, Any]] = []
    for idx, (layer, layer_cache) in enumerate(zip(model.layers, cache_list)):
        entry = _inventory_layer(layer, layer_cache)
        entry["layer_idx"] = idx
        layers_inventory.append(entry)

    summary = _summarise_layers(layers_inventory)
    summary.update({
        "prompt_n_tokens": len(raw_tokens),
        "batch_size": 1,
        "recurrent_bytes_from_state_delta": int(delta.recurrent_bytes()),
        "layers": layers_inventory,
    })
    return summary


def _capture_batched(adapter: Any, mx: Any) -> dict[str, Any]:
    """ContinuousBatcher-style batched-cache path — what C5.1 /
    C5.2 / C5.3 actually operate on. Uses
    ``adapter.make_batch_cache(left_padding=[...])`` to mint the
    per-layer cache list (ArraysCache for DeltaNet, BatchKVCache
    for GLOBAL), then drives one ``forward_batched`` pass with
    ``B=2``, different-length rows."""
    from silica.mlx.runner import forward_batched  # noqa: E402

    tokenizer = adapter.tokenizer()
    raw_long = tokenizer.encode(_PROMPT)
    raw_short = tokenizer.encode(_PROMPT_SHORT)
    max_len = max(len(raw_long), len(raw_short))
    # Row 0 is short (bigger left_padding); row 1 is long.
    left_padding = [max_len - len(raw_short), max_len - len(raw_long)]
    pad_id = 0  # exact id does not matter for the shape probe.
    tokens_2d = mx.array(
        [
            [pad_id] * left_padding[0] + list(raw_short),
            [pad_id] * left_padding[1] + list(raw_long),
        ],
        dtype=mx.int32,
    )

    make_batch_cache = getattr(adapter, "make_batch_cache", None)
    assert callable(make_batch_cache), (
        "adapter exposes no make_batch_cache — C5's batched path "
        "would not run here. Every Qwen3.5 adapter provides it "
        "(silica/models/qwen3_5.py:109)."
    )
    cache_list: list[Any] = make_batch_cache(left_padding)
    model = adapter._model
    assert len(cache_list) == len(model.layers)

    # forward_batched consumes (B, T) tokens directly. This is the
    # code path ContinuousBatcher drives for every prefill / decode
    # step on live batched rows.
    logits_batched = forward_batched(model, tokens_2d, cache_list)
    mx.eval(logits_batched)

    layers_inventory: list[dict[str, Any]] = []
    for idx, (layer, layer_cache) in enumerate(zip(model.layers, cache_list)):
        entry = _inventory_layer(layer, layer_cache)
        entry["layer_idx"] = idx
        layers_inventory.append(entry)

    summary = _summarise_layers(layers_inventory)
    summary.update({
        "prompt_n_tokens_per_row": [len(raw_short), len(raw_long)],
        "padded_n_tokens": max_len,
        "left_padding": left_padding,
        "batch_size": 2,
        "layers": layers_inventory,
    })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="HF model repo id")
    parser.add_argument("--out", required=True, help="output JSON path")
    args = parser.parse_args()

    if not _hf_cache_has_repo(args.repo):
        print(
            f"error: {args.repo} not in HF cache; skipping.",
            file=sys.stderr,
        )
        return 2

    import mlx.core as mx  # noqa: E402

    from silica.kvcache.manager import KVHandle  # noqa: E402
    from silica.models.factory import adapter_for_repo  # noqa: E402

    adapter, kv = adapter_for_repo(args.repo)
    model = adapter._model

    single = _capture_single_request(adapter, kv, mx, KVHandle)
    batched = _capture_batched(adapter, mx)

    result = {
        "repo": args.repo,
        "n_hidden_layers": len(model.layers),
        "single_request": single,
        "batched": batched,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")

    s_rec = single["total_recurrent_state_nbytes"]
    s_conv = single["total_conv_state_nbytes"]
    s_kv = single["total_kv_nbytes"]
    b_rec = batched["total_recurrent_state_nbytes"]
    b_conv = batched["total_conv_state_nbytes"]
    b_kv = batched["total_kv_nbytes"]
    print(
        f"[C5.0] {args.repo}: "
        f"layers={len(model.layers)} "
        f"(GLOBAL={single['n_global_layers']}, "
        f"DELTANET={single['n_deltanet_layers']})"
    )
    print(
        f"[C5.0]   single (B=1, T={single['prompt_n_tokens']}): "
        f"recurrent={s_rec / 1e6:.2f} MB, conv={s_conv / 1e6:.2f} MB, "
        f"kv={s_kv / 1e6:.2f} MB"
    )
    print(
        f"[C5.0]   batched (B=2, T={batched['padded_n_tokens']}): "
        f"recurrent={b_rec / 1e6:.2f} MB, conv={b_conv / 1e6:.2f} MB, "
        f"kv={b_kv / 1e6:.2f} MB"
    )
    print(f"[C5.0] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

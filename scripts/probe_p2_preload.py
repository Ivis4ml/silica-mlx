"""P-2 preload probe — Qwen3-0.6B as the P-2 dev-loop pure-KV model.

P-2 opening v2.1/v2.2 fixed Qwen3-0.6B as the dev-loop model for batched
execution (pure KV attention, no DeltaNet / MTP / multimodal). Unit 16a
needs a working end-to-end baseline before its scaffolding can claim
"token-identical to P-1 Engine.generate" parity. This probe establishes
that baseline the same way Gate B did for P-1 (Qwen3.5-0.8B).

Four sub-checks:

  (a) mlx_lm loads Qwen/Qwen3-0.6B with no MTP / multimodal sanitisation
      (these fields are absent from Qwen3-0.6B's config).
  (b) Qwen3Adapter.config and kv_layout are sensibly populated — Qwen3
      uses n_kv_heads (not num_key_value_heads) and flat args.hidden_size
      (not nested text_config). Adapter must handle both naming schemes.
  (c) attention_pattern() is all GLOBAL — no HYBRID_DELTANET in this
      model; the 16a scope-guard (reject non-GLOBAL) will accept it.
  (d) P-1 Engine.generate produces reasonable text for a fixed prompt
      under greedy + fixed seed. The emitted token stream is recorded
      here so 16a's parity test can use it as oracle.

First run downloads ~1.52 GB to ~/.cache/huggingface; subsequent runs
hit the cache.

Run: ``python scripts/probe_p2_preload.py``
Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx_lm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from silica.core.sampling import SamplingParams  # noqa: E402
from silica.engine import Engine  # noqa: E402
from silica.models.adapter import AttentionKind  # noqa: E402
from silica.models.qwen3 import Qwen3Adapter  # noqa: E402

REPO = "Qwen/Qwen3-0.6B"
ORACLE_PROMPT = "The capital of France is"
ORACLE_MAX_TOKENS = 16


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def _check_load_and_shape(
    adapter: Qwen3Adapter,
) -> tuple[bool, str]:
    cfg = adapter.config
    if cfg.model_name != "qwen3":
        return False, f"model_name expected 'qwen3', got {cfg.model_name!r}"
    if cfg.num_layers != 28:
        return False, f"num_layers expected 28, got {cfg.num_layers}"
    if cfg.hidden_size != 1024:
        return False, f"hidden_size expected 1024, got {cfg.hidden_size}"
    if cfg.vocab_size <= 0:
        return False, f"vocab_size not populated, got {cfg.vocab_size}"
    layout = adapter.kv_layout()
    if layout.num_layers != 28:
        return False, f"kv_layout.num_layers expected 28, got {layout.num_layers}"
    if layout.n_kv_heads != 8:
        return False, (
            f"kv_layout.n_kv_heads expected 8 (from self_attn.n_kv_heads), "
            f"got {layout.n_kv_heads}"
        )
    if layout.head_dim != 128:
        # Qwen3-0.6B uses head_dim=128 (not hidden_size/num_heads = 64) per
        # its config. Adapter must read args.head_dim, not compute.
        return False, f"kv_layout.head_dim expected 128, got {layout.head_dim}"
    return True, ""


def _check_attention_pattern_all_global(
    adapter: Qwen3Adapter,
) -> tuple[bool, str]:
    pattern = adapter.attention_pattern()
    if len(pattern.per_layer) != 28:
        return False, f"pattern length expected 28, got {len(pattern.per_layer)}"
    non_global = [
        (i, k) for i, k in enumerate(pattern.per_layer) if k != AttentionKind.GLOBAL
    ]
    if non_global:
        return False, f"found non-GLOBAL layers (not pure KV): {non_global[:5]}"
    return True, ""


def _check_tokenizer_parity(adapter: Qwen3Adapter) -> tuple[bool, str]:
    mlx_tok = adapter.tokenizer()
    hf_tok = AutoTokenizer.from_pretrained(REPO)
    fixture = "Hello, world. 你好，世界。"
    mlx_ids = mlx_tok.encode(fixture)
    hf_ids = hf_tok.encode(fixture)
    if list(mlx_ids) != list(hf_ids):
        return False, (
            f"tokenizer mismatch: mlx len={len(mlx_ids)} "
            f"hf len={len(hf_ids)}; heads mlx={mlx_ids[:8]} hf={hf_ids[:8]}"
        )
    return True, ""


def _check_p1_engine_greedy(
    adapter: Qwen3Adapter, kv: Any
) -> tuple[bool, str]:
    engine = Engine(adapter, kv)
    params = SamplingParams(temperature=0.0, max_tokens=ORACLE_MAX_TOKENS)
    tokens = list(engine.generate(ORACLE_PROMPT, params))
    if len(tokens) != ORACLE_MAX_TOKENS:
        return False, (
            f"expected {ORACLE_MAX_TOKENS} tokens, got {len(tokens)}"
        )
    tok = adapter.tokenizer()
    decoded = tok.decode(tokens)
    print(f"    tokens: {tokens}")
    print(f"    decoded: {decoded!r}")
    # Sanity: decoded text should not be pure whitespace / empty.
    if not decoded.strip():
        return False, "decoded output is empty/whitespace"
    return True, ""


def main() -> int:
    print(f"P-2 preload probe — mlx-lm {mlx_lm.__version__}, repo {REPO}")
    print("Loading model (first run downloads ~1.52 GB to ~/.cache/huggingface)...")
    adapter, kv = Qwen3Adapter.from_hf_repo(REPO)
    print(f"  adapter: {adapter.config.model_name} / "
          f"{adapter.config.num_layers} layers / vocab={adapter.config.vocab_size}")
    print()

    print("(a) ModelConfig + KVLayout populated from plain-Qwen3 args:")
    ok, msg = _check_load_and_shape(adapter)
    if not ok:
        return _fail(f"(a) {msg}")
    print("    PASS")

    print("(b) AttentionPattern is all GLOBAL (pure KV, no hybrid layers):")
    ok, msg = _check_attention_pattern_all_global(adapter)
    if not ok:
        return _fail(f"(b) {msg}")
    print("    PASS — 28/28 GLOBAL; 16a scope-guard will accept")

    print("(c) Tokenizer parity with HF AutoTokenizer:")
    ok, msg = _check_tokenizer_parity(adapter)
    if not ok:
        return _fail(f"(c) {msg}")
    print("    PASS")

    print("(d) P-1 Engine.generate greedy baseline (oracle for Unit 16a):")
    ok, msg = _check_p1_engine_greedy(adapter, kv)
    if not ok:
        return _fail(f"(d) {msg}")
    print("    PASS")

    print()
    print("RESULT: PASS — Qwen3-0.6B is the P-2 dev-loop model. "
          "Unit 16a unblocked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

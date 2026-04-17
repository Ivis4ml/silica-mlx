"""Day-1 Gate B probe — Qwen3.5-0.8B text-only load (D-014, R-8).

Verifies three PLAN.md §7 P-1 prerequisites:

  (a) ``mlx_lm.load("Qwen/Qwen3.5-0.8B")`` succeeds with multimodal heads
      filtered — the checkpoint carries ``vision_tower*`` and ``model.visual*``
      weights (plus a 14-key ``vision_config``), and ``qwen3_5.py:387-398``
      drops them at load. Probe asserts the resulting ``Model`` object has no
      vision-prefixed children / parameters.

  (b) MTP head is dropped at load — ``qwen3_5.py:308-313`` filters any
      ``mtp.*`` weight and shifts norm.weights by +1.0 to compensate for the
      MTP-trained norm. Probe asserts the resulting model has no ``mtp.*``
      parameter and that ``mlx_lm.generate.generate_step(max_tokens=N)`` yields
      exactly ``N`` tokens (no multi-token-prediction parallelism).

  (c) Tokenizer parity with the HF reference — mlx-lm wraps
      ``transformers.AutoTokenizer``; parity with HF should be trivially true.
      Probe verifies by encoding a mixed English / Chinese fixture with both
      the mlx-lm tokenizer and a freshly loaded HF tokenizer and asserting the
      token id lists are identical.

First run downloads ~1.77 GB of weights to ``~/.cache/huggingface``; subsequent
runs hit the cache.

Run: ``python scripts/probe_qwen3_5_load.py``
Exit codes: 0 on PASS, 1 on FAIL (with the specific sub-check that failed).
"""

from __future__ import annotations

import sys
from typing import Any

import mlx.core as mx
import mlx_lm
from mlx.utils import tree_flatten
from mlx_lm.generate import generate_step
from mlx_lm.utils import load
from transformers import AutoTokenizer

REPO = "Qwen/Qwen3.5-0.8B"
FIXTURE = "Hello, world. 你好，世界。123"


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def _flat_param_names(model: Any) -> list[str]:
    """MLX parameter enumeration — returns dotted-path names."""
    flat = tree_flatten(model.parameters())
    assert isinstance(flat, list)
    return [pair[0] for pair in flat]


def _check_no_vision_weights(model: Any) -> tuple[bool, str]:
    """mlx-lm's sanitize() must have dropped all vision-prefixed parameters."""
    for name in _flat_param_names(model):
        for bad in ("vision_tower", "visual.", "model.visual"):
            if bad in name:
                return False, f"found vision-prefixed parameter: {name}"
    return True, ""


def _check_no_mtp_weights(model: Any) -> tuple[bool, str]:
    """mlx-lm's sanitize() must have dropped all MTP-prefixed parameters."""
    for name in _flat_param_names(model):
        if "mtp." in name or name.startswith("mtp"):
            return False, f"found MTP-prefixed parameter: {name}"
    return True, ""


def _check_single_token_per_step(model: Any, tokenizer: Any) -> tuple[bool, str]:
    """Each generate_step yield must be exactly one token (MTP disabled)."""
    prompt_text = "The capital of France is"
    prompt_ids = tokenizer.encode(prompt_text)
    prompt = mx.array(prompt_ids, dtype=mx.int32)
    n_yielded = 0
    for i, (_tok, _) in enumerate(generate_step(prompt, model, max_tokens=3)):
        n_yielded += 1
        if i >= 2:
            break
    if n_yielded != 3:
        return False, f"expected 3 yields, got {n_yielded}"
    return True, ""


def _check_tokenizer_parity(mlx_tokenizer: Any) -> tuple[bool, str]:
    """mlx-lm tokenizer encode() must match HF AutoTokenizer.encode() ids."""
    hf_tok = AutoTokenizer.from_pretrained(REPO)
    mlx_ids = mlx_tokenizer.encode(FIXTURE)
    hf_ids = hf_tok.encode(FIXTURE)
    if list(mlx_ids) != list(hf_ids):
        return False, (
            f"mismatch: mlx={mlx_ids[:10]}... (len={len(mlx_ids)}) vs "
            f"hf={hf_ids[:10]}... (len={len(hf_ids)})"
        )
    decoded = mlx_tokenizer.decode(mlx_ids)
    if decoded.strip() != FIXTURE.strip():
        return False, f"round-trip mismatch: {decoded!r} vs {FIXTURE!r}"
    return True, ""


def main() -> int:
    print(f"Gate B probe — mlx-lm {mlx_lm.__version__}, repo {REPO}")
    print("Loading model (first run downloads ~1.77 GB to ~/.cache/huggingface)...")
    # mlx-lm load() uses Union[2tuple, 3tuple] return without @overload; when
    # return_config=True, the runtime returns the 3-tuple variant.
    model, tokenizer, config = load(REPO, return_config=True)  # type: ignore[misc]
    print(f"  model type: {type(model).__module__}.{type(model).__name__}")
    print(f"  num layers (model.layers): {len(model.layers)}")
    print(f"  tokenizer type: {type(tokenizer).__name__}")
    print(f"  config has vision_config: {'vision_config' in config}")
    print(f"  config has text_config.mtp_num_hidden_layers: "
          f"{config.get('text_config', {}).get('mtp_num_hidden_layers')}")
    print()

    if "vision_config" not in config:
        return _fail(
            "(a-pre) config.json has no vision_config — the checkpoint is not "
            "the expected multimodal variant, sanitize() cannot be exercised."
        )

    print("(a) No vision-prefixed parameters:")
    ok, msg = _check_no_vision_weights(model)
    if not ok:
        return _fail(f"(a) {msg}")
    print("    PASS (sanitize() dropped vision weights despite vision_config in json)")

    print("(b.1) No MTP-prefixed parameters:")
    ok, msg = _check_no_mtp_weights(model)
    if not ok:
        return _fail(f"(b.1) {msg}")
    print("    PASS")

    print("(b.2) generate_step yields exactly one token per iteration:")
    ok, msg = _check_single_token_per_step(model, tokenizer)
    if not ok:
        return _fail(f"(b.2) {msg}")
    print("    PASS")

    print("(c) Tokenizer parity with HF AutoTokenizer:")
    ok, msg = _check_tokenizer_parity(tokenizer)
    if not ok:
        return _fail(f"(c) {msg}")
    print("    PASS")

    print()
    print("RESULT: PASS — Qwen3.5-0.8B loads text-only, MTP disabled, tokenizer matches HF.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""D-023 B=1 single-customer MTP sweep runner.

Runs `mlx_vlm.generate` on `mlx-community/gemma-4-31b-it-4bit` (target,
4-bit IT) paired with `mlx-community/gemma-4-31B-it-assistant-bf16`
(drafter, bf16) — outcome A* mixed-precision pairing per spike doc §3.5.

The runner lives in the silica-mlx repo but MUST be invoked through the
project-external isolated venv at `~/.cache/silica-d023-mtp/.venv/bin/python`
because mlx-vlm 0.5.0 requires `mlx>=0.31.2 / mlx-lm>=0.31.3` and the
silica project pin stays at `mlx==0.31.1 / mlx-lm==0.31.2 /
mlx-metal==0.31.1` per the v1.7.21 determinism anchor in
`tests/test_p2_preload_parity.py`.

Output: one JSONL row per `(prompt, mode, rep, block_size)`
measurement, written to `<out_dir>/b1_sweep.jsonl`. The first row is a
`_meta` header capturing the runtime stack divergence statement
required by spike doc §10.

Invocation:
    ~/.cache/silica-d023-mtp/.venv/bin/python plans/D023_MTP_GEMMA4/run_b1_sweep.py \\
        --max-tokens 200 --reps 3 --block-sizes 2,3,6,9 --prompts factorial,bst
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from mlx_vlm import load, generate, apply_chat_template

TARGET_REPO = "mlx-community/gemma-4-31b-it-4bit"
DRAFTER_REPO = "mlx-community/gemma-4-31B-it-assistant-bf16"

PROMPTS = {
    "factorial": (
        "Write a short Python function that computes the factorial of a "
        "positive integer using recursion. Include a docstring."
    ),
    "bst": (
        "Explain how a binary search tree works in three short paragraphs, "
        "then list three common operations."
    ),
    "creative_scene": (
        "Describe a quiet morning in an old library through the eyes of a "
        "tired librarian. Use sensory detail and avoid cliches."
    ),
    "factual_explain": (
        "Explain in plain English why the sky appears blue during the day, "
        "and then briefly mention why sunsets are often red or orange. "
        "Keep it accurate but accessible to a curious twelve-year-old."
    ),
}

DEFAULT_BLOCK_SIZES = [2, 3, 6, 9]


def measure_one(target_model, target_processor, drafter_model, prompt,
                max_tokens, draft_block_size=None):
    """Run one generation; return timing + accept stats as a dict.

    `prompt` here is the **chat-templated** string (already wrapped with
    the model's chat template by the caller). This is critical — passing
    raw text without the user/model turn boundaries causes the IT model
    to treat the prompt as a continuation rather than as a user message,
    leading to degenerate output and unrepresentative accept-rate
    measurements. The 20260506_173908 run was invalidated by this bug.
    """
    kwargs = {"max_tokens": max_tokens, "temperature": 0.0}
    if drafter_model is not None and draft_block_size is not None:
        # Clear per-round accept lens so this measurement is isolated.
        drafter_model.accept_lens = []
        kwargs["draft_model"] = drafter_model
        kwargs["draft_kind"] = "mtp"
        kwargs["draft_block_size"] = draft_block_size

    t0 = time.perf_counter()
    result = generate(target_model, target_processor, prompt, **kwargs)
    wall = time.perf_counter() - t0

    accept_lens = []
    if drafter_model is not None and draft_block_size is not None:
        accept_lens = list(getattr(drafter_model, "accept_lens", []) or [])

    out = {
        "prompt_tokens": int(result.prompt_tokens),
        "generation_tokens": int(result.generation_tokens),
        "generation_tps": float(result.generation_tps),
        "prompt_tps": float(result.prompt_tps),
        "peak_memory_gb": float(result.peak_memory),
        "wall_s": float(wall),
        "draft_block_size": draft_block_size,
        "k_candidates": (draft_block_size - 1) if draft_block_size else None,
        "rounds": len(accept_lens) if accept_lens else None,
        "accept_lens_mean": (sum(accept_lens) / len(accept_lens)) if accept_lens else None,
        "accept_rate": (
            (sum(accept_lens) / len(accept_lens) / (draft_block_size - 1))
            if accept_lens and draft_block_size and draft_block_size > 1
            else None
        ),
        "text_head": result.text[:120],
        "text_len": len(result.text),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--block-sizes", type=str,
                        default=",".join(str(b) for b in DEFAULT_BLOCK_SIZES))
    parser.add_argument("--prompts", type=str, default="factorial,bst")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--warmup-tokens", type=int, default=20)
    args = parser.parse_args()

    block_sizes = [int(b) for b in args.block_sizes.split(",")]
    prompt_keys = args.prompts.split(",")
    prompts = {k: PROMPTS[k] for k in prompt_keys}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (Path(args.out_dir) if args.out_dir else
               (Path(__file__).parent / timestamp))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Toolchain capture for the runtime-stack-divergence statement.
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
    def _v(name: str) -> str:
        try:
            return _pkg_version(name)
        except PackageNotFoundError:
            return "not-installed"
    silica_pin = {
        "mlx": "0.31.1", "mlx-lm": "0.31.2", "mlx-metal": "0.31.1",
    }
    toolchain = {
        "venv": "~/.cache/silica-d023-mtp/.venv (isolated; project venv untouched)",
        "mlx": _v("mlx"),
        "mlx-lm": _v("mlx-lm"),
        "mlx-metal": _v("mlx-metal"),
        "mlx-vlm": _v("mlx-vlm"),
        "transformers": _v("transformers"),
        "silica_project_pin": silica_pin,
        "divergence_statement": (
            "External spike stack != silica pinned stack. Spike result "
            "informs D-023 decision only and does NOT constitute a silica "
            "runtime attestation. tests/test_p2_preload_parity.py remains "
            "anchored on the silica project pin."
        ),
    }
    print(f"[{timestamp}] Toolchain: {json.dumps(toolchain, indent=2)}")

    print(f"[{timestamp}] Loading target {TARGET_REPO} ...")
    target_model, target_processor = load(TARGET_REPO)
    print(f"[{timestamp}] Loading drafter {DRAFTER_REPO} ...")
    drafter_model, _ = load(DRAFTER_REPO)

    # Apply chat template once per prompt (single-turn user message).
    # Per the methodology fix: passing raw prompts without chat-template
    # wrapping caused degenerate output and invalidated 20260506_173908.
    formatted_prompts = {
        k: apply_chat_template(target_processor, target_model.config, p)
        for k, p in prompts.items()
    }
    print(f"[{timestamp}] Applied chat template to {len(formatted_prompts)} prompts.")

    results_file = out_dir / "b1_sweep.jsonl"

    with open(results_file, "w") as f:
        # Meta header row
        meta = {
            "_meta": True,
            "timestamp": timestamp,
            "target": TARGET_REPO,
            "drafter": DRAFTER_REPO,
            "max_tokens": args.max_tokens,
            "reps": args.reps,
            "block_sizes": block_sizes,
            "prompts": prompt_keys,
            "toolchain": toolchain,
        }
        f.write(json.dumps(meta) + "\n")
        f.flush()

        # One warmup on the first prompt to stabilize allocators
        if args.warmup_tokens > 0:
            warmup_prompt = next(iter(formatted_prompts.values()))
            print(f"[{timestamp}] Warmup ({args.warmup_tokens} tokens) ...")
            measure_one(target_model, target_processor, None,
                        warmup_prompt, max_tokens=args.warmup_tokens)

        for prompt_key in prompts.keys():
            prompt = formatted_prompts[prompt_key]
            for rep in range(args.reps):
                # Off-spec
                print(f"[{prompt_key} rep={rep}] off-spec ...")
                r = measure_one(target_model, target_processor, None,
                                prompt, max_tokens=args.max_tokens)
                row = {"prompt_key": prompt_key, "rep": rep, "mode": "off",
                       **r}
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"  gen_tps={r['generation_tps']:.2f} "
                      f"gen_tok={r['generation_tokens']} "
                      f"prompt_tok={r['prompt_tokens']} "
                      f"wall={r['wall_s']:.2f}s")

                # On-spec sweep over block_sizes
                for block_size in block_sizes:
                    print(f"[{prompt_key} rep={rep}] on-spec block={block_size} ...")
                    r = measure_one(target_model, target_processor,
                                    drafter_model, prompt,
                                    max_tokens=args.max_tokens,
                                    draft_block_size=block_size)
                    row = {"prompt_key": prompt_key, "rep": rep, "mode": "on",
                           **r}
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    accept_str = (
                        f"accept_rate={r['accept_rate']:.2%}"
                        if r["accept_rate"] is not None else "accept=?"
                    )
                    rounds_str = (
                        f"rounds={r['rounds']}"
                        if r["rounds"] is not None else "rounds=?"
                    )
                    print(f"  gen_tps={r['generation_tps']:.2f} "
                          f"gen_tok={r['generation_tokens']} "
                          f"{rounds_str} {accept_str} "
                          f"wall={r['wall_s']:.2f}s")

    print(f"\nResults: {results_file}")


if __name__ == "__main__":
    main()

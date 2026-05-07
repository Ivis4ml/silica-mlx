#!/usr/bin/env python3
"""D-023 decision-row parity audit.

The main sweep runner (`run_b1_sweep.py`) only stores `text[:120]` per
measurement, which is insufficient to substantiate a "byte-identical"
greedy-parity claim. This standalone audit re-runs each prompt's
decision row off-spec vs on-spec at temperature=0 with full text +
sha256 capture, and emits a per-(prompt, rep) parity verdict.

Run from the isolated venv:
    ~/.cache/silica-d023-mtp/.venv/bin/python plans/D023_MTP_GEMMA4/parity_audit.py
"""

import argparse
import hashlib
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

# Decision rows determined from the 2-session aggregated REPORT.md.
DECISION_BLOCK_SIZE = {
    "factorial": 3,
    "bst": 3,
    "creative_scene": 2,
    "factual_explain": 3,
}


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def run_one(target_model, target_processor, drafter_model, prompt,
            max_tokens, draft_block_size=None):
    kwargs = {"max_tokens": max_tokens, "temperature": 0.0}
    if drafter_model is not None and draft_block_size is not None:
        drafter_model.accept_lens = []
        kwargs["draft_model"] = drafter_model
        kwargs["draft_kind"] = "mtp"
        kwargs["draft_block_size"] = draft_block_size

    t0 = time.perf_counter()
    result = generate(target_model, target_processor, prompt, **kwargs)
    wall = time.perf_counter() - t0

    return {
        "text": result.text,
        "text_sha256": sha256(result.text),
        "text_len": len(result.text),
        "generation_tokens": int(result.generation_tokens),
        "generation_tps": float(result.generation_tps),
        "wall_s": float(wall),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (Path(args.out_dir) if args.out_dir else
               (Path(__file__).parent / f"parity_audit_{timestamp}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    from importlib.metadata import version as _pkg_version
    toolchain = {
        "venv": "~/.cache/silica-d023-mtp/.venv (isolated)",
        "mlx": _pkg_version("mlx"),
        "mlx-lm": _pkg_version("mlx-lm"),
        "mlx-metal": _pkg_version("mlx-metal"),
        "mlx-vlm": _pkg_version("mlx-vlm"),
    }
    print(f"[{timestamp}] Toolchain: {json.dumps(toolchain)}")

    print(f"[{timestamp}] Loading target {TARGET_REPO} ...")
    target_model, target_processor = load(TARGET_REPO)
    print(f"[{timestamp}] Loading drafter {DRAFTER_REPO} ...")
    drafter_model, _ = load(DRAFTER_REPO)

    formatted_prompts = {
        k: apply_chat_template(target_processor, target_model.config, p)
        for k, p in PROMPTS.items()
    }

    audit_path = out_dir / "parity_audit.jsonl"
    summary_rows = []

    with open(audit_path, "w") as f:
        f.write(json.dumps({
            "_meta": True,
            "timestamp": timestamp,
            "target": TARGET_REPO,
            "drafter": DRAFTER_REPO,
            "max_tokens": args.max_tokens,
            "reps": args.reps,
            "decision_blocks": DECISION_BLOCK_SIZE,
            "toolchain": toolchain,
        }) + "\n")

        for prompt_key, prompt in formatted_prompts.items():
            block = DECISION_BLOCK_SIZE[prompt_key]
            for rep in range(args.reps):
                print(f"[{prompt_key} rep={rep}] off-spec ...")
                off = run_one(target_model, target_processor, None,
                              prompt, max_tokens=args.max_tokens)
                f.write(json.dumps({
                    "prompt_key": prompt_key, "rep": rep, "mode": "off",
                    "draft_block_size": None, **off,
                }) + "\n")
                f.flush()
                print(f"  sha256={off['text_sha256'][:16]}... len={off['text_len']} "
                      f"tps={off['generation_tps']:.2f}")

                print(f"[{prompt_key} rep={rep}] on-spec block={block} ...")
                on = run_one(target_model, target_processor, drafter_model,
                             prompt, max_tokens=args.max_tokens,
                             draft_block_size=block)
                f.write(json.dumps({
                    "prompt_key": prompt_key, "rep": rep, "mode": "on",
                    "draft_block_size": block, **on,
                }) + "\n")
                f.flush()
                parity = (off["text_sha256"] == on["text_sha256"])
                print(f"  sha256={on['text_sha256'][:16]}... len={on['text_len']} "
                      f"tps={on['generation_tps']:.2f} parity={parity}")
                summary_rows.append({
                    "prompt_key": prompt_key,
                    "rep": rep,
                    "block": block,
                    "off_sha256": off["text_sha256"],
                    "on_sha256": on["text_sha256"],
                    "off_len": off["text_len"],
                    "on_len": on["text_len"],
                    "parity": parity,
                })

    # Render a small summary markdown
    summary_path = out_dir / "PARITY.md"
    with open(summary_path, "w") as f:
        f.write("# D-023 decision-row parity audit\n\n")
        f.write(f"Sessions aggregated by main sweep: 20260506_175802, 20260506_181244\n\n")
        f.write(f"This audit re-runs each prompt's decision row off-spec vs on-spec at "
                f"`temperature=0` with **full text + sha256** capture (n={args.reps} reps "
                f"per prompt). The sha256 verdict is the load-bearing parity claim.\n\n")
        f.write("| prompt | block | rep | off sha256 (head) | on sha256 (head) | off len | on len | parity |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for r in summary_rows:
            f.write(
                f"| `{r['prompt_key']}` | {r['block']} | {r['rep']} | "
                f"`{r['off_sha256'][:16]}...` | `{r['on_sha256'][:16]}...` | "
                f"{r['off_len']} | {r['on_len']} | "
                f"{'**PASS**' if r['parity'] else '**FAIL**'} |\n"
            )
        all_pass = all(r["parity"] for r in summary_rows)
        f.write(f"\n**Verdict: {'PASS' if all_pass else 'FAIL'}** "
                f"({sum(r['parity'] for r in summary_rows)}/{len(summary_rows)} "
                f"decision-row pairs match bytewise across reps).\n")

    print(f"\nAudit JSONL: {audit_path}")
    print(f"Audit summary: {summary_path}")


if __name__ == "__main__":
    main()

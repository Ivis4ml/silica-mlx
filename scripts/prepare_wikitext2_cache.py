"""scripts/prepare_wikitext2_cache.py — one-shot WikiText-2 extractor.

P-5-C.2 step 3b. Writes the pre-extracted text file the
``qwen3-0.6b-wikitext-ppl-*`` bench rows read at runtime. Run once
per developer machine:

    python scripts/prepare_wikitext2_cache.py

Default output path is ``~/.cache/silica/wikitext2-test.txt`` —
:data:`silica.bench.scenarios._WIKITEXT2_DEFAULT_PATH` matches this
exactly so no argument is needed for the common case. Override with
``--out /other/path.txt`` if a different location is required.

Design notes:

- The runtime bench harness must not pull in HuggingFace ``datasets``.
  Runtime code (``silica.bench.wikitext``) only reads a plain text
  file; this script is the separate one-shot that bridges
  ``datasets.load_dataset`` to that flat file. ``from datasets import
  load_dataset`` lives inside :func:`main` so an import-time ``datasets``
  absence still lets ``--help`` / argument parsing succeed with a
  clean ``SystemExit`` message pointing at the install command.
- Text format mirrors vqbench's ``validation/streaming_ppl.py``
  pipeline verbatim: ``"\n\n".join(row["text"] for row in ds)``.
  No ``.strip()``, no empty-line filtering — WikiText-2's own
  article-boundary formatting (blank rows between articles) is
  load-bearing for tokenizer consistency with vqbench. Silent
  post-processing here would make silica's PPL drift from the
  vqbench REPORT numbers even when the two share the same model +
  codec.
- Write is atomic via a tempfile in the same parent directory +
  ``os.replace``. The bench runner's ``_check_gates`` only checks
  ``is_file()``, so a truncated / half-written file would satisfy
  the gate and then explode inside ``load_wikitext_text``. Atomic
  replace removes the partial-write hazard: either the final file
  exists with the full contents, or the old one (or no file) stays
  in place.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_DEFAULT_OUT = (
    Path.home() / ".cache" / "silica" / "wikitext2-test.txt"
)
_DATASET_NAME = "wikitext"
_DATASET_CONFIG = "wikitext-2-raw-v1"
_DATASET_SPLIT = "test"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the WikiText-2 raw test split via the "
            "HuggingFace datasets library and write it to a flat "
            "UTF-8 text file for the silica bench PPL rows to read."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=(
            f"Output path. Default: {_DEFAULT_OUT}. Silica's "
            "bench scenarios default ``wikitext_path`` to this "
            "value, so the common case requires no argument."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Rewrite the file even if it already exists. Without "
            "this flag an existing file is left untouched (no-op)."
        ),
    )
    return parser.parse_args(argv)


def _import_load_dataset() -> Any:
    """Lazy import of ``datasets.load_dataset`` with a human-friendly
    error.

    Returns the ``load_dataset`` callable; raises ``SystemExit`` (not
    ``ImportError``) so the CLI exits cleanly when the optional
    dependency is absent, pointing the user at the install command
    instead of a bare traceback.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "datasets library not installed. Install it once with "
            "`uv pip install datasets` (or `pip install datasets`) "
            "and re-run this script. The silica bench runtime itself "
            "does not depend on datasets — this is a prepare-only "
            "dependency for extracting WikiText-2."
        ) from exc
    return load_dataset


def _extract_text(load_dataset: Any) -> str:
    """Load the WikiText-2 raw test split and join every row's
    ``text`` field with ``\\n\\n``.

    Mirrors vqbench's streaming_ppl preprocessing verbatim. Do NOT
    strip whitespace or filter empty rows — WikiText's own blank-line
    article boundaries are part of the tokenizer-consumed signal; any
    divergence here makes silica's PPL drift from vqbench's REPORT
    numbers for the same model + codec.
    """
    ds = load_dataset(
        _DATASET_NAME, _DATASET_CONFIG, split=_DATASET_SPLIT
    )
    return "\n\n".join(row["text"] for row in ds)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via a tempfile in the
    same parent directory followed by ``os.replace``.

    The same-parent constraint ensures ``os.replace`` is a rename
    within one filesystem (guaranteed atomic on POSIX and Windows).
    A partial write on interrupt leaves the tempfile (with a
    predictable prefix so developers can see and remove it) but
    never a truncated final file — matters because
    ``silica.bench.runner._check_gates`` only calls ``is_file()`` to
    decide whether the PPL row is runnable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=".wikitext2-test.", suffix=".txt.tmp", dir=path.parent
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except Exception:
        # Best-effort cleanup; ignore failures (the tempfile prefix
        # makes any leftover obvious).
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_path: Path = args.out

    if out_path.exists() and not args.overwrite:
        print(
            f"{out_path} already exists; skipping (pass --overwrite "
            f"to rewrite)."
        )
        return 0

    load_dataset = _import_load_dataset()
    text = _extract_text(load_dataset)
    _atomic_write_text(out_path, text)
    print(f"Wrote {len(text):,} chars to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

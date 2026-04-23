"""silica.bench.wikitext — WikiText loader for the PPL oracle.

P-5-C.2 step 2. Provides a deterministic, offline tokenization path
from a locally-cached WikiText text file to the ``(1, seq_len)``
``mx.array`` shape
:func:`silica.bench.ppl_oracle.teacher_forced_chunked_nll` and
:func:`silica.bench.ppl_oracle.teacher_forced_chunked_nll_with_codec`
consume.

Two entry points:

- :func:`load_wikitext_text` — read a pre-extracted WikiText text file
  off disk. The bench harness points this at either a vendored
  pre-extracted cache file or the HuggingFace datasets cache location;
  unit tests point it at ``tests/fixtures/wikitext2_tiny.txt``.
- :func:`tokenize_for_ppl` — tokenize text to ``(1, N)`` int32
  ``mx.array``, truncating to ``max_tokens`` and optionally enforcing
  a ``min_tokens`` floor.

Offline-only by design. The loader does not talk to the network, does
not call ``datasets.load_dataset``, does not prompt for downloads.
Keeping the I/O surface to "read a text file off disk" means unit tests
run without any dataset / tokenizer dependency beyond the tiny fixture
and the adapter-bound tokenizer the caller supplies. The
`HF-cache-path` logic (locate the cached WikiText arrow file, handle
the case where it is not cached) belongs to the C.2-step-3 bench
scenario where the decision of "which pre-extracted file to read" is
made.

vqbench's WikiText-2 PPL workload uses ``wikitext-2-raw-v1`` test
split, concatenated into one document, tokenized with the model's own
tokenizer, then streamed at ``chunk_size=256`` for the oracle. silica
follows this shape; the default ``max_tokens=512`` (configured at the
bench-row layer, not here) matches vqbench REPORT's headline
2-chunk run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx


def load_wikitext_text(path: Path | str) -> str:
    """Read a pre-extracted WikiText text file into a single string.

    Args:
        path: filesystem path to a plain-text UTF-8 file. Typically
            a vendored fixture (for tests) or a cached pre-extraction
            of HuggingFace's ``wikitext-2-raw-v1`` test split (for
            bench).

    Returns:
        The file's contents as a single string. No line-breaks are
        stripped or collapsed — the caller's tokenizer owns whatever
        normalization happens next.

    Raises:
        FileNotFoundError: with a message that includes the path and
            the expected content shape. The bench harness catches
            this to print a one-shot "how to populate the cache"
            hint; unit tests should never hit it (fixture is vendored
            in tree).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"WikiText text file not found: {p}. Expected a UTF-8 "
            f"plain-text file containing the pre-extracted WikiText-2 "
            f"test split (for the bench row) or a vendored fixture "
            f"under tests/fixtures/ (for unit tests)."
        )
    if not p.is_file():
        raise FileNotFoundError(
            f"WikiText text path is not a regular file: {p}"
        )
    return p.read_text(encoding="utf-8")


def tokenize_for_ppl(
    tokenizer: Any,
    text: str,
    *,
    max_tokens: int | None = None,
    min_tokens: int = 1,
) -> mx.array:
    """Tokenize ``text`` into the ``(1, seq_len)`` int32 shape the PPL
    oracle consumes.

    Duck-typed on the tokenizer: accepts anything exposing a callable
    ``encode(text: str) -> list[int] | Sequence[int]``. mlx-lm's
    ``TokenizerWrapper`` satisfies this directly, as does any HF-style
    tokenizer. The token-count truncation lives here (not in the
    tokenizer) so a long WikiText body can be tokenized once and reused
    across bench rows with different ``max_tokens``.

    Args:
        tokenizer: object with ``encode(text) -> Sequence[int]``.
        text: input string to tokenize.
        max_tokens: truncate to at most this many tokens from the
            start. ``None`` returns all tokens. Must be ``>= 0``
            when provided.
        min_tokens: raise ``ValueError`` if the tokenized length
            (post-truncation) is below this floor. Defaults to ``1``
            — a zero-token document is almost always a caller-side
            bug (empty file, wrong path, tokenizer misconfiguration).
            Set to ``0`` to accept empty input (the PPL oracle
            returns ``(0.0, 0)`` on ``seq_len == 0``).

    Returns:
        ``mx.array`` of shape ``(1, N)`` with ``dtype=mx.int32``. ``N``
        equals the tokenizer's output length, capped at ``max_tokens``
        if provided.

    Raises:
        ValueError: if ``max_tokens`` is negative, or the tokenized
            length is below ``min_tokens``.
        TypeError: if ``tokenizer.encode`` does not exist / is not
            callable.
    """
    if max_tokens is not None and max_tokens < 0:
        raise ValueError(
            f"max_tokens must be >= 0 when provided; got {max_tokens}"
        )
    if min_tokens < 0:
        raise ValueError(f"min_tokens must be >= 0; got {min_tokens}")

    encode = getattr(tokenizer, "encode", None)
    if encode is None or not callable(encode):
        raise TypeError(
            f"tokenizer must expose a callable `encode(text)` method; "
            f"got {type(tokenizer).__name__}"
        )

    raw_ids = encode(text)
    ids = list(raw_ids)
    if max_tokens is not None:
        ids = ids[:max_tokens]

    n = len(ids)
    if n < min_tokens:
        raise ValueError(
            f"tokenized length {n} is below min_tokens={min_tokens}. "
            f"Likely causes: empty text, wrong file path, or a "
            f"tokenizer that emits zero tokens for this input."
        )

    return mx.array([ids], dtype=mx.int32)


__all__ = ["load_wikitext_text", "tokenize_for_ppl"]

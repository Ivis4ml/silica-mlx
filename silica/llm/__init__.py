"""silica.llm — high-level Python facade (P-8 sub-unit (g)).

Surfaces :class:`silica.llm.LLM`, an mlx-lm-style ergonomic wrapper
over :class:`silica.engine.Engine` (and optional
:class:`silica.chat.session.ChatSession`) so notebook / script
callers can drop silica in with one import.

The facade is **lazy-loading** (model loaded on first call, not
construction) and **single-thread** (one :class:`LLM` instance is
not safe to drive concurrently — use the HTTP server in
:mod:`silica.server.openai_api` for concurrent serving). See
:mod:`silica.llm._facade` for the full contract.
"""

from silica.llm._facade import LLM

__all__ = ["LLM"]

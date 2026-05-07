"""silica.llm._facade — :class:`LLM` ergonomic wrapper (P-8 sub-unit (g)).

mlx-lm-style facade: ``LLM(model="Qwen/Qwen3.5-0.8B").generate("hi")``.
The class wraps :class:`silica.engine.Engine` (single-request path) and
optional :class:`silica.chat.session.ChatSession` (multi-turn) so that
notebook / script callers can drop silica in with one import.

Lazy-load semantics
-------------------

``LLM(...)`` does **not** load weights — a notebook that does
``llm = LLM("Qwen/Qwen3.5-0.8B")`` should not pay the multi-GB
checkpoint cost just to type-check downstream code. The first call
to :meth:`generate` / :meth:`chat` runs
:func:`silica.models.factory.adapter_for_repo` (which calls
``mlx_lm.load``); subsequent calls reuse the loaded model.
:meth:`unload` clears the loaded fields so the next call re-loads —
useful for swapping models in interactive sessions.

The facade is **not** thread-safe — concurrent generate calls
against one :class:`LLM` instance share the underlying
:class:`Engine` and would race on :class:`KVHandle` lifecycle. The
same single-active-decode invariant the HTTP server enforces (G-1
under ``runtime.engine_lock``) applies here, but the facade does
not own the lock — callers wanting concurrency must use
:mod:`silica.server.openai_api` instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from silica.core.logger import get_logger
from silica.core.sampling import SamplingParams
from silica.engine import Engine

if TYPE_CHECKING:
    from silica.kvcache.simple import SimpleKVCache
    from silica.models.adapter import ModelAdapter

log = get_logger(__name__)


class LLM:
    """Ergonomic wrapper around :class:`silica.engine.Engine`.

    Constructor is cheap: weights load on the first call to
    :meth:`generate` / :meth:`chat`, so import-time cost stays bounded
    for callers that conditionally instantiate (notebooks, CLI
    dispatchers, fixtures).

    Parameters
    ----------
    model:
        HuggingFace repo id (or local path) understood by
        :func:`silica.models.factory.adapter_for_repo`.
    sampling_params:
        Optional default :class:`SamplingParams`. Per-call ``params``
        argument overrides; absent both, the engine's
        :class:`SamplingParams()` defaults apply (greedy, max_tokens=128).

    Example
    -------

    >>> llm = LLM("Qwen/Qwen3.5-0.8B")
    >>> llm.generate("Once upon a time")  # doctest: +SKIP
    "..."
    >>> for delta in llm.generate("Once upon a time", stream=True):
    ...     print(delta, end="", flush=True)  # doctest: +SKIP
    """

    def __init__(
        self,
        model: str,
        *,
        sampling_params: SamplingParams | None = None,
    ) -> None:
        if not model:
            raise ValueError("LLM(model=...) must be a non-empty repo id")
        self._model_repo = model
        self._default_params = sampling_params
        # Lazy fields — populated on first :meth:`_ensure_loaded` call.
        # Typed as ``Any`` here because the public ``loaded`` property
        # is the one the user reads; the internal fields are guarded
        # by ``_ensure_loaded`` and never None during a generate call.
        self._adapter: ModelAdapter | None = None
        self._kv_manager: SimpleKVCache | None = None
        self._engine: Engine | None = None

    # --- introspection -------------------------------------------------

    @property
    def model_repo(self) -> str:
        """Repo id passed at construction."""
        return self._model_repo

    @property
    def loaded(self) -> bool:
        """Whether the underlying model has been loaded.

        ``False`` after construction and after :meth:`unload`; ``True``
        between the first :meth:`generate` / :meth:`chat` call and the
        next :meth:`unload`.
        """
        return self._engine is not None

    # --- lifecycle -----------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy load via the adapter factory.

        Idempotent; the second call short-circuits on ``self._engine``.
        Imports :mod:`silica.models.factory` lazily so a bare
        :class:`LLM` import does not pull mlx-lm load machinery into
        callers that never generate.
        """
        if self._engine is not None:
            return
        from silica.models.factory import adapter_for_repo

        log.info("llm.load model=%s", self._model_repo)
        adapter, kv = adapter_for_repo(self._model_repo)
        self._adapter = adapter
        self._kv_manager = kv
        self._engine = Engine(adapter=adapter, kv_manager=kv)

    def unload(self) -> None:
        """Drop the loaded model so the next call re-loads.

        Useful for notebook-style model swaps without re-importing.
        Idempotent; calling :meth:`unload` on an already-unloaded
        :class:`LLM` is a no-op. The MLX arrays held by the adapter /
        engine become GC-eligible once their references drop here.
        """
        if self._engine is None:
            return
        log.info("llm.unload model=%s", self._model_repo)
        self._engine = None
        self._adapter = None
        self._kv_manager = None

    # --- generation ----------------------------------------------------

    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        *,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Single-prompt generation.

        Parameters
        ----------
        prompt:
            Raw prompt text. The adapter's tokenizer encodes it; no
            chat template is applied (use :meth:`chat` for that).
        sampling_params:
            Per-call override; falls back to the constructor's
            ``sampling_params``, then to the engine default
            (:class:`SamplingParams()` — greedy, max_tokens=128).
        stream:
            ``False`` returns the full decoded reply text once
            generation finishes. ``True`` returns an iterator that
            yields incremental decoded deltas as tokens arrive (same
            shape as :meth:`ChatSession.chat`'s ``stream_to``
            callback, but produced as a generator).

        EOS / UTF-8 handling matches
        :meth:`silica.chat.session.ChatSession.chat`: the trailing
        EOS token is dropped from the decoded reply, and trailing
        replacement chars (``\\ufffd``) — partial multi-byte UTF-8
        sequences cut by EOS / max_tokens — are stripped so the
        returned text equals what a streaming consumer would see.
        """
        self._ensure_loaded()
        params = self._effective_params(sampling_params)
        if stream:
            return self._stream_generate(prompt, params)
        return self._collect_generate(prompt, params)

    def chat(
        self,
        messages: list[dict[str, str]],
        sampling_params: SamplingParams | None = None,
        *,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Multi-turn chat.

        Each call constructs a fresh :class:`ChatSession` seeded with
        ``messages``. The last entry must have ``role='user'``; the
        preceding entries become history. The returned reply is the
        assistant's text.

        ``messages`` shape mirrors the OpenAI chat-completions wire
        format: a list of ``{"role": str, "content": str}`` dicts. A
        leading system message is honoured; consecutive system
        messages are concatenated with ``\\n\\n``.

        Parameters
        ----------
        stream:
            ``False`` returns the full reply once the turn finishes.
            ``True`` yields incremental decoded deltas (same shape as
            the streaming :meth:`generate`).
        """
        self._ensure_loaded()
        params = self._effective_params(sampling_params)
        if not messages:
            raise ValueError("messages must not be empty")
        last = messages[-1]
        if last.get("role") != "user":
            raise ValueError(
                "the last message must have role='user', got "
                f"role={last.get('role')!r}"
            )
        user_text = last["content"]
        if not isinstance(user_text, str):
            raise TypeError(
                "the last user message's content must be a string"
            )

        system_parts: list[str] = []
        history: list[dict[str, str]] = []
        seen_non_system = False
        for m in messages[:-1]:
            role = m.get("role")
            content = m.get("content")
            if not isinstance(content, str):
                raise TypeError(
                    f"message content must be a string, got {type(content)}"
                )
            if role == "system":
                if seen_non_system:
                    raise ValueError(
                        "system messages must precede all user / "
                        "assistant turns"
                    )
                system_parts.append(content)
            elif role in ("user", "assistant"):
                seen_non_system = True
                history.append({"role": role, "content": content})
            else:
                raise ValueError(
                    f"unsupported role={role!r} (only system / user / "
                    "assistant)"
                )
        system_prompt = (
            "\n\n".join(system_parts) if system_parts else None
        )

        if stream:
            return self._stream_chat(
                system_prompt, history, user_text, params
            )
        return self._collect_chat(
            system_prompt, history, user_text, params
        )

    # --- internals -----------------------------------------------------

    def _effective_params(
        self, override: SamplingParams | None
    ) -> SamplingParams | None:
        if override is not None:
            return override
        return self._default_params

    def _eos_ids(self) -> set[int]:
        assert self._adapter is not None
        return set(
            getattr(self._adapter.tokenizer(), "eos_token_ids", set()) or ()
        )

    def _collect_generate(
        self, prompt: str, params: SamplingParams | None
    ) -> str:
        assert self._engine is not None and self._adapter is not None
        ids = list(self._engine.generate(prompt, params))
        eos = self._eos_ids()
        # Drop the trailing stop token before decoding so the returned
        # text mirrors what a streaming consumer sees (the stop token
        # is the natural terminator, not part of the reply).
        if ids and ids[-1] in eos:
            ids = ids[:-1]
        text = self._adapter.tokenizer().decode(ids).rstrip("�")
        return text

    def _stream_generate(
        self, prompt: str, params: SamplingParams | None
    ) -> Iterator[str]:
        assert self._engine is not None and self._adapter is not None
        eos = self._eos_ids()
        accumulated: list[int] = []
        printed = 0
        tokenizer = self._adapter.tokenizer()
        for tok in self._engine.generate(prompt, params):
            if tok in eos:
                # The trailing stop token is the natural terminator;
                # do not emit it to the consumer (decoding it would
                # leak ``<|im_end|>`` etc. into user-facing output).
                break
            accumulated.append(tok)
            current = tokenizer.decode(accumulated)
            safe = current.rstrip("�")
            if len(safe) > printed:
                yield safe[printed:]
                printed = len(safe)

    def _collect_chat(
        self,
        system_prompt: str | None,
        history: list[dict[str, str]],
        user_text: str,
        params: SamplingParams | None,
    ) -> str:
        session = self._build_chat_session(system_prompt, history)
        metrics = session.chat(user_text, sampling_params=params)
        # ``ChatSession.chat`` returns :class:`TurnMetrics` (typed),
        # but the facade builder returns ``Any`` to keep the import
        # surface narrow. Cast to ``str`` to satisfy strict mypy.
        return str(metrics.reply)

    def _stream_chat(
        self,
        system_prompt: str | None,
        history: list[dict[str, str]],
        user_text: str,
        params: SamplingParams | None,
    ) -> Iterator[str]:
        # Bridge ChatSession's stream_to callback (called from the
        # engine's caller thread, which is the user's thread here)
        # into a generator. The session runs synchronously, so we
        # accumulate deltas in a list during the call and yield them
        # afterwards. A more sophisticated bridge (asyncio queue +
        # worker thread) is the HTTP server's territory; the facade
        # is single-thread by design (G-1 under runtime.engine_lock
        # not applicable here — see module docstring).
        deltas: list[str] = []
        session = self._build_chat_session(system_prompt, history)
        session.chat(
            user_text,
            sampling_params=params,
            stream_to=deltas.append,
        )
        yield from deltas

    def _build_chat_session(
        self,
        system_prompt: str | None,
        history: list[dict[str, str]],
    ) -> Any:
        from silica.chat.session import ChatSession

        assert self._engine is not None and self._adapter is not None
        session = ChatSession(
            adapter=self._adapter,
            engine=self._engine,  # type: ignore[arg-type]
            system_prompt=system_prompt,
        )
        if history:
            full: list[dict[str, str]] = []
            if system_prompt:
                full.append({"role": "system", "content": system_prompt})
            full.extend(history)
            session.replace_messages(full)
        return session


__all__ = ["LLM"]

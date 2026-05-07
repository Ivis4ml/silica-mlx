"""silica.server.ratelimit — token-bucket rate limiting (P-8 sub-unit (h)).

Per-key token-bucket: each authentication key (or per-IP when auth
is disabled) gets its own bucket. The bucket starts full at the
configured RPM cap and refills linearly over time; each request
deducts one token. Empty bucket → 429 with the OpenAI-shaped
``rate_limit_error`` envelope.

When the configured RPM is ``None`` or ``<= 0``, the helper is a
pass-through. v0.1 default is unlimited; ``--rate-limit-rpm`` opts
in.

The token-bucket algorithm is the right shape for chat workloads:

- Smooth bursts (e.g. an SDK retry storm after a transient 503) up
  to the bucket capacity, instead of the hard cliff a sliding-window
  counter would impose.
- Refill is linear in elapsed wall-time, so a 60-RPM bucket allows
  exactly one request per second on average regardless of arrival
  pattern.
- Per-key isolation means one noisy client cannot starve the
  others — important once API keys become per-user in v0.2.

The implementation is **in-memory** and lost on process restart.
For a multi-process deployment a shared Redis-backed bucket is
post-announce; v0.1 is single-process per
``plans/P8_OPENING.md`` §6.1.2 G-1.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse

from silica.core.logger import get_logger
from silica.server.errors import openai_error_payload

log = get_logger(__name__)


_RATELIMIT_EXEMPT_PATHS: frozenset[str] = frozenset({"/healthz"})
"""Routes that bypass rate limiting.

Same logic as auth: liveness probes must not consume budget."""


class _Bucket:
    """One token bucket. Internal — the manager owns lifecycle."""

    __slots__ = ("tokens", "last_refill_s")

    def __init__(self, *, tokens: float, now_s: float) -> None:
        self.tokens = tokens
        self.last_refill_s = now_s


class _RateLimiter:
    """Per-key token-bucket store, scoped to one ``rpm`` cap.

    Cached per-RPM by :func:`consume_rate_limit_token` so the
    bucket state survives across requests. A reset of the
    process-level :data:`_LIMITERS` cache is the test seam.
    """

    def __init__(
        self,
        *,
        rpm: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rpm <= 0:
            raise ValueError(f"rpm must be > 0, got {rpm}")
        self._capacity = float(rpm)
        # Tokens per second — derived once at construction so the
        # hot path skips the divide.
        self._refill_per_s = self._capacity / 60.0
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        # In-memory dict + a single lock is fine for the v0.1
        # single-process target; concurrent route handlers contend
        # only on the lookup + refill arithmetic, which is
        # microseconds.
        self._lock = threading.Lock()

    @property
    def rpm(self) -> int:
        return int(self._capacity)

    def consume(self, key: str) -> bool:
        """Refill the bucket per elapsed time, deduct one token,
        return whether the deduction succeeded."""
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=self._capacity, now_s=now)
                self._buckets[key] = bucket
            else:
                elapsed = now - bucket.last_refill_s
                if elapsed > 0:
                    bucket.tokens = min(
                        self._capacity,
                        bucket.tokens + elapsed * self._refill_per_s,
                    )
                    bucket.last_refill_s = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True
            return False


# Per-RPM limiter cache. The first request at a given RPM
# constructs the limiter; subsequent requests reuse it so bucket
# state persists across the process lifetime. Tests reset via
# :func:`_reset_limiters` between runs.
_LIMITERS: dict[int, _RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def _get_or_make_limiter(rpm: int) -> _RateLimiter:
    with _LIMITERS_LOCK:
        existing = _LIMITERS.get(rpm)
        if existing is not None:
            return existing
        new = _RateLimiter(rpm=rpm)
        _LIMITERS[rpm] = new
        return new


def _reset_limiters() -> None:
    """Drop the per-RPM limiter cache.

    Called between tests so a fresh RPM bucket starts each
    parametrised case. Production callers do not invoke this.
    """
    with _LIMITERS_LOCK:
        _LIMITERS.clear()


def consume_rate_limit_token(
    request: Request, *, rpm: int | None
) -> JSONResponse | None:
    """Enforce the configured RPM cap for ``request``.

    Returns ``None`` to allow the request through; returns a 429
    :class:`JSONResponse` (OpenAI envelope, ``Retry-After`` header)
    when the bucket is empty.

    Behaviour:

    - ``rpm`` is ``None`` or ``<= 0`` — rate limit disabled, always
      allow.
    - Path is in :data:`_RATELIMIT_EXEMPT_PATHS` — always allow.
    - Otherwise — consume one token from the per-key bucket; reject
      with 429 if empty.
    """
    if rpm is None or rpm <= 0:
        return None
    if request.url.path in _RATELIMIT_EXEMPT_PATHS:
        return None

    limiter = _get_or_make_limiter(rpm)
    key = _default_key_fn(request)
    if limiter.consume(key):
        return None

    log.info(
        "ratelimit.deny key=%r path=%s rpm=%d",
        _redact_key(key),
        request.url.path,
        rpm,
    )
    return JSONResponse(
        status_code=429,
        content=openai_error_payload(
            message=(
                f"rate limit exceeded ({rpm} requests per minute)"
            ),
            error_type="rate_limit_error",
            code="rate_limit_exceeded",
        ),
        # OpenAI's docs cite ``Retry-After`` on 429s so SDK
        # exponential-backoff implementations can read a
        # server-supplied delay. One token refills every
        # ``60 / rpm`` seconds.
        headers={"Retry-After": f"{max(1, int(60.0 / rpm))}"},
    )


def _default_key_fn(request: Request) -> str:
    """Pick the bucket key from the request.

    Preference order:

    1. ``Authorization`` header (entire value, not just the bearer
       token — keeps shape future-compatible with non-Bearer schemes).
    2. ``X-Forwarded-For`` first-hop IP if present (typical reverse-
       proxy deployment).
    3. Direct ``request.client.host``.
    4. Literal ``"anonymous"`` if none of the above resolve (only
       happens in unusual transport configurations).
    """
    auth = request.headers.get("authorization")
    if auth:
        return f"auth:{auth}"
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        first = fwd.split(",")[0].strip()
        if first:
            return f"ip:{first}"
    if request.client is not None and request.client.host:
        return f"ip:{request.client.host}"
    return "anonymous"


def _redact_key(key: str) -> str:
    """Truncate a bucket key for logging.

    The key is often an ``Authorization`` header value; logging it
    in full leaks the API key into the server log. Keep the
    namespace prefix (``auth:`` / ``ip:``) so an operator can still
    distinguish auth- vs IP-keyed buckets.
    """
    if ":" not in key:
        return "<unscoped>"
    scope, _, _ = key.partition(":")
    return f"{scope}:<redacted>"


__all__ = [
    "consume_rate_limit_token",
]

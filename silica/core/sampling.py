"""SamplingParams: immutable sampling configuration for a generation request.

The field set is driven by D-013 (PLAN.md): the sampler applies logit processors
in the fixed order `temperature -> repetition penalty -> top-k -> top-p -> sample`,
plus the standard generation controls (max_tokens, stop, seed, ignore_eos).

Greedy decoding is triggered by `temperature == 0`, matching mlx-lm and vLLM.
The sampler reads `is_greedy` to take the argmax fast path and skip the chain.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SamplingParams(BaseModel):
    """Immutable sampling configuration for a single generation request.

    A processor is skipped when its parameter is the identity:
      - `temperature == 1.0`
      - `repetition_penalty == 1.0`
      - `top_k is None`
      - `top_p is None`

    When `is_greedy` is True the entire chain is skipped and argmax is taken.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float = Field(default=1.0, ge=0.0)
    top_k: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)

    max_tokens: int = Field(default=256, ge=1)
    stop: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False
    seed: int | None = None

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= 0.0

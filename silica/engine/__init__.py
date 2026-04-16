"""Silica engine — inference loop (stub, Phase 1 will fill this in)."""


class Engine:
    """Entry point for Silica inference.

    Phase 0 stub — raises NotImplementedError for all operations.
    Phase 1 will implement generate() on top of SimpleKVCache + mlx-lm.
    """

    def generate(self, prompt: str, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError("Engine.generate() not yet implemented (Phase 1)")

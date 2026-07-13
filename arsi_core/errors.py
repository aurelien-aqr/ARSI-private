"""Error taxonomy — behaviours are part of docs/SPEC.md.

Job-fatal (abort before/at start):      OllamaUnreachable, ModelMissing
Frame-level, retried then skipped:      ParseError, VLMCallError
Frame-level, skipped without retry:     FrameError
"""


class ArsiError(Exception):
    """Base class for every error this package raises on purpose."""


class OllamaUnreachable(ArsiError):
    def __init__(self, detail=""):
        super().__init__(
            "could not reach the Ollama server — start it with: ollama serve"
            + (f" ({detail})" if detail else ""))


class ModelMissing(ArsiError):
    def __init__(self, model: str):
        self.model = model
        super().__init__(
            f"model '{model}' is not installed — install it with: ollama pull {model}")


class ParseError(ArsiError):
    """The VLM answered, but not in the required format (bad JSON / missing
    structure). The runner retries with a format reminder, then marks the
    frame failed."""

    def __init__(self, detail: str, raw: str = ""):
        self.raw = raw
        super().__init__(detail)


class VLMCallError(ArsiError):
    """Transport-level failure during one VLM call (timeout, connection drop).
    Retried, then the frame is marked failed."""


class FrameError(ArsiError):
    """The frame itself is unusable (unreadable file, size mismatch that cannot
    be reconciled). Not retried; the batch continues."""

"""Thin wrapper around the ollama client: health, model checks (":latest"
aware), pull with progress, and chat with a per-call timeout.

Adapters inject this object as the `ollama` attribute of the vlm_0x modules,
so every script call goes through the same timeout/error handling — and tests
inject a fake with the same .chat()/.list() surface.
"""
import socket
import time

from .errors import OllamaUnreachable, ModelMissing, VLMCallError


class OllamaClient:
    def __init__(self, host=None, timeout: float = 120.0, impl=None):
        if impl is None:
            import ollama
            impl = ollama.Client(host=host, timeout=timeout)
        self._impl = impl
        self.timeout = timeout
        self.call_seconds = []          # wall-clock of successful chat calls

    # -- server / models ------------------------------------------------------

    def list(self):
        """Raw passthrough (vlm_0x check_model calls ollama.list())."""
        return self._impl.list()

    def model_names(self):
        try:
            data = self._impl.list()
        except Exception as exc:
            raise OllamaUnreachable(str(exc)) from exc
        models = getattr(data, "models", None)
        if models is None and isinstance(data, dict):
            models = data.get("models", [])
        names = []
        for m in models or []:
            name = getattr(m, "model", None) or getattr(m, "name", None)
            if name is None and isinstance(m, dict):
                name = m.get("model") or m.get("name")
            if name:
                names.append(name)
        return names

    def has_model(self, model: str) -> bool:
        # Ollama stores tag-less pulls as "name:latest" — accept both forms.
        names = self.model_names()
        return model in names or f"{model}:latest" in names

    def ensure_model(self, model: str):
        if not self.has_model(model):
            raise ModelMissing(model)

    def health(self) -> dict:
        try:
            names = self.model_names()
            return {"reachable": True, "models": names}
        except OllamaUnreachable as exc:
            return {"reachable": False, "models": [], "detail": str(exc)}

    def pull(self, model: str):
        """Yield {status, completed, total} progress dicts."""
        try:
            for part in self._impl.pull(model, stream=True):
                get = part.get if isinstance(part, dict) else lambda k, d=None: getattr(part, k, d)
                yield {"status": get("status", ""),
                       "completed": get("completed", 0) or 0,
                       "total": get("total", 0) or 0}
        except Exception as exc:
            raise OllamaUnreachable(str(exc)) from exc

    # -- interruption ---------------------------------------------------------

    def abort(self) -> int:
        """Interrupt the HTTP call in flight (from ANOTHER thread) and return how
        many sockets were shut down. Used by the force-stop path: a crop call on
        CPU takes 2-4 min and there is no way to ask Ollama to stop generating,
        so the only real lever is the connection.

        Measured, because the obvious things do not work: `Client.close()`
        returns instantly but leaves the reader blocked (120 s+), and closing the
        socket's file descriptor does not wake it either. Only
        `shutdown(SHUT_RDWR)` does — it unblocked the read in 0.0 s and the call
        raised RemoteProtocolError. The pool is private API of httpx/httpcore, so
        every step is defensive and a failure just means "nothing aborted".
        """
        n = 0
        pool = getattr(getattr(getattr(self._impl, "_client", None),
                               "_transport", None), "_pool", None)
        for conn in list(getattr(pool, "connections", []) or []):
            inner = getattr(conn, "_connection", conn)
            stream = getattr(inner, "_network_stream", None)
            sock = getattr(stream, "_sock", None)
            if sock is None:
                continue
            try:
                sock.shutdown(socket.SHUT_RDWR)
                n += 1
            except OSError:
                pass                # already dead: nothing to abort
        return n

    # -- inference ------------------------------------------------------------

    def chat(self, *args, **kwargs):
        """Same signature the scripts use (ollama.chat(model=..., messages=...,
        think=..., options=...)). Transport failures become VLMCallError."""
        t0 = time.time()
        try:
            resp = self._impl.chat(*args, **kwargs)
        except Exception as exc:
            raise VLMCallError(f"{type(exc).__name__}: {exc}") from exc
        self.call_seconds.append(time.time() - t0)
        return resp

"""OpenAI-compatible impl for OllamaClient, targeting `llama-server --mmproj`.

This is the prepared fallback of RUNBOOK_LORA.md step 6: if Ollama's GGUF
vision import drops the projector, serve the fine-tuned judge with
llama.cpp's server instead and swap this in — everything downstream
(adapters, runner, benchmark) keeps calling the same OllamaClient surface:

    from arsi_core.ollama_client import OllamaClient
    from arsi_core.llamacpp_client import LlamaCppServer
    client = OllamaClient(impl=LlamaCppServer("http://localhost:11435"))
    # or for the raw scripts / benchmark:
    #   vlm_05_reference_diff.ollama = client

Only what the vlm_0x call sites use is implemented: .chat(model, messages
with image paths, think, options) and .list(). Ollama-style image paths are
inlined as base64 data URLs, num_predict/temperature map onto max_tokens/
temperature (num_ctx is a server-side -c flag here, so it is ignored).
"""
import base64
import mimetypes
from pathlib import Path

import httpx


class LlamaCppServer:
    def __init__(self, base_url: str = "http://localhost:11435",
                 timeout: float = 300.0, client: httpx.Client = None):
        self.base_url = base_url.rstrip("/")
        self._http = client or httpx.Client(timeout=timeout)

    # -- OllamaClient surface -------------------------------------------------

    def list(self):
        r = self._http.get(self.base_url + "/v1/models")
        r.raise_for_status()
        return {"models": [{"model": m["id"]} for m in r.json().get("data", [])]}

    def chat(self, model=None, messages=None, think=None, options=None, **kw):
        payload = {"model": model or "default",
                   "messages": [self._convert(m) for m in (messages or [])]}
        options = options or {}
        if options.get("num_predict"):
            payload["max_tokens"] = options["num_predict"]
        if options.get("temperature") is not None:
            payload["temperature"] = options["temperature"]
        r = self._http.post(self.base_url + "/v1/chat/completions", json=payload)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"].get("content", "")
        return {"message": {"content": content}}

    def pull(self, model, stream=True):
        raise NotImplementedError("llama-server serves one fixed model — "
                                  "pulling happens at server start")

    def delete(self, model):
        raise NotImplementedError("llama-server serves one fixed model")

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _convert(msg: dict) -> dict:
        """Ollama message (content + image PATHS) -> OpenAI content parts."""
        images = msg.get("images") or []
        if not images:
            return {"role": msg.get("role", "user"),
                    "content": msg.get("content", "")}
        parts = [{"type": "text", "text": msg.get("content", "")}]
        for img in images:
            data = Path(img).read_bytes()
            mime = mimetypes.guess_type(str(img))[0] or "image/jpeg"
            url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        return {"role": msg.get("role", "user"), "content": parts}

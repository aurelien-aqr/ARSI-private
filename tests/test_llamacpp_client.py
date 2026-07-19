"""LlamaCppServer shim: same OllamaClient surface as the real Ollama impl,
speaking OpenAI to llama-server (the RUNBOOK_LORA step-6 fallback)."""
import base64
import json

import httpx
import pytest

from arsi_core.llamacpp_client import LlamaCppServer
from arsi_core.ollama_client import OllamaClient


def make_server(handler):
    return LlamaCppServer("http://fake:11435",
                          client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_list_maps_models():
    def handler(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "arsi-judge"}]})
    impl = make_server(handler)
    assert impl.list() == {"models": [{"model": "arsi-judge"}]}
    # and through the OllamaClient surface, ":latest"-aware like the real one
    oc = OllamaClient(impl=impl)
    assert oc.has_model("arsi-judge")


def test_chat_inlines_images_and_maps_options(img_factory):
    img = img_factory("crop.jpg", size=(60, 40))
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "YES phone"}}]})
    impl = make_server(handler)
    resp = impl.chat(model="arsi-judge",
                     messages=[{"role": "user", "content": "judge this",
                                "images": [str(img)]}],
                     think=False,
                     options={"num_predict": 128, "temperature": 0.0,
                              "num_ctx": 8192})
    assert resp == {"message": {"content": "YES phone"}}
    assert seen["max_tokens"] == 128 and seen["temperature"] == 0.0
    assert "num_ctx" not in seen                     # server-side flag, dropped
    parts = seen["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "judge this"}
    url = parts[1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == img.read_bytes()


def test_text_only_message_stays_plain_string():
    def handler(request):
        body = json.loads(request.content)
        assert body["messages"][0]["content"] == "hello"
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
    impl = make_server(handler)
    assert impl.chat(messages=[{"role": "user", "content": "hello"}]) \
        == {"message": {"content": "hi"}}


def test_pull_is_refused():
    impl = make_server(lambda r: httpx.Response(500))
    with pytest.raises(NotImplementedError):
        impl.pull("x")

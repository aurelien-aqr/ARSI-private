import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arsi_core.ollama_client import OllamaClient  # noqa: E402


class FakeOllama:
    """Programmable stand-in for ollama.Client: .chat pops replies from a
    queue (or calls a function), .list reports installed models."""

    def __init__(self, replies=None, models=("qwen3-vl:8b-instruct",)):
        self.replies = list(replies or [])
        self.models = list(models)
        self.calls = []

    def chat(self, model=None, messages=None, **kw):
        self.calls.append({"model": model, "messages": messages})
        if not self.replies:
            raise AssertionError("FakeOllama: no reply queued for this call")
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(messages)
        if isinstance(reply, Exception):
            raise reply
        return {"message": {"content": reply}}

    def list(self):
        return {"models": [{"model": m} for m in self.models]}


@pytest.fixture
def fake_client():
    def make(replies=None, models=("qwen3-vl:8b-instruct",)):
        return OllamaClient(impl=FakeOllama(replies, models))
    return make


@pytest.fixture
def img_factory(tmp_path):
    """Create simple RGB test images on disk."""
    from PIL import Image, ImageDraw

    def make(name="img.jpg", size=(400, 300), color=(128, 128, 128), rects=()):
        img = Image.new("RGB", size, color)
        draw = ImageDraw.Draw(img)
        for (box, col) in rects:
            draw.rectangle(box, fill=col)
        path = tmp_path / name
        img.save(path)
        return path
    return make

import atexit
import os
import shutil
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Isolate all app runtime state (jobs, masks, videos, cache, settings) from
# the real data/app. Must happen BEFORE the first arsi_core import, and must
# live under data/ so the backend's media guard and REPO_ROOT-relative URLs
# still work on test files.
if "ARSI_APP_DATA" not in os.environ:
    _tmp_root = REPO_ROOT / "data" / "app" / ".pytest-tmp"
    _tmp = _tmp_root / uuid.uuid4().hex[:8]
    _tmp.mkdir(parents=True, exist_ok=True)
    os.environ["ARSI_APP_DATA"] = str(_tmp)
    atexit.register(shutil.rmtree, _tmp_root, True)

from arsi_core.ollama_client import OllamaClient  # noqa: E402
from arsi_core.adapters import get_module  # noqa: E402

#: A job that names no model runs the script's own MODEL_NAME, and the backend
#: refuses to start a job whose model is not installed - so the fake server must
#: report that default as present or every default-model test dies as a job that
#: never finishes. Read from the module instead of typed here: changing the
#: shipped default must not mean editing four tuples in three files.
PIPELINE_DEFAULT_MODEL = get_module("vlm_05").MODEL_NAME
INSTALLED = ("qwen3-vl:8b-instruct", PIPELINE_DEFAULT_MODEL)


class FakeOllama:
    """Programmable stand-in for ollama.Client: .chat pops replies from a
    queue (or calls a function), .list reports installed models."""

    def __init__(self, replies=None, models=INSTALLED):
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
    def make(replies=None, models=INSTALLED):
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

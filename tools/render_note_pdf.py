"""Print a standalone note to PDF with headless Chrome.

    venv/bin/python tools/render_note_pdf.py                # all of them
    venv/bin/python tools/render_note_pdf.py vlm-benchmark  # one

The page geometry lives in the note's own @page / @media print rules - the
shared sheet in tools/report_style.py, so all three print alike - and this
drives the browser and adds the one thing CSS cannot give in Chrome, a running
foot with page numbers. Build the note first (tools/build_*_doc.py).

Chrome is driven through the DevTools protocol so that Page.printToPDF can be
called with real arguments; if that handshake fails for any reason the plain
--print-to-pdf command line is used instead, minus the page numbers.

The Studio serves the result at /notes/<key>.pdf and calls render() itself when
the PDF is missing or older than the page, so the button in the note's bar
never hands back a stale document.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.notes import NOTES, Note                                # noqa: E402

BROWSERS = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "brave-browser")
FLAGS = ["--headless=new", "--disable-gpu", "--no-sandbox", "--force-color-profile=srgb",
         "--run-all-compositor-stages-before-draw", "--hide-scrollbars",
         "--disable-extensions", "--disable-background-networking"]

FOOTER = """
<div style="width:100%;padding:0 14mm;font:7pt -apple-system,'DejaVu Sans',sans-serif;
            color:#8A9899;display:flex;justify-content:space-between;">
  <span>{head}</span>
  <span><span class="pageNumber"></span> / <span class="totalPages"></span></span>
</div>"""

# gallery images are marked loading="lazy"; printing never scrolls past them,
# so they have to be pulled in and decoded by hand before the page is printed
EAGER_IMAGES = """(async () => {
  const imgs = [...document.images];
  for (const img of imgs) img.removeAttribute('loading');
  await new Promise(r => requestAnimationFrame(r));
  await Promise.all(imgs.map(i => i.decode().catch(() => {})));
  return imgs.filter(i => !i.complete || !i.naturalWidth).length;
})()"""

# printToPDF wants inches; the sheet is A4 with 14 mm side margins
MM = 1 / 25.4
PRINT_ARGS = {
    "printBackground": True,
    "preferCSSPageSize": True,
    "paperWidth": 210 * MM, "paperHeight": 297 * MM,
    "marginTop": 15 * MM, "marginBottom": 14 * MM,
    "marginLeft": 14 * MM, "marginRight": 14 * MM,
    "displayHeaderFooter": True,
    "headerTemplate": "<div></div>",
}

#: one Chrome at a time per note. Two readers clicking the button together used
#: to start two renders writing the same file.
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def find_browser() -> str:
    for name in BROWSERS:
        path = shutil.which(name)
        if path:
            return path
    sys.exit("no Chrome/Chromium found; tried " + ", ".join(BROWSERS))


async def _print_via_cdp(browser: str, url: str, pdf: Path, footer: str) -> None:
    import websockets

    with tempfile.TemporaryDirectory(prefix="chrome-pdf-") as profile:
        proc = subprocess.Popen(
            [browser, *FLAGS, f"--user-data-dir={profile}", "--remote-debugging-port=0", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        try:
            port = None
            for _ in range(200):                       # the port is announced on stderr
                line = proc.stderr.readline()
                if not line:
                    break
                m = re.search(r"127\.0\.0\.1:(\d+)", line)
                if m:
                    port = m.group(1)
                    break
            if port is None:
                raise RuntimeError("Chrome never announced its debugging port")

            target = _page_target(port)

            async with websockets.connect(target, max_size=None) as ws:
                seq = 0

                async def call(method, **params):
                    nonlocal seq
                    seq += 1
                    await ws.send(json.dumps({"id": seq, "method": method, "params": params}))
                    while True:
                        msg = json.loads(await ws.recv())
                        if msg.get("id") == seq:
                            if "error" in msg:
                                raise RuntimeError(f"{method}: {msg['error']}")
                            return msg["result"]

                await call("Page.enable")
                await call("Page.navigate", url=url)
                # every figure is an inlined data: URI, so loading is local but not instant
                await _wait_loaded(ws)
                await call("Runtime.enable")
                pending = await call("Runtime.evaluate", expression=EAGER_IMAGES,
                                     awaitPromise=True, returnByValue=True, timeout=180000)
                missing = pending["result"].get("value")
                if missing:
                    raise RuntimeError(f"{missing} figures never decoded")
                await asyncio.sleep(1.0)
                result = await call("Page.printToPDF", footerTemplate=footer, **PRINT_ARGS)
                pdf.write_bytes(base64.b64decode(result["data"]))
        finally:
            proc.terminate()
            proc.wait(timeout=30)


def _page_target(port: str, tries: int = 40) -> str:
    """The websocket of the blank tab; the list also holds extension pages."""
    for _ in range(tries):
        targets = json.loads(urllib.request.urlopen(
            f"http://127.0.0.1:{port}/json/list", timeout=20).read())
        for t in targets:
            if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                return t["webSocketDebuggerUrl"]
        time.sleep(0.25)
    raise RuntimeError("no page target exposed by Chrome")


async def _wait_loaded(ws, timeout: float = 180.0) -> None:
    async def loop():
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("method") == "Page.loadEventFired":
                return
    await asyncio.wait_for(loop(), timeout)


def _print_via_cli(browser: str, url: str, pdf: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="chrome-pdf-") as profile:
        subprocess.run([browser, *FLAGS, f"--user-data-dir={profile}",
                        "--virtual-time-budget=60000", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", url],
                       check=True, capture_output=True, timeout=600)


def is_stale(note: Note) -> bool:
    """True when the PDF is missing or older than the page it prints."""
    return (not note.pdf.is_file()
            or note.pdf.stat().st_mtime < note.html.stat().st_mtime)


def render(key: str, force: bool = False) -> Path:
    """The note's PDF, printing it first unless a current one is on disk."""
    note = NOTES[key]
    if not note.html.is_file():
        raise FileNotFoundError(f"{note.html} missing - build the note first")
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.Lock())
    with lock:
        if not force and not is_stale(note):
            return note.pdf
        browser = find_browser()
        # ?theme=light rather than trusting the renderer's colour scheme; the
        # print palette also forces it, and one of the two is enough
        url = note.html.as_uri() + "?theme=light"
        footer = FOOTER.format(head=escape(note.running_head))
        try:
            asyncio.run(_print_via_cdp(browser, url, note.pdf, footer))
        except Exception as exc:              # noqa: BLE001 - any failure falls back
            print(f"devtools print failed ({exc}); falling back to --print-to-pdf",
                  file=sys.stderr)
            _print_via_cli(browser, url, note.pdf)
    return note.pdf


def _describe(pdf: Path) -> str:
    pages = ""
    if shutil.which("pdfinfo"):
        info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        pages = next((ln.split()[1] for ln in info.splitlines() if ln.startswith("Pages:")), "")
    return f"{pdf}  ({pdf.stat().st_size / 1e6:.1f} MB{', ' + pages + ' pages' if pages else ''})"


if __name__ == "__main__":
    keys = sys.argv[1:] or list(NOTES)
    for k in keys:
        if k not in NOTES:
            sys.exit(f"unknown note {k!r}; known: {', '.join(NOTES)}")
    for k in keys:
        print(_describe(render(k, force=True)))

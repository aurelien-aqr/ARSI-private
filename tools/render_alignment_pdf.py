"""Print the framing-drift report to PDF with headless Chrome.

The page geometry lives in the document's own @page / @media print rules; this
drives the browser and adds the one thing CSS cannot give in Chrome, a footer
with page numbers. Run tools/build_alignment_doc.py first.

Chrome is driven through the DevTools protocol so that Page.printToPDF can be
called with real arguments; if that handshake fails for any reason the plain
--print-to-pdf command line is used instead, minus the page numbers.

Output: docs/camera_alignment/camera_framing_drift_1760_39T.pdf
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
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "camera_alignment"
DOC = OUT / "camera_framing_drift_1760_39T.html"
PDF = DOC.with_suffix(".pdf")

BROWSERS = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "brave-browser")
FLAGS = ["--headless=new", "--disable-gpu", "--no-sandbox", "--force-color-profile=srgb",
         "--run-all-compositor-stages-before-draw", "--hide-scrollbars",
         "--disable-extensions", "--disable-background-networking"]

FOOTER = """
<div style="width:100%;padding:0 14mm;font:7pt -apple-system,'DejaVu Sans',sans-serif;
            color:#8A9899;display:flex;justify-content:space-between;">
  <span>Framing drift 1760 &rarr; 39T &middot; ARSI</span>
  <span><span class="pageNumber"></span> / <span class="totalPages"></span></span>
</div>"""

# the gallery images are marked loading="lazy"; printing never scrolls past them,
# so they have to be pulled in and decoded by hand before the page is printed
EAGER_IMAGES = """(async () => {
  const imgs = [...document.images];
  for (const img of imgs) img.removeAttribute('loading');
  await new Promise(r => requestAnimationFrame(r));
  await Promise.all(imgs.map(i => i.decode().catch(() => {})));
  return imgs.filter(i => !i.complete || !i.naturalWidth).length;
})()"""

# printToPDF wants inches; the document is A4 with 14 mm side margins
MM = 1 / 25.4
PRINT_ARGS = {
    "printBackground": True,
    "preferCSSPageSize": True,
    "paperWidth": 210 * MM, "paperHeight": 297 * MM,
    "marginTop": 16 * MM, "marginBottom": 15 * MM,
    "marginLeft": 14 * MM, "marginRight": 14 * MM,
    "displayHeaderFooter": True,
    "headerTemplate": "<div></div>",
    "footerTemplate": FOOTER,
}


def find_browser() -> str:
    for name in BROWSERS:
        path = shutil.which(name)
        if path:
            return path
    sys.exit("no Chrome/Chromium found; tried " + ", ".join(BROWSERS))


async def _print_via_cdp(browser: str, url: str, pdf: Path) -> None:
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
                result = await call("Page.printToPDF", **PRINT_ARGS)
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


def render(doc: Path = DOC, pdf: Path = PDF) -> Path:
    if not doc.exists():
        sys.exit(f"{doc} missing — run tools/build_alignment_doc.py first")
    browser, url = find_browser(), doc.as_uri()
    try:
        asyncio.run(_print_via_cdp(browser, url, pdf))
    except Exception as exc:                            # noqa: BLE001 - any failure falls back
        print(f"devtools print failed ({exc}); falling back to --print-to-pdf", file=sys.stderr)
        _print_via_cli(browser, url, pdf)
    return pdf


if __name__ == "__main__":
    out = render()
    pages = ""
    if shutil.which("pdfinfo"):
        info = subprocess.run(["pdfinfo", str(out)], capture_output=True, text=True).stdout
        pages = next((l.split()[1] for l in info.splitlines() if l.startswith("Pages:")), "")
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB{', ' + pages + ' pages' if pages else ''})")

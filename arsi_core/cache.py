"""vlm_05 verdict cache.

Same key scheme as benchmark/run_benchmark.py
(`ref_name|img_name|[bbox]|model|prompt_fingerprint`, plus `|mask:<hash>`
when a mask is active), so the app profits from the existing benchmark
cache. The benchmark file is read as a SEED (never written); new verdicts
go to the app's own file.
"""
import hashlib
import json
from pathlib import Path

from . import APP_DATA, REPO_ROOT

APP_CACHE_PATH = APP_DATA / "cache" / "verdicts.json"
BENCH_CACHE_PATH = REPO_ROOT / "benchmark" / "cache.json"
#: name -> sha1 of the file the cached verdicts were earned on. The key scheme
#: identifies an image by its NAME, so rewriting a file in place (a benchmark
#: frame rebuilt through a different mask, a re-exported crop) silently scores
#: the new pixels against the old verdicts. This is what catches that.
FINGERPRINTS_PATH = APP_DATA / "cache" / "image_fingerprints.json"


class VerdictCache:
    def __init__(self, path=APP_CACHE_PATH, seed_paths=(BENCH_CACHE_PATH,)):
        self.path = Path(path)
        self._data = {}
        for seed in seed_paths:
            if seed and Path(seed).exists():
                with open(seed, encoding="utf-8") as fh:
                    self._data.update(json.load(fh))
        self._own = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                self._own = json.load(fh)
            self._data.update(self._own)

    def get(self, key):
        return self._data.get(key)

    def put(self, key, value):
        self._data[key] = value
        self._own[key] = value
        self._flush()

    def _flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._own, fh, indent=1, ensure_ascii=False)

    def drop_changed(self, paths):
        """Forget every verdict earned on an image whose bytes have changed.

        Compares each path against the recorded fingerprint and, on a mismatch,
        drops the entries naming that file - the key carries the basename, which
        is exactly what stopped identifying the pixels. Returns one dict per
        changed image; an empty list is the normal case and costs one sha1 per
        input image, not per region.

        `seed_only` marks entries that live in a read-only seed file (the
        benchmark cache): they are gone for this run, but the seed still holds
        them and only rebuilding it removes them for good.
        """
        try:
            known = json.loads(FINGERPRINTS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            known = {}

        changed = []
        for path in {str(p) for p in paths if p}:
            f = Path(path)
            if not f.exists():
                continue
            digest = hashlib.sha1(f.read_bytes()).hexdigest()
            name = f.name
            if known.get(name, digest) != digest:
                marker = "|" + name + "|"
                hit = [k for k in self._data if marker in k]
                own = [k for k in hit if k in self._own]
                for k in hit:
                    self._data.pop(k, None)
                for k in own:
                    self._own.pop(k, None)
                if hit:
                    changed.append({"image": name, "dropped": len(hit),
                                    "seed_only": len(hit) - len(own)})
            known[name] = digest

        if changed:
            self._flush()
        FINGERPRINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FINGERPRINTS_PATH, "w", encoding="utf-8") as fh:
            json.dump(known, fh, indent=1, sort_keys=True)
        return changed

    def __len__(self):
        return len(self._data)

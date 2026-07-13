"""vlm_05 verdict cache.

Same key scheme as benchmark/run_benchmark.py
(`ref_name|img_name|[bbox]|model|prompt_fingerprint`, plus `|mask:<hash>`
when a mask is active), so the app profits from the existing benchmark
cache. The benchmark file is read as a SEED (never written); new verdicts
go to the app's own file.
"""
import json
from pathlib import Path

from . import APP_DATA, REPO_ROOT

APP_CACHE_PATH = APP_DATA / "cache" / "verdicts.json"
BENCH_CACHE_PATH = REPO_ROOT / "benchmark" / "cache.json"


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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._own, fh, indent=1, ensure_ascii=False)

    def __len__(self):
        return len(self._data)

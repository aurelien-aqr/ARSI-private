"""Benchmark API: browse and correct a dataset, launch a scored run.

Everything here runs against a synthetic two-case dataset in a temp directory —
the shipped protocols are the measurement, and a test suite that edits them
would be editing the results.
"""
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.backend.main as backend
from app.backend import bench_runs
from arsi_core import benchmarks, runner
from arsi_core.ollama_client import OllamaClient

from conftest import FakeOllama

APP_DATA = Path(os.environ["ARSI_APP_DATA"])
REPO_ROOT = Path(__file__).resolve().parent.parent
CLEAN_CROP = "NO, the same empty seat."


def _img(name, rects=()):
    """An image under data/ (so the media guard and repo-relative paths work)."""
    path = APP_DATA / "benchimgs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (400, 300), (90, 95, 100))
    if rects:
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        for box in rects:
            d.rectangle(box, fill=(240, 240, 240))
    img.save(path)
    return str(path.relative_to(REPO_ROOT))


@pytest.fixture
def bench(monkeypatch, tmp_path):
    """A temp datasets dir holding `demo`, a temp runs dir, and a TestClient."""
    monkeypatch.setattr(benchmarks, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(benchmarks, "BACKUPS_DIR", tmp_path / "datasets" / ".backups")
    monkeypatch.setattr(benchmarks, "LEGACY_PATHS", {})
    monkeypatch.setattr(bench_runs, "RUNS_DIR", tmp_path / "runs")

    ref = _img("ref.jpg")
    anom = _img("anom.jpg", rects=[(150, 100, 250, 200)])
    clean = _img("clean.jpg")
    doc = {"_about": "synthetic", "name": "demo",
           "references": {"cam": ref},
           "cases": [
               {"id": "a_anom", "image": anom, "reference": "cam", "source": "real",
                "has_anomaly": True, "types": ["object"], "note": "white square",
                "instances": [{"type": "object", "label": "square",
                               "bbox": [150, 100, 250, 200]}]},
               {"id": "b_clean", "image": clean, "reference": "cam", "source": "real",
                "has_anomaly": False, "types": [], "note": "nothing",
                "instances": []},
           ]}
    benchmarks.save("demo", doc)

    def make(replies=None, models=("qwen3-vl:8b-instruct",)):
        impl = FakeOllama(replies, models)
        oc = OllamaClient(impl=impl)
        monkeypatch.setattr(backend, "client_for", lambda timeout=None: oc)
        monkeypatch.setattr(runner, "OllamaClient", lambda *a, **k: oc)
        return TestClient(backend.app), impl
    return make


def wait_run(client, run_id, timeout=90.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = client.get(f"/api/benchmarks/runs/{run_id}").json()
        if st["run"]["status"] in ("completed", "failed", "cancelled", "interrupted"):
            return st
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


# ---------------------------------------------------------------- browsing

def test_index_lists_datasets_with_their_counts(bench):
    client, _ = bench()
    rows = {d["id"]: d for d in client.get("/api/benchmarks").json()["datasets"]}
    assert rows["demo"]["n_cases"] == 2
    assert rows["demo"]["n_anomalous"] == 1 and rows["demo"]["n_instances"] == 1
    assert rows["demo"]["errors"] == []


def test_detail_carries_image_urls_for_case_and_reference(bench):
    client, _ = bench()
    d = client.get("/api/benchmarks/demo").json()
    case = d["cases"][0]
    assert case["img"].startswith("/api/media/data/")
    assert case["ref_img"].startswith("/api/media/data/")
    assert client.get(case["img"]).status_code == 200
    assert d["references_map"]["cam"]["img"] == case["ref_img"]


def test_unknown_dataset_is_404(bench):
    client, _ = bench()
    assert client.get("/api/benchmarks/nope").status_code == 404


def test_candidates_reports_the_dirs_it_scanned(bench):
    """An empty candidate list must be explainable — for 39T it means the
    unlabelled moments were never extracted, not that the endpoint is broken."""
    client, _ = bench()
    r = client.get("/api/benchmarks/demo/candidates").json()
    assert r["dirs"] and all(d.startswith("data/") for d in r["dirs"])
    assert "n_total" in r


# ---------------------------------------------------------------- editing

def test_put_case_saves_and_marks_it_human_reviewed(bench):
    client, _ = bench()
    case = client.get("/api/benchmarks/demo").json()["cases"][0]
    case["instances"][0]["bbox"] = [160, 110, 240, 190]
    case["note"] = "corrected by hand"
    r = client.put("/api/benchmarks/demo/cases/a_anom", json=case)
    assert r.status_code == 200, r.text
    saved = r.json()["case"]
    assert saved["instances"][0]["bbox"] == [160, 110, 240, 190]
    assert saved["note"] == "corrected by hand"
    assert saved["reviewed"]                       # the point of the screen
    assert r.json()["dataset"]["n_reviewed"] == 1
    # and it is on disk, not just in the response
    assert benchmarks.load("demo")[1]["cases"][0]["instances"][0]["bbox"] \
        == [160, 110, 240, 190]


def test_put_case_backs_up_the_previous_version(bench):
    client, _ = bench()
    case = client.get("/api/benchmarks/demo").json()["cases"][0]
    client.put("/api/benchmarks/demo/cases/a_anom", json={**case, "note": "v2"})
    assert list(benchmarks.BACKUPS_DIR.glob("demo-*.json"))


@pytest.mark.parametrize("mutate,expected", [
    (lambda c: c.update(instances=[{"type": "object", "bbox": [10, 10, 5, 5]}]), 400),
    (lambda c: c.update(instances=[{"type": "object", "bbox": [1, 2, 3]}]), 400),
    (lambda c: c.update(reference="nosuchcam"), 400),
    (lambda c: c.update(has_anomaly=False), 400),      # clean but keeps its box
])
def test_put_case_rejects_unusable_edits(bench, mutate, expected):
    client, _ = bench()
    case = client.get("/api/benchmarks/demo").json()["cases"][0]
    mutate(case)
    r = client.put("/api/benchmarks/demo/cases/a_anom", json=case)
    assert r.status_code == expected, r.text
    # the file must be untouched by a rejected edit
    assert benchmarks.load("demo")[1]["cases"][0]["has_anomaly"] is True


def test_put_unknown_case_is_404(bench):
    client, _ = bench()
    case = client.get("/api/benchmarks/demo").json()["cases"][0]
    assert client.put("/api/benchmarks/demo/cases/ghost", json=case).status_code == 404


def test_marking_an_anomalous_case_clean_drops_its_boxes(bench):
    """The legitimate way to declare a mislabelled case clean: send no
    instances. A clean case that keeps boxes is what gets rejected above."""
    client, _ = bench()
    case = client.get("/api/benchmarks/demo").json()["cases"][0]
    r = client.put("/api/benchmarks/demo/cases/a_anom",
                   json={**case, "has_anomaly": False, "instances": []})
    assert r.status_code == 200
    assert r.json()["case"]["types"] == []
    assert r.json()["dataset"]["n_anomalous"] == 0


def test_post_and_delete_a_case(bench):
    client, _ = bench()
    new = {"id": "c_extra", "image": _img("extra.jpg"), "reference": "cam",
           "source": "real", "has_anomaly": False, "note": "added from the UI",
           "instances": []}
    r = client.post("/api/benchmarks/demo/cases", json=new)
    assert r.status_code == 200, r.text
    assert r.json()["dataset"]["n_cases"] == 3
    assert client.post("/api/benchmarks/demo/cases", json=new).status_code == 409
    assert client.delete("/api/benchmarks/demo/cases/c_extra").status_code == 200
    assert client.delete("/api/benchmarks/demo/cases/c_extra").status_code == 404
    assert benchmarks.summary(*benchmarks.load("demo"))["n_cases"] == 2


# ---------------------------------------------------------------- runs

def test_localize_run_scores_without_touching_ollama(bench):
    """No model, no Ollama, seconds per case: this is the mode that makes
    threshold tuning a measurement instead of a guess."""
    client, _ = bench()
    r = client.post("/api/benchmarks/runs", json={
        "dataset": "demo", "mode": "localize", "localizer": "photo",
        "params": {"PERSON_FILTER": False}})
    assert r.status_code == 200, r.text
    assert r.json()["job_id"] is None                 # no job queue involved
    st = wait_run(client, r.json()["run_id"])
    assert st["run"]["status"] == "completed"
    assert st["run"]["progress"] == {"done": 2, "total": 2}
    score = st["score"]
    assert score["mode"] == "localize"
    assert score["localization"]["inst_total"] == 1
    assert [c["id"] for c in score["cases"]] == ["a_anom", "b_clean"]


def test_full_run_reproduces_the_frame_matrix(bench):
    client, impl = bench([CLEAN_CROP] * 40)
    r = client.post("/api/benchmarks/runs", json={
        "dataset": "demo", "mode": "full", "script": "vlm_05",
        "model": "qwen3-vl:8b-instruct", "localizer": "photo",
        "params": {"PERSON_FILTER": False}})
    assert r.status_code == 200, r.text
    run_id, job_id = r.json()["run_id"], r.json()["job_id"]
    st = wait_run(client, run_id)
    assert st["run"]["status"] == "completed"
    # the judge said NO to everything, so nothing is flagged: the anomalous case
    # is a miss and the clean one is a true negative
    assert st["score"]["frame"]["counts"] == {"TP": 0, "FP": 0, "TN": 1, "FN": 1}
    assert st["score"]["objects"]["inst_detected"] == 0
    # and it is an ordinary job, browsable in the Results screen
    assert client.get(f"/api/jobs/{job_id}").json()["config"]["bench"]["run_id"] == run_id


def test_full_run_uses_each_case_own_reference(bench):
    """The 39T protocol has one reference per camera. The run must pass the
    case's reference, not the first one it saw."""
    client, _ = bench([CLEAN_CROP] * 40)
    ref2 = _img("ref2.jpg", rects=[(0, 0, 60, 60)])
    _, doc = benchmarks.load("demo")
    doc["references"]["cam2"] = ref2
    doc["cases"][1]["reference"] = "cam2"
    benchmarks.save("demo", doc)

    r = client.post("/api/benchmarks/runs", json={
        "dataset": "demo", "mode": "full", "model": "qwen3-vl:8b-instruct",
        "params": {"PERSON_FILTER": False}})
    run_id = r.json()["run_id"]
    wait_run(client, run_id)
    job = client.get(f"/api/jobs/{r.json()['job_id']}").json()
    assert job["config"]["n_references"] == 2
    rows = client.get(f"/api/benchmarks/runs/{run_id}").json()["score"]["cases"]
    assert [row["reference"] for row in rows] == ["cam", "cam2"]


def test_run_on_a_subset_of_cases(bench):
    client, _ = bench()
    r = client.post("/api/benchmarks/runs", json={
        "dataset": "demo", "mode": "localize", "cases": ["b_clean"]})
    st = wait_run(client, r.json()["run_id"])
    assert st["run"]["config"]["subset"] is True
    assert [c["id"] for c in st["score"]["cases"]] == ["b_clean"]


def test_report_md_is_written_and_served(bench):
    client, _ = bench()
    r = client.post("/api/benchmarks/runs", json={"dataset": "demo",
                                                 "mode": "localize"})
    run_id = r.json()["run_id"]
    wait_run(client, run_id)
    md = client.get(f"/api/benchmarks/runs/{run_id}/report.md").text
    assert "Localization only" in md and "demo" in md
    assert "blob canary" in md


def test_editing_the_dataset_marks_earlier_runs_stale(bench):
    """A run keeps the numbers it measured; what it must not do is silently
    compare as if the labels had not moved."""
    client, _ = bench()
    r = client.post("/api/benchmarks/runs", json={"dataset": "demo",
                                                 "mode": "localize"})
    run_id = r.json()["run_id"]
    st = wait_run(client, run_id)
    assert st["stale"] is False
    before = st["score"]["localization"]["inst_total"]

    case = client.get("/api/benchmarks/demo").json()["cases"][0]
    case["instances"][0]["bbox"] = [0, 0, 40, 40]
    client.put("/api/benchmarks/demo/cases/a_anom", json=case)

    after = client.get(f"/api/benchmarks/runs/{run_id}").json()
    assert after["stale"] is True
    assert after["score"]["localization"]["inst_total"] == before
    assert after["score"]["cases"][0]["instances"][0]["bbox"] == [150, 100, 250, 200]
    row = next(x for x in client.get("/api/benchmarks/runs").json()["runs"]
               if x["run_id"] == run_id)
    assert row["stale"] is True and row["headline"]["mode"] == "localize"


def test_runs_index_carries_the_headline_numbers(bench):
    client, _ = bench()
    r = client.post("/api/benchmarks/runs", json={"dataset": "demo",
                                                 "mode": "localize"})
    wait_run(client, r.json()["run_id"])
    row = client.get("/api/benchmarks/runs").json()["runs"][0]
    assert set(row["headline"]) >= {"localized", "strict", "regions", "recall"}
    assert row["config"]["localizer"] == "photo"


def test_delete_run(bench):
    client, _ = bench()
    r = client.post("/api/benchmarks/runs", json={"dataset": "demo",
                                                 "mode": "localize"})
    run_id = r.json()["run_id"]
    wait_run(client, run_id)
    assert client.delete(f"/api/benchmarks/runs/{run_id}").status_code == 200
    assert client.get(f"/api/benchmarks/runs/{run_id}").status_code == 404
    assert client.delete(f"/api/benchmarks/runs/{run_id}").status_code == 404


@pytest.mark.parametrize("payload,code", [
    ({"dataset": "ghost"}, 404),
    ({"dataset": "demo", "mode": "sideways"}, 400),
    ({"dataset": "demo", "localizer": "telepathy"}, 400),
    ({"dataset": "demo", "cases": ["nope"]}, 400),
    ({"dataset": "demo", "mode": "localize", "script": "vlm_01"}, 400),
])
def test_bad_run_requests(bench, payload, code):
    client, _ = bench()
    assert client.post("/api/benchmarks/runs", json=payload).status_code == code


def test_full_run_model_missing_is_409(bench):
    client, _ = bench(models=("something-else",))
    r = client.post("/api/benchmarks/runs", json={
        "dataset": "demo", "mode": "full", "model": "qwen3-vl:8b-instruct"})
    assert r.status_code == 409

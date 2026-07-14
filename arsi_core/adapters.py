"""Uniform run_frame() over the five vlm_0x scripts (docs/SPEC.md "Pipeline
adapters").

The scripts are imported as modules and configured by temporarily setting
their module attributes (MODEL_NAME, PROMPT, thresholds...) — including their
`ollama` attribute, which is swapped for the caller's OllamaClient so every
call shares one timeout/error policy (and tests inject a fake). Jobs run one
at a time (the Ollama server is the bottleneck), so this is safe; `configured`
restores every attribute afterwards either way.
"""
import hashlib
import importlib
import json
import re
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

from .errors import ParseError, FrameError
from .schema import Detection, FrameResult, guess_type

SCRIPTS = {
    "vlm_01": "vlm_01_single_image",
    "vlm_02": "vlm_02_reference_compare",
    "vlm_03": "vlm_03_bounding_box",
    "vlm_04": "vlm_04_hybrid_detect",
    "vlm_05": "vlm_05_reference_diff",
}
NEEDS_REFERENCE = {"vlm_02": True, "vlm_05": True, "vlm_04": False}

_modules = {}
_detectors = {}     # (weights, classes) -> loaded YOLO-World model (vlm_04)


def get_module(script: str):
    if script not in SCRIPTS:
        raise ValueError(f"unknown script '{script}' (choose from {sorted(SCRIPTS)})")
    if script not in _modules:
        _modules[script] = importlib.import_module(SCRIPTS[script])
    return _modules[script]


def default_prompt(script: str) -> str:
    return get_module(script).PROMPT


@contextmanager
def configured(module, **attrs):
    saved = {k: getattr(module, k) for k in attrs}
    try:
        for k, v in attrs.items():
            setattr(module, k, v)
        yield module
    finally:
        for k, v in saved.items():
            setattr(module, k, v)


def _module_overrides(module, params: dict) -> dict:
    """params keys that name UPPER_CASE module attributes become overrides
    (the generic 'advanced params' mechanism: DIFF_THRESHOLD, PERSON_FILTER,
    DETECTOR_CONF...)."""
    out = {}
    for k, v in (params or {}).items():
        if k.isupper() and hasattr(module, k):
            out[k] = v
    return out


def _open_image(path) -> Image.Image:
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception as exc:
        raise FrameError(f"cannot read image {path}: {exc}") from exc


def _chat_text(client, model: str, prompt: str, images: list, module) -> str:
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [str(p) for p in images]}],
        think=False,
        options={"num_ctx": module.NUM_CTX, "num_predict": module.NUM_PREDICT,
                 "temperature": module.TEMPERATURE},
    )
    message = response.get("message", {}) if isinstance(response, dict) \
        else getattr(response, "message", {})
    get = message.get if isinstance(message, dict) else lambda k, d="": getattr(message, k, d)
    return (get("content", "") or "").strip() or (get("thinking", "") or "").strip()


# ---------------------------------------------------------------------------
# vlm_01 / vlm_02 — structured whole-frame report
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"^\s*(GRAFFITI|VANDALISM|FORGOTTEN OBJECT)\s*:\s*(yes|no)\b\s*-?\s*(.*)$",
    re.IGNORECASE)
_ITEM_RE = re.compile(r"^\s*-\s+(.*?)(?:\(([^)]*)\))?\s*$")
_SEVERITY_RE = re.compile(r"^\s*SEVERITY\s*:\s*(\d)", re.IGNORECASE)

_TYPE_OF_SECTION = {"GRAFFITI": "graffiti", "VANDALISM": "damage",
                    "FORGOTTEN OBJECT": "object"}


def parse_structured_report(text: str):
    """Parse the GRAFFITI/VANDALISM/FORGOTTEN OBJECT format of vlm_01/02.
    Raises ParseError when none of the three section lines is present."""
    sections, detections, severity = {}, [], None
    current, forgotten_note = None, ""
    for line in text.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            name = m.group(1).upper()
            yes = m.group(2).lower() == "yes"
            sections[name] = yes
            note = m.group(3).strip()
            current = name if name == "FORGOTTEN OBJECT" else None
            if yes and name != "FORGOTTEN OBJECT":
                detections.append(Detection(label=note or name.lower(),
                                            type=_TYPE_OF_SECTION[name]))
            elif yes:
                forgotten_note = note    # fallback if no bullet list follows
            continue
        s = _SEVERITY_RE.match(line)
        if s:
            severity = int(s.group(1))
            current = None
            continue
        if line.strip().upper().startswith("DESCRIPTION"):
            current = None
            continue
        if current == "FORGOTTEN OBJECT":
            m = _ITEM_RE.match(line)
            if m and m.group(1).strip():
                detections.append(Detection(label=m.group(1).strip(" .,-"),
                                            type="object",
                                            zone=(m.group(2) or "").strip() or None))
    if not sections:
        raise ParseError("no GRAFFITI/VANDALISM/FORGOTTEN OBJECT line in reply", raw=text)
    # model answered "FORGOTTEN OBJECT: yes - a phone" inline, without the
    # bulleted list: keep the name rather than an empty detections list
    if sections.get("FORGOTTEN OBJECT") and not any(d.type == "object" for d in detections):
        detections.append(Detection(label=forgotten_note or "unnamed object",
                                    type="object"))
    if severity is not None:
        for d in detections:
            d.severity = severity
    anomaly = any(sections.values())
    return anomaly, detections


def _run_whole_frame(script, image, reference, model, prompt, params, client):
    module = get_module(script)
    images = [reference, image] if script == "vlm_02" else [image]
    if script == "vlm_02" and reference is None:
        raise FrameError("vlm_02 needs a reference image")
    with configured(module, ollama=client, **_module_overrides(module, params)):
        text = _chat_text(client, model, prompt, images, module)
    anomaly, detections = parse_structured_report(text)
    return FrameResult(frame_id=Path(image).stem, image=str(image),
                       anomaly=anomaly, detections=detections, raw_response=text)


# ---------------------------------------------------------------------------
# vlm_03 — whole-frame JSON bounding boxes
# ---------------------------------------------------------------------------

_LABEL_TO_TYPE = {"graffiti": "graffiti", "vandalism": "damage",
                  "forgotten_object": "object", "forgotten_left_object": "object"}


def parse_bbox_json(text: str):
    """Tolerant-but-honest version of vlm_03's parse_json. Unlike the script
    (which returns [] on any decode error, silently scoring a broken reply as
    'clean'), an unusable reply raises ParseError so the runner retries.
    Tolerated deviations, all observed in real model output:
    - Markdown code fences around the JSON;
    - a single object instead of an array, or {"detections": [...]};
    - an array TRUNCATED mid-object by the generation limit -> repaired by
      cutting at the last complete element (partial results beat a failure).
    """
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    start = t.find("[")
    obj_start = t.find("{")
    if start == -1 and obj_start == -1:
        raise ParseError("no JSON in reply", raw=text)
    if start == -1 or (obj_start != -1 and obj_start < start):
        # a bare object: either a wrapper ({"detections": [...]}) or one item
        end = t.rfind("}")
        try:
            data = json.loads(t[obj_start:end + 1])
        except json.JSONDecodeError as exc:
            raise ParseError(f"invalid JSON: {exc}", raw=text) from exc
        if isinstance(data, dict):
            for key in ("detections", "anomalies", "items", "results"):
                if isinstance(data.get(key), list):
                    return data[key]
            return [data]
        data = [data]
        return data
    end = t.rfind("]")
    try:
        data = json.loads(t[start:end + 1] if end > start else t[start:])
    except json.JSONDecodeError:
        # likely truncated by the token limit (note: rfind("]") may have hit an
        # inner bbox array) — repair on the FULL tail: keep complete elements.
        tail = t[start:]
        cut = tail.rfind("}")
        if cut == -1:
            raise ParseError("unparseable JSON array", raw=text)
        try:
            data = json.loads(tail[:cut + 1] + "]")
        except json.JSONDecodeError as exc:
            raise ParseError(f"invalid JSON: {exc}", raw=text) from exc
    if not isinstance(data, list):
        raise ParseError("JSON is not an array", raw=text)
    return data


def scale_bboxes(items: list, width: int, height: int):
    """Models answer 0-1 normalized, 0-1000 (Qwen default) or absolute pixels
    despite the prompt; detect the scale from the largest coordinate (same
    heuristic vlm_03 ships)."""
    coords = [c for d in items for c in (d.get("bbox") or [])
              if isinstance(c, (int, float))]
    max_c = max(coords) if coords else 0.0
    if max_c <= 1.0:
        sx, sy = float(width), float(height)
    elif max_c <= 1000.0:
        sx, sy = width / 1000.0, height / 1000.0
    else:
        sx = sy = 1.0
    out = []
    for d in items:
        bbox = d.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        out.append(Detection(
            label=str(d.get("label", "?")),
            type=_LABEL_TO_TYPE.get(str(d.get("label", "")).lower(), "unknown"),
            bbox=[round(bbox[0] * sx), round(bbox[1] * sy),
                  round(bbox[2] * sx), round(bbox[3] * sy)],
            severity=d.get("severity") if isinstance(d.get("severity"), int) else None))
    return out


def _run_vlm03(image, model, prompt, params, client):
    module = get_module("vlm_03")
    img = _open_image(image)
    overrides = _module_overrides(module, params)
    # A busy frame can need more than the script's 512-token budget for its
    # JSON array; a truncated array was the main cause of failed frames.
    overrides.setdefault("NUM_PREDICT", max(module.NUM_PREDICT, 1024))
    with configured(module, ollama=client, **overrides):
        text = _chat_text(client, model, prompt, [image], module)
    detections = scale_bboxes(parse_bbox_json(text), img.width, img.height)
    return FrameResult(frame_id=Path(image).stem, image=str(image),
                       anomaly=len(detections) > 0, detections=detections,
                       raw_response=text)


# ---------------------------------------------------------------------------
# vlm_04 — YOLO-World localizer + VLM confirmation
# ---------------------------------------------------------------------------

def _get_detector(module):
    key = (module.DETECTOR_WEIGHTS, tuple(module.DETECTOR_CLASSES))
    if key not in _detectors:
        try:
            _detectors[key] = module.load_detector(module.DETECTOR_WEIGHTS,
                                                   module.DETECTOR_CLASSES)
        except SystemExit as exc:   # the script exits on load failure; a library must not
            raise FrameError(f"YOLO-World detector unavailable "
                             f"({module.DETECTOR_WEIGHTS})") from exc
    return _detectors[key]


def _run_vlm04(image, reference, model, prompt, params, client):
    module = get_module("vlm_04")
    overrides = _module_overrides(module, params)
    overrides.update(ollama=client, MODEL_NAME=model, PROMPT=prompt)
    with configured(module, **overrides):
        detector = _get_detector(module)
        candidates = module.detect(detector, str(image), module.DETECTOR_CLASSES,
                                   module.DETECTOR_CONF, module.IMGSZS)
        if module.USE_REFERENCE and reference:
            ref_dets = module.detect(detector, str(reference), module.DETECTOR_CLASSES,
                                     module.DETECTOR_CONF, module.IMGSZS)
            candidates = module.filter_new(candidates, ref_dets, module.REFERENCE_IOU)
        img = _open_image(image)
        detections, raw_lines = [], []
        for cand in candidates:
            confirmed, label = True, cand["label"]
            if module.USE_VLM:
                confirmed, label = module.confirm_with_vlm(
                    img, cand, module.CROP_MARGIN, module.CROP_CONTEXT)
                raw_lines.append(f"{cand['label']} {cand['bbox']} -> "
                                 f"{'YES' if confirmed else 'NO'} {label}")
            if confirmed:
                detections.append(Detection(
                    label=label or cand["label"], type="object",
                    bbox=[round(v) for v in cand["bbox"]],
                    confidence=round(cand["confidence"], 3)))
    return FrameResult(frame_id=Path(image).stem, image=str(image),
                       anomaly=len(detections) > 0, detections=detections,
                       raw_response="\n".join(raw_lines))


# ---------------------------------------------------------------------------
# vlm_05 — reference-diff localizer + crop judge (mirrors benchmark/run_benchmark)
# ---------------------------------------------------------------------------

def prompt_fingerprint(model: str, prompt: str) -> str:
    """Identical to benchmark/run_benchmark.prompt_fingerprint, so the app
    reuses the existing verdict cache."""
    return hashlib.sha1((model + "||" + prompt).encode("utf-8")).hexdigest()[:12]


def _run_vlm05(image, reference, model, prompt, params, client, cache, mask_hash):
    module = get_module("vlm_05")
    if reference is None:
        raise FrameError("vlm_05 needs a reference image")
    overrides = _module_overrides(module, params)
    overrides.update(ollama=client, MODEL_NAME=model, PROMPT=prompt)
    with configured(module, **overrides):
        ref_img = _open_image(reference)
        img = _open_image(image)
        if img.size != ref_img.size:
            img = img.resize(ref_img.size)
        regions, loc = module.localize(str(reference), str(image))

        fp = prompt_fingerprint(model, prompt)
        ref_key, img_key = Path(reference).name, Path(image).name
        mask_part = f"|mask:{mask_hash}" if mask_hash else ""
        kept, raw_lines = [], []
        for r in regions:
            key = f"{ref_key}|{img_key}|{r['bbox']}|{model}|{fp}{mask_part}"
            hit = cache.get(key) if cache is not None else None
            if hit is not None:
                is_obj, label = hit["yes"], hit["label"]
            else:
                is_obj, label = module.classify_with_vlm(
                    img, ref_img, r, module.CROP_MARGIN, module.CROP_CONTEXT)
                if cache is not None:
                    cache.put(key, {"yes": bool(is_obj), "label": label})
            if module.is_non_anomaly(label) or module.is_implausible(label, r["area"]):
                is_obj = False
            raw_lines.append(f"{r['bbox']} [{r.get('channel', 'photo')}] -> "
                             f"{'YES' if is_obj else 'NO'} {label}")
            if is_obj:
                r = dict(r, vlm_label=label)
                kept.append(r)
        kept = module.dedupe_regions(kept)

    detections = [Detection(label=r["vlm_label"], type=guess_type(r["vlm_label"]),
                            bbox=list(r["bbox"]), channel=r.get("channel"))
                  for r in kept]
    raw = (f"localizer: {loc}\n" + "\n".join(raw_lines))
    return FrameResult(frame_id=Path(image).stem, image=str(image),
                       anomaly=len(detections) > 0, detections=detections,
                       raw_response=raw)


# ---------------------------------------------------------------------------

def run_frame(script: str, image, reference=None, model: str = None,
              prompt: str = None, params: dict = None, client=None,
              cache=None, mask_hash: str = "") -> FrameResult:
    """Run one pipeline on one frame. Raises the docs/SPEC.md error taxonomy;
    the runner owns retries and batch isolation."""
    module = get_module(script)
    model = model or module.MODEL_NAME
    prompt = prompt or module.PROMPT
    params = params or {}
    if client is None:
        from .ollama_client import OllamaClient
        client = OllamaClient(timeout=float(params.get("timeout_s", 120)))
    if not Path(image).exists():
        raise FrameError(f"image not found: {image}")
    if reference is not None and not Path(reference).exists():
        raise FrameError(f"reference not found: {reference}")

    if script in ("vlm_01", "vlm_02"):
        return _run_whole_frame(script, image, reference, model, prompt, params, client)
    if script == "vlm_03":
        return _run_vlm03(image, model, prompt, params, client)
    if script == "vlm_04":
        return _run_vlm04(image, reference, model, prompt, params, client)
    return _run_vlm05(image, reference, model, prompt, params, client, cache, mask_hash)

#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - tools/dinomaly2.py
#  Dinomaly2 (arXiv 2510.17611, preview code May 2026) - the deltas over the
#  CVPR-2025 Dinomaly we already have in tools/dinomaly.py.
#
#  WHY BOTHER. Our Dinomaly verdict (docs/DECISIONS.md) rests on a cost
#  argument - "one model to train per camera". Dinomaly2 is the authors' own
#  answer to exactly that: a single framework covering, in their words,
#  "single-class, multi-class, inference-unified multi-class and few-shot"
#  settings. One of its five elements, context-aware recentering, is also the
#  most plausible fix for the failure we actually observe on our fleet - a
#  global offset between cameras and between sessions (lighting, exposure).
#  So this is not "try the newer paper", it is a targeted hypothesis.
#
#  WHAT CHANGES, read from the reference implementation (guojiajeremy/Dinomaly2,
#  models/uad.py, models/vision_transformer.py, utils.py, dinomaly_2D.py):
#
#    1. CONTEXT-AWARE RECENTERING (new in v2, the headline). Each encoder target
#       token has the CLS token of its own image subtracted, then LayerNorm:
#           en = LayerNorm(patch_tokens - cls_token)
#       The CLS token is a global summary of the image. Subtracting it removes
#       whatever is common to the whole frame - which is precisely what a
#       different camera, a different exposure or a different hour of the day
#       changes. Our v1 has no equivalent.
#    2. LOOSE LOSS with gradient down-weighting, not masking. v1 (ours) optimises
#       the hardest 10% of points and throws the rest away. v2 keeps a global
#       cosine over the whole map and merely SCALES the gradient of the easy 90%
#       by 0.1, with the ratio warmed up over the first 1000 steps.
#    3. NOISY BOTTLENECK, different shape and twice the noise: Linear(D->256) ->
#       Linear(256->4D) -> GELU -> Linear(4D->D), dropout 0.4 (ours: 0.2, and a
#       single D->4D->D MLP).
#    4. REVERSED DECODER PAIRING. The decoder block outputs are reversed before
#       being grouped, so the LAST decoder block reconstructs the EARLY encoder
#       layers. Ours pairs them in order. This is a real architectural detail,
#       not a cosmetic one.
#    5. Linear attention with an explicit eps in the normaliser (their
#       LinearAttention2). Numerically identical to ours in practice - kept only
#       so the list of five matches the paper.
#
#  WHAT IS STILL SIMPLIFIED, same disclaimer as v1: ViT-S/14-reg rather than a
#  larger backbone, our own crop-based training loop, our fixed 1120 px input
#  instead of their 448/392. One further deviation worth naming because it is
#  not cosmetic: the reference keeps the CLS and register tokens inside the
#  sequence the bottleneck and decoder see, and strips them only at the end, so
#  their decoder has a global token to attend to. Ours drops them at encode
#  time and the decoder sees patch tokens only. This answers "does v2 help ON
#  OUR FLEET", not "does v2 reproduce their MVTec table".
#
#  Checkpoints live next to v1 as weights/dinomaly2_<camera>.pt, so both
#  variants can be scored on the same benchmark without either overwriting the
#  other.
# =============================================================================

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import tools.dinomaly as dm

# Ablation switch. Context-aware recentering is the one element of Dinomaly2
# that plausibly explains a cross-camera result, so it must be possible to turn
# it off without touching anything else - otherwise a null result cannot be
# attributed. DINOMALY2_CAR=0 trains and scores the same architecture with the
# raw encoder targets; checkpoints then land under dinomaly2nocar_<camera>.pt.
CAR = os.environ.get("DINOMALY2_CAR", "1") != "0"
TAG = "dinomaly2" if CAR else "dinomaly2nocar"

PATCH = dm.PATCH
INPUT_W = dm.INPUT_W
N_GROUPS = dm.N_GROUPS
DEVICE = dm.DEVICE
WEIGHTS_DIR = dm.WEIGHTS_DIR

DROPOUT = 0.4          # their default; v1 here used 0.2
LL_RATIO = 0.9         # fraction of points whose gradient is damped
LL_FACTOR = 0.1        # by how much
LL_WARMUP = 1000       # steps over which LL_RATIO ramps in


# --- encoding: same features as v1, plus the CLS token -----------------------

def encode(path: str, size=None):
    """(layers[L, N, D], cls[L, D], valid[N], (gh, gw)).

    Identical to dm.encode() except that the class token is returned as well -
    context-aware recentering needs it, and v1 discarded it.
    """
    size = size or Image.open(path).size
    gw, gh = dm.grid_size(size)
    img = Image.open(path).convert("RGB").resize((gw * PATCH, gh * PATCH),
                                                 Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / \
        np.array([0.229, 0.224, 0.225], np.float32)
    t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(DEVICE)
    with torch.no_grad():
        outs = dm._load_encoder().get_intermediate_layers(
            t, n=list(dm.LAYERS), reshape=False, return_class_token=True,
            norm=True)
    layers = torch.cat([o[0] for o in outs], 0).cpu()          # [L, N, D]
    cls = torch.cat([o[1] for o in outs], 0).cpu()             # [L, D]
    lum = np.asarray(Image.open(path).convert("L").resize((gw, gh), Image.BILINEAR),
                     dtype=np.float32).reshape(-1)
    return layers, cls, torch.from_numpy(lum >= dm.BLACK_LEVEL), (gh, gw)


def group_targets(layers: torch.Tensor, cls: torch.Tensor = None):
    """The N_GROUPS reconstruction targets, context-aware recentered.

    layers [L, N, D] (or [B, L, N, D]), cls [L, D] (or [B, L, D]).
    Recentering happens per layer BEFORE the group mean, mirroring the reference
    implementation, which recenters the fused encoder feature with the fused CLS
    of the same layers - identical up to the order of two linear operations.
    """
    per = len(dm.LAYERS) // N_GROUPS
    batched = layers.dim() == 4
    out = []
    for i in range(N_GROUPS):
        sl = layers[..., i * per:(i + 1) * per, :, :] if batched \
            else layers[i * per:(i + 1) * per]
        g = sl.mean(-3)                                        # [.., N, D]
        if cls is not None and CAR:
            c = cls[..., i * per:(i + 1) * per, :] if batched \
                else cls[i * per:(i + 1) * per]
            g = g - c.mean(-2).unsqueeze(-2)                    # broadcast over N
            g = F.layer_norm(g, (g.shape[-1],), eps=1e-8)
        out.append(g)
    return out


def bottleneck_input(layers: torch.Tensor):
    """What the bottleneck sees: the mean over all target layers (their
    fuse_layer_bottleneck = every layer). v1 used the sum; cosine losses do not
    care, but the bottleneck's Linear layers do, so the two variants must not
    share this."""
    return layers.mean(-3)


# --- the trainable half ------------------------------------------------------

class Dinomaly2(nn.Module):
    def __init__(self, dim=384, depth=8, heads=6, dropout=DROPOUT):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(dim, 256), nn.Dropout(dropout),
            nn.Linear(256, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout))
        self.blocks = nn.ModuleList([dm.Block(dim, heads) for _ in range(depth)])
        self.depth = depth

    def forward(self, x):
        x = self.bottleneck(x)
        outs = []
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        outs = outs[::-1]                    # element 4: reversed pairing
        per = self.depth // N_GROUPS
        return [torch.stack(outs[i * per:(i + 1) * per]).mean(0)
                for i in range(N_GROUPS)]


# --- loss --------------------------------------------------------------------

def _modify_grad(x, inds, factor):
    inds = inds.expand_as(x)
    out = x.clone()
    out[inds] = out[inds] * factor
    return out


def loose_loss(preds, targets, valid=None, p: float = LL_RATIO,
               factor: float = LL_FACTOR):
    """Global cosine distance per sample, with the easy points' gradient damped.

    The difference with v1's hard_cosine_loss is where the selection acts. v1
    selects in the LOSS: the easy points contribute nothing at all. v2 selects in
    the GRADIENT: every point contributes to the value, but the easy ones push
    ten times less hard. The model therefore still knows what "already correct"
    looks like, which is what lets a single decoder stay calibrated over
    heterogeneous inputs instead of chasing whichever patch happens to be worst.
    """
    total = 0.0
    for pred, tgt in zip(preds, targets):
        a = tgt.detach()
        b = pred
        if valid is not None:
            m = valid if valid.dim() == b.dim() - 1 else valid.unsqueeze(0)
            a = a * m.unsqueeze(-1)
            b = b * m.unsqueeze(-1)
        with torch.no_grad():
            point = (1 - F.cosine_similarity(a, b, dim=-1)).unsqueeze(-1)
        k = max(1, int(point.numel() * (1 - p)))
        thresh = torch.topk(point.reshape(-1), k=k).values[-1]
        flat_a = a.reshape(a.shape[0], -1) if a.dim() == 3 else a.reshape(1, -1)
        flat_b = b.reshape(b.shape[0], -1) if b.dim() == 3 else b.reshape(1, -1)
        total = total + torch.mean(1 - F.cosine_similarity(flat_a, flat_b, dim=-1))
        if b.requires_grad:
            b.register_hook(lambda g, i=point < thresh, f=factor: _modify_grad(g, i, f))
    return total / len(preds)


def warm_ratio(step: int, p_final: float = LL_RATIO, warmup: int = LL_WARMUP):
    return min(p_final * max(step, 1) / warmup, p_final)


# --- score -------------------------------------------------------------------

@torch.no_grad()
def anomaly_map(model: nn.Module, layers: torch.Tensor, cls: torch.Tensor,
                valid: torch.Tensor, grid) -> np.ndarray:
    model.eval()
    layers, cls = layers.to(DEVICE), cls.to(DEVICE)
    targets = group_targets(layers, cls)
    preds = model(bottleneck_input(layers)[None])
    score = torch.zeros(layers.shape[1], device=DEVICE)
    for p, t in zip(preds, targets):
        score = score + (1.0 - F.cosine_similarity(p[0], t, dim=-1))
    score[~valid.to(DEVICE)] = 0.0
    gh, gw = grid
    return score.cpu().numpy().reshape(gh, gw)


# --- checkpoints -------------------------------------------------------------

def checkpoint_path(camera: str = "tram_1762"):
    return WEIGHTS_DIR / f"{TAG}_{camera}.pt"


def save(model, camera, meta: dict):
    WEIGHTS_DIR.mkdir(exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state": state, "meta": meta, "variant": TAG, "car": CAR,
                "input_w": INPUT_W, "layers": list(dm.LAYERS),
                "dropout": DROPOUT, "ll_ratio": LL_RATIO, "ll_factor": LL_FACTOR},
               checkpoint_path(camera))


def load(camera: str = "tram_1762"):
    p = checkpoint_path(camera)
    if not p.exists():
        raise FileNotFoundError(
            f"no {TAG} checkpoint for '{camera}' ({p}). Train one first:\n"
            f"    DINOMALY2_CAR={int(CAR)} python tools/dinomaly2_train.py "
            f"--camera {camera}")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if ck.get("input_w") != INPUT_W:
        raise ValueError(f"checkpoint was trained at INPUT_W={ck.get('input_w')} "
                         f"but INPUT_W is {INPUT_W}")
    model = Dinomaly2()
    model.load_state_dict(ck["state"])
    return model.to(DEVICE).eval(), ck.get("meta", {})

#!/usr/bin/env python3
# =============================================================================
#  ARSI-VLM - tools/dinomaly.py
#  Dinomaly (CVPR 2025) - the model itself. Training lives in dinomaly_train.py,
#  the vlm_05-compatible wrapper in dinomaly_localizer.py.
#
#  WHY a third localizer at all. photo compares PIXELS to one reference frame,
#  photo+dino compares FEATURES to one reference frame. Both inherit that
#  reference's session: its lighting, its exposure, its time of day. The measured
#  consequence is in benchmark/README - a cross-session empty frame still yields
#  12 kept regions after the DINOv2 gate. Dinomaly needs NO reference at
#  inference: it learns what this camera looks like when nothing is wrong, from
#  MANY nominal frames, and flags what it cannot reconstruct.
#
#  HOW it works (Dinomaly, CVPR 2025 - "the less is more philosophy"):
#    encoder  frozen DINOv2, 8 middle layers (the same backbone dino_localizer
#             already downloads - nothing new to fetch)
#    bottleneck  one MLP with heavy dropout ("noisy bottleneck"): the noise is
#             what stops the network from learning the identity function
#    decoder  transformer blocks with LINEAR attention, deliberately "unfocused"
#             so a token cannot copy one specific token of the encoder
#    loss     cosine between encoder and decoder features, on the HARDEST points
#             only - a decoder that reconstructs everything perfectly also
#             reconstructs anomalies perfectly, which is the failure mode of
#             every reconstruction-based detector
#    score    1 - cos(encoder, decoder) per patch, summed over the two layer
#             groups: what the model could not rebuild is what it never saw.
#
#  WHAT IS SIMPLIFIED vs the paper, so the numbers are read for what they are:
#    - linear attention here is the elu+1 kernel, not their exact variant;
#    - hard mining keeps the worst HARD_Q quantile instead of their gradient
#      down-weighting schedule;
#    - no multi-scale training crops (our camera is fixed - the scale is fixed);
#    - ViT-S/14-reg, not ViT-B/14: it is what is already cached on this machine.
#  This is enough to answer "does the reference-free direction pay?", not to
#  reproduce their MVTec table.
#
#  The encoder is FROZEN, which is what makes this trainable on a CPU laptop:
#  its features are computed once and cached, and only the bottleneck + decoder
#  (~14 M params) see gradients.
# =============================================================================

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

PATCH = 14
INPUT_W = int(os.environ.get("DINOMALY_INPUT_W", 1120))   # same grid as dino_localizer
BACKBONE = "dinov2_vits14_reg"
LAYERS = (2, 3, 4, 5, 6, 7, 8, 9)     # 8 middle blocks of the 12; last blocks are
#                                       too task-specific, first ones too generic
N_GROUPS = 2                          # LAYERS split in two, one loss term each
BLACK_LEVEL = 12                      # masked windows, same cutoff as vlm_05
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

_encoder = None


# --- frozen encoder ----------------------------------------------------------

def _load_encoder():
    """The same cached DINOv2 checkout dino_localizer uses (offline-capable)."""
    global _encoder
    if _encoder is None:
        torch.set_num_threads(max(1, os.cpu_count() or 4))
        local = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        if local.exists():
            _encoder = torch.hub.load(str(local), BACKBONE, source="local",
                                      verbose=False).eval()
        else:
            _encoder = torch.hub.load("facebookresearch/dinov2", BACKBONE,
                                      verbose=False).eval()
        for p in _encoder.parameters():
            p.requires_grad_(False)
    return _encoder


def grid_size(size):
    """(gw, gh) for an image of `size`, snapped to whole patches."""
    w, h = size
    gw = max(1, INPUT_W // PATCH)
    gh = max(1, int(round(h * (gw * PATCH) / w / PATCH)))
    return gw, gh


def encode(path: str, size=None):
    """Frozen-encoder features of one image.

    Returns (layers[L, N, D] float32, valid[N] bool, (gh, gw)). `valid` is False
    on the blacked-out mask areas - they carry no information and must not enter
    either the training loss or the anomaly map, exactly as vlm_05 ignores them.
    """
    size = size or Image.open(path).size
    gw, gh = grid_size(size)
    img = Image.open(path).convert("RGB").resize((gw * PATCH, gh * PATCH),
                                                 Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / \
        np.array([0.229, 0.224, 0.225], np.float32)
    t = torch.from_numpy(x.transpose(2, 0, 1))[None]
    with torch.no_grad():
        outs = _load_encoder().get_intermediate_layers(
            t, n=list(LAYERS), reshape=False, return_class_token=False, norm=True)
    layers = torch.cat([o for o in outs], 0)              # [L, N, D]
    lum = np.asarray(Image.open(path).convert("L").resize((gw, gh), Image.BILINEAR),
                     dtype=np.float32).reshape(-1)
    return layers, torch.from_numpy(lum >= BLACK_LEVEL), (gh, gw)


def group_targets(layers: torch.Tensor):
    """The two reconstruction targets: the layer stack summed within each group.
    Summing (rather than concatenating) keeps the decoder width at D."""
    per = len(LAYERS) // N_GROUPS
    return [layers[i * per:(i + 1) * per].sum(0) for i in range(N_GROUPS)]


# --- the trainable half ------------------------------------------------------

class LinearAttention(nn.Module):
    """elu+1 kernel attention. The point is NOT speed: softmax attention is
    sharp enough for a decoder token to copy the one encoder token it should be
    reconstructing, anomaly included. A smooth kernel cannot single one out, so
    an unfamiliar patch has to be rebuilt from its context - and fails."""

    def __init__(self, dim, heads=6):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = F.elu(q) + 1, F.elu(k) + 1
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(2)) + 1e-6)
        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    def __init__(self, dim, heads=6, mlp=4, drop=0.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(dim * mlp, dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class Dinomaly(nn.Module):
    """Noisy bottleneck + linear-attention decoder over frozen DINOv2 features."""

    def __init__(self, dim=384, depth=8, heads=6, dropout=0.2):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.depth = depth

    def forward(self, layers_sum):
        """layers_sum: [B, N, D] - the encoder stack summed. Returns the N_GROUPS
        decoder aggregates, matching group_targets()."""
        x = self.bottleneck(layers_sum)
        outs = []
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        per = self.depth // N_GROUPS
        return [torch.stack(outs[i * per:(i + 1) * per]).sum(0)
                for i in range(N_GROUPS)]


# --- loss and score ----------------------------------------------------------

HARD_Q = 0.9      # optimise the worst 10% of points only


def hard_cosine_loss(preds, targets, valid=None, q: float = HARD_Q):
    """Point-wise cosine distance, averaged over the HARDEST (1-q) fraction.

    A decoder trained to reconstruct every point perfectly reconstructs anomalies
    perfectly too - that is how reconstruction detectors quietly stop detecting.
    Dropping the points it already gets right keeps the pressure where it is not
    yet good, and leaves the model deliberately loose everywhere else."""
    total = 0.0
    for p, t in zip(preds, targets):
        d = 1.0 - F.cosine_similarity(p, t, dim=-1)        # [B, N]
        if valid is not None:
            d = d[valid]
        d = d.flatten()
        # top-k BY COUNT, not by value: thresholding at the q-quantile keeps
        # everything as soon as the low end has ties (a decoder that nails most
        # points has thousands of them at ~0), which silently turns hard mining
        # back into a plain mean - the exact behaviour this loss exists to avoid
        k = max(1, int(round(d.numel() * (1.0 - q))))
        total = total + torch.topk(d, k).values.mean()
    return total / len(preds)


@torch.no_grad()
def anomaly_map(model: nn.Module, layers: torch.Tensor, valid: torch.Tensor,
                grid) -> np.ndarray:
    """Per-patch anomaly score = summed cosine distance between the encoder
    groups and what the decoder rebuilt of them. Masked patches score 0."""
    model.eval()
    targets = group_targets(layers)
    preds = model(layers.sum(0)[None])
    score = torch.zeros(layers.shape[1])
    for p, t in zip(preds, targets):
        score = score + (1.0 - F.cosine_similarity(p[0], t, dim=-1))
    score[~valid] = 0.0
    gh, gw = grid
    return score.numpy().reshape(gh, gw)


# --- checkpoints -------------------------------------------------------------

def checkpoint_path(camera: str = "tram_1762") -> Path:
    return WEIGHTS_DIR / f"dinomaly_{camera}.pt"


def save(model, camera, meta: dict):
    WEIGHTS_DIR.mkdir(exist_ok=True)
    torch.save({"state": model.state_dict(), "meta": meta,
                "input_w": INPUT_W, "layers": list(LAYERS)},
               checkpoint_path(camera))


def load(camera: str = "tram_1762"):
    """(model, meta). Raises FileNotFoundError with the training command - a
    missing checkpoint is a setup step, not a bug."""
    p = checkpoint_path(camera)
    if not p.exists():
        raise FileNotFoundError(
            f"no Dinomaly checkpoint for '{camera}' ({p}). Train one first:\n"
            f"    python tools/dinomaly_train.py --camera {camera}")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if ck.get("input_w") != INPUT_W:
        raise ValueError(f"checkpoint was trained at INPUT_W={ck.get('input_w')} "
                         f"but DINOMALY_INPUT_W is {INPUT_W} - the grid must match")
    model = Dinomaly()
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck.get("meta", {})

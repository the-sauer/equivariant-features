# Affine Equivariant Features, the main implementation of my master thesis.
# Copyright (C) 2026 Hendrik Sauer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""What is board-validity masking actually worth? — an evaluation-only ablation.

A trained masked descriptor that does not beat its unmasked twin admits two very
different readings: the mask buys nothing on this data, or it buys something the
*predictor* cannot collect. This script separates them without training anything. It
pushes one fixed set of validation batches (`BlobTrackData`'s seeded eval sampler, so
the numbers are on the same footing as the training loop's `FPR95@all`) through a
checkpoint under a matrix of mask conditions, built from two entry points:

    gate    what multiplies the PATCH, inside `input_norm`  (the `cascade` position)
    weight  what multiplies the FEATURE FIELD before the head  (the late position)

each fed from {nothing, the true mask, the model's own `m_pred`, another patch's true
mask}. `none` — a literal 1 at both — is the ablation the model's own `forward` cannot
express: called with `mask=None` it falls back to the *prediction*, not to no masking.

`--content-blind` replaces every patch by ONE fixed texture, so the mask is the only
thing that still varies between samples. Whatever FPR95 survives there is not background
suppression — it is the mask *shape* identifying the blob, a shortcut a mask-fed model
can learn instead of a descriptor. Read it against `late_shuffled`/`gate_shuffled` in the
same run, which is that mode's chance level.

`--occlusion` additionally overwrites an angular wedge of every target patch with
another patch's content before any of that. Off-board background alone is a weak test —
on canonicalized track patches it is a thin rim, and often bland — whereas the question
masking exists to answer is what happens when a sizeable, *structured* piece of the
neighbourhood is not the board. `--blind-mask` then keeps the "true" mask ignorant of
the occluder, which is what a board-in-frame mask would actually give you.

Run it in the pixi env (h5py + CUDA)::

    pixi run --manifest-path deps/BlobBoards.jl/pixi.toml python src/mask_ablation.py \
      --run /raid/data/hsa/logs/EXP/track_logpolar_fftmask_s96 \
      --run /raid/data/hsa/logs/EXP/track_logpolar_fft_s96 \
      --tracks .../iteration_4_logpolar_s96.tracks --occlusion 0 0.1 0.2 0.4

`docs/mask_ceiling_experiment.md` is the full write-up of what this measured on the
`2026_08_19_MaskCeiling` sweep, including the reproduction commands.

Pass several `--run`s to compare arms; a mask-free checkpoint simply skips the
conditions that need `m_pred`, which is itself the interesting comparison (does a
perfect mask help a trunk that never saw one?). `HardNetLogPolar` only — the steerable
family applies its weight inside `self.net`, so it needs its own hook.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from aef import export, models
from aef.data.track import BlobTrackData
from aef.evaluate import fpr
from aef.models.hardnet import input_norm
from aef.models.utils import L2Norm

FPR = fpr()

# Forward-pass chunk. The METRIC still ranks the whole batch's pairs — `input_norm` and
# the head are per-patch, so splitting the forward is exact — this only keeps a 1024-
# patch batch off the GPU's memory ceiling.
CHUNK = 128


# --------------------------------------------------------------------------- model


def load_model(run_dir, checkpoint="best", device="cuda"):
    """Rebuild a run's model from its own `cfg.yaml` and load the checkpoint's weights."""
    name, weights, params = export.from_run(run_dir, checkpoint)
    model = getattr(models, name)(**params)
    state = torch.load(weights, map_location="cpu", weights_only=False)
    state = state.get("model_state_dict", state.get("state_dict", state))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ! load_state_dict missing={list(missing)} unexpected={list(unexpected)}")
    return model.eval().to(device), name, params


def embed(model, patches, gate=None, weight=None):
    """`HardNetLogPolar.forward` with both mask entry points made explicit.

    `gate` (B,1,P,P) multiplies the patch inside `input_norm` — invalid pixels become the
    valid region's mean, so they carry no contrast. `weight` (B,1,*,*) multiplies the
    feature field before the head, averaged down to its grid. Either may be None, meaning
    no masking there. Returns the descriptor and the model's own `m_pred` (None for a
    mask-free model).
    """
    if patches.shape[0] > CHUNK:
        outs = [embed(model, patches[i:i + CHUNK],
                      None if gate is None else gate[i:i + CHUNK],
                      None if weight is None else weight[i:i + CHUNK])
                for i in range(0, patches.shape[0], CHUNK)]
        m = None if outs[0][1] is None else torch.cat([o[1] for o in outs])
        return torch.cat([o[0] for o in outs]), m
    x = input_norm(patches, mask=gate) if gate is not None else input_norm(patches)
    if hasattr(model, "_trunk_and_mask"):
        feat, m_pred = model._trunk_and_mask(x)
    else:
        feat, m_pred = model.trunk(x), None
    if weight is not None:
        w = weight
        if w.shape[-2:] != feat.shape[-2:]:
            w = F.adaptive_avg_pool2d(w, feat.shape[-2:])
        feat = feat * w
    return L2Norm()(model.head(feat).view(patches.size(0), -1)), m_pred


def predict_mask(model, patches):
    """`m_pred` from an unmasked pass — what the model actually has at test time."""
    if not getattr(model, "learned_mask", False):
        return None
    with torch.no_grad():
        return embed(model, patches)[1]


# ----------------------------------------------------------------------- occlusion


def occlude(patches, masks, is_pdf, frac, generator):
    """Overwrite an angular wedge of every TARGET patch with another patch's content.

    Log-polar layout: dim -2 is angular (and wraps), dim -1 radial. A wedge across all
    radii is what an occluder crossing the blob's neighbourhood in the image plane maps
    to — a board edge, a cable, a hand. The filler is a *different* patch rather than a
    constant, so the occluder carries structure the trunk cannot dismiss as flat fill,
    and `input_norm`'s statistics move with it.

    PDF views are left alone: the board's own rendering is clean by construction, and
    the deployment question is what an occluded *observation* costs.

    Returns `(patches, occluded)` with `occluded` 1 where content was replaced — the
    part a truthful validity mask would have to zero.
    """
    b, _, a_dim, _ = patches.shape
    occ = torch.zeros_like(masks)
    width = int(round(frac * a_dim)) if frac > 0 else 0
    if width <= 0:
        return patches, occ
    starts = torch.randint(0, a_dim, (b,), generator=generator, device=patches.device)
    idx = (starts.view(-1, 1) + torch.arange(width, device=patches.device)) % a_dim
    rows = torch.arange(b, device=patches.device).view(-1, 1).expand(-1, width)
    target = (~is_pdf.view(-1).bool()).view(-1, 1).expand(-1, width)
    rows, idx = rows[target], idx[target]
    filler = patches[torch.randperm(b, generator=generator, device=patches.device)]
    out = patches.clone()
    out[rows, :, idx, :] = filler[rows, :, idx, :]
    occ[rows, :, idx, :] = 1.0
    return out, occ


def blank_like(patches, seed=1234):
    """One fixed texture, repeated for every patch in the batch.

    Not a constant: `input_norm` would divide a flat patch by ~0 std, and a gate that
    fills the invalid region with the valid region's mean would leave it flat too, so
    nothing — not even the mask boundary — would reach the trunk. A fixed random field
    is identical across samples (so the image carries zero identity) while still giving
    the mask an edge to imprint on.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    ref = torch.randn((1, *patches.shape[1:]), generator=g)
    return ref.to(patches.device, patches.dtype).expand_as(patches).contiguous()


# ---------------------------------------------------------------------- conditions

# (name, gate source, weight source). Sources:
#   None       no masking there — a literal 1
#   gt         the true validity map, on every view
#   pred       the model's own m_pred, on every view
#   shuffled   another patch's true map: right statistics, wrong content (the control
#              that says whether a gain is the mask's information or just its shape)
#   pdfgt      GT on the PDF views, 1 on the targets — what the trained forward gates with
#   routed     GT on the PDF views, m_pred on the targets — what it weights with
CONDITIONS = [
    ("none",          None,       None),
    ("native",        "pdfgt",    "routed"),   # == model.forward(patches, mask, is_pdf)
    ("native_gt",     "pdfgt",    "gt"),       # native, but the target's mask is perfect
    ("late_gt",       None,       "gt"),
    ("late_pred",     None,       "pred"),
    ("late_shuffled", None,       "shuffled"),
    ("gate_gt",       "gt",       None),
    ("gate_pred",     "pred",     None),
    ("gate_shuffled", "shuffled", None),
    ("both_gt",       "gt",       "gt"),
]


def source_map(kind, gt, m_pred, shuffled_gt, is_pdf, patch_res):
    """One condition's mask map at patch resolution, or None for "no masking here"."""
    if kind is None:
        return None
    if kind == "gt":
        return gt
    if kind == "shuffled":
        return shuffled_gt
    a = is_pdf.view(-1, 1, 1, 1).to(gt.dtype)
    if kind == "pdfgt":
        return a * gt + (1.0 - a) * torch.ones_like(gt)
    if m_pred is None:
        return None                              # mask-free model: no predicted path
    up = F.interpolate(m_pred, size=patch_res, mode="bilinear", align_corners=False)
    if kind == "pred":
        return up
    if kind == "routed":
        return a * gt + (1.0 - a) * up
    raise ValueError(f"unknown mask source {kind!r}")


# ------------------------------------------------------------------------- measure


def tails(model, loader, device, gate_gt=False, quantiles=(50, 90, 95, 99)):
    """The distance distributions behind FPR95, for one condition.

    FPR95 is set by a SINGLE quantile of the positives — the threshold that recalls 95% of
    them — so a model can be tighter on positives almost everywhere and still lose, if its
    worst 5% are worse or its negatives did not move out with them. Positives are split by
    pair type, since the PDF<->image ones are a different (much easier) problem.
    """
    parts = {"pos_img_img": [], "pos_pdf_img": [], "neg": []}
    for data in loader:
        patches = data["patches"].to(device)
        gate = data["masks"].to(device) if gate_gt else None
        labels = data["keypoints"][:, 1].to(device)
        pdf = data["is_pdf"].to(device).view(-1).bool()
        with torch.no_grad():
            d, _ = embed(model, patches, gate=gate)
        dist = torch.cdist(d, d)
        same = labels.view(-1, 1) == labels.view(1, -1)
        upper = torch.triu(torch.ones_like(same), diagonal=1).bool()
        img = (~pdf).view(-1, 1) & (~pdf).view(1, -1)
        parts["pos_img_img"].append(dist[same & upper & img].cpu())
        parts["pos_pdf_img"].append(dist[same & upper & ~img].cpu())
        parts["neg"].append(dist[~same & upper].cpu())
    v = {k: torch.cat(x).numpy() for k, x in parts.items()}
    for k, x in v.items():
        print(f"    {k:>11} n={x.size:>9}  "
              + "  ".join(f"p{q}={np.percentile(x, q):.3f}" for q in quantiles))
    pos = np.concatenate([v["pos_img_img"], v["pos_pdf_img"]])
    thr = float(np.percentile(pos, 95))
    print(f"    threshold@95% recall = {thr:.3f}  ->  FPR = {(v['neg'] < thr).mean():.4f}")


def run(model, loader, device, occlusion, seed=0, mask_sees_occlusion=True,
        content_blind=False, conditions=CONDITIONS):
    """FPR95 per condition over one fixed pass, plus predictor diagnostics."""
    acc = {c[0]: ([], []) for c in conditions}
    diag = {"m_pred_mean": [], "gt_mean": [], "corr": [], "iou": [], "d_shift": []}
    gen = torch.Generator(device=device).manual_seed(seed)
    for data in loader:
        patches = data["patches"].to(device)
        masks = data["masks"].to(device)
        labels = data["keypoints"][:, 1].to(device)      # the contrastive index
        is_pdf = data["is_pdf"].to(device)
        patches, occ = occlude(patches, masks, is_pdf, occlusion, gen)
        gt = masks * (1 - occ) if mask_sees_occlusion else masks
        if content_blind:
            patches = blank_like(patches)        # the mask is now the only variable
        shuffled = gt[torch.randperm(gt.shape[0], generator=gen, device=device)]
        res = patches.shape[-2:]

        with torch.no_grad():
            m_pred = predict_mask(model, patches)
            for name, gk, wk in conditions:
                g = source_map(gk, gt, m_pred, shuffled, is_pdf, res)
                w = source_map(wk, gt, m_pred, shuffled, is_pdf, res)
                if (gk is not None and g is None) or (wk is not None and w is None):
                    continue                     # mask-free model: skip predicted paths
                d, _ = embed(model, patches, gate=g, weight=w)
                acc[name][0].append(float(FPR(d, labels).item()))
                acc[name][1].append(float(FPR(d, labels, is_proxy=is_pdf).item()))

            if m_pred is None:
                continue
            # How good is the predictor, and how much does the weight move the
            # descriptor at all? A `d_shift` near 0 would mean the mask path is inert.
            gt_small = F.adaptive_avg_pool2d(gt, m_pred.shape[-2:])
            t = ~is_pdf.view(-1).bool()          # the views the predictor exists for
            mp, gs = m_pred[t].flatten(), gt_small[t].flatten()
            diag["m_pred_mean"].append(float(mp.mean()))
            diag["gt_mean"].append(float(gs.mean()))
            diag["corr"].append(float(torch.corrcoef(torch.stack([mp, gs]))[0, 1]))
            pb, gb = mp > 0.5, gs > 0.5
            diag["iou"].append(float((pb & gb).sum() / (pb | gb).sum().clamp(min=1)))
            d0, _ = embed(model, patches)
            d1, _ = embed(model, patches, weight=gt)
            diag["d_shift"].append(float(((d1 - d0).norm(dim=1) / d0.norm(dim=1)).mean()))
    return acc, diag


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True, metavar="DIR",
                    help="training run directory (repeatable); model + weights come "
                         "from its cfg.yaml, as with to_onnx.py --run")
    ap.add_argument("--tracks", required=True, help="the .tracks file to validate on")
    ap.add_argument("--checkpoint", default="best", help="best|latest (default: best)")
    ap.add_argument("--boards", nargs="*", default=["08d", "5b6", "cf2", "f75", "f80"],
                    help="validation board uids (default: the track_descriptor_base split)")
    ap.add_argument("--batch-size", type=int, default=1024,
                    help="FPR95's pair population — keep it at the training value (1024)")
    ap.add_argument("--occlusion", type=float, nargs="*", default=[0.0, 0.2, 0.4],
                    help="fractions of the angular axis to overwrite on target patches")
    ap.add_argument("--blind-mask", action="store_true",
                    help="the 'true' mask covers off-board area only, not the occluder "
                         "— i.e. what a board-in-frame mask really gives you")
    ap.add_argument("--tails", action="store_true",
                    help="also print the positive/negative distance quantiles that set "
                         "FPR95, with and without a perfect input gate")
    ap.add_argument("--content-blind", action="store_true",
                    help="blank every patch to one fixed texture, leaving the mask as "
                         "the only per-sample signal — measures how much blob identity "
                         "the mask alone leaks")
    ap.add_argument("--max-batches", type=int, default=0,
                    help="cap the pass for a quick look (0 = the whole split)")
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = BlobTrackData(h5_path=args.tracks, min_track_length=2, load_into_memory=True,
                       include_untracked=True, sequences=args.boards,
                       patch_type="logpolar", with_mask=True)
    sampler = ds.get_eval_batch_sampler(args.batch_size, confuser_fraction=0.5)
    if args.max_batches:
        sampler.batches = sampler.batches[:args.max_batches]
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler,
                                         collate_fn=ds.get_collate_func(), num_workers=4)

    for run_dir in args.run:
        model, name, params = load_model(run_dir, args.checkpoint, device)
        print(f"\n### {os.path.basename(run_dir.rstrip('/'))}  {name}({params})",
              flush=True)
        for occ in args.occlusion:
            t = time.time()
            acc, diag = run(model, loader, device, occ,
                            mask_sees_occlusion=not args.blind_mask,
                            content_blind=args.content_blind)
            print(f"  occlusion={occ:.0%}  ({time.time() - t:.0f}s)"
                  f"{'  [mask blind to it]' if args.blind_mask else ''}"
                  f"{'  [content blind]' if args.content_blind else ''}")
            base = np.nanmean(acc["none"][0])
            for cname, _, _ in CONDITIONS:
                plain, proxy = acc[cname]
                if not plain:
                    continue
                p, pa = np.nanmean(plain), np.nanmean(proxy)
                print(f"    {cname:<14} FPR95 {p:8.5f}  ({p / base:5.2f}x)   "
                      f"proxy-anchored {pa:8.5f}")
            if diag["corr"]:
                print("    m_pred: mean %.3f (gt %.3f)  corr %.3f  IoU@.5 %.3f   "
                      "|d(gt)-d(1)|/|d| %.4f" % (
                          np.mean(diag["m_pred_mean"]), np.mean(diag["gt_mean"]),
                          np.mean(diag["corr"]), np.mean(diag["iou"]),
                          np.mean(diag["d_shift"])), flush=True)
        if args.tails:
            for gate in (False, True):
                print(f"  distance tails, gate={'gt' if gate else 'none'}", flush=True)
                tails(model, loader, device, gate_gt=gate)


if __name__ == "__main__":
    main()

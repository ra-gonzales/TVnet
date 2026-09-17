"""Dual-stage TVnet inference.

Ports ``pipeline`` / ``pipeline_steps`` from ``legacy/TV tracking/AV_functions.m``.

Stage 1 sees the whole cine resized to 160x160 and produces a coarse annotation. That
annotation defines a standardising transform (centre on the valve, 1.5 mm isotropic, valve
plane horizontal, apex down, cropped to 59x81, upsampled x2 to 118x162 at 0.75 mm), and
stage 2 annotates the standardised image. The result is mapped back to the original frame.

Stage 2 can then be re-run on its own output — re-standardising from the *refined* points —
which is what the paper's "stage 1+2+2", "1+2+2+2" and "1+2+2+2+2" rows mean.
``pipeline_steps`` in the MATLAB returns all five as ``TV_1..TV_5``; :func:`predict_stages`
here returns the same list.

Every frame of a cine is annotated independently by both networks, but the standardising
transform is derived from **frame 1 only** and applied to the whole cine, so the geometry is
computed once per iteration, not once per frame.
"""

from __future__ import annotations

import numpy as np
import torch

from . import geometry as geo

__all__ = ["predict_stage1", "predict_stages", "predict_subject", "sampling_of", "TTA1", "TTA2",
           "tta_variants"]

STAGE1_SIZE = (160, 160)
ROW_HALF = 29
COL_HALF = 40


def sampling_of(ckpt: dict) -> str:
    """The resampling convention a checkpoint was trained with (geometry_affine.SAMPLING).
    Checkpoints from before the fix carry no key; they were trained with the shifted one."""
    return str(ckpt.get("sampling", "shifted"))


# Test-time augmentation. Each variant is predicted, mapped back exactly, and the variants are
# combined coordinate by coordinate. Stage 1 perturbs the canonical grid: (rotation deg, scale,
# row shift, col shift) in canonical pixels. Stage 2 perturbs the reference points that set the
# standardisation: (rotation deg about the frame-1 valve midpoint, row shift, col shift) in mm.
# Every perturbation stays inside the training augmentation (10 deg, 10 %, 3 px).
TTA1 = ((0.0, 1.0, 0.0, 0.0), (5.0, 1.0, 0.0, 0.0), (-5.0, 1.0, 0.0, 0.0),
        (0.0, 1.05, 0.0, 0.0), (0.0, 0.95, 0.0, 0.0), (0.0, 1.0, 2.0, 0.0),
        (0.0, 1.0, -2.0, 0.0), (0.0, 1.0, 0.0, 2.0), (0.0, 1.0, 0.0, -2.0))
TTA2 = ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (-4.0, 0.0, 0.0), (0.0, 1.5, 0.0),
        (0.0, -1.5, 0.0), (0.0, 0.0, 1.5), (0.0, 0.0, -1.5))


def tta_variants(stage: int, name):
    """A named test-time augmentation set for one stage: none, or std (TTA1 / TTA2)."""
    if name in (None, "none"):
        return None
    if name == "std":
        return TTA1 if stage == 1 else TTA2
    raise ValueError(f"unknown test-time augmentation set {name!r}")


def _combine(preds, agg: str) -> np.ndarray:
    stack = np.stack(preds)                     # (variants, frames, 4)
    if agg == "median":
        return np.median(stack, axis=0)
    if agg == "mean":
        return stack.mean(axis=0)
    raise ValueError(f"tta_agg must be median or mean, got {agg!r}")


def _jitter_reference(TV, Rxy, rot_deg: float, dr_mm: float, dc_mm: float) -> np.ndarray:
    """Rotate every frame of TV about the frame-1 valve midpoint and shift it, in mm."""
    Rx, Ry = float(Rxy[0]), float(Rxy[1])
    a = np.asarray(TV, dtype=np.float64).copy()
    mid_r, mid_c = (a[0, 0] + a[0, 2]) / 2.0, (a[0, 1] + a[0, 3]) / 2.0
    cs, sn = np.cos(np.deg2rad(rot_deg)), np.sin(np.deg2rad(rot_deg))
    for k in (0, 2):
        yr, yc = (a[:, k] - mid_r) * Ry, (a[:, k + 1] - mid_c) * Rx
        a[:, k] = mid_r + (cs * yr - sn * yc + dr_mm) / Ry
        a[:, k + 1] = mid_c + (sn * yr + cs * yc + dc_mm) / Rx
    return a


@torch.no_grad()
def _run(model, x: np.ndarray, device, batch_size: int = 64) -> np.ndarray:
    """Feed ``(F, 1, H, W)`` float32 through ``model`` -> ``(F, 4)`` float64."""
    model.eval()
    out = []
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    for i in range(0, t.shape[0], batch_size):
        chunk = t[i:i + batch_size].to(device, non_blocking=True)
        out.append(model(chunk).detach().float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def predict_stage1(IM: np.ndarray, model, device="cuda", stage1_mode: str = "legacy",
                   Rxy=None, sampling: str = "exact", tta=None,
                   tta_agg: str = "median", canon=None) -> np.ndarray:
    """Stage 1: prepare the cine, regress, map the points back to original coordinates.

    ``stage1_mode='legacy'`` mirrors steps 3-6 of ``AV_functions('pipeline', ...)``: resize the
    whole cine to 160x160 regardless of its shape, which changes scale and aspect ratio
    together.

    ``stage1_mode='canonical'`` instead resamples onto the fixed millimetre grid of
    :mod:`tvnet.canonical` (192x192 at 1.75 mm, zero-padded or centre-cropped), so the network
    sees one physical scale and one aspect ratio. ``Rxy`` is then required.
    """
    if stage1_mode == "canonical":
        from .canonical import (MM, SIZE, from_canonical, normalise_valid, to_canonical,
                                unwarp_points, warp_matrix)

        if Rxy is None:
            raise ValueError("stage1_mode='canonical' needs the in-plane resolution Rxy")
        preds = []
        for var in (tta or (None,)):
            warp = None if var is None else (warp_matrix(var[0], var[1]), (var[2], var[3]))
            IM_c, _, mask, ctx = to_canonical(IM, None, Rxy, sampling=sampling, warp=warp,
                                              size=int((canon or (SIZE, MM))[0]),
                                              mm=float((canon or (SIZE, MM))[1]))
            x = np.stack([normalise_valid(IM_c[:, :, f], mask) for f in range(IM_c.shape[2])])
            y = _run(model, x[:, None].astype(np.float32), device)
            if warp is not None:
                y = unwarp_points(y, warp, ctx)
            preds.append(from_canonical(y, ctx))
        return preds[0] if len(preds) == 1 else _combine(preds, tta_agg)

    if tta:
        raise ValueError("stage-1 test-time augmentation needs stage1_mode canonical")

    if stage1_mode != "legacy":
        raise ValueError(f"stage1_mode must be 'legacy' or 'canonical', got {stage1_mode!r}")

    IM_1, _ = geo.resize_dims(IM, None, IM.shape, STAGE1_SIZE)
    x_1 = geo.prepare_data_network(IM_1)
    y_1 = _run(model, x_1, device)
    _, TV_1 = geo.resize_dims(None, y_1, IM_1.shape, IM.shape)
    return TV_1


def predict_stages(
    IM: np.ndarray,
    Rxy,
    model_stage1,
    model_stage2,
    n_iter: int = 4,
    device="cuda",
    row_half: int = ROW_HALF,
    col_half: int = COL_HALF,
    geometry: str = "legacy",
    stage1_mode: str = "legacy",
    apex_hint=None,
    sampling1: str = "exact",
    sampling2: str = "exact",
    tta1=None,
    tta2=None,
    tta_agg: str = "median",
    canon=None,
    crop=None,
) -> list[np.ndarray]:
    """Return ``[TV_1, TV_2, ..., TV_{1+n_iter}]`` in the original image's coordinates.

    ``TV_1`` is stage 1 alone; ``TV_2`` is stage 1+2; ``TV_3`` is 1+2+2, and so on, matching
    ``TV_1..TV_5`` of the original MATLAB pipeline when ``n_iter == 4``.

    Parameters
    ----------
    IM : (H, W, frames) float array — the cine, values as stored (already in [0, 1]).
    Rxy : (Rx, Ry) in-plane resolution in mm/pixel.
    geometry : ``'legacy'`` reproduces ``AV_functions.m`` exactly. ``'affine'`` uses the
        continuous single-resample standardisation of :mod:`tvnet.geometry_affine`; it must
        match whatever the stage-2 network was trained with.
    sampling1, sampling2 : resampling convention of the canonical stage-1 grid and of the
        affine stage-2 crop (geometry_affine.SAMPLING). Each must match the training of its
        network, which sampling_of reads from a checkpoint.
    tta1, tta2, tta_agg : test-time augmentation for stage 1 and for every stage-2 pass (TTA1,
        TTA2 or None), combined per coordinate by median or mean.
    canon, crop : input geometry of the canonical stage-1 grid (size, mm) and of the affine
        stage-2 crop (rows, cols, mm); None for the defaults. Each must match its network.
    """
    if IM.ndim != 3:
        raise ValueError(f"IM must be (H, W, frames), got shape {IM.shape}")
    if geometry not in ("legacy", "affine"):
        raise ValueError(f"geometry must be 'legacy' or 'affine', got {geometry!r}")
    if apex_hint is not None and geometry != "affine":
        # the legacy chain has its own flip rule; silently ignoring a prior would mislead
        raise ValueError("apex_hint applies only to geometry='affine'")

    if tta2 and geometry != "affine":
        raise ValueError("stage-2 test-time augmentation needs geometry affine")
    results = [predict_stage1(IM, model_stage1, device=device, stage1_mode=stage1_mode,
                              Rxy=Rxy, sampling=sampling1, tta=tta1, tta_agg=tta_agg,
                              canon=canon)]

    if geometry == "affine":
        from .geometry_affine import destandardize_affine, standardize_affine

    crop_kw = {} if crop is None else {"out_size": (int(crop[0]), int(crop[1])),
                                       "target_mm": float(crop[2])}
    for _ in range(n_iter):
        # The previous iteration's points drive the standardisation.
        if geometry == "affine":
            preds = []
            for var in (tta2 or (None,)):
                ref = results[-1] if var is None else _jitter_reference(results[-1], Rxy, *var)
                IM_2, _ref, ctx = standardize_affine(IM, ref, Rxy, apex_hint=apex_hint,
                                                     sampling=sampling2, **crop_kw)
                x_2 = geo.prepare_data_network(IM_2)
                y_2 = _run(model_stage2, x_2, device)
                preds.append(destandardize_affine(y_2, ctx))
            results.append(preds[0] if len(preds) == 1 else _combine(preds, tta_agg))
        else:
            IM_2, _tv_ref_2, ctx = geo.standardize(
                IM, results[-1], Rxy, row_half=row_half, col_half=col_half
            )
            x_2 = geo.prepare_data_network(IM_2)
            y_2 = _run(model_stage2, x_2, device)
            results.append(geo.destandardize(y_2, ctx))

    return results


def predict_subject(subject, model_stage1, model_stage2, n_iter: int = 4, device="cuda", **kw):
    """Convenience wrapper over a :class:`tvnet.data.Subject`."""
    return predict_stages(
        subject.IM,
        (subject.Rx, subject.Ry),
        model_stage1,
        model_stage2,
        n_iter=n_iter,
        device=device,
        **kw,
    )

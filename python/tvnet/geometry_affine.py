"""A continuous, single-resample replacement for the legacy standardisation chain.

Motivation
----------
``AV_functions.m`` standardises the cine in six steps — pad to centre the valve, resample to
1.5 mm, rotate, flip, crop 59x81, upsample x2 to 118x162 — and two of them quantise:

* ``center_valve`` rounds the frame-1 valve midpoint to a whole pixel,
* ``crop_heart`` rounds the image centre to a whole pixel.

That makes the pipeline a **discontinuous function of its own input**. Feeding stage 2 its
own output (the paper's "1+2+2", "1+2+2+2" rows) can therefore flip a rounding decision
between iterations and shift the entire standardised image by one pixel. It is not
hypothetical: validating the port against the original MATLAB networks, subject 21 of fold 1
has a frame-1 midpoint of 99.4998 in one run and 99.5003 in the other, and everything
downstream moves by ~0.3 px (see ``results/matlab_port_validation.md``).

The chain also resamples the image three times — 1.5 mm, then rotation, then a x2 upsample —
where one would do.

This module composes the whole transform analytically and applies it as a **single**
sub-pixel resample:

* the valve midpoint is used at full precision, never rounded,
* the rotation, flip and crop become a choice of orthonormal basis rather than three
  separate operations,
* the output grid is defined directly in millimetres, so the 1.5 mm intermediate disappears,
* the inverse is the exact algebraic inverse, so the round trip is exact to machine
  precision rather than to interpolation accuracy.

The geometry it produces is the *same* geometry the legacy chain aims at — valve plane
horizontal, valve centred, apex pointing down, point 0 to the left — so a network trained on
one is directly comparable with a network trained on the other.

Conventions match :mod:`tvnet.geometry`: images are ``(H, W, frames)``, landmark rows are
``[row0, col0, row1, col1]`` in **MATLAB 1-based** pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .matlab_compat import imrotate as _imrotate  # noqa: F401  (kernel reference)

__all__ = ["AffineContext", "standardize_affine", "destandardize_affine", "apex_direction",
           "STAGE2_SIZE", "TARGET_MM", "APEX_LPS", "SAMPLING", "source_offset"]

STAGE2_SIZE = (118, 162)   # rows, cols — same as the legacy pipeline
TARGET_MM = 0.75           # effective resolution of the standardised image

# Generic direction of the cardiac apex in DICOM patient coordinates (LPS: +x left, +y
# posterior, +z superior): left, anterior, inferior. Projected onto an image plane through
# ImageOrientationPatient it gives the apex side of the valve without any prediction
# (scripts/orientation_prior_check.py: correct for 140/140 subjects, robust to 30 degrees).
APEX_LPS = (0.6, -0.5, -0.6)

# Below this |cos| between the projected apex and the valve normal the header says little
# about which side is which, and the motion rule decides instead.
PRIOR_MIN_COS = 0.2

# How _sample_bicubic is addressed. It reads 0-based coordinates, while the maps here and in
# tvnet.canonical are written in MATLAB 1-based pixels. Until 2026-09-11 they were passed
# unconverted ("shifted"), so every resampled image sat one source pixel away from its
# landmarks. On the square stage-1 grid that offset is nearly the same for every subject; in the
# standardised stage-2 crop its direction follows the valve of each subject, which made it
# per-subject label noise that no network can learn (median 0.74 mm, up to 3.9 mm over the 140
# subjects; FINDINGS section 10). "exact" is the correct convention. Checkpoints trained before
# the fix carry no "sampling" key and are evaluated with "shifted", the convention they learned.
SAMPLING = ("exact", "shifted")


def source_offset(sampling: str) -> float:
    """What to subtract from a 1-based source coordinate before _sample_bicubic reads it."""
    if sampling == "exact":
        return 1.0
    if sampling == "shifted":
        return 0.0
    raise ValueError(f"sampling must be one of {SAMPLING}, got {sampling!r}")


@dataclass
class AffineContext:
    """Everything needed to map between original and standardised coordinates."""

    midpoint: np.ndarray      # (2,) frame-1 valve midpoint, original pixels, 1-based
    u: np.ndarray             # (2,) unit vector along the valve, in mm (row, col) components
    v: np.ndarray             # (2,) unit vector towards the apex, in mm
    Rxy: tuple                # (Rx, Ry) mm per pixel of the ORIGINAL image
    out_size: tuple           # (rows, cols) of the standardised image
    target_mm: float
    out_centre: np.ndarray    # (2,) centre of the output grid, 1-based


def _keys_cubic(x: np.ndarray) -> np.ndarray:
    """Keys cubic convolution kernel with a = -0.5 — the kernel MATLAB's imresize uses."""
    ax = np.abs(x)
    ax2 = ax * ax
    ax3 = ax2 * ax
    return np.where(ax <= 1, 1.5 * ax3 - 2.5 * ax2 + 1.0,
                    np.where(ax < 2, -0.5 * ax3 + 2.5 * ax2 - 4.0 * ax + 2.0, 0.0))


def _sample_bicubic(IM: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Sample ``IM`` (H, W, F) at 0-based fractional ``(rows, cols)`` with Keys bicubic.

    Out-of-bounds samples read 0, matching the zero fill MATLAB's ``imrotate``/padding use.
    Returns ``(len(rows), F)``.
    """
    H, W, F = IM.shape
    r0 = np.floor(rows).astype(np.int64) - 1
    c0 = np.floor(cols).astype(np.int64) - 1
    fr = rows - np.floor(rows)
    fc = cols - np.floor(cols)

    # weights for the 4 taps in each direction
    wr = np.stack([_keys_cubic(fr + 1), _keys_cubic(fr),
                   _keys_cubic(fr - 1), _keys_cubic(fr - 2)], axis=1)  # (N, 4)
    wc = np.stack([_keys_cubic(fc + 1), _keys_cubic(fc),
                   _keys_cubic(fc - 1), _keys_cubic(fc - 2)], axis=1)  # (N, 4)

    out = np.zeros((rows.size, F), dtype=np.float64)
    flat = IM.reshape(H * W, F)
    for i in range(4):
        ri = r0 + i
        ok_r = (ri >= 0) & (ri < H)
        for j in range(4):
            cj = c0 + j
            ok = ok_r & (cj >= 0) & (cj < W)
            if not ok.any():
                continue
            w = (wr[:, i] * wc[:, j])[ok]
            idx = ri[ok] * W + cj[ok]
            out[ok] += w[:, None] * flat[idx]
    return out


def apex_direction(image_orientation, apex_lps=APEX_LPS) -> np.ndarray | None:
    """Unit apex direction in image axes, (row, col) components in mm, from the DICOM header.

    ``image_orientation`` is ImageOrientationPatient: the row direction cosines (the direction
    in which the column index grows) followed by the column direction cosines (the direction
    in which the row index grows). Returns None when the header is missing or malformed, or
    when the apex direction is perpendicular to the image plane.
    """
    if image_orientation is None:
        return None
    io = np.asarray(image_orientation, dtype=np.float64).ravel()
    if io.size != 6 or not np.all(np.isfinite(io)):
        return None
    a = np.asarray(apex_lps, dtype=np.float64)
    a = a / np.linalg.norm(a)
    d = np.array([io[3:6] @ a, io[:3] @ a])
    n = np.linalg.norm(d)
    return None if n < 1e-6 else d / n


def _basis(TV_ref: np.ndarray, Rxy, apex_hint=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Valve midpoint and the orthonormal (along-valve, towards-apex) basis, in mm.

    This single choice replaces ``rotate_heart`` + ``flip_heart``: taking ``u`` to point from
    point 0 to point 1 puts point 0 on the left, and choosing the sign of the perpendicular
    ``v`` by the direction the valve actually travels puts the apex at the bottom. There is
    no conditional and nothing to quantise.
    """
    Rx, Ry = float(Rxy[0]), float(Rxy[1])
    TV = np.asarray(TV_ref, dtype=np.float64)
    if TV.ndim == 1:
        TV = TV[None, :]

    mid = np.array([(TV[0, 0] + TV[0, 2]) / 2.0, (TV[0, 1] + TV[0, 3]) / 2.0])

    # along-valve direction, point 0 -> point 1, in millimetres
    u = np.array([(TV[0, 2] - TV[0, 0]) * Ry, (TV[0, 3] - TV[0, 1]) * Rx])
    n = np.linalg.norm(u)
    if n < 1e-12:
        raise ValueError("the two landmarks coincide in frame 1; no valve direction")
    u = u / n

    # Perpendicular; its sign decides which way the apex points, so it must be decided from
    # the most reliable evidence available.
    #
    # An earlier version used the MEAN displacement over the cardiac cycle. That is a poor
    # statistic here: systolic descent and diastolic recovery partly cancel, so the mean can
    # land near zero, and when the reference points come from a noisy stage-1 prediction the
    # sign then flips on noise. Measured on the fold-1 test set with a baseline stage-1
    # prediction, one subject decided its orientation on a margin of 0.46 mm against a cohort
    # median of 7.80 mm, came out inverted, and stage 2 saw that cine upside down — a
    # multi-millimetre error that grew with every iteration.
    #
    # Instead, take the sign from the frame of MAXIMUM excursion: the moment in the cycle when
    # the signal is largest and least ambiguous. `margin` reports how decisive that was, so a
    # caller can tell a confident orientation from a coin-flip.
    v = np.array([-u[1], u[0]])
    margin = 0.0
    if TV.shape[0] > 1:
        d_row = ((TV[:, 0] - TV[0, 0]) + (TV[:, 2] - TV[0, 2])) / 2.0 * Ry
        d_col = ((TV[:, 1] - TV[0, 1]) + (TV[:, 3] - TV[0, 3])) / 2.0 * Rx
        proj = d_row * v[0] + d_col * v[1]
        k = int(np.argmax(np.abs(proj)))
        margin = float(abs(proj[k]))
        if proj[k] < 0:
            v = -v
    # An orientation prior, when given and informative, overrides the motion. The motion is
    # read off the reference points, so an inverted stage-1 prediction inverts it; the
    # scanner geometry cannot be fooled that way.
    # ``last_prior_agrees`` is None when no prior was used, else whether it matched the motion.
    agrees = None
    if apex_hint is not None:
        h = np.asarray(apex_hint, dtype=np.float64)
        if abs(h @ v) >= PRIOR_MIN_COS * np.linalg.norm(h):
            agrees = bool(h @ v > 0)
            if not agrees:
                v = -v
    _basis.last_margin = margin
    _basis.last_prior_agrees = agrees
    return mid, u, v


def standardize_affine(
    IM: np.ndarray,
    TV_ref: np.ndarray,
    Rxy,
    out_size: tuple = STAGE2_SIZE,
    target_mm: float = TARGET_MM,
    TV_extra: np.ndarray | None = None,
    apex_hint=None,
    sampling: str = "exact",
):
    """Standardise a cine with one continuous resample.

    Returns ``(IM_std, TV_std, ctx)``, or ``(IM_std, TV_std, TV_extra_std, ctx)`` when
    ``TV_extra`` is given. ``TV_ref`` drives the transform — the stage-1 prediction at
    inference, the ground truth when building a training set.
    """
    IM = np.asarray(IM, dtype=np.float64)
    if IM.ndim != 3:
        raise ValueError(f"IM must be (H, W, frames), got {IM.shape}")
    # ``apex_hint`` (see :func:`apex_direction`) fixes which side of the valve line is the
    # apex; without it the side is read from the motion of ``TV_ref``.
    mid, u, v = _basis(TV_ref, Rxy, apex_hint)
    n_row, n_col = out_size
    centre = np.array([(n_row + 1) / 2.0, (n_col + 1) / 2.0])

    ctx = AffineContext(midpoint=mid, u=u, v=v, Rxy=(float(Rxy[0]), float(Rxy[1])),
                        out_size=(n_row, n_col), target_mm=float(target_mm),
                        out_centre=centre)

    # inverse map: every output pixel -> where to read in the original image
    ii, jj = np.meshgrid(np.arange(1, n_row + 1, dtype=np.float64),
                        np.arange(1, n_col + 1, dtype=np.float64), indexing="ij")
    d_row_mm = (ii - centre[0]) * target_mm
    d_col_mm = (jj - centre[1]) * target_mm
    # offset in mm = d_row_mm * v + d_col_mm * u
    off_r_mm = d_row_mm * v[0] + d_col_mm * u[0]
    off_c_mm = d_row_mm * v[1] + d_col_mm * u[1]
    src_r = mid[0] + off_r_mm / ctx.Rxy[1]
    src_c = mid[1] + off_c_mm / ctx.Rxy[0]

    # src_r / src_c are 1-based and the sampler 0-based: see SAMPLING
    off = source_offset(sampling)
    sampled = _sample_bicubic(IM, src_r.ravel() - off, src_c.ravel() - off)
    IM_std = sampled.reshape(n_row, n_col, IM.shape[2])

    TV_std = _forward_points(np.asarray(TV_ref, dtype=np.float64), ctx)
    if TV_extra is None:
        return IM_std, TV_std, ctx
    return IM_std, TV_std, _forward_points(np.asarray(TV_extra, float), ctx), ctx


def _forward_points(TV: np.ndarray, ctx: AffineContext) -> np.ndarray:
    """Original pixel coordinates -> standardised pixel coordinates (exact, affine)."""
    a = np.atleast_2d(np.asarray(TV, dtype=np.float64))
    out = np.empty_like(a)
    Rx, Ry = ctx.Rxy
    for k in range(a.shape[1] // 2):
        dr_mm = (a[:, 2 * k] - ctx.midpoint[0]) * Ry
        dc_mm = (a[:, 2 * k + 1] - ctx.midpoint[1]) * Rx
        # project onto the orthonormal basis
        out[:, 2 * k] = (dr_mm * ctx.v[0] + dc_mm * ctx.v[1]) / ctx.target_mm + ctx.out_centre[0]
        out[:, 2 * k + 1] = (dr_mm * ctx.u[0] + dc_mm * ctx.u[1]) / ctx.target_mm + ctx.out_centre[1]
    return out.reshape(np.asarray(TV).shape)


def destandardize_affine(TV_std: np.ndarray, ctx: AffineContext) -> np.ndarray:
    """Standardised pixel coordinates -> original pixel coordinates.

    The exact algebraic inverse of :func:`_forward_points`: because ``u`` and ``v`` are
    orthonormal the inverse is the transpose, so the round trip is exact to machine
    precision — there is no rounding anywhere and no interpolation involved.
    """
    a = np.atleast_2d(np.asarray(TV_std, dtype=np.float64))
    out = np.empty_like(a)
    Rx, Ry = ctx.Rxy
    for k in range(a.shape[1] // 2):
        d_row = (a[:, 2 * k] - ctx.out_centre[0]) * ctx.target_mm
        d_col = (a[:, 2 * k + 1] - ctx.out_centre[1]) * ctx.target_mm
        dr_mm = d_row * ctx.v[0] + d_col * ctx.u[0]
        dc_mm = d_row * ctx.v[1] + d_col * ctx.u[1]
        out[:, 2 * k] = ctx.midpoint[0] + dr_mm / Ry
        out[:, 2 * k + 1] = ctx.midpoint[1] + dc_mm / Rx
    return out.reshape(np.asarray(TV_std).shape)

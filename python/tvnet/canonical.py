"""A canonical, FOV-preserving input grid for stage 1.

The published stage 1 resizes every cine to 160x160 *regardless of its shape*, which mixes
two unrelated things into one operation: it changes the scale, and it changes the aspect
ratio. Measured over the 140 subjects:

* in-plane resolution runs 1.429 - 2.212 mm/px (median 1.786),
* the field of view runs 141-375 mm vertically and 155-460 mm horizontally,
* so the anisotropy the network has to absorb varies by ~29% from subject to subject, on top
  of a 2.6x range in physical scale.

None of that is anatomy; it is acquisition and reconstruction settings. This module removes
it by resampling onto a **fixed millimetre grid** and then zero-padding or centre-cropping to
a fixed square, so the network sees one scale, one aspect ratio, and anatomy of a consistent
physical size.

Defaults
--------
``SIZE = 192`` at ``MM = 1.75`` mm/px, i.e. a 336 x 336 mm field of view.

* 1.75 mm/px sits at the cohort's median native resolution (1.786), so the common case is
  resampled at essentially 1:1 rather than up- or down-sampled.
* A half-FOV of 168 mm clears the worst-case landmark in the whole dataset, which sits
  117.9 mm from the image centre, with ~50 mm of margin for augmentation to move things.
* 192 = 2^6 x 3 halves cleanly six times (192, 96, 48, 24, 12, 6), so every stride-2 stage of
  the network divides exactly - no ragged feature maps, which 160 also satisfies but many
  otherwise-sensible sizes (e.g. 180, 200, 220) do not.

Normalisation caveat
--------------------
Padding is not cosmetic here: a 141 mm cine padded to 336 mm is more than half zeros, and
``(x - median)/iqr`` computed over the padded frame would then be driven by the padding rather
than the anatomy - a brand-new source of exactly the inter-subject variability this module
exists to remove. :func:`to_canonical` therefore also returns a **validity mask**, and
:func:`normalise_valid` uses only real samples for the median and IQR.

Conventions match the rest of the package: images are ``(H, W, frames)``, landmarks are
``(frames, 4) = [row0, col0, row1, col1]`` in MATLAB 1-based pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry_affine import _sample_bicubic, source_offset
from .matlab_compat import mat_iqr, mat_median

__all__ = ["CanonicalContext", "SIZE", "MM", "to_canonical", "from_canonical",
           "normalise_valid", "warp_matrix", "warp_points", "unwarp_points"]

SIZE = 192
MM = 1.75


@dataclass
class CanonicalContext:
    """Everything needed to map landmarks between original and canonical coordinates."""

    centre_in: np.ndarray   # (2,) centre of the source image, 1-based (row, col)
    centre_out: float       # centre of the canonical grid, 1-based
    Rxy: tuple              # (Rx, Ry) mm/px of the source image
    size: int
    mm: float
    in_shape: tuple         # (H, W, frames) of the source


def to_canonical(IM, TV=None, Rxy=(1.0, 1.0), size: int = SIZE, mm: float = MM,
                 sampling: str = "exact", warp=None):
    """Resample a cine onto the canonical millimetre grid.

    Returns ``(IM_c, TV_c, mask, ctx)`` where ``IM_c`` is ``(size, size, frames)``, ``TV_c``
    is the transformed landmarks (or ``None``), and ``mask`` is a ``(size, size)`` boolean
    array that is ``True`` wherever the sample came from inside the source image.

    Samples outside the source read 0, so a field of view smaller than ``size * mm`` is
    zero-padded and a larger one is centre-cropped - both handled by the same resample.

    warp (test-time augmentation) is a pair (A, t): canonical point p is shown at
    A (p - c) + c + t, c the grid centre. TV_c is returned unwarped; landmarks predicted on the
    warped grid map back with unwarp_points.
    """
    IM = np.asarray(IM, dtype=np.float64)
    if IM.ndim != 3:
        raise ValueError(f"IM must be (H, W, frames), got {IM.shape}")
    H, W, F = IM.shape
    Rx, Ry = float(Rxy[0]), float(Rxy[1])

    centre_in = np.array([(H + 1) / 2.0, (W + 1) / 2.0])
    centre_out = (size + 1) / 2.0
    ctx = CanonicalContext(centre_in=centre_in, centre_out=centre_out, Rxy=(Rx, Ry),
                           size=size, mm=mm, in_shape=(H, W, F))

    ii, jj = np.meshgrid(np.arange(1, size + 1, dtype=np.float64),
                         np.arange(1, size + 1, dtype=np.float64), indexing="ij")
    if warp is not None:
        # each output pixel shows the canonical point that the warp carries onto it
        ii, jj = _unwarp_rc(ii, jj, warp, centre_out)
    src_r = centre_in[0] + (ii - centre_out) * mm / Ry
    src_c = centre_in[1] + (jj - centre_out) * mm / Rx

    # src_r / src_c are 1-based and the sampler 0-based (geometry_affine.SAMPLING)
    off = source_offset(sampling)
    sampled = _sample_bicubic(IM, src_r.ravel() - off, src_c.ravel() - off)
    IM_c = sampled.reshape(size, size, F)

    # A sample is "real" when its 4x4 cubic footprint sits inside the source image; use the
    # centre with a half-pixel guard, which is what matters for the intensity statistics.
    mask = ((src_r >= 1.0) & (src_r <= H) & (src_c >= 1.0) & (src_c <= W))

    TV_c = None if TV is None else _forward(np.asarray(TV, dtype=np.float64), ctx)
    return IM_c, TV_c, mask, ctx


def _forward(TV: np.ndarray, ctx: CanonicalContext) -> np.ndarray:
    """Original pixel coordinates -> canonical pixel coordinates."""
    a = np.atleast_2d(TV)
    out = np.empty_like(a)
    Rx, Ry = ctx.Rxy
    for k in range(a.shape[1] // 2):
        out[:, 2 * k] = (a[:, 2 * k] - ctx.centre_in[0]) * Ry / ctx.mm + ctx.centre_out
        out[:, 2 * k + 1] = (a[:, 2 * k + 1] - ctx.centre_in[1]) * Rx / ctx.mm + ctx.centre_out
    return out.reshape(TV.shape)


def from_canonical(TV_c, ctx: CanonicalContext) -> np.ndarray:
    """Canonical pixel coordinates -> original pixel coordinates (exact inverse)."""
    a = np.atleast_2d(np.asarray(TV_c, dtype=np.float64))
    out = np.empty_like(a)
    Rx, Ry = ctx.Rxy
    for k in range(a.shape[1] // 2):
        out[:, 2 * k] = (a[:, 2 * k] - ctx.centre_out) * ctx.mm / Ry + ctx.centre_in[0]
        out[:, 2 * k + 1] = (a[:, 2 * k + 1] - ctx.centre_out) * ctx.mm / Rx + ctx.centre_in[1]
    return out.reshape(np.asarray(TV_c).shape)


def warp_matrix(rot_deg: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Rotation and scale acting on (row, col) offsets from the grid centre."""
    th = np.deg2rad(rot_deg)
    return scale * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def _unwarp_rc(r, c, warp, centre):
    """Inverse of p -> A (p - centre) + centre + t, on row and column arrays."""
    A, t = warp
    Ai = np.linalg.inv(np.asarray(A, dtype=np.float64))
    dr = r - centre - t[0]
    dc = c - centre - t[1]
    return Ai[0, 0] * dr + Ai[0, 1] * dc + centre, Ai[1, 0] * dr + Ai[1, 1] * dc + centre


def warp_points(TV_c, warp, ctx: CanonicalContext) -> np.ndarray:
    """Canonical landmarks -> where the warped grid shows them."""
    A, t = warp
    A = np.asarray(A, dtype=np.float64)
    a = np.atleast_2d(np.asarray(TV_c, dtype=np.float64)).copy()
    c = ctx.centre_out
    for k in range(a.shape[1] // 2):
        dr, dc = a[:, 2 * k] - c, a[:, 2 * k + 1] - c
        a[:, 2 * k] = A[0, 0] * dr + A[0, 1] * dc + c + t[0]
        a[:, 2 * k + 1] = A[1, 0] * dr + A[1, 1] * dc + c + t[1]
    return a.reshape(np.asarray(TV_c).shape)


def unwarp_points(TV_w, warp, ctx: CanonicalContext) -> np.ndarray:
    """Landmarks predicted on a warped canonical grid -> plain canonical coordinates."""
    a = np.atleast_2d(np.asarray(TV_w, dtype=np.float64)).copy()
    for k in range(a.shape[1] // 2):
        a[:, 2 * k], a[:, 2 * k + 1] = _unwarp_rc(a[:, 2 * k], a[:, 2 * k + 1], warp,
                                                  ctx.centre_out)
    return a.reshape(np.asarray(TV_w).shape)


def normalise_valid(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """``(x - median)/iqr`` with the statistics taken over real samples only.

    With ``mask=None`` this is exactly the legacy whole-frame normalisation.
    """
    vals = img if (mask is None or mask.all()) else img[mask]
    if vals.size < 8:
        vals = img
    iqr = mat_iqr(vals)
    if not np.isfinite(iqr) or iqr <= 0:
        # A frame whose interquartile range is zero — e.g. a heavily zero-padded canonical
        # frame normalised over the WHOLE frame rather than the valid region, where both
        # quartiles land on the padding. Dividing by it yields NaN and silently poisons the
        # batch, so fall back to a spread that exists.
        iqr = float(np.std(vals))
        if not np.isfinite(iqr) or iqr <= 0:
            return np.zeros_like(img)
    return (img - mat_median(vals)) / iqr

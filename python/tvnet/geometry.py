"""Geometry of the TVnet standardisation, ported from ``legacy/TV tracking/AV_functions.m``.

Every public function here mirrors one MATLAB subfunction of ``AV_functions.m``, keeping
its variable meanings, its argument order and — deliberately — its off-by-one and
off-by-half quirks, because the forward transform builds the stage-2 training targets and
the inverse transform maps stage-2 predictions back to the original image. The two only
agree if both reproduce the same quirks.

Conventions (SPEC.md sections 1 and 3)
--------------------------------------
* Images are ``(H, W, F)`` float64 — MATLAB's ``(rows, cols, frames)``.
* Landmarks are ``(F, 2k)`` float64 in **MATLAB 1-based pixel coordinates**, laid out
  ``[row1, col1, row2, col2, ...]``. Everything below uses ``TV[:, 0::2]`` for rows and
  ``TV[:, 1::2]`` for columns, exactly as MATLAB's ``TV(:,1:2:end)`` / ``TV(:,2:2:end)``.
* ``Rxy`` is ``[Rx, Ry]`` = ``[ResolutionX, ResolutionY]`` mm/pixel, i.e. index 0 is the
  **column** resolution and index 1 the **row** resolution. ``fix_resolution`` relies on
  that ordering.
* ``None`` stands in for MATLAB's ``[]``: ``resize_dims([], TV, ...)`` becomes
  ``resize_dims(None, TV, ...)``.

Known MATLAB quirks reproduced verbatim (each is flagged at its site)
---------------------------------------------------------------------
1. ``resize_dims`` / ``fix_resolution`` scale landmark coordinates by a plain size ratio
   (``r * row_out/row_in``) rather than the half-pixel-correct
   ``(r - 0.5) * ratio + 0.5``. A ~0.5 px image/point offset follows on every resize.
2. ``crop_heart`` offsets points by ``row_out/2`` (= 29.5) where the pixel-exact crop
   offset is ``row_half + 1`` (= 30). Another exact half-pixel bias.
3. ``rotate_heart`` rotates its points about ``(H/2, W/2)`` while MATLAB's ``imrotate``
   rotates the image about the world centre ``((H+1)/2, (W+1)/2)``. The resulting
   image/point disagreement is ``(I - R) @ (0.5, 0.5)``, up to 0.71 px.
4. ``flip_heart`` uses the pixel-exact 1-based mirror ``c -> 2*x_m - c + 1``. The ``+1``
   is kept (``legacy/TVnet/main.py`` drops it); see ``tests/test_geometry.py``.

Resampling method
-----------------
``AV_functions.m`` calls ``imresize3(V, [r c f])`` with no method. SPEC.md section 13
(which overrides section 10) establishes from ``toolbox/images/images/imresize3.m`` that
R2019a's default is **cubic with antialiasing**, not trilinear, and that bicubic agrees
with the original networks' stored predictions to 0.031 px against 0.081 px for trilinear.
So ``resize_dims`` and ``fix_resolution`` default to ``method="cubic"``; pass
``method="linear"`` for the trilinear reading (``matlab_compat.imresize3_linear``).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .matlab_compat import imresize3_matlab, imrotate, mat_iqr, mat_median

__all__ = [
    "resize_dims",
    "prepare_data_network",
    "center_valve",
    "fix_resolution",
    "rotate_heart",
    "flip_heart",
    "crop_heart",
    "unflip_heart",
    "unrotate_heart",
    "apply_offset",
    "apply_scale",
    "apply_rotation",
    "apply_flip",
    "standardize",
    "destandardize",
    "ROW_HALF",
    "COL_HALF",
    "RXY_FIXED",
    "STAGE2_SIZE",
]

# The constants hard-coded in every copy of the pipeline inside AV_functions.m.
ROW_HALF = 29
COL_HALF = 40
RXY_FIXED = (1.5, 1.5)
STAGE2_SIZE = ((ROW_HALF * 2 + 1) * 2, (COL_HALF * 2 + 1) * 2)  # (118, 162)


# =====================================================================
# small helpers
# =====================================================================

def _mat_round(x):
    """MATLAB ``round`` — half away from zero.

    DEVIATION from ``numpy.round``, which rounds half to even: MATLAB's
    ``round(29.5) == 30`` and ``round(30.5) == 31`` while ``numpy.round(30.5) == 30``.
    ``center_valve`` and ``crop_heart`` both round exact ``.5`` values, so the rule
    matters.
    """
    a = np.asarray(x, dtype=np.float64)
    return np.sign(a) * np.floor(np.abs(a) + 0.5)


def _snap(value, rtol: float = 1e-9):
    """Round to the nearest integer when within ``rtol`` relative of one, else pass through.

    Used only by ``fix_resolution(..., snap_size=True)``; see the warning there.
    """
    v = float(value)
    n = float(np.round(v))
    return n if abs(v - n) <= rtol * max(1.0, abs(v)) else v


def _as_av(TV):
    """Normalise a landmark array to ``(n_rows, 2k)`` float64, remembering its ndim.

    Returns ``(array, was_1d)``. A 1-D ``(2k,)`` input is treated as MATLAB's single-row
    ``1 x 2k`` matrix and restored to 1-D on output by :func:`_restore_av`.
    """
    a = np.asarray(TV, dtype=np.float64)
    if a.ndim == 1:
        return a.reshape(1, -1), True
    if a.ndim != 2:
        raise ValueError(f"landmark array must be 1-D or 2-D, got shape {a.shape}")
    if a.shape[1] % 2 != 0:
        raise ValueError(
            f"landmark array must have an even number of columns, got {a.shape[1]}"
        )
    return a, False


def _restore_av(a, was_1d):
    return a.reshape(-1) if was_1d else a


def _size3(size_in) -> tuple[int, int, int]:
    """MATLAB ``size(IM)`` -> ``(rows, cols, frames)``; a 2-D size means one frame."""
    s = tuple(int(v) for v in size_in)
    if len(s) == 2:
        return s[0], s[1], 1
    if len(s) != 3:
        raise ValueError(f"size_in must have 2 or 3 elements, got {size_in!r}")
    return s  # type: ignore[return-value]


def _stack(IM):
    """Coerce an image to ``(H, W, F)`` float64; a 2-D image becomes one frame."""
    a = np.asarray(IM, dtype=np.float64)
    if a.ndim == 2:
        return a[:, :, None]
    if a.ndim != 3:
        raise ValueError(f"image must be (H, W) or (H, W, F), got shape {a.shape}")
    return a


def _reference(TV, TV_ref, who: str):
    """Pick the landmark array that *drives* a transform.

    ``center_valve``, ``rotate_heart`` and ``flip_heart`` derive their transform from the
    landmarks themselves, so MATLAB cannot be called with ``TV_in = []`` there (indexing
    ``[](1,1)`` errors). ``TV_ref`` is the Python escape hatch: give the reference
    landmarks explicitly and ``TV`` may then be ``None``, in which case the point maths is
    skipped and only the image is transformed.
    """
    ref = TV if TV_ref is None else TV_ref
    if ref is None:
        raise ValueError(
            f"{who}: the transform is derived from the landmarks, so TV=None requires "
            "TV_ref=<landmarks>. (MATLAB errors outright if TV_in is [] here.)"
        )
    ref_a, _ = _as_av(ref)
    if ref_a.shape[0] < 1:
        raise ValueError(f"{who}: reference landmark array has no rows")
    return ref_a


# ---------------------------------------------------------------- point-only transforms
# These are the point halves of the transforms above, factored out so that a second
# landmark array (``TV_extra`` in :func:`standardize`) can be carried through exactly the
# same transform without recomputing the image.

def apply_offset(TV, info):
    """``TV + repmat(info, 1, size(TV,2)/2)`` — the point half of centring/cropping."""
    if TV is None:
        return None
    a, was_1d = _as_av(TV)
    info = np.asarray(info, dtype=np.float64).reshape(-1)
    out = a + np.tile(info, a.shape[1] // 2)
    return _restore_av(out, was_1d)


def apply_scale(TV, row_ratio, col_ratio):
    """``TV .* repmat([row_ratio col_ratio], 1, size(TV,2)/2)``.

    The point half of ``resize_dims`` and ``fix_resolution``. Reproduced quirk (1): a
    plain ratio, not a half-pixel-corrected resampling map.
    """
    if TV is None:
        return None
    a, was_1d = _as_av(TV)
    out = a * np.tile(np.array([row_ratio, col_ratio], dtype=np.float64), a.shape[1] // 2)
    return _restore_av(out, was_1d)


def apply_rotation(TV, info_rotated):
    """The point half of :func:`rotate_heart`, given its ``info_rotated``.

    Mirrors, verbatim::

        TV_rotated(:,1:2:end) = a*TV(:,1:2:end)-a*e + b*TV(:,2:2:end)-b*f + e
        TV_rotated(:,2:2:end) = c*TV(:,1:2:end)-c*e + d*TV(:,2:2:end)-d*f + f

    i.e. ``p' = R @ (p - ImCenter) + ImCenter`` with ``ImCenter = [H/2; W/2]``.
    """
    if TV is None:
        return None
    a_arr, was_1d = _as_av(TV)
    info = np.asarray(info_rotated, dtype=np.float64)
    a, b = info[0, 0], info[0, 1]
    c, d = info[1, 0], info[1, 1]
    e, f = info[0, 2], info[1, 2]

    rows = a_arr[:, 0::2]
    cols = a_arr[:, 1::2]
    out = np.empty_like(a_arr)
    out[:, 0::2] = a * rows - a * e + b * cols - b * f + e
    out[:, 1::2] = c * rows - c * e + d * cols - d * f + f
    return _restore_av(out, was_1d)


def apply_flip(TV, info_flipped):
    """The point half of :func:`flip_heart`, given its ``info_flipped``.

    ``info_flipped`` is the 2x2 matrix MATLAB builds: row 0 is the left-right flip
    ``[done, 2*x_m]``, row 1 the up-down flip ``[done, 2*y_m]``. The maps are the
    pixel-exact 1-based mirrors ``c -> 2*x_m - c + 1`` and ``r -> 2*y_m - r + 1``.
    Order matters not at all (the two axes are independent), but MATLAB applies
    left-right first, so this does too.
    """
    if TV is None:
        return None
    a, was_1d = _as_av(TV)
    info = np.asarray(info_flipped, dtype=np.float64)
    out = a.copy()
    if info[0, 0] == 1:
        # DEVIATION from legacy/TVnet/main.py, which drops this "+1".
        out[:, 1::2] = info[0, 1] - out[:, 1::2] + 1.0
    if info[1, 0] == 1:
        out[:, 0::2] = info[1, 1] - out[:, 0::2] + 1.0
    return _restore_av(out, was_1d)


# =====================================================================
# [3] resize_dims
# =====================================================================

def resize_dims(IM_in, TV_in, size_in, size_out, method: str = "cubic",
                antialiasing=None):
    """MATLAB ``AV_functions('resize_dims', IM_in, TV_in, size_in, size_out)``.

    Resamples the image to ``size_out`` and scales the landmarks by the size ratio.

    Parameters
    ----------
    IM_in : (H, W, F) array or None
        ``None`` is MATLAB's ``[]``.
    TV_in : (F, 2k) array or None
    size_in : (rows, cols, frames)
        MATLAB passes ``size(IM_in)`` explicitly, so the inverse calls can supply the size
        of an image that is no longer around. The landmark ratio uses *this*, not
        ``IM_in.shape``.
    size_out : (rows, cols)
    method, antialiasing
        Passed to :func:`matlab_compat.imresize3_matlab`. Default cubic + antialiasing,
        R2019a's ``imresize3`` default (SPEC.md section 13).

    Returns
    -------
    (IM_resized, TV_resized)

    Notes
    -----
    DEVIATION: MATLAB pre-initialises both outputs with ``zeros`` and returns those when
    an input is ``[]``; here an absent input yields ``None``. The MATLAB zeros are never
    consumed — every call site uses ``~`` for the output whose input was empty.
    """
    row_in, col_in, fr = _size3(size_in)
    size_out = tuple(int(v) for v in np.asarray(size_out).ravel()[:2])
    row_out, col_out = size_out

    IM_resized = None
    if IM_in is not None:
        IM = _stack(IM_in)
        IM_resized = imresize3_matlab(IM, (row_out, col_out, IM.shape[2]),
                                      method=method, antialiasing=antialiasing)

    TV_resized = apply_scale(TV_in, row_out / row_in, col_out / col_in)
    return IM_resized, TV_resized


# =====================================================================
# [4] prepare_data_network
# =====================================================================

def prepare_data_network(x_in):
    """MATLAB ``AV_functions('prepare_data_network', x_in)``.

    Per-frame normalisation ``(x - median(x(:))) / iqr(x(:))`` with MATLAB's ``median``
    and ``iqr`` (``matlab_compat.mat_median`` / ``mat_iqr``; MATLAB's quantile rule is not
    numpy's — SPEC.md section 13.4).

    Parameters
    ----------
    x_in : (H, W, F) or (H, W) array

    Returns
    -------
    (F, 1, H, W) float32

    Notes
    -----
    DEVIATION: MATLAB returns ``(H, W, 1, F)`` — its ``trainNetwork``/``predict`` layout.
    PyTorch wants NCHW, so the batch axis is moved to the front. Same numbers, same
    order along the frame axis.

    DEVIATION: MATLAB accumulates in double and stores a double array; this returns
    float32 because that is what the network consumes and what MATLAB's ``predict`` casts
    to anyway.
    """
    x = _stack(x_in)
    H, W, F = x.shape
    out = np.empty((F, 1, H, W), dtype=np.float32)
    for i in range(F):
        frame = x[:, :, i]
        out[i, 0] = ((frame - mat_median(frame)) / mat_iqr(frame)).astype(np.float32)
    return out


# =====================================================================
# [7] center_valve
# =====================================================================

def center_valve(IM_in, TV_in, TV_ref=None):
    """MATLAB ``AV_functions('center_valve', IM_in, TV_in)``.

    Zero-pads the cine on one side of each axis so that the frame-1 valve midpoint lands
    on the image centre. MATLAB::

        center_valve_xy = [round((TV(1,1)+TV(1,end-1))/2) round((TV(1,2)+TV(1,end))/2)]
        dx = col_in - 2*center_valve_xy(2);  a_dx = abs(dx)
        dy = row_in - 2*center_valve_xy(1);  a_dy = abs(dy)
        IM_centered = zeros(row_in+a_dy, col_in+a_dx, fr)
        dx = max(dx,0);  dy = max(dy,0)
        IM_centered(1+dy:dy+row_in, 1+dx:dx+col_in, :) = IM_in
        info_centered = [dy dx]

    The midpoint is taken between the **first** and the **last** landmark
    (``TV(1,1)``/``TV(1,end-1)`` for rows), which for the 2-point tricuspid arrays is
    simply points 1 and 2.

    Parameters
    ----------
    IM_in : (H, W, F) array — required (its size defines the padding).
    TV_in : (F, 2k) array or None — the landmarks to carry through.
    TV_ref : (F, 2k) array or None
        Landmarks that *drive* the centring. Defaults to ``TV_in``. Supply it to
        transform the image with ``TV_in=None``; MATLAB has no such option.

    Returns
    -------
    (IM_centered, TV_centered, info_centered) with ``info_centered = [dy, dx]`` as a
    length-2 float64 array.
    """
    ref = _reference(TV_in, TV_ref, "center_valve")
    IM = _stack(IM_in)
    row_in, col_in, fr = IM.shape

    cy = int(_mat_round((ref[0, 0] + ref[0, -2]) / 2.0))
    cx = int(_mat_round((ref[0, 1] + ref[0, -1]) / 2.0))

    dx = col_in - 2 * cx
    dy = row_in - 2 * cy
    a_dx, a_dy = abs(dx), abs(dy)

    IM_centered = np.zeros((row_in + a_dy, col_in + a_dx, fr), dtype=np.float64)
    dx = max(dx, 0)
    dy = max(dy, 0)
    IM_centered[dy:dy + row_in, dx:dx + col_in, :] = IM

    info_centered = np.array([dy, dx], dtype=np.float64)
    TV_centered = apply_offset(TV_in, info_centered)
    return IM_centered, TV_centered, info_centered


# =====================================================================
# [8] fix_resolution
# =====================================================================

def fix_resolution(IM_in, TV_in, size_in, Rxy_in, Rxy_out, method: str = "cubic",
                   antialiasing=None, snap_size: bool = False):
    """MATLAB ``AV_functions('fix_resolution', IM_in, TV_in, size_in, Rxy_in, Rxy_out)``.

    Resamples to a target mm/pixel resolution. MATLAB::

        row_out = ceil(Rxy_in(2)*row_in/Rxy_out(2))
        col_out = ceil(Rxy_in(1)*col_in/Rxy_out(1))
        Rxy_tmp(2) = Rxy_in(2)*row_in/row_out      % achieved row resolution
        Rxy_tmp(1) = Rxy_in(1)*col_in/col_out      % achieved col resolution

    The ``ceil`` is why the achieved resolution ``Rxy_tmp`` differs from the requested
    1.5 mm; the inverse call has to be given ``Rxy_tmp``, not ``Rxy_out``.

    ``Rxy`` ordering is ``[Rx, Ry] = [column resolution, row resolution]`` — index 1
    drives the row count, index 0 the column count.

    Parameters
    ----------
    IM_in : (H, W, F) array or None
    TV_in : (F, 2k) array or None
    size_in : (rows, cols, frames)
    Rxy_in, Rxy_out : (Rx, Ry) mm/pixel
    snap_size : bool
        ``False`` (default) reproduces MATLAB exactly. See the warning below.

    Returns
    -------
    (IM_fixed, TV_fixed, Rxy_tmp) — ``Rxy_tmp`` is a length-2 float64 array ``[Rx, Ry]``.

    Warning
    -------
    **Reproduced MATLAB floating-point bug.** The inverse call in the pipeline is
    ``fix_resolution([], TV, size(IM_fixed), Rxy_tmp, Rxy_in)``, and algebraically
    ``Rxy_tmp(2)*row_fixed/Rxy_in(2)`` is exactly ``row_centered``. In IEEE double it is
    sometimes ``row_centered + 3e-14``, and ``ceil`` then returns ``row_centered + 1``.
    The landmarks are consequently rescaled by ``(row_centered+1)/row_fixed`` instead of
    ``row_centered/row_fixed``, so the whole inverse is off by ~0.5 px (~0.4 mm) for the
    affected subjects. MATLAB does the identical arithmetic in the identical order, so it
    has the identical bug, and the default here reproduces it.

    ``snap_size=True`` rounds the size to the nearest integer when it is within 1e-9
    relative of one, before the ``ceil``. That is an ablation, not MATLAB behaviour; it
    makes the point round trip exact to ~1e-13 px.
    """
    row_in, col_in, fr = _size3(size_in)
    Rxy_in = np.asarray(Rxy_in, dtype=np.float64).reshape(-1)
    Rxy_out = np.asarray(Rxy_out, dtype=np.float64).reshape(-1)

    row_target = Rxy_in[1] * row_in / Rxy_out[1]
    col_target = Rxy_in[0] * col_in / Rxy_out[0]
    if snap_size:
        # DEVIATION (opt-in): defuse the ceil-of-a-float bug documented above.
        row_target = _snap(row_target)
        col_target = _snap(col_target)
    row_out = int(np.ceil(row_target))
    col_out = int(np.ceil(col_target))

    Rxy_tmp = np.zeros(2, dtype=np.float64)
    Rxy_tmp[1] = Rxy_in[1] * row_in / row_out
    Rxy_tmp[0] = Rxy_in[0] * col_in / col_out

    IM_fixed = None
    if IM_in is not None:
        IM = _stack(IM_in)
        IM_fixed = imresize3_matlab(IM, (row_out, col_out, IM.shape[2]),
                                    method=method, antialiasing=antialiasing)

    TV_fixed = apply_scale(TV_in, row_out / row_in, col_out / col_in)
    return IM_fixed, TV_fixed, Rxy_tmp


# =====================================================================
# [9] rotate_heart
# =====================================================================

def rotate_heart(IM_in, TV_in, TV_ref=None):
    """MATLAB ``AV_functions('rotate_heart', IM_in, TV_in)``.

    Rotates so the frame-1 valve plane becomes horizontal. MATLAB::

        rot = atan((TV(1,1)-TV(1,end-1)) / (TV(1,2)-TV(1,end))) * 180/pi
        RotMatrix = [cosd(rot) -sind(rot); sind(rot) cosd(rot)]
        ImCenter  = [row_in; col_in]/2
        IM_rotated(:,:,i) = imrotate(IM_in(:,:,i), rot, 'bicubic', 'crop')
        info_rotated = [RotMatrix ImCenter]              % 2x3

    ``rot`` is ``atan``, not ``atan2``, so it lies in (-90, 90] and the valve is made
    horizontal without regard to which end is which — that is what ``flip_heart`` sorts
    out afterwards. A vertical plane (``c1 == c2``) gives ``atan(Inf) = 90``, matching
    MATLAB; a fully degenerate plane (both points identical) gives ``atan(0/0) = NaN``,
    also matching MATLAB.

    DEVIATION (reproduced quirk 3): the point formula rotates about ``(H/2, W/2)`` while
    MATLAB's ``imrotate`` rotates the image about the world centre ``((H+1)/2, (W+1)/2)``.
    MATLAB is internally inconsistent here and reproducing MATLAB means reproducing it,
    so ``imrotate`` is called with its own default centre. The image/point disagreement is
    exactly ``(I - R) @ (0.5, 0.5)``, at most 0.71 px.

    DEVIATION: MATLAB loops ``imrotate`` over frames; this calls
    ``matlab_compat.imrotate`` once on the whole ``(H, W, F)`` stack. The resampling is
    per-plane and independent of ``F``, so the numbers are identical — only faster.

    Returns
    -------
    (IM_rotated, TV_rotated, info_rotated) with ``info_rotated`` the 2x3
    ``[RotMatrix ImCenter]``.
    """
    ref = _reference(TV_in, TV_ref, "rotate_heart")
    IM = _stack(IM_in)
    row_in, col_in, fr = IM.shape

    with np.errstate(divide="ignore", invalid="ignore"):
        # atan((r1 - r2) / (c1 - c2)); MATLAB's Inf and NaN behaviour is inherited.
        rot = float(np.degrees(np.arctan(
            (ref[0, 0] - ref[0, -2]) / (ref[0, 1] - ref[0, -1])
        )))

    th = np.deg2rad(rot)
    RotMatrix = np.array([[np.cos(th), -np.sin(th)],
                          [np.sin(th), np.cos(th)]], dtype=np.float64)
    ImCenter = np.array([[row_in / 2.0], [col_in / 2.0]], dtype=np.float64)
    info_rotated = np.hstack([RotMatrix, ImCenter])

    IM_rotated = imrotate(IM, rot, method="bicubic", crop=True, fill=0.0)
    TV_rotated = apply_rotation(TV_in, info_rotated)
    return IM_rotated, TV_rotated, info_rotated


# =====================================================================
# [10] flip_heart
# =====================================================================

def flip_heart(IM_in, TV_in, TV_ref=None):
    """MATLAB ``AV_functions('flip_heart', IM_in, TV_in)``.

    Two conditional flips, in this order:

    * ``fliplr`` if ``TV(1,2) > TV(1,end)`` — i.e. if point 1's column is to the right of
      point 2's, so that afterwards ``col1 < col2`` in frame 1.
    * ``flipud`` if ``mean(TV(2:end,1)) < TV(1,1) && mean(TV(2:end,end-1)) < TV(1,end-1)``
      — i.e. if *both* points move up (to smaller row indices) over the cycle, so that
      afterwards the valve moves down and the apex is at the bottom.

    The ``flipud`` test is applied to the **already left-right-flipped** landmarks, as in
    MATLAB; rows are untouched by ``fliplr`` so the outcome is the same either way.

    The point maps are the pixel-exact 1-based mirrors::

        TV(:,2:2:end) = 2*x_m - TV(:,2:2:end) + 1,   x_m = size(IM_in,2)/2
        TV(:,1:2:end) = 2*y_m - TV(:,1:2:end) + 1,   y_m = size(IM_in,1)/2

    Since ``2*x_m == col_in``, that is ``c -> col_in + 1 - c``, which is precisely where
    ``fliplr`` sends 1-based column ``c``. DEVIATION from ``legacy/TVnet/main.py``, which
    comments the ``+1`` out in both the flip and the unflip: that keeps the point round
    trip intact but leaves every standardised landmark one pixel off the anatomy it is
    supposed to mark. The MATLAB ``+1`` is kept here.

    Returns
    -------
    (IM_flipped, TV_flipped, info_flipped) with the 2x2 ``info_flipped``:
    row 0 = ``[1, 2*x_m]`` if left-right flipped else ``[0, 0]``,
    row 1 = ``[1, 2*y_m]`` if up-down flipped else ``[0, 0]``.
    """
    ref = _reference(TV_in, TV_ref, "flip_heart")
    IM = _stack(IM_in)
    row_in, col_in = IM.shape[0], IM.shape[1]

    IM_flipped = IM
    info_flipped = np.zeros((2, 2), dtype=np.float64)

    # --- left/right -------------------------------------------------------------------
    if ref[0, 1] > ref[0, -1]:
        x_m = col_in / 2.0
        IM_flipped = IM_flipped[:, ::-1, :]
        info_flipped[0, :] = [1.0, 2.0 * x_m]
        ref = ref.copy()
        ref[:, 1::2] = 2.0 * x_m - ref[:, 1::2] + 1.0

    # --- up/down ----------------------------------------------------------------------
    if ref.shape[0] > 1 and (float(np.mean(ref[1:, 0])) < ref[0, 0]
                             and float(np.mean(ref[1:, -2])) < ref[0, -2]):
        y_m = row_in / 2.0
        IM_flipped = IM_flipped[::-1, :, :]
        info_flipped[1, :] = [1.0, 2.0 * y_m]

    IM_flipped = np.ascontiguousarray(IM_flipped)
    TV_flipped = apply_flip(TV_in, info_flipped) if TV_in is not None else None
    return IM_flipped, TV_flipped, info_flipped


# =====================================================================
# [11] crop_heart
# =====================================================================

def crop_heart(IM_in, TV_in, row_half: int = ROW_HALF, col_half: int = COL_HALF):
    """MATLAB ``AV_functions('crop_heart', IM_in, TV_in, row_half, col_half)``.

    Crops ``(2*row_half+1) x (2*col_half+1)`` about the image centre, zero-padding
    symmetrically first if the image is too small. MATLAB::

        row_out = row_half*2+1;  col_out = col_half*2+1
        if row_in < row_out: row_part = ceil((row_out-row_in)/2); pad both sides
        if col_in < col_out: col_part = ceil((col_out-col_in)/2); pad both sides
        center = [round(size(IM,1)/2) round(size(IM,2)/2)]
        IM_cropped = IM(center(1)-row_half:center(1)+row_half, ...)
        final_dX = row_out/2 - center(1) + row_part
        final_dY = col_out/2 - center(2) + col_part
        info_cropped = [final_dX final_dY]

    Note the padding adds ``row_part`` on *both* sides, so a padded axis can end up one
    pixel larger than ``row_out`` when the deficit is odd; ``round(size/2)`` then picks
    the crop centre.

    DEVIATION (reproduced quirk 2): the point offset uses ``row_out/2`` = 29.5 where the
    pixel-exact offset for a 1-based crop starting at ``center(1)-row_half`` is
    ``row_half + 1`` = 30. So MATLAB's cropped landmarks sit exactly half a pixel above
    and half a pixel left of the feature they mark. The inverse (step 16 of the pipeline)
    subtracts the same ``info_cropped``, so the bias cancels end to end.

    Returns
    -------
    (IM_cropped, TV_cropped, info_cropped) with ``info_cropped`` a length-2 float64 array.
    """
    IM = _stack(IM_in)
    row_in, col_in, fr = IM.shape
    row_half = int(row_half)
    col_half = int(col_half)
    row_out = row_half * 2 + 1
    col_out = col_half * 2 + 1

    row_part = 0
    col_part = 0
    if row_in < row_out:
        row_part = int(np.ceil((row_out - row_in) / 2.0))
        pad = np.zeros((row_part, IM.shape[1], fr), dtype=np.float64)
        IM = np.concatenate([pad, IM, pad], axis=0)
    if col_in < col_out:
        col_part = int(np.ceil((col_out - col_in) / 2.0))
        pad = np.zeros((IM.shape[0], col_part, fr), dtype=np.float64)
        IM = np.concatenate([pad, IM, pad], axis=1)

    # MATLAB round (half away from zero) on a positive value.
    center_r = int(_mat_round(IM.shape[0] / 2.0))
    center_c = int(_mat_round(IM.shape[1] / 2.0))

    r0 = center_r - row_half          # 1-based start -> 0-based index is r0 - 1
    c0 = center_c - col_half
    if r0 < 1 or c0 < 1 or center_r + row_half > IM.shape[0] \
            or center_c + col_half > IM.shape[1]:
        raise ValueError(
            "crop_heart: crop window falls outside the (padded) image — "
            f"want rows {r0}:{center_r + row_half}, cols {c0}:{center_c + col_half} "
            f"of a {IM.shape[0]}x{IM.shape[1]} image"
        )
    IM_cropped = np.ascontiguousarray(
        IM[r0 - 1:center_r + row_half, c0 - 1:center_c + col_half, :]
    )

    dX = row_out / 2.0
    dY = col_out / 2.0
    info_cropped = np.array([dX - center_r + row_part, dY - center_c + col_part],
                            dtype=np.float64)
    TV_cropped = apply_offset(TV_in, info_cropped)
    return IM_cropped, TV_cropped, info_cropped


# =====================================================================
# [17] unflip_heart
# =====================================================================

def unflip_heart(TV_flipped, info_flipped):
    """MATLAB ``AV_functions('unflip_heart', TV_flipped, info_flipped)``.

    Inverts :func:`flip_heart` for points only (no image). MATLAB applies the up-down
    unflip first and the left-right unflip second — the reverse of the forward order,
    though the two axes are independent so it makes no difference::

        if info_flipped(2,1) == 1: TV(:,1:2:end) = info_flipped(2,2) - TV(:,1:2:end) + 1
        if info_flipped(1,1) == 1: TV(:,2:2:end) = info_flipped(1,2) - TV(:,2:2:end) + 1

    Each mirror is its own inverse because ``info_flipped(k,2) = 2*x_m`` is the same
    constant used forward, ``+1`` included.
    """
    if TV_flipped is None:
        return None
    a, was_1d = _as_av(TV_flipped)
    info = np.asarray(info_flipped, dtype=np.float64)
    out = a.copy()
    if info[1, 0] == 1:
        out[:, 0::2] = info[1, 1] - out[:, 0::2] + 1.0
    if info[0, 0] == 1:
        out[:, 1::2] = info[0, 1] - out[:, 1::2] + 1.0
    return _restore_av(out, was_1d)


# =====================================================================
# [18] unrotate_heart
# =====================================================================

def unrotate_heart(TV_rotated, info_rotated):
    """MATLAB ``AV_functions('unrotate_heart', TV_rotated, info_rotated)``.

    Inverts :func:`rotate_heart` for points only. MATLAB writes the inverse out
    algebraically rather than transposing the matrix::

        TV_un(:,1:2:end) = (d*TV(:,1:2:end) + a*d*e - d*e - b*TV(:,2:2:end) - b*c*e + b*f)
                           / (a*d - b*c)
        TV_un(:,2:2:end) = (TV(:,2:2:end) + c*e + d*f - f - c*TV_un(:,1:2:end)) / d

    Note the second line consumes the **already unrotated** rows, so the two statements
    must run in this order. It also divides by ``d = cos(rot)``, which is 0 for a
    perfectly vertical valve plane (``rot = +-90``); MATLAB has the same singularity, and
    it is left in place rather than silently patched.
    """
    if TV_rotated is None:
        return None
    a_arr, was_1d = _as_av(TV_rotated)
    info = np.asarray(info_rotated, dtype=np.float64)
    a, b = info[0, 0], info[0, 1]
    c, d = info[1, 0], info[1, 1]
    e, f = info[0, 2], info[1, 2]

    rows = a_arr[:, 0::2]
    cols = a_arr[:, 1::2]
    out = np.empty_like(a_arr)
    out[:, 0::2] = (d * rows + a * d * e - d * e - b * cols - b * c * e + b * f) / (a * d - b * c)
    out[:, 1::2] = (cols + c * e + d * f - f - c * out[:, 0::2]) / d
    return _restore_av(out, was_1d)


# =====================================================================
# Composites: standardisation and its inverse
# =====================================================================

def standardize(IM, TV_ref, Rxy_in, row_half: int = ROW_HALF, col_half: int = COL_HALF,
                Rxy_fixed: Sequence[float] = RXY_FIXED, TV_extra=None,
                method: str = "cubic"):
    """Steps 7-12 of the ``AV_functions.m`` pipeline: the stage-2 standardisation.

    ``center_valve`` -> ``fix_resolution`` (to 1.5 mm) -> ``rotate_heart`` ->
    ``flip_heart`` -> ``crop_heart`` (59x81) -> ``resize_dims`` x2 (118x162, ~0.75 mm).

    The whole chain is driven by ``TV_ref``: the stage-1 (or previous stage-2) prediction
    at inference time (``pipeline`` / ``pipeline_steps``), and the ground truth when
    building the stage-2 training set (``get_gt_2nd``, ``partition_2nd_RV.m``).

    Parameters
    ----------
    IM : (H, W, F) array
    TV_ref : (F, 2k) array
        Landmarks that drive the transform. They are also the ones returned as ``TV_2``.
    Rxy_in : (Rx, Ry) mm/pixel of ``IM``.
    row_half, col_half : int
        Crop half-widths; MATLAB hard-codes 29 and 40.
    Rxy_fixed : (Rx, Ry)
        Intermediate isotropic resolution; MATLAB hard-codes ``[1.5 1.5]``.
    TV_extra : (F, 2k) array or None
        A second landmark array carried through the identical transform without
        influencing it — e.g. the ground truth alongside a stage-1 prediction. When given,
        a fourth output ``TV_extra_2`` is returned.
    method : str
        Resampling method for the two resize steps (see the module docstring).

    Returns
    -------
    ``(IM_2, TV_2, ctx)``, or ``(IM_2, TV_2, ctx, TV_extra_2)`` when ``TV_extra`` is given.
    ``ctx`` is a dict holding everything :func:`destandardize` needs.
    """
    IM = _stack(IM)
    Rxy_in = np.asarray(Rxy_in, dtype=np.float64).reshape(-1)
    Rxy_fixed = np.asarray(Rxy_fixed, dtype=np.float64).reshape(-1)
    extra = None if TV_extra is None else np.asarray(TV_extra, dtype=np.float64)

    # 7. centre the frame-1 valve midpoint
    IM_c, TV_c, info_centered = center_valve(IM, TV_ref)
    extra = apply_offset(extra, info_centered)
    shape_centered = IM_c.shape

    # 8. resample to the fixed (nominally 1.5 mm) resolution
    IM_f, TV_f, Rxy_tmp = fix_resolution(IM_c, TV_c, shape_centered, Rxy_in, Rxy_fixed,
                                         method=method)
    extra = apply_scale(extra, IM_f.shape[0] / shape_centered[0],
                        IM_f.shape[1] / shape_centered[1])
    shape_fixed = IM_f.shape

    # 9. rotate the valve plane horizontal
    IM_r, TV_r, info_rotated = rotate_heart(IM_f, TV_f)
    extra = apply_rotation(extra, info_rotated)

    # 10. flip so point 1 is left of point 2 and the valve moves down
    IM_fl, TV_fl, info_flipped = flip_heart(IM_r, TV_r)
    extra = apply_flip(extra, info_flipped)

    # 11. crop about the centre
    IM_cr, TV_cr, info_cropped = crop_heart(IM_fl, TV_fl, row_half, col_half)
    extra = apply_offset(extra, info_cropped)
    shape_cropped = IM_cr.shape

    # 12. upsample x2 to the stage-2 input size
    size_2 = ((row_half * 2 + 1) * 2, (col_half * 2 + 1) * 2)
    IM_2, TV_2 = resize_dims(IM_cr, TV_cr, shape_cropped, size_2, method=method)
    extra = apply_scale(extra, size_2[0] / shape_cropped[0], size_2[1] / shape_cropped[1])

    ctx: dict[str, Any] = {
        "info_centered": info_centered,
        "shape_centered": shape_centered,
        "Rxy_in": Rxy_in,
        "Rxy_fixed": Rxy_fixed,
        "Rxy_tmp": Rxy_tmp,
        "shape_fixed": shape_fixed,
        "info_rotated": info_rotated,
        "info_flipped": info_flipped,
        "info_cropped": info_cropped,
        "shape_cropped": shape_cropped,
        "shape_2": IM_2.shape,
        "row_half": row_half,
        "col_half": col_half,
        "method": method,
        # Effective mm/pixel of IM_2: the fixed-resolution pixel, halved by the x2 resize.
        "Rxy_2": Rxy_tmp * np.array([shape_cropped[1] / size_2[1],
                                     shape_cropped[0] / size_2[0]]),
    }

    if TV_extra is None:
        return IM_2, TV_2, ctx
    return IM_2, TV_2, ctx, extra


def destandardize(TV_2, ctx, snap_size: bool = False):
    """Steps 15-19 of the ``AV_functions.m`` pipeline: back to original image coordinates.

    ``resize_dims`` (118x162 -> 59x81) -> ``- info_cropped`` -> :func:`unflip_heart` ->
    :func:`unrotate_heart` -> ``fix_resolution`` (back to ``Rxy_in``) -> ``- info_centered``.

    Exactly inverts :func:`standardize`, quirk for quirk: the half-pixel biases of
    ``crop_heart`` and of the plain-ratio resizes are re-applied with the opposite sign,
    so a landmark that went through ``standardize`` comes back unchanged.

    Parameters
    ----------
    TV_2 : (F, 2k) array — landmarks in the 118x162 standardised frame.
    ctx : dict from :func:`standardize`.
    snap_size : bool
        Passed to the step-18 :func:`fix_resolution`. ``False`` (default) reproduces
        MATLAB, including the ``ceil``-of-a-float bug documented there, which costs about
        0.5 px on the subjects it hits. ``True`` is the ablation that removes it.

    Returns
    -------
    (F, 2k) float64 landmarks in the original image's 1-based pixel coordinates.
    """
    method = ctx.get("method", "cubic")

    # 15. back to the cropped 59x81 frame
    _, TV = resize_dims(None, TV_2, ctx["shape_2"], ctx["shape_cropped"][:2],
                        method=method)
    # 16. undo the crop offset
    TV = apply_offset(TV, -np.asarray(ctx["info_cropped"], dtype=np.float64))
    # 17a. undo the flips
    TV = unflip_heart(TV, ctx["info_flipped"])
    # 17b. undo the rotation
    TV = unrotate_heart(TV, ctx["info_rotated"])
    # 18. back to the original resolution
    _, TV, _ = fix_resolution(None, TV, ctx["shape_fixed"], ctx["Rxy_tmp"],
                              ctx["Rxy_in"], method=method, snap_size=snap_size)
    # 19. undo the centring pad
    TV = apply_offset(TV, -np.asarray(ctx["info_centered"], dtype=np.float64))
    return TV

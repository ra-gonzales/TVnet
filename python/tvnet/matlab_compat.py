"""MATLAB-compatible numerics for the TVnet reproduction.

Every function here mirrors a specific MATLAB routine used by the legacy code in
``legacy/TV tracking``.  The reference implementations were read directly from
the MATLAB R2019a toolbox sources installed on this machine (MATLAB itself is
*not* licensed here and is never invoked -- only numpy/scipy are used):

===========================  ==========================================================
Python                       MATLAB reference source
===========================  ==========================================================
``mat_median``               ``median`` (built-in)
``mat_quantile``             ``toolbox/stats/eml/prctile.m`` -> ``percentile_vector``
``mat_iqr``                  ``toolbox/stats/.../iqr`` = ``diff(prctile(x,[25 75]))``
``imresize``                 ``toolbox/matlab/images/imresize.m`` +
                             ``+matlab/+images/+internal/+resize/{contributions,cubic,
                             triangle,box,dimensionOrder}.m``
``imresize_scale``           same, ``imresize(A,SCALE)`` / ``imresize(A,SCALE,'OutputSize',SZ)``
``imresize3_linear``         ``toolbox/images/images/imresize3.m`` with ``'linear'``
``imresize3_matlab``         ``imresize3.m`` (default method, see note below)
``imrotate``                 ``toolbox/images/images/imrotate.m`` (-> ``imwarp`` ->
                             ``+images/+internal/interp2d.m``)
``imtranslate``              ``toolbox/images/images/imtranslate.m`` (integer-shift path)
``csaps_matlab``             ``toolbox/curvefit/curvefit/private/cfsmthspl.m``
                             (what ``fit(x,y,'smoothingspline')`` actually calls)
===========================  ==========================================================

Conventions
-----------
* Images are ``(H, W)`` or ``(H, W, n_frames)`` float64.  Everything is computed
  in float64; the caller keeps track of the physical meaning.
* MATLAB pixel coordinates are 1-based and refer to pixel *centres*; the spatial
  arguments of :func:`imrotate` (``centre``) are 1-based for that reason.
  Array indexing inside is of course 0-based.

Three findings that contradict the project SPEC are flagged with ``DEVIATION``
comments in the code and repeated here because downstream modules depend on them:

1. ``imresize`` mirrors (symmetric-pads) out-of-range input indices.  MATLAB
   versions before ~R2016b *clamped* them instead; R2019a mirrors.
2. ``imresize3``'s default interpolation method in R2019a is **cubic**, not
   linear.  ``resize_dims`` / ``fix_resolution`` in ``AV_functions.m`` call
   ``imresize3(IM,[r c fr])`` with no method, so they get cubic.
3. MATLAB's default smoothing parameter is ``p = 1/(1 + trace(R)/(6*trace(Q'WQ)))``
   which for *uniformly* spaced ``x`` equals ``1/(1 + h**3/9)``, not the
   ``1/(1 + h**3/6)`` quoted in SPEC.md sections 6 and 10.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "mat_median",
    "mat_quantile",
    "mat_iqr",
    "imresize",
    "imresize_scale",
    "imresize3_linear",
    "imresize3_matlab",
    "imrotate",
    "imtranslate",
    "csaps_matlab",
    "MatlabSmoothingSpline",
]


# =====================================================================
# Order statistics
# =====================================================================

def mat_median(x):
    """MATLAB ``median(x(:))`` -- the median over *all* elements.

    Mirrors MATLAB's built-in ``median``.  For real, finite input this is
    identical to :func:`numpy.median` (average of the two central order
    statistics when the count is even), so numpy is used directly.

    NaN handling deliberately differs from :func:`mat_quantile`: MATLAB's
    ``median`` defaults to ``'includenan'`` and returns NaN if any element is
    NaN (as ``numpy.median`` does), whereas ``prctile`` drops NaNs.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size == 0:
        raise ValueError("mat_median: empty input")
    return float(np.median(a))


def mat_quantile(x, q):
    """MATLAB ``quantile(x(:), q)``.

    Mirrors ``toolbox/stats/eml/prctile.m::percentile_vector``, whose exact
    arithmetic is::

        r = q * n
        i = round(r)                       % round() is half-away-from-zero
        if    i < 1  : y = x_(1)
        elseif i >= n: y = x_(n)
        else         : t = r - i
                       y = (0.5 - t)*x_(i) + (0.5 + t)*x_(i+1)

    Equivalently: the ``n`` sorted values are assigned cumulative probabilities
    ``(k - 0.5)/n`` for ``k = 1..n``, linear interpolation is used in between,
    and values outside ``[0.5/n, 1 - 0.5/n]`` are clamped to the extremes.
    This is *not* what ``numpy.percentile`` computes (numpy uses ``(k-1)/(n-1)``
    and never clamps).

    Parameters
    ----------
    x : array_like
        Flattened before use, exactly as MATLAB's ``quantile(x(:), q)``.
        NaNs are dropped first (``prctile.m`` counts only non-NaN values); an
        all-NaN input gives NaN.
    q : float or array_like
        Probability/probabilities in ``[0, 1]``.

    Returns
    -------
    float if ``q`` is a scalar, otherwise an ``ndarray`` shaped like ``q``.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size == 0:
        raise ValueError("mat_quantile: empty input")
    qa = np.asarray(q, dtype=np.float64)
    scalar = qa.ndim == 0
    qf = np.atleast_1d(qa)
    if np.any(qf < 0.0) or np.any(qf > 1.0):
        raise ValueError("mat_quantile: probabilities must lie in [0, 1]")

    # ``prctile.m``: ``n = sum(~isnan(x))`` -- NaNs are *excluded* from the
    # count, and since sort puts them last the interpolation only ever indexes
    # into the non-NaN prefix.  (MATLAB's ``median`` behaves the other way and
    # propagates NaN; see :func:`mat_median`.)  Without this, one NaN in a
    # 5-element vector silently returns 1.75 for q=0.25 where MATLAB returns
    # 1.5, and NaN for q=0.75 where MATLAB returns 3.5.
    a = a[~np.isnan(a)]
    if a.size == 0:
        out = np.full(qf.shape, np.nan, dtype=np.float64)
        return float(out[0]) if scalar else out.reshape(qa.shape)

    xs = np.sort(a)
    n = xs.size
    if n == 1:
        out = np.full(qf.shape, xs[0], dtype=np.float64)
        return float(out.reshape(())) if scalar else out.reshape(qa.shape)

    r = qf * n
    # DEVIATION: MATLAB's round() is half-away-from-zero; numpy.round is
    # half-to-even.  r >= 0 always here, so floor(r + 0.5) reproduces MATLAB.
    i = np.floor(r + 0.5)  # 1-based index of the left neighbour
    t = r - i
    i_int = i.astype(np.int64)

    lo = np.clip(i_int - 1, 0, n - 1)  # -> 0-based
    hi = np.clip(i_int, 0, n - 1)      # 0-based index of x_(i+1)
    out = (0.5 - t) * xs[lo] + (0.5 + t) * xs[hi]

    # Clamp outside the (0.5/n, 1-0.5/n) range, as MATLAB does.
    out = np.where(i_int < 1, xs[0], out)
    out = np.where(i_int >= n, xs[-1], out)

    if scalar:
        return float(out[0])
    return out.reshape(qa.shape)


def mat_iqr(x):
    """MATLAB ``iqr(x(:))`` = ``diff(prctile(x, [25 75]))``."""
    return float(mat_quantile(x, 0.75) - mat_quantile(x, 0.25))


# =====================================================================
# imresize -- separable "contributions" resampling
# =====================================================================

def _cubic(x):
    """Mirrors ``+matlab/+images/+internal/+resize/cubic.m`` (Keys, a = -0.5)."""
    absx = np.abs(x)
    absx2 = absx * absx
    absx3 = absx2 * absx
    return ((1.5 * absx3 - 2.5 * absx2 + 1.0) * (absx <= 1.0)
            + (-0.5 * absx3 + 2.5 * absx2 - 4.0 * absx + 2.0)
            * ((1.0 < absx) & (absx <= 2.0)))


def _triangle(x):
    """Mirrors ``.../resize/triangle.m``.  Note the asymmetric interval ends."""
    x = np.asarray(x, dtype=np.float64)
    return ((x + 1.0) * ((-1.0 <= x) & (x < 0.0))
            + (1.0 - x) * ((0.0 <= x) & (x <= 1.0)))


def _box(x):
    """Mirrors ``.../resize/box.m``: 1 on ``[-0.5, 0.5)``, 0 elsewhere."""
    x = np.asarray(x, dtype=np.float64)
    return ((-0.5 <= x) & (x < 0.5)).astype(np.float64)


# name -> (kernel, kernel_width, default antialiasing).  Mirrors
# imresize.m::getMethodInfo (and imresize3.m::getMethodInfo, which adds
# 'linear'/'trilinear'/'tricubic' as aliases).
_METHODS = {
    "nearest": (_box, 1.0, False),
    "bilinear": (_triangle, 2.0, True),
    "linear": (_triangle, 2.0, True),
    "trilinear": (_triangle, 2.0, True),
    "triangle": (_triangle, 2.0, True),
    "bicubic": (_cubic, 4.0, True),
    "cubic": (_cubic, 4.0, True),
    "tricubic": (_cubic, 4.0, True),
    "box": (_box, 1.0, True),
}


def _method_info(method):
    key = str(method).lower()
    if key not in _METHODS:
        raise ValueError(
            "unsupported interpolation method %r; expected one of %s"
            % (method, sorted(_METHODS))
        )
    return _METHODS[key]


def _contributions(in_length, out_length, scale, kernel, kernel_width,
                   antialiasing):
    """Port of ``.../resize/contributions.m``.

    Returns ``(weights, indices)`` where ``weights`` is ``(out_length, P)`` and
    ``indices`` is the matching **0-based** input index matrix.  Row ``k`` holds
    everything needed for output sample ``k``.
    """
    in_length = int(in_length)
    out_length = int(out_length)
    scale = float(scale)

    if scale < 1.0 and antialiasing:
        # Stretched kernel: simultaneously interpolate and anti-alias.
        def h(t):
            return scale * kernel(scale * t)
        kernel_width = kernel_width / scale
    else:
        h = kernel

    x = np.arange(1, out_length + 1, dtype=np.float64)          # output coords
    # Inverse mapping: 0.5 in output space -> 0.5 in input space.  In 0-based
    # terms this is the (k + 0.5)/scale - 0.5 rule required by the spec.
    u = x / scale + 0.5 * (1.0 - 1.0 / scale)

    left = np.floor(u - kernel_width / 2.0)
    P = int(np.ceil(kernel_width)) + 2

    indices = left[:, None] + np.arange(P, dtype=np.float64)[None, :]  # 1-based
    weights = np.asarray(h(u[:, None] - indices), dtype=np.float64)

    wsum = weights.sum(axis=1, keepdims=True)
    if np.any(wsum == 0.0):
        raise ArithmeticError("imresize: degenerate zero-sum interpolation row")
    weights = weights / wsum

    # Mirror out-of-bounds indices; equivalent to symmetric padding.  This is
    # verbatim R2019a behaviour:  aux = [1:n, n:-1:1];
    #                             indices = aux(mod(indices-1, 2n) + 1)
    # DEVIATION from older MATLAB (<= ~R2016a), which *clamped* instead
    # (replicating the end points).  R2019a -- the version whose toolbox source
    # this port was written against -- mirrors.
    aux = np.concatenate([np.arange(1, in_length + 1),
                          np.arange(in_length, 0, -1)])
    idx1 = aux[np.mod(indices.astype(np.int64) - 1, aux.size)]

    keep = np.any(weights != 0.0, axis=0)
    weights = weights[:, keep]
    idx1 = idx1[:, keep]

    return weights, idx1 - 1  # 0-based


def _resize_along_dim(A, dim, weights, indices):
    """Port of ``imresize.m::resizeAlongDim`` for float arrays."""
    A_ = np.moveaxis(A, dim, 0)
    gathered = A_[indices]                       # (out_length, P, rest...)
    w = weights.reshape(weights.shape + (1,) * (gathered.ndim - 2))
    out = (gathered * w).sum(axis=1)
    return np.moveaxis(out, 0, dim)


def _resize_core(A, out_size, scale, method, antialiasing):
    """Shared body of imresize / imresize3.

    ``out_size`` and ``scale`` are per-spatial-dimension sequences of the same
    length (2 here -- see :func:`imresize3_matlab` for why 3-D reduces to 2-D).
    Mirrors the main body of ``imresize.m``.
    """
    kernel, kernel_width, aa_default = _method_info(method)
    if antialiasing is None:
        antialiasing = aa_default
    antialiasing = bool(antialiasing)

    weights = []
    indices = []
    for k in range(len(out_size)):
        w, i = _contributions(A.shape[k], out_size[k], scale[k], kernel,
                              kernel_width, antialiasing)
        weights.append(w)
        indices.append(i)

    B = A
    # imresize.m: order = dimensionOrder(scale) = [~, order] = sort(scale), i.e.
    # resize first along the dimension with the smallest scale factor.  MATLAB's
    # sort is stable, so ties keep the natural dimension order.
    order = np.argsort(np.asarray(scale, dtype=np.float64), kind="stable")
    for dim in order:
        B = _resize_along_dim(B, int(dim), weights[dim], indices[dim])
    return B


def _as_image(A):
    A = np.asarray(A, dtype=np.float64)
    if A.ndim not in (2, 3):
        raise ValueError("expected a 2-D (H,W) or 3-D (H,W,F) array, got shape %r"
                         % (A.shape,))
    if A.size == 0:
        raise ValueError("empty input image")
    return A


def imresize(A, out_shape, method="bicubic", antialiasing=None):
    """MATLAB ``imresize(A, [NUMROWS NUMCOLS], METHOD)``.

    Full port of the R2019a separable "contributions" algorithm, including the
    antialiasing kernel stretch on downscale and the symmetric (mirrored) edge
    handling.

    Parameters
    ----------
    A : (H, W) or (H, W, F) float array
        Only the first two dimensions are resized, exactly as MATLAB does for
        N-D input.
    out_shape : (rows, cols)
        Non-integer sizes are ``ceil``-ed, mirroring ``imresize.m::fixupSize``.
    method : {'bicubic', 'bilinear', 'nearest', 'box', 'triangle', 'cubic'}
        Default ``'bicubic'``, MATLAB's default.
    antialiasing : bool or None
        ``None`` selects MATLAB's default: ``False`` for ``'nearest'``,
        ``True`` for every other method.  Only has an effect where the scale
        factor along an axis is < 1.

    Notes
    -----
    DEVIATION: MATLAB returns the input class and saturates/rounds integer
    types.  This port is float64-only, which is all the project needs (the cine
    data is float).
    """
    A = _as_image(A)
    out_shape = np.asarray(out_shape, dtype=np.float64).ravel()
    if out_shape.size != 2:
        raise ValueError("out_shape must have two elements (rows, cols)")
    out_size = np.ceil(out_shape).astype(np.int64)
    if np.any(out_size < 1):
        raise ValueError("out_shape must be positive")
    # imresize.m::deriveScaleFromSize
    scale = out_size / np.array(A.shape[:2], dtype=np.float64)
    return _resize_core(A, out_size, scale, method, antialiasing)


def imresize_scale(A, scale, method="bicubic", antialiasing=None,
                   out_shape=None):
    """MATLAB ``imresize(A, SCALE, ...)``.

    With ``out_shape=None`` this is ``imresize(A, SCALE)``: the output size is
    ``ceil(SCALE * size(A))`` (``imresize.m::deriveSizeFromScale``).

    With ``out_shape`` given this is the exact equivalent of MATLAB's
    ``imresize(A, SCALE, 'OutputSize', SZ)`` -- the call made by
    ``before_code/ImageAugmenter.m``.  In that syntax MATLAB keeps the
    *user-supplied* scale for the interpolation weights and only changes how
    many output samples are produced (``imresize.m::fixupSizeAndScale`` leaves
    ``params.scale`` alone when it is non-empty).  The effect is "resample by
    ``scale``, then crop/pad back to ``SZ`` anchored at the top-left", which is
    the reading SPEC.md section 5 requires -- except that where the sampling
    window runs off the input, MATLAB mirrors rather than zero-pads.

    Parameters
    ----------
    scale : float or (row_scale, col_scale)
    out_shape : (rows, cols) or None
    """
    A = _as_image(A)
    sc = np.asarray(scale, dtype=np.float64).ravel()
    if sc.size == 1:
        sc = np.repeat(sc, 2)
    if sc.size != 2:
        raise ValueError("scale must be a scalar or a 2-element sequence")
    if np.any(sc <= 0):
        raise ValueError("scale must be positive")

    if out_shape is None:
        out_size = np.ceil(sc * np.array(A.shape[:2], dtype=np.float64))
        out_size = out_size.astype(np.int64)
    else:
        out_size = np.ceil(np.asarray(out_shape, dtype=np.float64).ravel())
        if out_size.size != 2:
            raise ValueError("out_shape must have two elements (rows, cols)")
        out_size = out_size.astype(np.int64)
    if np.any(out_size < 1):
        raise ValueError("output size must be positive")

    return _resize_core(A, out_size, sc, method, antialiasing)


def imresize3_matlab(V, out_shape, method="cubic", antialiasing=None):
    """MATLAB ``imresize3(V, [NUMROWS NUMCOLS NUMPLANES], METHOD)``.

    Restricted -- deliberately -- to the case this project uses, where the third
    dimension is unchanged (``out_shape[2] == V.shape[2]``); a violation raises.

    Why the restriction makes this exact rather than approximate: with a plane
    scale of exactly 1, ``contributions`` produces a single surviving column of
    weight 1 for the third dimension (every kernel here is an interpolating
    kernel that vanishes at all non-zero integers), so ``resizeAlongDim`` along
    dimension 3 is the identity.  MATLAB resizes dimensions in ascending order
    of scale factor, and inserting an identity step anywhere in that order
    cannot change the result, so ``imresize3`` reduces exactly to the separable
    2-D resize of each frame.

    DEVIATION / IMPORTANT: MATLAB R2019a's ``imresize3`` default method is
    ``'cubic'`` (``imresize3.m::parseInputs`` sets ``params.kernel = @cubic``
    and its help text says "cubic ... the default method"), **not** ``'linear'``
    as SPEC.md section 10 states.  ``AV_functions.m`` calls
    ``imresize3(IM_in,[row_out col_out fr])`` with no method in both
    ``resize_dims`` and ``fix_resolution``, so the legacy pipeline actually used
    cubic there.  This function therefore defaults to cubic;
    :func:`imresize3_linear` is provided for the linear reading.
    """
    V = _as_image(V)
    out_shape = np.asarray(out_shape, dtype=np.float64).ravel()
    if out_shape.size != 3:
        raise ValueError("out_shape must have three elements (rows, cols, planes)")
    out_size3 = np.ceil(out_shape).astype(np.int64)

    n_planes = V.shape[2] if V.ndim == 3 else 1
    if int(out_size3[2]) != int(n_planes):
        raise ValueError(
            "imresize3 port only supports an unchanged third dimension "
            "(got %d planes in, %d requested). The TVnet pipeline never "
            "resamples the frame axis." % (n_planes, int(out_size3[2]))
        )
    if np.any(out_size3[:2] < 1):
        raise ValueError("out_shape must be positive")

    scale = out_size3[:2] / np.array(V.shape[:2], dtype=np.float64)
    return _resize_core(V, out_size3[:2], scale, method, antialiasing)


def imresize3_linear(V, out_shape):
    """MATLAB ``imresize3(V, [r c f], 'linear')`` with the frame axis unchanged.

    See :func:`imresize3_matlab` for why this is exactly a per-frame separable
    bilinear resize, and for the note that R2019a's *default* method is cubic
    rather than linear.  Antialiasing follows the MATLAB default for a named
    non-nearest method: on (i.e. the triangle kernel is stretched on downscale).
    """
    return imresize3_matlab(V, out_shape, method="linear", antialiasing=None)


# =====================================================================
# imrotate
# =====================================================================

# tap offsets (relative to floor(coordinate)) for each kernel
_TAPS = {
    _cubic: (-1, 0, 1, 2),
    _triangle: (0, 1),
    _box: (0, 1),
}


def _sample2d(A, r0, c0, kernel, fill):
    """Separable resampling of ``A`` at 0-based float coords ``(r0, c0)``.

    Out-of-range taps contribute ``fill``.  This reproduces MATLAB's
    ``images.internal.interp2d`` with ``SmoothEdges = true`` (imwarp's default,
    and what ``imrotate`` uses): the image is padded with the fill value and the
    kernel is then applied normally, so a kernel straddling the border blends
    real samples with the fill value rather than renormalising.
    """
    H, W = A.shape[:2]
    taps = _TAPS[kernel]

    br = np.floor(r0).astype(np.int64)
    bc = np.floor(c0).astype(np.int64)
    fr = r0 - br
    fc = c0 - bc

    wr = [np.asarray(kernel(fr - m), dtype=np.float64) for m in taps]
    wc = [np.asarray(kernel(fc - m), dtype=np.float64) for m in taps]

    extra = (1,) * (A.ndim - 2)
    out = np.zeros(r0.shape + A.shape[2:], dtype=np.float64)

    for mi, m in enumerate(taps):
        rr = br + m
        ok_r = (rr >= 0) & (rr < H)
        rr_c = np.clip(rr, 0, H - 1)
        for li, l in enumerate(taps):
            cc = bc + l
            ok = ok_r & (cc >= 0) & (cc < W)
            cc_c = np.clip(cc, 0, W - 1)
            w = wr[mi] * wc[li]
            if not np.any(w):
                continue
            vals = np.where(ok.reshape(ok.shape + extra),
                            A[rr_c, cc_c], fill)
            out += w.reshape(w.shape + extra) * vals
    return out


def _rot_matrix(angle_deg):
    """``R = [[cos, -sin], [sin, cos]]`` acting on ``(row, col)``.

    Derived from ``imrotate.m``, which builds
    ``tform = affine2d([cosd -sind 0; sind cosd 0; 0 0 1])`` and applies it to
    world coordinates ``[x y 1] * T`` with ``x = col``, ``y = row``.  Writing
    that out in ``(row, col)`` gives exactly the matrix above -- the same one
    ``rotate_heart`` uses for its points (SPEC.md section 10).
    """
    th = np.deg2rad(float(angle_deg))
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def imrotate(A, angle_deg, method="bicubic", crop=True, fill=0.0, centre=None):
    """MATLAB ``imrotate(A, ANGLE, METHOD, BBOX)``.

    Rotates the image *content* counter-clockwise by ``angle_deg`` with zero
    fill.  With ``crop=True`` the output has the same size as the input
    (MATLAB's ``'crop'``); with ``crop=False`` it is MATLAB's ``'loose'``
    bounding box.

    Geometry (derived from ``imrotate.m`` -> ``imwarp`` ->
    ``applyGeometricTransformToSpatialRef`` / ``snapWorldLimitsToSatisfyResolution``):
    a feature at 1-based pixel ``p = (row, col)`` moves to::

        p' = R @ (p - centre) + centre,     R = [[cos, -sin], [sin, cos]]

    with ``centre = ((H + 1) / 2, (W + 1) / 2)``.  The centre is the *world*
    centre of the image, i.e. the midpoint of ``[0.5, H + 0.5]``.

    DEVIATION worth knowing about: ``rotate_heart`` in ``AV_functions.m`` rotates
    its landmark points about ``(H/2, W/2)`` instead -- half a pixel away in each
    axis.  MATLAB is internally inconsistent here, and reproducing MATLAB means
    reproducing that: images go through this function (world centre) while the
    landmarks go through the legacy formula (``size/2``).  The resulting
    image-vs-point offset is exactly ``(I - R) @ (0.5, 0.5)``, up to ~0.71 px.
    Pass ``centre=(H/2, W/2)`` to make the image agree with the legacy point
    formula instead; that is an ablation, not MATLAB behaviour.

    Parameters
    ----------
    A : (H, W) or (H, W, F) float array
    angle_deg : float
        Counter-clockwise, degrees.
    method : {'bicubic', 'bilinear', 'nearest'}
        DEVIATION: defaults to ``'bicubic'``, whereas MATLAB's ``imrotate``
        defaults to ``'nearest'``.  Chosen because every legacy call site
        (``AV_functions.m::rotate_heart``, ``ImageAugmenter.m``) passes
        ``'bicubic','crop'`` explicitly; do not read ``imrotate(A, ang)`` here
        as equivalent to bare ``imrotate(A, ang)`` in MATLAB.
    crop : bool
        True -> MATLAB ``'crop'`` (same output size); False -> ``'loose'``.
        DEVIATION: defaults to ``'crop'``; MATLAB's default BBOX is
        ``'loose'``.
    fill : float
        Value outside the source image.  MATLAB's ``imrotate`` always uses 0.
    centre : (row, col) or None
        1-based rotation centre.  ``None`` -> MATLAB's ``((H+1)/2, (W+1)/2)``.
    """
    A = _as_image(A)
    H, W = A.shape[:2]
    angle_deg = float(angle_deg)
    kernel, _, _ = _method_info(method)
    if kernel not in _TAPS:
        raise ValueError("imrotate: unsupported method %r" % (method,))

    # --- exact 90-degree short-circuits (imrotate.m lines 76-131) -----------
    if centre is None and np.remainder(angle_deg, 90.0) == 0.0:
        k = int(np.mod(np.floor(angle_deg / 90.0), 4))
        if k == 0:
            return A.copy()
        if k == 2:
            return A[::-1, ::-1].copy()
        # k == 1 or 3: +-90 degrees
        if crop and H != W:
            # MATLAB rotates only the *central square* and zero-fills the rest.
            # imbegin = (max(twod_size) == so) * abs(diff(floor(twod_size/2)))
            d = abs(int(np.floor(W / 2)) - int(np.floor(H / 2)))
            m = min(H, W)
            r0 = d if H > W else 0
            c0 = d if W > H else 0
            B = np.zeros_like(A)
            sub = A[r0:r0 + m, c0:c0 + m]
            B[r0:r0 + m, c0:c0 + m] = np.rot90(sub, k)
            return B
        return np.ascontiguousarray(np.rot90(A, k))

    R = _rot_matrix(angle_deg)
    grid_centre = np.array([(H + 1.0) / 2.0, (W + 1.0) / 2.0])
    if centre is None:
        C = grid_centre
    else:
        C = np.array([float(centre[0]), float(centre[1])])

    if crop:
        nr, nc = H, W
        # MATLAB re-centres the cropped output on the rotated image centre;
        # with MATLAB's own centre that is a no-op, so output index == input
        # world coordinate.  Keeping that identity for a custom ``centre`` too
        # is what makes the ablation ("same grid, rotate about C") meaningful.
        out_centre = grid_centre
    else:
        # imrotate.m::getOutputBound for 'loose': forward-map the world bbox
        # and take ceil of its extent.
        corners = np.array([[0.5, 0.5], [0.5, W + 0.5],
                            [H + 0.5, 0.5], [H + 0.5, W + 0.5]],
                           dtype=np.float64) - C
        mapped = corners @ R.T
        nr = max(int(np.ceil(mapped[:, 0].max() - mapped[:, 0].min())), 1)
        nc = max(int(np.ceil(mapped[:, 1].max() - mapped[:, 1].min())), 1)
        out_centre = R @ (grid_centre - C) + C

    # Output pixel (i, j), 1-based, sits at offset delta from the output grid
    # centre, whose world position is ``out_centre``.  Inverting
    # ``p_out = R @ (p_in - C) + C`` gives the sampling coordinates.
    di = np.arange(1, nr + 1, dtype=np.float64) - (nr + 1.0) / 2.0 + out_centre[0]
    dj = np.arange(1, nc + 1, dtype=np.float64) - (nc + 1.0) / 2.0 + out_centre[1]
    DI, DJ = np.meshgrid(di - C[0], dj - C[1], indexing="ij")

    Rinv = R.T  # rotation matrices are orthogonal
    r_in = Rinv[0, 0] * DI + Rinv[0, 1] * DJ + C[0]
    c_in = Rinv[1, 0] * DI + Rinv[1, 1] * DJ + C[1]

    # 1-based -> 0-based for array sampling
    return _sample2d(A, r_in - 1.0, c_in - 1.0, kernel, float(fill))


# =====================================================================
# imtranslate
# =====================================================================

def imtranslate(A, shift_xy, fill=0.0):
    """MATLAB ``imtranslate(A, [tx ty], 'FillValues', fill)``, integer shifts.

    ``tx`` shifts along **columns** (world x), ``ty`` along **rows** (world y);
    positive values move the content right / down.  Output has the same size as
    the input (MATLAB's default ``'OutputView', 'same'``), and vacated pixels
    take ``fill``.  Mirrors ``imtranslate.m::translateIntegerShift2D``, where
    ``NonFillOutputLoc = InputLoc + translation``.

    DEVIATION: only integer shifts are supported.  MATLAB falls back to
    ``imwarp`` for fractional shifts; ``ImageAugmenter.m`` only ever draws
    ``randi([-3,3])``, so a fractional shift is a caller bug and raises.
    """
    A = _as_image(A)
    s = np.asarray(shift_xy, dtype=np.float64).ravel()
    if s.size != 2:
        raise ValueError("shift_xy must be (tx, ty)")
    if np.any(s != np.round(s)):
        raise ValueError("imtranslate port supports integer shifts only, got %r"
                         % (tuple(s),))
    tx, ty = int(s[0]), int(s[1])

    H, W = A.shape[:2]
    out = np.full(A.shape, float(fill), dtype=np.float64)

    # destination row range = source row range + ty
    dr0, dr1 = max(0, ty), min(H, H + ty)
    dc0, dc1 = max(0, tx), min(W, W + tx)
    if dr1 > dr0 and dc1 > dc0:
        out[dr0:dr1, dc0:dc1] = A[dr0 - ty:dr1 - ty, dc0 - tx:dc1 - tx]
    return out


# =====================================================================
# Cubic smoothing spline (MATLAB fit(x, y, 'smoothingspline'))
# =====================================================================

class MatlabSmoothingSpline:
    """Piecewise-cubic result of :func:`csaps_matlab`, in MATLAB ``pp`` form.

    ``coefs[i] = [a, b, c, d]`` describes the piece on
    ``[breaks[i], breaks[i+1]]`` as ``a*t**3 + b*t**2 + c*t + d`` with
    ``t = x - breaks[i]``, matching ``mkpp``/``ppval``.
    """

    __slots__ = ("breaks", "coefs", "p", "x", "y", "w")

    def __init__(self, breaks, coefs, p, x, y, w):
        self.breaks = np.asarray(breaks, dtype=np.float64)
        self.coefs = np.asarray(coefs, dtype=np.float64)
        self.p = float(p)
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.w = np.asarray(w, dtype=np.float64)

    def _pieces(self, xq):
        xq = np.asarray(xq, dtype=np.float64)
        n_piece = self.coefs.shape[0]
        # MATLAB ppval: everything below breaks[1] uses piece 1, everything at
        # or above breaks[-2] uses the last piece (i.e. it extrapolates with the
        # end cubics rather than returning NaN).
        idx = np.searchsorted(self.breaks, xq.ravel(), side="right") - 1
        idx = np.clip(idx, 0, n_piece - 1)
        t = xq.ravel() - self.breaks[idx]
        return xq.shape, idx, t

    def eval(self, xq):
        """Value of the spline at ``xq`` (MATLAB ``fitresult(xq)``)."""
        shape, idx, t = self._pieces(xq)
        c = self.coefs[idx]
        v = ((c[:, 0] * t + c[:, 1]) * t + c[:, 2]) * t + c[:, 3]
        return v.reshape(shape)

    def deriv(self, xq):
        """First derivative at ``xq`` (MATLAB ``differentiate(fitresult, xq)``)."""
        shape, idx, t = self._pieces(xq)
        c = self.coefs[idx]
        v = (3.0 * c[:, 0] * t + 2.0 * c[:, 1]) * t + c[:, 2]
        return v.reshape(shape)

    def __call__(self, xq):
        return self.eval(xq)

    def __repr__(self):
        return ("MatlabSmoothingSpline(p=%.12g, n_breaks=%d, range=[%.6g, %.6g])"
                % (self.p, self.breaks.size, self.breaks[0], self.breaks[-1]))


def _smoothing_matrices(h, w):
    """Reinsch matrices ``R`` (n-2, n-2) and ``Qt`` (n-2, n) from de Boor XIV.6.

    Same scaling as ``cfsmthspl.m``: ``R`` is six times the Green & Silverman
    ``R`` so that at ``p = 1`` the system ``R u = diff(divdif)`` yields
    ``u = sigma / 6`` with ``sigma`` the natural-cubic-spline second derivatives.
    """
    n = h.size + 1
    main = 2.0 * (h[:-1] + h[1:])           # length n-2
    off = h[1:-1]                            # length n-3
    R = (np.diag(main) + np.diag(off, 1) + np.diag(off, -1))

    Qt = np.zeros((n - 2, n), dtype=np.float64)
    inv = 1.0 / h
    rows = np.arange(n - 2)
    Qt[rows, rows] = inv[:-1]
    Qt[rows, rows + 1] = -(inv[:-1] + inv[1:])
    Qt[rows, rows + 2] = inv[1:]
    return R, Qt


def _preprocess(x, y, w):
    """Port of ``cfsmthspl.m::preprocess``: sort, then merge duplicate sites.

    Duplicate ``x`` values collapse to their weighted mean ``y`` with summed
    weight, which leaves the smoothing spline unchanged for any ``p``.
    """
    order = np.argsort(x, kind="stable")
    x, y, w = x[order], y[order], w[order]
    if x.size < 2 or np.all(np.diff(x) > 0):
        return x, y, w
    ux, first = np.unique(x, return_index=True)
    sw = np.zeros(ux.size, dtype=np.float64)
    sy = np.zeros(ux.size, dtype=np.float64)
    inv = np.searchsorted(ux, x)
    np.add.at(sw, inv, w)
    np.add.at(sy, inv, w * y)
    return ux, sy / sw, sw


def csaps_matlab(x, y, p=None, w=None):
    """MATLAB ``fit(x, y, 'smoothingspline')`` -- a cubic smoothing spline.

    Port of ``toolbox/curvefit/curvefit/private/cfsmthspl.m``, the routine
    ``fit`` actually dispatches to (``fit.m`` line 457).  It minimises::

        p * sum(w .* (y - f(x)).^2)  +  (1 - p) * integral(f''(t)^2 dt)

    Parameters
    ----------
    x, y : 1-D array_like
        Data sites and values.  ``x`` need not be sorted or distinct.
    p : float in [0, 1] or None
        ``None`` reproduces MATLAB's automatic choice (see below).
        ``p = 1`` interpolates (natural cubic spline); ``p = 0`` is the weighted
        least-squares straight line.
    w : 1-D array_like or None
        Observation weights; ``None`` -> all ones, MATLAB's default.  Sites
        whose weight is ``<= 1e-13 * max(w)`` are dropped before fitting,
        exactly as ``cfsmthspl.m`` does (they would otherwise divide by zero
        in ``W = diag(1/w)`` and turn the whole fit into NaN).

    Returns
    -------
    MatlabSmoothingSpline
        with ``.eval(xq)``, ``.deriv(xq)`` and the chosen ``.p``.

    Notes
    -----
    DEVIATION from SPEC.md sections 6 and 10, which state
    ``p = 1/(1 + h**3/6)``.  ``cfsmthspl.m`` (and ``csaps.m`` line 227) actually
    use::

        p = 1 / (1 + trace(R) / (6 * trace(Qt*W*Qt')))

    For uniformly spaced ``x`` with spacing ``h`` and unit weights,
    ``trace(R) = 4*h*(n-2)`` and ``trace(Qt*Qt') = 6*(n-2)/h**2``, so that
    closed form is ``p = 1 / (1 + h**3 / 9)`` -- a denominator of 9, not 6.
    This port uses the exact trace expression, which is also correct for
    non-uniform spacing (where no closed form exists).

    Do not read the resulting number as "almost 1, therefore almost
    interpolating".  The default balances the two terms of ``M``: by
    construction ``6*(1-p)*trace(Qt*W*Qt') == p*trace(R)``.  Since ``Qt*W*Qt'``
    scales like ``1/h**2`` while ``R`` scales like ``h``, a small ``h`` forces
    ``p`` very close to 1 while still smoothing appreciably.  For the TVnet AVPD
    curves (~30 frames over ~0.9 s, h ~ 0.03 s) ``p ~ 0.999997`` yet the fitted
    values still move off the data by ~1% of the curve amplitude.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if w is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(w, dtype=np.float64).ravel()
        if w.size != x.size:
            raise ValueError("w must have the same length as x")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")

    x, y, w = _preprocess(x, y, w)

    # cfsmthspl.m: "remove all points corresponding to relatively small
    # weights since a (near-)zero weight in effect asks for the corresponding
    # datum to be disregarded while, at the same time, leading to bad
    # condition and even division by zero."  Without this step ``W = 1/w``
    # below is +Inf and the whole fit comes back NaN -- silently.
    maxw = float(np.max(np.abs(w))) if w.size else 0.0
    keep = w > 1e-13 * maxw
    if not np.all(keep):
        x, y, w = x[keep], y[keep], w[keep]
        if x.size < 2:
            raise ValueError(
                "csaps_matlab: fewer than 2 sites have a non-negligible weight")

    n = x.size
    if n < 2:
        raise ValueError("csaps_matlab needs at least 2 distinct sites")
    if p is not None and not (0.0 <= float(p) <= 1.0):
        raise ValueError("p must lie in [0, 1]")

    h = np.diff(x)
    divdif = np.diff(y) / h

    if n == 2:
        # cfsmthspl.m: the smoothing spline is the straight-line interpolant.
        coefs = np.array([[0.0, 0.0, divdif[0], y[0]]], dtype=np.float64)
        return MatlabSmoothingSpline(x, coefs, 1.0 if p is None else float(p),
                                     x, y, w)

    R, Qt = _smoothing_matrices(h, w)
    W = np.diag(1.0 / w)
    QtWQ = Qt @ W @ Qt.T

    if p is None:
        p = 1.0 / (1.0 + np.trace(R) / (6.0 * np.trace(QtWQ)))
    p = float(p)

    M = 6.0 * (1.0 - p) * QtWQ + p * R
    u = np.linalg.solve(M, np.diff(divdif))

    # Smoothed ordinates:  yhat = y - 6*(1-p) * W * Qt' * u
    yhat = y - (6.0 * (1.0 - p)) * (W @ (Qt.T @ u))

    c3 = np.concatenate(([0.0], p * u, [0.0]))          # length n
    c2 = np.diff(yhat) / h - h * (2.0 * c3[:-1] + c3[1:])

    coefs = np.column_stack([
        np.diff(c3) / h,       # cubic coefficient
        3.0 * c3[:-1],         # quadratic coefficient
        c2,                    # linear coefficient
        yhat[:-1],             # constant term
    ])
    return MatlabSmoothingSpline(x, coefs, p, x, y, w)

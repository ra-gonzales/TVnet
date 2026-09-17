"""Clinical-metric derivation for the TVnet reproduction (SPEC.md section 6).

Port of the two analysis routines at the bottom of
``legacy/TV tracking/analysis_pipeline.m``:

======================================  ==================================================
MATLAB                                  Python
======================================  ==================================================
``get_AVPD(TV_in, Rxy, endT)``          :func:`displacement_curve` + :func:`smooth_curve`
                                        + :func:`tapse`
``findpeakvelocities(curve, t, EST)``   :func:`find_peak_velocities`
``fit(x, y, 'smoothingspline')``        :func:`tvnet.matlab_compat.csaps_matlab`
======================================  ==================================================

Conventions
-----------
* Landmark arrays are ``(n_frames, 4) = [row1, col1, row2, col2]`` in MATLAB 1-based
  pixel coordinates, float64 (SPEC.md 1).  ``Rxy = (Rx, Ry)`` in mm/pixel, with
  ``Rx`` the *column* resolution and ``Ry`` the *row* resolution -- exactly the
  MATLAB ``Rxy_in = [ResolutionX ResolutionY]`` ordering, so ``Rxy[0]`` is MATLAB's
  ``Rxy(1)`` and ``Rxy[1]`` is MATLAB's ``Rxy(2)``.
* **All frame indices returned by this module are 0-based**, and ``EST`` passed to
  :func:`find_peak_velocities` is 0-based too.  MATLAB's are 1-based.  This is the
  single systematic deviation; every MATLAB slice ``v(a:b)`` is translated as
  ``v[a-1:b]`` so that the selected elements are identical.
* The sampling grid is ``t = linspace(0, TimeVector[-1], n_frames)``, *not* the
  recorded trigger times -- ``get_AVPD`` builds ``x = linspace(0, endT, fr)`` and
  ``get_accuracy`` builds the matching ``t = linspace(0, endT, size(TV_gt,1))``.
  See :func:`time_grid`.

Units: displacement in **mm**, velocity in **cm/s** (the spline derivative is
mm/s, divided by 10 exactly as ``get_AVPD`` does), time in **seconds**.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # package import
    from .matlab_compat import csaps_matlab
except ImportError:  # pragma: no cover - direct-script import
    from tvnet.matlab_compat import csaps_matlab  # type: ignore[no-redef]

__all__ = [
    "time_grid",
    "displacement_curve",
    "smooth_curve",
    "tapse",
    "find_peak_velocities",
    "rv_eprime",
    "clinical_metrics",
]


# ======================================================================================
# helpers
# ======================================================================================


def time_grid(TimeVector, n_frames: int) -> np.ndarray:
    """The MATLAB analysis time base: ``linspace(0, endT, n_frames)``.

    Mirrors ``get_AVPD``'s ``x = linspace(0,endT,fr)'`` and ``get_accuracy``'s
    ``t = linspace(0,endT,size(TV_gt,1))'``, where ``endT = SET(i).TimeVector(end)``
    (``analysis_pipeline.m`` lines 73 / 276 / 336).

    DEVIATION (none, but worth stating): the legacy code deliberately *re-samples*
    the time axis uniformly between 0 and the last trigger time rather than using the
    recorded ``TimeVector`` itself.  For the TVnet cines the trigger times are already
    near-uniform, but the two are not bit-identical, so this function reproduces
    MATLAB's uniform grid.

    Parameters
    ----------
    TimeVector : array_like or float
        The subject's trigger times in seconds (only the last element is used), or a
        bare scalar already holding ``endT``.
    n_frames : int
        Number of frames, ``fr`` in MATLAB.

    Returns
    -------
    ndarray, shape (n_frames,), float64
    """
    tv = np.asarray(TimeVector, dtype=np.float64).ravel()
    if tv.size == 0:
        raise ValueError("TimeVector is empty")
    end_t = float(tv[-1])
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    if int(n_frames) == 1:
        # numpy-vs-MATLAB semantic difference: ``numpy.linspace(a, b, 1)`` returns
        # ``[a]`` whereas MATLAB's ``linspace(a, b, 1)`` returns ``b`` (documented:
        # "If n is 1, linspace returns x2").  Unreachable through get_AVPD -- a
        # one-frame track cannot be spline-fitted -- but this function is public and
        # advertises MATLAB's grid, so it returns MATLAB's answer.
        return np.array([end_t], dtype=np.float64)
    return np.linspace(0.0, end_t, int(n_frames), dtype=np.float64)


def _check_tv(TV) -> np.ndarray:
    tv = np.asarray(TV, dtype=np.float64)
    if tv.ndim != 2 or tv.shape[1] != 4:
        raise ValueError(f"TV must be (n_frames, 4) = [r1, c1, r2, c2], got {tv.shape}")
    if tv.shape[0] < 1:
        raise ValueError("TV must have at least one frame")
    return tv


def _check_rxy(Rxy) -> tuple[float, float]:
    r = np.asarray(Rxy, dtype=np.float64).ravel()
    if r.size != 2:
        raise ValueError(f"Rxy must be (Rx, Ry), got {r.size} element(s)")
    return float(r[0]), float(r[1])


def _first_nearest(t: np.ndarray, target: float) -> int:
    """0-based index of MATLAB ``find(abs(t-target)==min(abs(t-target)))``.

    MATLAB's ``find`` returns *all* tied indices; every subsequent use is either a
    colon endpoint or a scalar offset, both of which silently take the first element,
    so the first (lowest) index is the faithful translation.
    """
    d = np.abs(t - target)
    return int(np.argmin(d))  # argmin returns the first minimiser, like find(...)(1)


# ======================================================================================
# get_AVPD, part 1: the raw displacement curve
# ======================================================================================


def displacement_curve(TV, Rxy, mode: str = "mean", point: int = 0) -> dict:
    """Perpendicular displacement of the tricuspid annulus from its frame-1 plane.

    Port of the geometry half of ``get_AVPD`` (``analysis_pipeline.m`` lines 302-330).
    Verbatim MATLAB::

        TV_in = TV_in.*[Rxy(2) Rxy(1) Rxy(2) Rxy(1)];
        pe_1 = [TV_in(1,1) TV_in(1,2)];  pe_2 = [TV_in(1,3) TV_in(1,4)];
        m_i = (pe_2(2)-pe_1(2))/(pe_2(1)-pe_1(1));
        b_i = pe_1(2)-m_i*pe_1(1);
        for k=2:fr
            if abs(m_i) == Inf
                dist_lat(k,1) = pe_1(1)-p_lat(1);
                dist_sep(k,1) = pe_2(1)-p_sep(1);
            else
                dist_lat(k,1) = (m_i*p_lat(1)-p_lat(2)+b_i)/sqrt(1+m_i^2);
                ...
        if mean(dist_lat) > 0, dist_lat = -dist_lat; end   % likewise dist_sep
        AVPD = (dist_lat+dist_sep)/2;

    Note the axis swap: the line is fitted with **row as x and column as y**, so the
    slope is ``m = (c2-c1)/(r2-r1)`` and the signed point-line distance is
    ``(m*r - c + b)/sqrt(1+m^2)``.  Frame 1 is never touched by the loop, so both
    per-point curves are exactly 0 there (which is also what the geometry gives,
    since both frame-1 points lie on the plane by construction).

    Parameters
    ----------
    TV : array_like, shape (n_frames, 4)
        ``[row1, col1, row2, col2]``, 1-based pixel coordinates.
    Rxy : (float, float)
        ``(Rx, Ry)`` mm/pixel.  Rows are scaled by ``Ry``, columns by ``Rx``.
    mode : {'mean', 'single'}
        ``'mean'``    -> ``curve = (d_point0 + d_point1)/2``, i.e. ``AVPD`` exactly as
        the legacy code computes it.
        ``'single'``  -> ``curve = d_point[point]``, the lateral-point-only definition
        the paper's Fig. 1(d) describes.  SPEC.md 6 requires both to be available and
        the matching one selected empirically.
    point : {0, 1}
        Which annulus point ``mode='single'`` uses.  Point 0 is ``TV[:, 0:2]``
        (MATLAB ``dist_lat``), point 1 is ``TV[:, 2:4]`` (MATLAB ``dist_sep``).
        Ignored when ``mode='mean'``.

    Returns
    -------
    dict with keys

    ``curve``
        (n_frames,) float64 -- the selected displacement curve, mm.
    ``d0``, ``d1``
        (n_frames,) float64 -- the per-point curves after sign correction
        (MATLAB ``dist_lat`` / ``dist_sep``).
    ``mean``
        (n_frames,) float64 -- ``(d0 + d1)/2``, the legacy ``AVPD``, always present.
    ``slope``, ``intercept``
        The frame-1 plane's ``m_i`` and ``b_i`` in mm space (``b_i`` is ``nan`` when
        the vertical fallback is taken, matching MATLAB's ``-Inf*x`` being unused).
    ``vertical``
        bool -- True when ``abs(m_i) == Inf`` and the ``r1 - r`` fallback was used.
    ``flipped``
        (bool, bool) -- whether each per-point curve was negated.
    ``mode``, ``point``
        Echo of the arguments.
    """
    if mode not in ("mean", "single"):
        raise ValueError(f"mode must be 'mean' or 'single', got {mode!r}")
    if point not in (0, 1):
        raise ValueError(f"point must be 0 or 1, got {point!r}")

    tv = _check_tv(TV)
    rx, ry = _check_rxy(Rxy)
    fr = tv.shape[0]

    # Pixel -> mm.  MATLAB: TV_in .* [Rxy(2) Rxy(1) Rxy(2) Rxy(1)] -- rows by Ry,
    # columns by Rx.
    av_mm = tv * np.array([ry, rx, ry, rx], dtype=np.float64)

    r1, c1 = av_mm[0, 0], av_mm[0, 1]      # pe_1
    r2, c2 = av_mm[0, 2], av_mm[0, 3]      # pe_2

    dr = r2 - r1
    with np.errstate(divide="ignore", invalid="ignore"):
        m_i = (c2 - c1) / dr               # +-Inf when dr == 0, NaN when both are 0
    b_i = c1 - m_i * r1

    d0 = np.zeros(fr, dtype=np.float64)
    d1 = np.zeros(fr, dtype=np.float64)

    vertical = bool(np.isinf(m_i))
    # MATLAB tests `abs(m_i) == Inf`, which is False for NaN; a NaN slope therefore
    # falls through to the general branch and produces NaN distances.  Reproduced.
    if vertical:
        # Frame 1 stays 0 (the loop is `for k=2:fr`); each point is referenced to its
        # own frame-1 row.
        d0[1:] = r1 - av_mm[1:, 0]
        d1[1:] = r2 - av_mm[1:, 2]
        b_i = np.nan
    else:
        denom = np.sqrt(1.0 + m_i * m_i)
        d0[1:] = (m_i * av_mm[1:, 0] - av_mm[1:, 1] + b_i) / denom
        d1[1:] = (m_i * av_mm[1:, 2] - av_mm[1:, 3] + b_i) / denom

    # Adjust curve direction: MATLAB flips only on a strictly positive mean.
    flip0 = bool(np.mean(d0) > 0)
    flip1 = bool(np.mean(d1) > 0)
    if flip0:
        d0 = -d0
    if flip1:
        d1 = -d1

    avpd = (d0 + d1) / 2.0
    curve = avpd if mode == "mean" else (d0 if point == 0 else d1)

    return {
        "curve": curve.copy(),
        "d0": d0,
        "d1": d1,
        "mean": avpd,
        "slope": float(m_i),
        "intercept": float(b_i),
        "vertical": vertical,
        "flipped": (flip0, flip1),
        "mode": mode,
        "point": int(point),
    }


# ======================================================================================
# get_AVPD, part 2: the smoothing spline
# ======================================================================================


def smooth_curve(curve, t) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a displacement curve and differentiate it, as ``get_AVPD`` does.

    Port of ``analysis_pipeline.m`` lines 336-341::

        x = linspace(0,endT,fr)';
        ft = fittype('smoothingspline');
        [fitresult, ~] = fit(x,AVPD,ft);
        y_d = differentiate(fitresult,x)/10;
        y   = fitresult(x);

    The spline is :func:`tvnet.matlab_compat.csaps_matlab` with MATLAB's automatic
    smoothing parameter ``p = 1/(1 + trace(R)/(6*trace(Qt*W*Qt')))`` (SPEC.md 14).

    Parameters
    ----------
    curve : array_like, shape (n,)
        Displacement in mm.
    t : array_like, shape (n,)
        Sample times in seconds (use :func:`time_grid`).

    Returns
    -------
    (y, dydt)
        ``y``    -- (n,) smoothed displacement, mm.
        ``dydt`` -- (n,) velocity in **cm/s**; the spline derivative is mm/s and is
        divided by 10 exactly as MATLAB does.
    """
    y_in = np.asarray(curve, dtype=np.float64).ravel()
    x = np.asarray(t, dtype=np.float64).ravel()
    if x.size != y_in.size:
        raise ValueError(f"curve ({y_in.size}) and t ({x.size}) must be the same length")
    if x.size < 2:
        raise ValueError("need at least 2 samples to fit a smoothing spline")

    fit = csaps_matlab(x, y_in)
    y = fit.eval(x)
    dydt = fit.deriv(x) / 10.0  # mm/s -> cm/s
    return y, dydt


def tapse(TV, Rxy, TimeVector, mode: str = "mean", point: int = 0) -> dict:
    """TAPSE (tricuspid annular plane systolic excursion) and its end-systolic frame.

    Full port of ``get_AVPD`` (``analysis_pipeline.m`` line 302), i.e.
    :func:`displacement_curve` followed by :func:`smooth_curve` and::

        [PD(1,1),PD(1,2)] = min(y);
        PD(1,1) = abs(PD(1,1));

    so TAPSE is ``|min(smoothed curve)|`` and ``PD(2)`` -- the frame at which the
    minimum occurs -- is the end-systolic time ``EST`` that
    :func:`find_peak_velocities` needs.

    DEVIATION: ``index`` is 0-based (MATLAB's ``PD(1,2)`` is 1-based).

    Parameters
    ----------
    TV, Rxy, mode, point
        See :func:`displacement_curve`.
    TimeVector : array_like or float
        Trigger times in seconds; only the last entry is used (see :func:`time_grid`).

    Returns
    -------
    dict with keys

    ``value_mm``
        float -- TAPSE in mm, ``abs(min(smooth))``.
    ``index``
        int -- 0-based ``argmin(smooth)``; this is ``EST``.
    ``curve``
        (n,) raw displacement, mm (the ``mode``/``point`` selection).
    ``smooth``
        (n,) spline-smoothed displacement, mm.
    ``velocity``
        (n,) spline derivative, cm/s.
    ``t``
        (n,) the uniform time grid the fit used, s.
    ``geometry``
        the full :func:`displacement_curve` dict.
    """
    geom = displacement_curve(TV, Rxy, mode=mode, point=point)
    curve = geom["curve"]
    t = time_grid(TimeVector, curve.size)
    y, vel = smooth_curve(curve, t)

    idx = int(np.argmin(y))  # MATLAB min() also returns the first minimiser
    return {
        "value_mm": float(abs(y[idx])),
        "index": idx,
        "curve": curve,
        "smooth": y,
        "velocity": vel,
        "t": t,
        "geometry": geom,
    }


# ======================================================================================
# findpeakvelocities
# ======================================================================================


def find_peak_velocities(vel, t, EST: int) -> dict:
    """Locate s', e' and a' on an annular velocity curve.

    Verbatim port of ``findpeakvelocities`` (``analysis_pipeline.m`` line 351), whose
    phase durations come from Kovacs 2004 (*Duration of diastole during supine
    bicycle*)::

        RR = t(end);  HR = 60/RR;
        ERFT  = (313-0.957*HR)*10^-3;
        DiasT = (-1150+4.40*HR+65500/HR)*10^-3;
        ACT   = (166-0.454*HR)*10^-3;
        ES    = RR-(ERFT+DiasT+ACT);
        es   = EST;
        erf  = find(abs(t-(ES+ERFT))==min(abs(t-(ES+ERFT))));
        dias = find(abs(t-(ES+ERFT+DiasT))==min(abs(t-(ES+ERFT+DiasT))));
        ac   = find(abs(t-RR)==min(abs(t-RR)));
        if length(curve(es:end))<=3, sprime=[]; eprime=[]; aprime=[]; return; end
        if erf<es, es=find(curve==min(curve)); end
        [sprime(1,1),sprime(1,2)] = min(curve(1:es));
        [eprime(1,1),eprime(1,2)] = max(curve(es:erf));
        [aprime(1,1),aprime(1,2)] = max(curve(dias:end));
        eprime(1,2) = eprime(1,2)+es-1;
        aprime(1,2) = aprime(1,2)+dias-1;
        if eprime(1,2) == aprime(1,2), aprime = 0; end

    Index translation (the part that silently shifts e' if botched): MATLAB slices are
    1-based **inclusive**, so with ``es0 = es - 1`` and ``erf0 = erf - 1``,

    * ``curve(1:es)``     -> ``curve[0:es0+1]``   (``sprime`` needs no offset, its
      local index already is the global one)
    * ``curve(es:erf)``   -> ``curve[es0:erf0+1]``, global index = local + ``es0``
      (MATLAB's ``+es-1`` on 1-based indices)
    * ``curve(dias:end)`` -> ``curve[dias0:]``, global index = local + ``dias0``
    * the guard ``length(curve(es:end))<=3`` -> ``n - es0 <= 3``

    ``ac`` is computed by MATLAB and never used; it is returned here for reference.

    DEVIATION (1): all indices in and out are **0-based**, so ``EST`` must be the
    0-based ``tapse(...)['index']``.

    DEVIATION (2): if ``erf < es`` *and* the fallback ``es = argmin(curve)`` is still
    greater than ``erf``, MATLAB's ``max(curve(es:erf))`` returns ``[]`` and the
    assignment ``eprime(1,1) = []`` raises.  Rather than raise, ``eprime`` is returned
    as ``None`` (the same value the empty-diastole guard produces), so a whole-cohort
    sweep cannot be derailed by one pathological subject.

    Parameters
    ----------
    vel : array_like, shape (n,)
        Velocity curve, cm/s (``tapse(...)['velocity']``).
    t : array_like, shape (n,)
        Matching time grid in seconds (``tapse(...)['t']``).
    EST : int
        0-based end-systolic frame index.

    Returns
    -------
    dict with keys ``sprime``, ``eprime``, ``aprime`` -- each either ``None`` (MATLAB
    ``[]``) or ``{'value': float, 'index': int|None}``.  ``aprime`` is
    ``{'value': 0.0, 'index': None}`` in the case where MATLAB overwrites it with the
    scalar ``0`` because a' and e' landed on the same frame.  Also returned, for
    inspection: ``RR``, ``HR``, ``ERFT``, ``DiasT``, ``ACT``, ``ES``, and the 0-based
    phase indices ``es``, ``erf``, ``dias``, ``ac``.
    """
    curve = np.asarray(vel, dtype=np.float64).ravel()
    tt = np.asarray(t, dtype=np.float64).ravel()
    if curve.size != tt.size:
        raise ValueError(f"vel ({curve.size}) and t ({tt.size}) must be the same length")
    n = curve.size
    if n == 0:
        raise ValueError("vel is empty")
    est = int(EST)
    if not (0 <= est < n):
        raise ValueError(f"EST={est} out of range for {n} frames (0-based)")

    RR = float(tt[-1])
    HR = 60.0 / RR
    ERFT = (313.0 - 0.957 * HR) * 1e-3
    DiasT = (-1150.0 + 4.40 * HR + 65500.0 / HR) * 1e-3
    ACT = (166.0 - 0.454 * HR) * 1e-3
    MDD = ERFT + DiasT + ACT
    ES = RR - MDD

    es = est                                             # MATLAB: es = EST
    erf = _first_nearest(tt, ES + ERFT)
    dias = _first_nearest(tt, ES + ERFT + DiasT)
    ac = _first_nearest(tt, RR)

    phases = {
        "RR": RR, "HR": HR, "ERFT": ERFT, "DiasT": DiasT, "ACT": ACT, "ES": ES,
        "es": es, "erf": erf, "dias": dias, "ac": ac,
    }

    # MATLAB: length(curve(es:end)) <= 3  ->  (n - es_1based + 1) <= 3  ->  n - es0 <= 3
    if n - es <= 3:  # assume no diastole in signal
        return {"sprime": None, "eprime": None, "aprime": None, **phases}

    if erf < es:
        es = int(np.argmin(curve))  # MATLAB: es = find(curve==min(curve)); colon uses (1)
        phases["es"] = es
        phases["es_corrected"] = True
    else:
        phases["es_corrected"] = False

    # s' = min(curve(1:es))  ->  curve[0:es+1]; local index == global index.
    s_seg = curve[: es + 1]
    s_loc = int(np.argmin(s_seg))
    sprime = {"value": float(s_seg[s_loc]), "index": s_loc}

    # e' = max(curve(es:erf))  ->  curve[es:erf+1]; global = local + es.
    if erf < es:
        # DEVIATION (2) above: MATLAB would error on the empty slice.
        eprime = None
        e_idx = None
    else:
        e_seg = curve[es : erf + 1]
        e_loc = int(np.argmax(e_seg))
        e_idx = e_loc + es
        eprime = {"value": float(e_seg[e_loc]), "index": e_idx}

    # a' = max(curve(dias:end))  ->  curve[dias:]; global = local + dias.
    a_seg = curve[dias:]
    if a_seg.size == 0:  # unreachable for a valid `dias`, kept for total safety
        aprime = None
        a_idx = None
    else:
        a_loc = int(np.argmax(a_seg))
        a_idx = a_loc + dias
        aprime = {"value": float(a_seg[a_loc]), "index": a_idx}

    # MATLAB: if eprime(1,2) == aprime(1,2), aprime = 0; end  (a bare scalar, no index)
    if eprime is not None and aprime is not None and e_idx == a_idx:
        aprime = {"value": 0.0, "index": None}

    return {"sprime": sprime, "eprime": eprime, "aprime": aprime, **phases}


# ======================================================================================
# convenience wrappers
# ======================================================================================


def rv_eprime(TV, Rxy, TimeVector, **kw) -> float | None:
    """RV e' in cm/s, or ``None`` where MATLAB returns ``[]``.

    Composition of :func:`tapse` and :func:`find_peak_velocities`, mirroring
    ``get_accuracy``'s::

        [~,e_gt,~] = findpeakvelocities(vel_gt,t,PD_gt(2));

    ``get_accuracy`` additionally guards with ``size(TV_gt,1) > 2``; that guard is
    reproduced here (fewer than 3 frames -> ``None``).

    Parameters
    ----------
    TV, Rxy, TimeVector
        See :func:`tapse`.
    **kw
        ``mode`` and ``point``, forwarded to :func:`tapse`.
    """
    tv = _check_tv(TV)
    if tv.shape[0] <= 2:  # MATLAB: size(TV_gt,1) > 2
        return None
    res = tapse(tv, Rxy, TimeVector, **kw)
    peaks = find_peak_velocities(res["velocity"], res["t"], res["index"])
    e = peaks["eprime"]
    return None if e is None else float(e["value"])


def clinical_metrics(TV, Rxy, TimeVector, **kw) -> dict:
    """TAPSE and the peak annular velocities for one subject, in one call.

    Equivalent to the ``get_AVPD`` + ``findpeakvelocities`` pair that
    ``get_accuracy`` runs on each of the ground-truth and predicted landmark tracks.

    Parameters
    ----------
    TV, Rxy, TimeVector
        See :func:`tapse`.
    **kw
        ``mode`` (``'mean'`` | ``'single'``) and ``point`` (0 | 1).

    Returns
    -------
    dict with keys

    ``tapse_mm``
        float -- TAPSE, mm.
    ``est_index``
        int -- 0-based end-systolic frame (``argmin`` of the smoothed curve).
    ``rve_cm_s``
        float or None -- RV e', cm/s.
    ``rve_index``
        int or None -- 0-based frame of e'.
    ``sprime_cm_s``, ``sprime_index``, ``aprime_cm_s``, ``aprime_index``
        the other two peaks (``aprime_index`` is ``None`` when MATLAB zeroed a').
    ``curve``, ``smooth``, ``velocity``, ``t``
        the underlying arrays (mm, mm, cm/s, s).
    ``phases``
        the Kovacs timings and phase indices from :func:`find_peak_velocities`
        (``None`` when there were <= 2 frames).
    ``mode``, ``point``
        Echo of the arguments.
    """
    tv = _check_tv(TV)
    res = tapse(tv, Rxy, TimeVector, **kw)

    if tv.shape[0] <= 2:  # MATLAB get_accuracy's `size(TV_gt,1) > 2` guard
        peaks = None
    else:
        peaks = find_peak_velocities(res["velocity"], res["t"], res["index"])

    def _pick(name):
        if peaks is None or peaks[name] is None:
            return None, None
        return peaks[name]["value"], peaks[name]["index"]

    s_val, s_idx = _pick("sprime")
    e_val, e_idx = _pick("eprime")
    a_val, a_idx = _pick("aprime")

    phases: dict[str, Any] | None
    if peaks is None:
        phases = None
    else:
        phases = {k: v for k, v in peaks.items()
                  if k not in ("sprime", "eprime", "aprime")}

    return {
        "tapse_mm": res["value_mm"],
        "est_index": res["index"],
        "rve_cm_s": e_val,
        "rve_index": e_idx,
        "sprime_cm_s": s_val,
        "sprime_index": s_idx,
        "aprime_cm_s": a_val,
        "aprime_index": a_idx,
        "curve": res["curve"],
        "smooth": res["smooth"],
        "velocity": res["velocity"],
        "t": res["t"],
        "phases": phases,
        "mode": res["geometry"]["mode"],
        "point": res["geometry"]["point"],
    }

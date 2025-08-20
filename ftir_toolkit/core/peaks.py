from __future__ import annotations
import numpy as np

def local_linear_baseline(xw: np.ndarray, yw: np.ndarray) -> np.ndarray:
    x0, x1 = float(xw[0]), float(xw[-1])
    y0, y1 = float(yw[0]), float(yw[-1])
    if x1 == x0:
        return np.full_like(yw, (y0 + y1) / 2.0)
    m = (y1 - y0) / (x1 - x0)
    b = y0 - m * x0
    return m * xw + b

def band_peak_metrics(x: np.ndarray, y_abs: np.ndarray, center: float, fullwidth: float) -> tuple[float, float, float]:
    """
    In a descending-x window [center-½W, center+½W]:
      - Build local linear baseline
      - Find argmax on baseline-corrected signal
    Returns:
      xpk (cm-1), y_baseline_at_peak (absorbance), height (absorbance)
    (np.nan, np.nan, np.nan) if invalid window.
    """
    half = fullwidth / 2.0
    left = center + half
    right = center - half
    mask = (x <= left) & (x >= right)
    if np.sum(mask) < 5:
        return np.nan, np.nan, np.nan

    xw = x[mask]
    yw = y_abs[mask]
    bl = local_linear_baseline(xw, yw)
    yw_corr = yw - bl

    k = int(np.nanargmax(yw_corr))
    return float(xw[k]), float(bl[k]), float(yw_corr[k])

from __future__ import annotations
from scipy.signal import savgol_filter
import numpy as np

def absorbance_transform(I: np.ndarray) -> np.ndarray:
    eps = np.finfo(float).eps
    return 2.0 - np.log10(np.clip(I, eps, None))

def ensure_descending(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (x[::-1], y[::-1]) if x[0] < x[-1] else (x, y)

def second_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    return np.gradient(np.gradient(y, dx), dx)

def savgol_if_requested(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    from scipy.signal import savgol_filter
    if window % 2 == 0:
        window += 1
    if window < 5 or window > len(y):
        window = max(5, min(len(y)//2*2+1, len(y) - (1 - len(y)%2)))
    poly = min(poly, max(2, window - 2))
    return savgol_filter(y, window_length=window, polyorder=poly)

def savgol_smooth_and_derivative(y: np.ndarray, x: np.ndarray, window: int, poly: int):
    """
    Returns (y_smooth, d2) using Savitzky–Golay for both smoothing and second derivative.
    - Forces odd window and sane bounds.
    - Uses delta=|median(diff(x))| so it works whether x is descending or ascending.
    """
    if window % 2 == 0:
        window += 1
    if window < 5 or window > len(y):
        window = max(5, min(len(y)//2*2+1, len(y) - (1 - len(y)%2)))
    poly = min(poly, max(2, window - 2))

    dx = np.median(np.diff(x))
    delta = float(abs(dx)) if dx != 0 else 1.0

    y_smooth = savgol_filter(y, window_length=window, polyorder=poly, deriv=0)
    d2 = savgol_filter(y, window_length=window, polyorder=poly, deriv=2, delta=delta)
    return y_smooth, d2

def column_exponent(values: np.ndarray) -> int:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0
    med = np.median(np.abs(vals))
    if med <= 0:
        return 0
    return int(np.floor(np.log10(med)))

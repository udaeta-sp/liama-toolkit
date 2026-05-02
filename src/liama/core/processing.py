"""Spectral processing functions.

Each function documents the exact mathematical operation it performs.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Smoothing & Derivatives (mutually exclusive in pipeline)
# ---------------------------------------------------------------------------

def smooth_sg(y: np.ndarray, window: int = 15, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing.

    Operation: scipy.signal.savgol_filter(y, window, polyorder)
    Preserves peak shapes better than moving average.
    """
    window = _ensure_odd(window)
    polyorder = min(polyorder, window - 1)
    if len(y) < window:
        return y.copy()
    return savgol_filter(y, window, polyorder)


def second_derivative_sg(
    y: np.ndarray, window: int = 15, polyorder: int = 3, delta: float = 1.0
) -> np.ndarray:
    """Second derivative via Savitzky-Golay.

    Operation: scipy.signal.savgol_filter(y, window, polyorder, deriv=2, delta=delta)
    Combines smoothing and differentiation in one step.
    The delta parameter should be the spacing between wavenumber points.
    """
    window = _ensure_odd(window)
    polyorder = min(polyorder, window - 1)
    if polyorder < 2:
        polyorder = 2
        window = max(window, 3)
    if len(y) < window:
        return np.zeros_like(y)
    return savgol_filter(y, window, polyorder, deriv=2, delta=delta)


# ---------------------------------------------------------------------------
# Baseline correction
# ---------------------------------------------------------------------------

def baseline_polynomial(
    wn: np.ndarray, y: np.ndarray, degree: int = 2
) -> np.ndarray:
    """Polynomial baseline correction.

    Operation:
        coeffs = numpy.polyfit(wn, y, degree)
        baseline = numpy.polyval(coeffs, wn)
        y_corrected = y - baseline

    Fits a polynomial of given degree to the spectrum and subtracts it.
    """
    if len(y) < degree + 1:
        return y.copy()
    coeffs = np.polyfit(wn, y, degree)
    baseline = np.polyval(coeffs, wn)
    return y - baseline


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_row_sum(y: np.ndarray) -> np.ndarray:
    """Row-sum normalization (proportion).

    Operation: y_norm = y / sum(y)
    Each value becomes a proportion of the total. Row sums to 1.
    """
    total = np.sum(y)
    if total == 0:
        return y.copy()
    return y / total


def normalize_minmax(y: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0, 1].

    Operation: y_norm = (y - min(y)) / (max(y) - min(y))
    """
    ymin, ymax = np.min(y), np.max(y)
    rng = ymax - ymin
    if rng == 0:
        return np.zeros_like(y)
    return (y - ymin) / rng


def normalize_snv(y: np.ndarray) -> np.ndarray:
    """Standard Normal Variate (SNV).

    Operation: y_snv = (y - mean(y)) / std(y)
    Centers and scales each spectrum individually.
    """
    m = np.mean(y)
    s = np.std(y)
    if s == 0:
        return np.zeros_like(y)
    return (y - m) / s


def transform_hellinger(y: np.ndarray) -> np.ndarray:
    """Hellinger transformation.

    Operation:
        y_prop = y / sum(y)       (row-sum normalization)
        y_hell = sqrt(y_prop)     (square root of proportions)

    Suitable for compositional data. Requires non-negative values.
    Negative values are clipped to 0 before transformation.
    """
    y_pos = np.clip(y, 0, None)
    total = np.sum(y_pos)
    if total == 0:
        return np.zeros_like(y)
    return np.sqrt(y_pos / total)


# ---------------------------------------------------------------------------
# Scaling (applied column-wise to a matrix)
# ---------------------------------------------------------------------------

def scale_standard(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard scaling (UV-scaling / autoscaling).

    Operation: X_scaled = (X - mean) / std, per column
    Equivalent to sklearn.preprocessing.StandardScaler().fit_transform(X)

    Returns: (X_scaled, means, stds) — means and stds needed for projection.
    """
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0, ddof=0)
    stds[stds == 0] = 1.0
    return (X - means) / stds, means, stds


def scale_pareto(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pareto scaling.

    Operation: X_scaled = (X - mean) / sqrt(std), per column
    Less aggressive than standard scaling — preserves more of the
    original variance structure.

    Returns: (X_scaled, means, sqrt_stds)
    """
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0, ddof=0)
    stds[stds == 0] = 1.0
    sqrt_stds = np.sqrt(stds)
    return (X - means) / sqrt_stds, means, sqrt_stds


# ---------------------------------------------------------------------------
# Pipeline application
# ---------------------------------------------------------------------------

def apply_pipeline_to_spectrum(
    wn: np.ndarray, y: np.ndarray, steps: list[dict]
) -> np.ndarray:
    """Apply a sequence of processing steps to a single spectrum.

    Each step is a dict: {"name": str, "enabled": bool, "params": dict}
    Steps are applied in the order given.
    """
    result = y.copy()
    for step in steps:
        if not step.get("enabled", False):
            continue
        name = step["name"]
        params = step.get("params", {})

        if name == "smooth_sg":
            result = smooth_sg(result, **params)
        elif name == "second_derivative_sg":
            delta = abs(wn[1] - wn[0]) if len(wn) > 1 else 1.0
            result = second_derivative_sg(result, delta=delta, **params)
        elif name == "baseline_polynomial":
            result = baseline_polynomial(wn, result, **params)
        elif name == "normalize_row_sum":
            result = normalize_row_sum(result)
        elif name == "normalize_minmax":
            result = normalize_minmax(result)
        elif name == "normalize_snv":
            result = normalize_snv(result)
        elif name == "transform_hellinger":
            result = transform_hellinger(result)
    return result


def apply_pipeline_to_matrix(
    wn: np.ndarray, X: np.ndarray, steps: list[dict]
) -> tuple[np.ndarray, dict]:
    """Apply pipeline to a matrix (n_samples × n_wavenumbers).

    Row-wise operations (smooth, derivative, baseline, normalize) are
    applied per row. Column-wise operations (scaling) are applied to
    the whole matrix.

    Returns: (X_processed, fit_params) where fit_params contains
    scaling parameters needed for projecting new samples.
    """
    result = X.copy()
    fit_params = {}

    for step in steps:
        if not step.get("enabled", False):
            continue
        name = step["name"]
        params = step.get("params", {})

        # Row-wise operations
        if name in ("smooth_sg", "second_derivative_sg", "baseline_polynomial",
                     "normalize_row_sum", "normalize_minmax", "normalize_snv",
                     "transform_hellinger"):
            for i in range(result.shape[0]):
                result[i] = apply_pipeline_to_spectrum(
                    wn, result[i], [step]
                )

        # Column-wise operations
        elif name == "scale_standard":
            result, means, stds = scale_standard(result)
            fit_params["scale_means"] = means
            fit_params["scale_stds"] = stds
            fit_params["scale_type"] = "standard"

        elif name == "scale_pareto":
            result, means, sqrt_stds = scale_pareto(result)
            fit_params["scale_means"] = means
            fit_params["scale_stds"] = sqrt_stds
            fit_params["scale_type"] = "pareto"

    return result, fit_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_odd(n: int) -> int:
    """Ensure window is odd and >= 3."""
    n = max(3, n)
    return n if n % 2 == 1 else n + 1

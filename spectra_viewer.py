"""Utilities for plotting and saving spectra.

This module provides two helper functions, :func:`plot_spectrum` and
:func:`save_spectrum`, which optionally apply a Savitzky–Golay filter to
obtain the second derivative of a spectrum.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def _second_derivative(y: np.ndarray, x: np.ndarray, apply_smoothing: bool,
                       window: int, poly: int) -> np.ndarray:
    """Return the second derivative of *y*.

    Parameters
    ----------
    y, x : :class:`numpy.ndarray`
        Input spectral data.
    apply_smoothing : bool
        Whether to apply Savitzky–Golay smoothing. If ``False`` the second
        derivative is obtained using :func:`numpy.gradient`.
    window : int
        Window length for the Savitzky–Golay filter.  Will be incremented by
        one if an even value is supplied so that it is always odd.
    poly : int
        Polynomial order for the Savitzky–Golay filter.
    """
    if apply_smoothing:
        if window % 2 == 0:
            window += 1
        return savgol_filter(y, window_length=window, polyorder=poly, deriv=2)

    dx = x[1] - x[0] if len(x) > 1 else 1.0
    return np.gradient(np.gradient(y, dx), dx)


def plot_spectrum(x: np.ndarray, y: np.ndarray, *, apply_smoothing: bool = False,
                  window: int = 11, poly: int = 3, ax: plt.Axes | None = None) -> np.ndarray:
    """Plot a spectrum and its second derivative.

    Returns the second derivative array so callers (and tests) can inspect the
    values produced by the smoothing routine.
    """
    second_derivative = _second_derivative(y, x, apply_smoothing, window, poly)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x, y, label="spectrum")
    ax.plot(x, second_derivative, label="second derivative", color="red")
    ax.legend()
    return second_derivative


def save_spectrum(x: np.ndarray, y: np.ndarray, filename: str, *,
                  apply_smoothing: bool = False, window: int = 11,
                  poly: int = 3) -> np.ndarray:
    """Save a spectrum and its second derivative to ``filename``.

    The second derivative array is returned for convenience and testing.
    """
    second_derivative = _second_derivative(y, x, apply_smoothing, window, poly)

    fig, ax = plt.subplots()
    ax.plot(x, y, label="spectrum")
    ax.plot(x, second_derivative, label="second derivative", color="red")
    ax.legend()
    fig.savefig(filename)
    plt.close(fig)
    return second_derivative

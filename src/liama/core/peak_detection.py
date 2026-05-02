"""Peak detection for FTIR spectra."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def detect_peaks(
    wn: np.ndarray,
    y: np.ndarray,
    prominence: float = 0.05,
    distance: int = 10,
) -> list[dict]:
    """Detect peaks in a spectrum.

    Uses scipy.signal.find_peaks on the absorbance signal.
    Returns list of dicts with 'wavenumber', 'absorbance', 'index'.
    """
    if len(y) < 3:
        return []

    indices, properties = find_peaks(y, prominence=prominence, distance=distance)

    peaks = []
    for idx in indices:
        peaks.append({
            "wavenumber": float(wn[idx]),
            "absorbance": float(y[idx]),
            "index": int(idx),
        })

    # Sort by wavenumber descending (high to low, FTIR convention)
    peaks.sort(key=lambda p: p["wavenumber"], reverse=True)
    return peaks

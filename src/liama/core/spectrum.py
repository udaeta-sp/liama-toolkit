"""Spectrum data container."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


def interp_to_grid(
    target_wn: np.ndarray, src_wn: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Resample `values` onto `target_wn`, both grids in descending order.

    np.interp requires ascending x, so both are flipped in and the result is
    flipped back to stay aligned with target_wn.
    """
    return np.interp(target_wn[::-1], src_wn[::-1], values[::-1])[::-1]


@dataclass
class Spectrum:
    """Single FTIR-ATR spectrum with optional metadata."""

    name: str
    wavenumbers: np.ndarray  # cm⁻¹, descending order
    absorbance: np.ndarray   # absorbance units
    file_path: Path | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    load_error: str | None = None

    @property
    def n_points(self) -> int:
        return len(self.wavenumbers)

    @property
    def wn_min(self) -> float:
        return float(self.wavenumbers[-1])

    @property
    def wn_max(self) -> float:
        return float(self.wavenumbers[0])

    def interpolate_to(self, target_wn: np.ndarray) -> np.ndarray:
        """Interpolate absorbance onto a common (descending) wavenumber grid."""
        return interp_to_grid(target_wn, self.wavenumbers, self.absorbance)

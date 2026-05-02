"""Spectrum data container."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


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

    def slice_range(self, wn_low: float, wn_high: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (wavenumbers, absorbance) within [wn_low, wn_high]."""
        mask = (self.wavenumbers >= wn_low) & (self.wavenumbers <= wn_high)
        return self.wavenumbers[mask], self.absorbance[mask]

    def interpolate_to(self, target_wn: np.ndarray) -> np.ndarray:
        """Interpolate absorbance onto a common wavenumber grid."""
        return np.interp(target_wn, self.wavenumbers[::-1], self.absorbance[::-1])[::-1]

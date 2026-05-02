"""Read Thermo/Nicolet .SPA files via SpectroChemPy."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .spectrum import Spectrum

log = logging.getLogger(__name__)


def read_spa(path: Path) -> Spectrum:
    """Read a single .SPA file and return a Spectrum with absorbance data.

    Uses spectrochempy.read_omnic for robust binary parsing.
    Converts transmittance (%) to absorbance: A = 2 - log10(T).
    """
    try:
        import spectrochempy as scp

        dataset = scp.read_omnic(str(path))

        # Extract wavenumber axis (x coordinates)
        wn = dataset.x.data
        if hasattr(wn, "magnitude"):
            wn = wn.magnitude
        wn = np.asarray(wn, dtype=np.float64).ravel()

        # Extract intensity values
        y = dataset.data
        if hasattr(y, "magnitude"):
            y = y.magnitude
        y = np.asarray(y, dtype=np.float64).ravel()

        # Check units — convert transmittance to absorbance if needed
        units_str = str(getattr(dataset, "units", "")).lower()
        title_str = str(getattr(dataset, "title", "")).lower()

        is_transmittance = (
            "transmittance" in units_str
            or "transmittance" in title_str
            or "%" in units_str
        )

        if is_transmittance:
            eps = 1e-10
            y_clamped = np.clip(y, eps, None)
            absorbance = 2.0 - np.log10(y_clamped)
        else:
            absorbance = y.copy()

        # Ensure descending wavenumber order (convention: 4000 → 400)
        if len(wn) > 1 and wn[0] < wn[-1]:
            wn = wn[::-1]
            absorbance = absorbance[::-1]

        # Validate physical plausibility
        if np.any(np.abs(absorbance) > 100):
            log.warning(
                "%s: absorbance values exceed 100, likely misread", path.name
            )

        return Spectrum(
            name=path.stem,
            wavenumbers=wn,
            absorbance=absorbance,
            file_path=path,
        )

    except Exception as e:
        log.error("Failed to read %s: %s", path.name, e)
        return Spectrum(
            name=path.stem,
            wavenumbers=np.array([]),
            absorbance=np.array([]),
            file_path=path,
            load_error=str(e),
        )


def scan_folder(folder: Path) -> list[Spectrum]:
    """Recursively scan a folder for .SPA files and load them."""
    spa_files = sorted(folder.rglob("*.SPA"), key=lambda p: p.stem)
    # Also catch lowercase extension
    spa_files += sorted(folder.rglob("*.spa"), key=lambda p: p.stem)
    # Remove duplicates (case-insensitive filesystems)
    seen = set()
    unique = []
    for f in spa_files:
        key = str(f).lower()
        if key not in seen:
            seen.add(key)
            unique.append(f)

    spectra = []
    for f in unique:
        sp = read_spa(f)
        spectra.append(sp)
    return spectra

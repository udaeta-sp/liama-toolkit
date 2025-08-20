import numpy as np
from ftir_toolkit.core.peaks import band_peak_metrics

def test_band_peak_metrics_simple():
    x = np.linspace(2000, 600, 1401)  # descending if reversed
    x = x[::-1]  # make descending high->low cm-1
    # synthetic peak at 1313 cm-1
    center = 1313.0
    y = 0.2 + 0.5 * np.exp(-0.5*((x-center)/5.0)**2)
    xpk, ybl, h = band_peak_metrics(x, y, 1313.0, 100.0)
    assert np.isfinite(xpk) and np.isfinite(ybl) and np.isfinite(h)

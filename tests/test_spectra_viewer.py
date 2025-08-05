import numpy as np
from scipy.signal import savgol_filter

from spectra_viewer import plot_spectrum, save_spectrum


def test_plot_spectrum_applies_savgol():
    x = np.linspace(0, 10, 50)
    y = np.sin(x)
    result = plot_spectrum(x, y, apply_smoothing=True, window=10, poly=3)
    expected = savgol_filter(y, window_length=11, polyorder=3, deriv=2)
    assert np.allclose(result, expected)


def test_save_spectrum_applies_savgol(tmp_path):
    x = np.linspace(0, 10, 50)
    y = np.sin(x)
    outfile = tmp_path / "spec.png"
    result = save_spectrum(x, y, str(outfile), apply_smoothing=True, window=10, poly=2)
    expected = savgol_filter(y, window_length=11, polyorder=2, deriv=2)
    assert np.allclose(result, expected)
    assert outfile.exists()

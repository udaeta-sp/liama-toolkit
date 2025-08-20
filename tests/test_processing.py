import numpy as np
from ftir_toolkit.core.processing import absorbance_transform, ensure_descending, column_exponent

def test_absorbance_transform_no_neg_inf():
    I = np.array([1.0, 0.0, 10.0])
    A = absorbance_transform(I)
    assert np.isfinite(A).all()

def test_ensure_descending():
    x = np.array([1, 2, 3], float)
    y = np.array([10, 20, 30], float)
    xr, yr = ensure_descending(x, y)
    assert xr[0] > xr[-1]

def test_column_exponent():
    vals = np.array([1e-3, 2e-3, 5e-3])
    e = column_exponent(vals)
    assert e == -3

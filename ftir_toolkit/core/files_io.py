from __future__ import annotations
import os
from typing import List, Tuple
import numpy as np
import spectrochempy as scp

def list_spa_files(folder: str) -> Tuple[List[str], List[str]]:
    files = [f for f in os.listdir(folder) if f.lower().endswith(".spa")]
    paths = [os.path.join(folder, f) for f in files]
    return files, paths

def read_spa_absorbance(path: str) -> tuple[np.ndarray, np.ndarray, str, str]:
    """
    Read .SPA via SpectroChemPy and return (x_cm1, A_absorbance, x_units, y_units).
    A = 2 - log10(I); small epsilon clamp applied inside processing transform.
    """
    ds = scp.read_omnic(path)
    x = ds.x.data.astype(float)
    I = ds.data[0].astype(float)
    from .processing import absorbance_transform
    A = absorbance_transform(I)
    return x, A, str(ds.x.units), str(ds.units)

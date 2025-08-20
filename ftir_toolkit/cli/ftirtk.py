from __future__ import annotations
import argparse
import numpy as np
from ..core.files_io import list_spa_files, read_spa_absorbance
from ..core.processing import ensure_descending
from ..core.peaks import band_peak_metrics

def main():
    ap = argparse.ArgumentParser(description="FTIR-ATR CLI")
    ap.add_argument("folder", help="Folder with .SPA files")
    ap.add_argument("--bands", nargs="+", default=["1602:100", "1313:100", "778:100"],
                    help="List bands as center:fullwidth (cm-1)")
    args = ap.parse_args()

    files, paths = list_spa_files(args.folder)
    bands = []
    for b in args.bands:
        c, w = b.split(":")
        bands.append((float(c), float(w)))

    print("archivo," + ",".join([f"{int(c)}_height" for c, _ in bands]))
    for f, p in zip(files, paths):
        try:
            x, A, _, _ = read_spa_absorbance(p)
            x, A = ensure_descending(x, A)
            heights = []
            for c, w in bands:
                _, _, h = band_peak_metrics(x, A, c, w)
                heights.append(h)
            vals = [("nan" if not np.isfinite(h) else f"{h:.6e}") for h in heights]
            print(f"{f}," + ",".join(vals))
        except Exception:
            print(f"{f}," + ",".join(["nan"] * len(bands)))

if __name__ == "__main__":
    main()

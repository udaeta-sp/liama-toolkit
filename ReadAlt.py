import os
from typing import Dict, List, Sequence, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
import spectrochempy as scp
from scipy.signal import savgol_filter


def load_spa_files(folder: str) -> Tuple[List[str], List[str]]:
    """Return SPA file names and paths found in ``folder``."""
    files = [f for f in os.listdir(folder) if f.lower().endswith(".spa")]
    paths = [os.path.join(folder, f) for f in files]
    return files, paths


def get_default_range(path: str) -> Tuple[float, float]:
    """Return the default X range from the SPA file located at ``path``."""
    dataset = scp.read_omnic(path)
    x = dataset.x.data
    return float(np.max(x)), float(np.min(x))


def plot_spectrum(
    path: str,
    apply_smoothing: bool = False,
    window: int = 15,
    poly: int = 3,
    xmin: float | None = None,
    xmax: float | None = None,
    bands: Sequence[Tuple[float, float]] | None = None,
    band_colors: Sequence[str] | None = None,
    peaks: Sequence[float] | None = None,
    peak_colors: Sequence[str] | None = None,
    x_lines: Sequence[float] | None = None,
) -> None:
    """Display the spectrum and its second derivative."""
    dataset = scp.read_omnic(path)
    x = dataset.x.data
    y = dataset.data[0]
    absorbance = 2 - np.log10(y)
    y = absorbance

    if apply_smoothing:
        if window % 2 == 0:
            window += 1
        second_derivative = savgol_filter(y, window_length=window, polyorder=poly, deriv=2)
    else:
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        second_derivative = np.gradient(np.gradient(y, dx), dx)

    if xmin is None or xmax is None:
        xmin, xmax = float(np.max(x)), float(np.min(x))
    if xmin < xmax:
        xmin, xmax = xmax, xmin

    mask = (x <= xmin) & (x >= xmax)
    x_range = x[mask]
    y_range = y[mask]
    der_range = second_derivative[mask]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, dpi=130)
    ax1.plot(x_range, y_range, color="blue", lw=1)
    ax1.set_ylabel(f"{dataset.units}")
    ax1.set_title("FTIR-ATR")
    ax1.tick_params(labelsize=8)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(min(y_range) * 0.95, max(y_range) * 1.05)

    ax2.plot(x_range, der_range, color="red", lw=1)
    ax2.set_xlabel(f"{dataset.x.units}")
    ax2.set_ylabel("2nd derivative")
    ax2.set_title("Second derivative")
    ax2.tick_params(labelsize=8)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(min(der_range) * 1.1, max(der_range) * 1.1)

    bands = bands or []
    band_colors = band_colors or []
    for idx, (center, width) in enumerate(bands):
        color = band_colors[idx] if idx < len(band_colors) else "gray"
        ax1.axvspan(center - width / 2, center + width / 2, color=color, alpha=0.3)
        ax2.axvspan(center - width / 2, center + width / 2, color=color, alpha=0.3)

    x_lines = x_lines or []
    for line in x_lines:
        ax1.axvline(x=line, color="green", linestyle="--", linewidth=1)
        ax2.axvline(x=line, color="green", linestyle="--", linewidth=1)

    peaks = peaks or []
    peak_colors = peak_colors or []
    for idx, peak in enumerate(peaks):
        color = peak_colors[idx] if idx < len(peak_colors) else "black"
        if min(x_range) <= peak <= max(x_range):
            idx_close = np.argmin(np.abs(x - peak))
            y_val = y[idx_close]
            ax1.scatter(
                peak,
                y_val - (max(y_range) - min(y_range)) * 0.03,
                marker="^",
                color=color,
            )

    plt.tight_layout()
    plt.show()


def save_spectrum(
    path: str,
    apply_smoothing: bool = False,
    window: int = 15,
    poly: int = 3,
    xmin: float | None = None,
    xmax: float | None = None,
    bands: Sequence[Tuple[float, float]] | None = None,
    band_colors: Sequence[str] | None = None,
    peaks: Sequence[float] | None = None,
    peak_colors: Sequence[str] | None = None,
    x_lines: Sequence[float] | None = None,
    output_dir: str | None = None,
    include_derivative: bool = True,
) -> str:
    """Save the spectrum (and optional second derivative) as a PNG image."""
    dataset = scp.read_omnic(path)
    x = dataset.x.data
    y = dataset.data[0]
    absorbance = 2 - np.log10(y)
    y = absorbance

    if apply_smoothing:
        if window % 2 == 0:
            window += 1
        second_derivative = savgol_filter(y, window_length=window, polyorder=poly, deriv=2)
    else:
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        second_derivative = np.gradient(np.gradient(y, dx), dx)

    if xmin is None or xmax is None:
        xmin, xmax = float(np.max(x)), float(np.min(x))
    if xmin < xmax:
        xmin, xmax = xmax, xmin

    mask = (x <= xmin) & (x >= xmax)
    x_range = x[mask]
    y_range = y[mask]
    der_range = second_derivative[mask]

    fig, ax1 = plt.subplots(figsize=(6, 5), dpi=300)
    ax1.plot(x_range, y_range, color="blue", lw=1, label="FTIR-ATR")
    ax1.set_ylabel("Absorbance", color="blue")
    ax1.set_xlabel("Wavenumber (cm⁻¹)")
    ax1.tick_params(axis="y", labelcolor="blue", labelsize=8)
    ax1.tick_params(axis="x", labelsize=8)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(min(y_range) * 0.95, max(y_range) * 1.05)

    if include_derivative:
        ax2 = ax1.twinx()
        ax2.plot(x_range, der_range, color="red", lw=1, label="2nd derivative")
        ax2.set_ylabel("2nd derivative", color="red")
        ax2.tick_params(axis="y", labelcolor="red", labelsize=8)
        ax2.set_ylim(min(der_range) * 1.1, max(der_range) * 1.8)

    bands = bands or []
    band_colors = band_colors or []
    for idx, (center, width) in enumerate(bands):
        color = band_colors[idx] if idx < len(band_colors) else "gray"
        ax1.axvspan(center - width / 2, center + width / 2, color=color, alpha=0.3)
        if include_derivative:
            ax2.axvspan(center - width / 2, center + width / 2, color=color, alpha=0.3)

    x_lines = x_lines or []
    for line in x_lines:
        ax1.axvline(x=line, color="green", linestyle="--", linewidth=1)
        if include_derivative:
            ax2.axvline(x=line, color="green", linestyle="--", linewidth=1)

    peaks = peaks or []
    peak_colors = peak_colors or []
    for idx, peak in enumerate(peaks):
        color = peak_colors[idx] if idx < len(peak_colors) else "black"
        if min(x_range) <= peak <= max(x_range):
            idx_close = np.argmin(np.abs(x - peak))
            y_val = y[idx_close]
            ax1.scatter(
                peak,
                y_val - (max(y_range) - min(y_range)) * 0.03,
                marker="^",
                color=color,
            )

    plt.title(
        "FTIR-ATR and second derivative" if include_derivative else "FTIR-ATR"
    )
    plt.tight_layout()
    name_base = os.path.splitext(os.path.basename(path))[0]
    out_dir = output_dir or os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name_base}_range.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


class SpectraViewerApp:
    """Tkinter application for viewing and exporting SPA spectra."""

    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title("SPA Spectra Viewer")

        self.file_paths: Dict[str, str] = {}

        self.selected_file = tk.StringVar()
        self.smooth_var = tk.BooleanVar()
        self.window_var = tk.IntVar(value=15)
        self.poly_var = tk.IntVar(value=3)
        self.derivative_var = tk.BooleanVar(value=True)
        self.folder_var = tk.StringVar()

        self._build_ui()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.master, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.master.columnconfigure(0, weight=1)

        file_frame = ttk.Frame(main)
        file_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        ttk.Button(
            file_frame, text="Browse Folder", command=self.browse_folder
        ).grid(row=0, column=0, padx=5)
        self.file_combo = ttk.Combobox(
            file_frame, textvariable=self.selected_file, state="readonly", width=40
        )
        self.file_combo.grid(row=0, column=1, padx=5, sticky="ew")
        file_frame.columnconfigure(1, weight=1)
        ttk.Button(
            file_frame, text="View Spectrum", command=self.show_selected
        ).grid(row=0, column=2, padx=5)

        proc_frame = ttk.Labelframe(main, text="Processing")
        proc_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Checkbutton(
            proc_frame, text="Apply smoothing", variable=self.smooth_var
        ).grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(proc_frame, text="Window").grid(row=0, column=1, sticky="e")
        ttk.Entry(proc_frame, textvariable=self.window_var, width=6).grid(
            row=0, column=2, padx=5
        )
        ttk.Label(proc_frame, text="Poly order").grid(row=0, column=3, sticky="e")
        ttk.Entry(proc_frame, textvariable=self.poly_var, width=6).grid(
            row=0, column=4, padx=5
        )

        range_frame = ttk.Labelframe(main, text="X range (cm⁻¹)")
        range_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        self.min_entry = ttk.Entry(range_frame, width=10)
        self.min_entry.grid(row=0, column=0, padx=5)
        self.max_entry = ttk.Entry(range_frame, width=10)
        self.max_entry.grid(row=0, column=1, padx=5)

        annot_frame = ttk.Labelframe(main, text="Annotations")
        annot_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Label(annot_frame, text="Bands (center:width,...)").grid(
            row=0, column=0, sticky="e"
        )
        self.band_entry = ttk.Entry(annot_frame, width=28)
        self.band_entry.grid(row=0, column=1, columnspan=2, padx=5, sticky="ew")
        ttk.Label(annot_frame, text="Band colors").grid(row=1, column=0, sticky="e")
        self.band_color_entry = ttk.Entry(annot_frame, width=28)
        self.band_color_entry.insert(0, "gray")
        self.band_color_entry.grid(row=1, column=1, columnspan=2, padx=5, sticky="ew")
        ttk.Label(annot_frame, text="Peaks (x,...)").grid(row=2, column=0, sticky="e")
        self.peak_entry = ttk.Entry(annot_frame, width=28)
        self.peak_entry.grid(row=2, column=1, columnspan=2, padx=5, sticky="ew")
        ttk.Label(annot_frame, text="Peak colors").grid(row=3, column=0, sticky="e")
        self.peak_color_entry = ttk.Entry(annot_frame, width=28)
        self.peak_color_entry.insert(0, "black")
        self.peak_color_entry.grid(row=3, column=1, columnspan=2, padx=5, sticky="ew")
        ttk.Label(annot_frame, text="Vertical lines (x,...)").grid(
            row=4, column=0, sticky="e"
        )
        self.lines_entry = ttk.Entry(annot_frame, width=28)
        self.lines_entry.grid(row=4, column=1, columnspan=2, padx=5, sticky="ew")
        ttk.Checkbutton(
            annot_frame,
            text="Save second derivative",
            variable=self.derivative_var,
        ).grid(row=5, column=0, columnspan=3, sticky="w", padx=5, pady=2)

        export_frame = ttk.Frame(main)
        export_frame.grid(row=4, column=0, columnspan=3, pady=10)
        ttk.Button(
            export_frame, text="Export Image", command=self.export_image
        ).grid(row=0, column=0, padx=5)
        ttk.Button(
            export_frame, text="Export All", command=self.export_all
        ).grid(row=0, column=1, padx=5)

    def browse_folder(self) -> None:
        folder = filedialog.askdirectory()
        if not folder:
            return
        files, paths = load_spa_files(folder)
        if not files:
            messagebox.showinfo(
                "Information", "No .SPA files found in the selected folder."
            )
            return
        self.file_paths = dict(zip(files, paths))
        self.file_combo["values"] = files
        self.file_combo.current(0)
        self.set_default_range(paths[0])
        self.folder_var.set(folder)

    def set_default_range(self, path: str) -> None:
        xmin, xmax = get_default_range(path)
        self.min_entry.delete(0, tk.END)
        self.min_entry.insert(0, str(xmin))
        self.max_entry.delete(0, tk.END)
        self.max_entry.insert(0, str(xmax))

    def parse_bands(self) -> Tuple[List[Tuple[float, float]], List[str]]:
        raw = self.band_entry.get().strip()
        color_raw = self.band_color_entry.get().strip()
        bands: List[Tuple[float, float]] = []
        colors: List[str] = []
        if raw:
            for item in raw.split(','):
                if ':' in item:
                    center, width = item.split(':')
                    bands.append((float(center), float(width)))
        if color_raw:
            colors = [c.strip() for c in color_raw.split(',') if c.strip()]
        return bands, colors

    def parse_peaks(self) -> Tuple[List[float], List[str]]:
        raw = self.peak_entry.get().strip()
        color_raw = self.peak_color_entry.get().strip()
        peaks: List[float] = []
        colors: List[str] = []
        if raw:
            peaks = [float(p.strip()) for p in raw.split(',') if p.strip()]
        if color_raw:
            colors = [c.strip() for c in color_raw.split(',') if c.strip()]
        return peaks, colors

    def parse_lines(self) -> List[float]:
        raw = self.lines_entry.get().strip()
        if raw:
            return [float(v.strip()) for v in raw.split(',') if v.strip()]
        return []

    def read_range(self, path: str | None = None) -> Tuple[float, float]:
        xmin = self.min_entry.get().strip()
        xmax = self.max_entry.get().strip()
        if not xmin or not xmax:
            if path:
                return get_default_range(path)
            raise ValueError
        xmin_f = float(xmin)
        xmax_f = float(xmax)
        if xmin_f > xmax_f:
            xmin_f, xmax_f = xmax_f, xmin_f
        return xmin_f, xmax_f

    def show_selected(self) -> None:
        file = self.file_combo.get()
        path = self.file_paths.get(file)
        if not path:
            return
        try:
            xmin, xmax = self.read_range(path)
        except Exception:
            messagebox.showerror(
                "Error", "Please provide a valid range (xmin < xmax)"
            )
            return
        bands, band_colors = self.parse_bands()
        peaks, peak_colors = self.parse_peaks()
        lines = self.parse_lines()
        plot_spectrum(
            path,
            apply_smoothing=self.smooth_var.get(),
            window=self.window_var.get(),
            poly=self.poly_var.get(),
            xmin=xmin,
            xmax=xmax,
            bands=bands,
            band_colors=band_colors,
            peaks=peaks,
            peak_colors=peak_colors,
            x_lines=lines,
        )

    def export_image(self) -> None:
        file = self.file_combo.get()
        path = self.file_paths.get(file)
        if not path:
            return
        try:
            xmin, xmax = self.read_range(path)
        except Exception:
            messagebox.showerror(
                "Error", "Please provide a valid range (xmin < xmax)"
            )
            return
        bands, band_colors = self.parse_bands()
        peaks, peak_colors = self.parse_peaks()
        lines = self.parse_lines()
        out_path = save_spectrum(
            path,
            apply_smoothing=self.smooth_var.get(),
            window=self.window_var.get(),
            poly=self.poly_var.get(),
            xmin=xmin,
            xmax=xmax,
            bands=bands,
            band_colors=band_colors,
            peaks=peaks,
            peak_colors=peak_colors,
            x_lines=lines,
            include_derivative=self.derivative_var.get(),
        )
        messagebox.showinfo("Image saved", f"Saved as:\n{out_path}")

    def export_all(self) -> None:
        folder = self.folder_var.get()
        if not folder:
            return
        files, paths = load_spa_files(folder)
        if not files:
            messagebox.showinfo(
                "Information", "No .SPA files found in the selected folder."
            )
            return
        try:
            xmin, xmax = self.read_range(paths[0])
        except Exception:
            messagebox.showerror(
                "Error", "Please provide a valid range (xmin < xmax)"
            )
            return
        bands, band_colors = self.parse_bands()
        peaks, peak_colors = self.parse_peaks()
        lines = self.parse_lines()
        out_folder = os.path.join(folder, "exported_images")
        os.makedirs(out_folder, exist_ok=True)
        for path in paths:
            save_spectrum(
                path,
                apply_smoothing=self.smooth_var.get(),
                window=self.window_var.get(),
                poly=self.poly_var.get(),
                xmin=xmin,
                xmax=xmax,
                bands=bands,
                band_colors=band_colors,
                peaks=peaks,
                peak_colors=peak_colors,
                x_lines=lines,
                output_dir=out_folder,
                include_derivative=self.derivative_var.get(),
            )
        messagebox.showinfo(
            "Export complete", f"Images saved in:\n{out_folder}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = SpectraViewerApp(root)
    root.mainloop()

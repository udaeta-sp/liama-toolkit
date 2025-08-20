from __future__ import annotations
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from scipy.signal import find_peaks

from ..config import BandsConfig, DisplayConfig
from ..core.files_io import list_spa_files, read_spa_absorbance
from ..core.processing import ensure_descending, second_derivative, savgol_if_requested, column_exponent
from ..core.peaks import band_peak_metrics
from ..viz.plotting import plot_spectrum_with_derivative, plot_spectrum_overlay, plot_spectrum_only


class FTIRApp:
    def __init__(self, master):
        self.master = master
        master.title("Visualizador y Exportador de Espectros .SPA")

        self.frame = tk.Frame(master, padx=10, pady=10)
        self.frame.pack(fill="both", expand=True)

        # Estado básico
        self.folder = tk.StringVar()
        self.files = []
        self.paths = []
        self.sel_file = tk.StringVar()

        # Parámetros
        self.var_smooth = tk.BooleanVar(value=False)
        self.var_deriv_export = tk.BooleanVar(value=True)
        self.savgol_win = tk.IntVar(value=15)
        self.savgol_poly = tk.IntVar(value=3)
        self.var_show_heights = tk.BooleanVar(value=False)

        # Picos globales (find_peaks)
        self.var_peaks = tk.BooleanVar(value=False)
        self.peak_height = tk.StringVar(value="")       # float or empty
        self.peak_prom = tk.StringVar(value="0.05")     # float or empty
        self.peak_dist = tk.StringVar(value="")         # int or empty

        # Entradas rango X
        self.entry_min = tk.Entry(self.frame, width=10)
        self.entry_max = tk.Entry(self.frame, width=10)

        # Bandas y colores (texto)
        self.entry_bands = tk.Entry(self.frame, width=28)
        self.entry_colors = tk.Entry(self.frame, width=28)
        self.entry_colors.insert(0, "green,cyan,red")

        # Líneas verticales
        self.entry_vlines = tk.Entry(self.frame, width=28)

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        # carpeta + archivo
        tk.Button(self.frame, text="Seleccionar carpeta", command=self._select_folder).grid(row=0, column=0, padx=5, pady=5)
        self.opt_files = ttk.OptionMenu(self.frame, self.sel_file, "")
        self.opt_files.grid(row=0, column=1, padx=5)
        tk.Button(self.frame, text="Ver espectro", command=self._open_selected).grid(row=0, column=2, padx=5, pady=5)

        # suavizado
        tk.Checkbutton(self.frame, text="Aplicar suavizado", variable=self.var_smooth).grid(row=1, column=0, sticky="w")
        tk.Label(self.frame, text="Ventana SG:").grid(row=1, column=1, sticky="e")
        tk.Entry(self.frame, textvariable=self.savgol_win, width=6).grid(row=1, column=2)
        tk.Label(self.frame, text="Orden SG:").grid(row=2, column=1, sticky="e")
        tk.Entry(self.frame, textvariable=self.savgol_poly, width=6).grid(row=2, column=2)

        # rango x
        tk.Label(self.frame, text="Rango x (cm⁻¹):").grid(row=3, column=0, sticky="e")
        self.entry_min.grid(row=3, column=1)
        self.entry_max.grid(row=3, column=2)

        # alturas diagnósticas
        tk.Checkbutton(self.frame, text="Mostrar alturas 1602/1313/778", variable=self.var_show_heights)\
            .grid(row=4, column=0, columnspan=3, sticky="w")

        # bandas y colores
        tk.Label(self.frame, text="Bandas (centro:ancho, ...):").grid(row=5, column=0, sticky="e")
        self.entry_bands.grid(row=5, column=1, columnspan=2)
        tk.Label(self.frame, text="Colores (coma):").grid(row=6, column=0, sticky="e")
        self.entry_colors.grid(row=6, column=1, columnspan=2)

        # líneas verticales
        tk.Label(self.frame, text="Líneas verticales (x,x,...):").grid(row=7, column=0, sticky="e")
        self.entry_vlines.grid(row=7, column=1, columnspan=2)

        # picos globales
        tk.Checkbutton(self.frame, text="Mostrar picos (find_peaks)", variable=self.var_peaks).grid(row=8, column=0, sticky="w")
        tk.Label(self.frame, text="Altura mín:").grid(row=8, column=1, sticky="e")
        tk.Entry(self.frame, textvariable=self.peak_height, width=8).grid(row=8, column=2)
        tk.Label(self.frame, text="Prominencia mín:").grid(row=9, column=1, sticky="e")
        tk.Entry(self.frame, textvariable=self.peak_prom, width=8).grid(row=9, column=2)
        tk.Label(self.frame, text="Distancia mín:").grid(row=10, column=1, sticky="e")
        tk.Entry(self.frame, textvariable=self.peak_dist, width=8).grid(row=10, column=2)

        # acciones
        tk.Button(self.frame, text="Exportar imagen (PNG)", command=self._export_current).grid(row=11, column=0, pady=10)
        tk.Button(self.frame, text="Exportar todos", command=self._export_all).grid(row=11, column=1, pady=10)
        tk.Button(self.frame, text="Tabla alturas 1602/1313/778", command=self._show_heights_table).grid(row=11, column=2, pady=10)
        tk.Checkbutton(self.frame, text="Incluir segunda derivada en PNG", variable=self.var_deriv_export).grid(row=12, column=0, columnspan=3, sticky="w")

    # ------------- Utilidades GUI -------------
    def _select_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        self.folder.set(folder)
        self.files, self.paths = list_spa_files(folder)
        menu = self.opt_files["menu"]
        menu.delete(0, "end")
        for f in self.files:
            menu.add_command(label=f, command=tk._setit(self.sel_file, f))
        if self.paths:
            self.sel_file.set(self.files[0])
            self._load_default_range(self.paths[0])

    def _load_default_range(self, path: str):
        x, A, _, _ = read_spa_absorbance(path)
        xmin, xmax = float(np.max(x)), float(np.min(x))
        self.entry_min.delete(0, tk.END); self.entry_min.insert(0, f"{xmin}")
        self.entry_max.delete(0, tk.END); self.entry_max.insert(0, f"{xmax}")

    def _get_range(self, path: str) -> tuple[float, float]:
        xmin = self.entry_min.get().strip()
        xmax = self.entry_max.get().strip()
        if xmin == "" or xmax == "":
            x, _, _, _ = read_spa_absorbance(path)
            return float(np.max(x)), float(np.min(x))
        a, b = float(xmin), float(xmax)
        return (b, a) if a < b else (a, b)

    def _parse_bands(self):
        """
        Texto: '1602:100, 1313:100'  -> [(1602.0, 100.0), (1313.0, 100.0)]
        Colores: 'green,cyan'        -> ['green','cyan']
        Si vacío, usa BandsConfig.triple y BandsConfig.colors
        """
        txt = self.entry_bands.get().strip()
        col = self.entry_colors.get().strip()
        if not txt:
            return BandsConfig.triple, BandsConfig.colors
        bands = []
        for token in txt.split(","):
            token = token.strip()
            if ":" in token:
                c, w = token.split(":")
                bands.append((float(c), float(w)))
        colors = [c.strip() for c in col.split(",")] if col else []
        return bands, colors

    def _parse_vlines(self):
        txt = self.entry_vlines.get().strip()
        if not txt:
            return []
        vals = []
        for t in txt.split(","):
            t = t.strip()
            if t:
                vals.append(float(t))
        return vals

    def _read_peaks_params(self):
        def f2(s):
            try:
                return float(s) if s.strip() != "" else None
            except Exception:
                return None
        def i2(s):
            try:
                return int(s) if s.strip() != "" else None
            except Exception:
                return None
        return f2(self.peak_height.get()), f2(self.peak_prom.get()), i2(self.peak_dist.get())

    # ------------- Lógica de abrir/plotear -------------
    def _open_selected(self):
        fname = self.sel_file.get()
        if not fname:
            return
        path = dict(zip(self.files, self.paths)).get(fname)
        if not path:
            return

        xmin, xmax = self._get_range(path)

        x, A, x_units, y_units = read_spa_absorbance(path)
        x, A = ensure_descending(x, A)

        if self.var_smooth.get():
            from ..core.processing import savgol_smooth_and_derivative
            y_plot, d2 = savgol_smooth_and_derivative(A, x, self.savgol_win.get(), self.savgol_poly.get())
        else:
            y_plot = A
            from ..core.processing import second_derivative
            d2 = second_derivative(A, x)



        mask = (x <= xmin) & (x >= xmax)
        if np.sum(mask) < 2:
            messagebox.showerror("Error", "Rango seleccionado vacío o insuficiente.")
            return

        bands, band_colors = self._parse_bands()
        vlines = self._parse_vlines()

        # segmentos de altura diagnósticos
        height_segments = []
        if self.var_show_heights.get():
            # usa siempre absorbancia no suavizada para el cálculo
            for (center, fullw), color in zip(BandsConfig.triple, BandsConfig.colors):
                xpk, ybl, h = band_peak_metrics(x, A, center, fullw)
                height_segments.append((xpk, ybl, h, color))

        # picos globales (solo para anotación visual; no altera alturas)
        peaks_x = []
        if self.var_peaks.get():
            h, p, d = self._read_peaks_params()
            xr, yr = x[mask], y_plot[mask]
            idx, _ = find_peaks(yr, height=h, prominence=p, distance=d)
            peaks_x = xr[idx].tolist()

        # plot
        x_r, y_r, d2_r = x[mask], y_plot[mask], d2[mask]
        fig, (ax1, ax2) = plot_spectrum_with_derivative(
            x_r, y_r, d2_r, x_units, y_units, xmin, xmax,
            user_bands=bands, user_band_colors=band_colors,
            vlines=vlines, height_segments=height_segments
        )

        # anotar picos globales si se pidió
        if peaks_x:
            off = (np.max(y_r) - np.min(y_r)) * 0.03
            for xv in peaks_x:
                # encontrar y para ese x aproximado
                k = int(np.argmin(np.abs(x_r - xv)))
                ax1.scatter([x_r[k]], [y_r[k] + off], marker="v", color="black", s=16)
                ax1.annotate(f"{x_r[k]:.1f}", (x_r[k], y_r[k] + off), fontsize=9, color="black", va="bottom")

        fig.show()

    # ------------- Exportar -------------
    def _export_current(self):
        fname = self.sel_file.get()
        if not fname:
            messagebox.showinfo("Aviso", "Seleccione un archivo.")
            return
        path = dict(zip(self.files, self.paths)).get(fname)
        if not path:
            return
        try:
            self._export_one(path, out_dir=os.path.dirname(path))
            messagebox.showinfo("Imagen guardada", "Se exportó el PNG junto al archivo .SPA.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _export_all(self):
        if not self.paths:
            messagebox.showinfo("Aviso", "Seleccione una carpeta con archivos .SPA.")
            return
        out_root = os.path.join(self.folder.get(), "imagenes_exportadas")
        os.makedirs(out_root, exist_ok=True)
        count = 0
        for p in self.paths:
            try:
                self._export_one(p, out_dir=out_root)
                count += 1
            except Exception:
                pass
        messagebox.showinfo("Exportación completa", f"Se guardaron {count} imágenes en:\n{out_root}")

    def _show_heights_table(self):
        # Requiere: numpy as np; functions: list_spa_files, read_spa_absorbance,
        # ensure_descending, band_peak_metrics, column_exponent; DisplayConfig.
        folder = getattr(self, "folder", tk.StringVar()).get() if hasattr(self, "folder") else ""
        if not folder:
            messagebox.showerror("Error", "Seleccione una carpeta primero.")
            return

        try:
            files, paths = list_spa_files(folder)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo listar la carpeta.\n{e}")
            return

        if not paths:
            messagebox.showinfo("Aviso", "No se encontraron archivos .SPA en la carpeta.")
            return

        nombres, h1602, h1313, h778 = [], [], [], []

        for f, p in zip(files, paths):
            try:
                x, A, _, _ = read_spa_absorbance(p)
                x, A = ensure_descending(x, A)
                # alturas siempre desde absorbancia NO suavizada, baseline local
                xpk, ybl, a1 = band_peak_metrics(x, A, 1602.0, 100.0); h1602.append(a1)
                xpk, ybl, a2 = band_peak_metrics(x, A, 1313.0, 100.0); h1313.append(a2)
                xpk, ybl, a3 = band_peak_metrics(x, A,  778.0, 100.0); h778.append(a3)
                nombres.append(f)
            except Exception:
                nombres.append(f)
                h1602.append(np.nan); h1313.append(np.nan); h778.append(np.nan)

        dec = DisplayConfig().decimals_alturas
        e1 = column_exponent(np.array(h1602, dtype=float))
        e2 = column_exponent(np.array(h1313, dtype=float))
        e3 = column_exponent(np.array(h778,  dtype=float))

        s1 = 10.0**e1 if e1 != 0 else 1.0
        s2 = 10.0**e2 if e2 != 0 else 1.0
        s3 = 10.0**e3 if e3 != 0 else 1.0

        def fmt(v, s):
            return "nan" if not np.isfinite(v) else f"{(v/s):.{dec}f}"
        def elabel(k):
            return f"(×10^{k})" if k != 0 else ""

        top = tk.Toplevel(self.master if hasattr(self, "master") else None)
        top.title("Alturas de picos (1602/1313/778)")

        cols = [
            ("archivo", "archivo"),
            ("h1602", f"1602_height {elabel(e1)}"),
            ("h1313", f"1313_height {elabel(e2)}"),
            ("h778",  f"778_height  {elabel(e3)}"),
        ]
        tree = ttk.Treeview(top, columns=[c[0] for c in cols], show="headings")

        for key, label in cols:
            tree.heading(key, text=label)
            if key == "archivo":
                tree.column(key, anchor="w", width=360, stretch=True)
            else:
                tree.column(key, anchor="e", width=160, stretch=False)

        for name, a1, a2, a3 in zip(nombres, h1602, h1313, h778):
            tree.insert("", "end", values=[name, fmt(a1, s1), fmt(a2, s2), fmt(a3, s3)])

        tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(top, orient="vertical", command=tree.yview)
        tree.configure(yscroll=sb.set)
        sb.pack(side="right", fill="y")


    def _export_one(self, path: str, out_dir: str):
        xmin, xmax = self._get_range(path)
        x, A, x_units, y_units = read_spa_absorbance(path)
        x, A = ensure_descending(x, A)

        # Derivada coherente
        if self.var_smooth.get():
            from ..core.processing import savgol_smooth_and_derivative
            y_plot, d2 = savgol_smooth_and_derivative(A, x, self.savgol_win.get(), self.savgol_poly.get())
        else:
            y_plot = A
            from ..core.processing import second_derivative
            d2 = second_derivative(A, x)

        mask = (x <= xmin) & (x >= xmax)
        if np.sum(mask) < 2:
            raise RuntimeError("Rango seleccionado vacío o insuficiente.")

        x_r, y_r = x[mask], y_plot[mask]
        d2_r = d2[mask] if d2 is not None else None

        # Overlays comunes
        bands, band_colors = self._parse_bands()
        vlines = self._parse_vlines()

        # Segmentos de altura (siempre desde A no suavizada)
        height_segments = []
        from ..core.peaks import band_peak_metrics
        from ..config import BandsConfig
        for (center, fullw), color in zip(BandsConfig.triple, BandsConfig.colors):
            xpk, ybl, h = band_peak_metrics(x, A, center, fullw)
            height_segments.append((xpk, ybl, h, color))

        # Picos globales (mismos parámetros que en pantalla)
        peaks_x = []
        if self.var_peaks.get():
            from scipy.signal import find_peaks
            h, p, d = self._read_peaks_params()
            idx, _ = find_peaks(y_r, height=h, prominence=p, distance=d)
            peaks_x = x_r[idx].tolist()

        # Exportación: overlay twin-y si se incluye derivada, si no panel único
        if self.var_deriv_export.get():
            fig, (ax1, ax2) = plot_spectrum_overlay(
                x_r, y_r, d2_r, x_units, y_units, xmin, xmax,
                user_bands=bands, user_band_colors=band_colors,
                vlines=vlines, height_segments=height_segments,
                fig_size=(13, 8), dpi=600
            )
            # Anotar picos (arriba)
            if peaks_x:
                off = (float(np.max(y_r)) - float(np.min(y_r))) * 0.03
                for xv in peaks_x:
                    k = int(np.argmin(np.abs(x_r - xv)))
                    ax1.scatter([x_r[k]], [y_r[k] + off], marker="v", color="black", s=16)
                    ax1.annotate(f"{x_r[k]:.1f}", (x_r[k], y_r[k] + off), fontsize=9, color="black", va="bottom")
        else:
            fig, ax1 = plot_spectrum_only(
                x_r, y_r, x_units, y_units, xmin, xmax,
                user_bands=bands, user_band_colors=band_colors,
                vlines=vlines, height_segments=height_segments,
                fig_size=(13, 8), dpi=600
            )
            if peaks_x:
                off = (float(np.max(y_r)) - float(np.min(y_r))) * 0.03
                for xv in peaks_x:
                    k = int(np.argmin(np.abs(x_r - xv)))
                    ax1.scatter([x_r[k]], [y_r[k] + off], marker="v", color="black", s=16)
                    ax1.annotate(f"{x_r[k]:.1f}", (x_r[k], y_r[k] + off), fontsize=9, color="black", va="bottom")

        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, f"{base}_rango.png")
        fig.savefig(out_path, bbox_inches="tight")
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass



    # ------------- Tabla alturas -------------
    def _show_heights_table(self):
        # Requiere: numpy as np; functions: list_spa_files, read_spa_absorbance,
        # ensure_descending, band_peak_metrics, column_exponent; DisplayConfig.
        folder = self.folder.get()
        if not folder:
            messagebox.showerror("Error", "Seleccione una carpeta primero.")
            return

        try:
            files, paths = list_spa_files(folder)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo listar la carpeta.\n{e}")
            return

        if not paths:
            messagebox.showinfo("Aviso", "No se encontraron archivos .SPA en la carpeta.")
            return

        nombres, h1602, h1313, h778 = [], [], [], []

        for f, p in zip(files, paths):
            try:
                x, A, _, _ = read_spa_absorbance(p)
                x, A = ensure_descending(x, A)
                xpk, ybl, a1 = band_peak_metrics(x, A, 1602.0, 100.0); h1602.append(a1)
                xpk, ybl, a2 = band_peak_metrics(x, A, 1313.0, 100.0); h1313.append(a2)
                xpk, ybl, a3 = band_peak_metrics(x, A,  778.0, 100.0); h778.append(a3)
                nombres.append(f)
            except Exception:
                nombres.append(f)
                h1602.append(np.nan); h1313.append(np.nan); h778.append(np.nan)

        dec = DisplayConfig().decimals_alturas
        e1 = column_exponent(np.array(h1602, dtype=float))
        e2 = column_exponent(np.array(h1313, dtype=float))
        e3 = column_exponent(np.array(h778,  dtype=float))

        s1 = 10.0**e1 if e1 != 0 else 1.0
        s2 = 10.0**e2 if e2 != 0 else 1.0
        s3 = 10.0**e3 if e3 != 0 else 1.0

        def fmt(v, s):
            return "nan" if not np.isfinite(v) else f"{(v/s):.{dec}f}"
        def elabel(k):
            return f"(×10^{k})" if k != 0 else ""

        top = tk.Toplevel(self.master)
        top.title("Alturas de picos (1602/1313/778)")

        cols = [
            ("archivo", "archivo"),
            ("h1602", f"1602_height {elabel(e1)}"),
            ("h1313", f"1313_height {elabel(e2)}"),
            ("h778",  f"778_height  {elabel(e3)}"),
        ]
        col_ids = [c[0] for c in cols]
        col_headers = [c[1] for c in cols]

        frame = tk.Frame(top)
        frame.pack(fill="both", expand=True, padx=6, pady=6)

        tree = ttk.Treeview(frame, columns=col_ids, show="headings")
        for key, label in cols:
            tree.heading(key, text=label)
            if key == "archivo":
                tree.column(key, anchor="w", width=420, stretch=True)
            else:
                tree.column(key, anchor="e", width=160, stretch=False)

        for name, a1, a2, a3 in zip(nombres, h1602, h1313, h778):
            tree.insert("", "end", values=[name, fmt(a1, s1), fmt(a2, s2), fmt(a3, s3)])

        tree.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        # --- Copiar al portapapeles (TSV): botón y Ctrl+C ---
        def _copy_to_clipboard(event=None):
            # si hay selección, copiar solo seleccionadas; si no, todas
            selected = tree.selection()
            items = selected if selected else tree.get_children("")
            lines = []

            # encabezados
            lines.append("\t".join(col_headers))

            # filas
            for iid in items:
                vals = tree.item(iid, "values")
                # asegurar que sea texto
                row = [str(v) for v in vals]
                lines.append("\t".join(row))

            tsv = "\n".join(lines)
            top.clipboard_clear()
            top.clipboard_append(tsv)
            # mantener en el portapapeles al cerrar la ventana
            top.update()  # fuerza copiar en Windows
            messagebox.showinfo("Copiado", "Tabla copiada al portapapeles (TSV).")

        # Botón copiar
        btn_frame = tk.Frame(top)
        btn_frame.pack(fill="x", padx=6, pady=(0,6))
        tk.Button(btn_frame, text="Copiar tabla al portapapeles (TSV)", command=_copy_to_clipboard)\
            .pack(side="left")

        # Atajo Ctrl+C
        tree.bind("<Control-c>", _copy_to_clipboard)
        tree.bind("<Control-C>", _copy_to_clipboard)



def main():
    root = tk.Tk()
    app = FTIRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

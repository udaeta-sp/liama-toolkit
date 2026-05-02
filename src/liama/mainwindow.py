"""Main window: orchestrates all widgets, manages data flow."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QToolBar, QPushButton,
    QComboBox, QLabel, QFileDialog, QStatusBar, QMessageBox,
)

from .core.spectrum import Spectrum
from .core.spa_reader import scan_folder
from .core.processing import smooth_sg, second_derivative_sg, apply_pipeline_to_matrix
from .core.peak_detection import detect_peaks
from .core.multivariate import run_pca, run_plsda, run_random_forest

from .widgets.spectrum_list import SpectrumListPanel
from .widgets.canvas_widget import CanvasWidget, SpectrumViewConfig, DerivativeViewConfig
from .widgets.vista_tab import VistaTab
from .widgets.processing_tab import ProcessingTab
from .widgets.annotations_tab import AnnotationsTab
from .widgets.export_tab import ExportTab
from .widgets.multivariate_panel import MultivariatePanel
from .widgets.metadata_dialog import MetadataDialog

from .utils.colors import get_spectrum_color, get_derivative_color, SPECTRUM_COLORS
from .utils.theme import (
    QSS, QSS_LIGHT, apply_mpl_dark_theme, apply_mpl_light_theme,
    BG_DARK, BG_MID, BORDER, FG_TEXT, FG_DIM,
    LT_BG, LT_BG_MID, LT_BORDER, LT_FG_TEXT, LT_FG_DIM,
)

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LIAMA Toolkit — FTIR-ATR")
        self.setMinimumSize(1200, 750)

        # ── Data state ──
        self._all_spectra: list[Spectrum] = []       # full list from folder
        self._stage_indices: list[int] = []           # indices into _all_spectra
        self._spectrum_configs: list[SpectrumViewConfig] = []
        self._derivative_configs: list[DerivativeViewConfig] = []
        self._metadata_categories: list[str] = []
        self._color_counter = 0
        self._deriv_color_counter = 0
        self._color_memory: dict[int, str] = {}  # index → last assigned color
        self._current_mode = "spectra"  # "spectra" or "multivariate"
        self._dark_theme = True

        apply_mpl_dark_theme()
        self._build_ui()
        self._connect_signals()

    # ─────────────────────────────────────────────────────────────────
    # UI Construction
    # ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        btn_open = QPushButton("Abrir carpeta")
        btn_open.clicked.connect(self._open_folder)
        toolbar.addWidget(btn_open)

        btn_meta = QPushButton("Cargar metadatos")
        btn_meta.clicked.connect(self._load_metadata)
        toolbar.addWidget(btn_meta)

        toolbar.addWidget(QLabel("  Modo: "))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Espectros", "Análisis Multivariado"])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        toolbar.addWidget(self._mode_combo)

        self._btn_theme = QPushButton("Tema claro")
        self._btn_theme.setToolTip("Alternar entre tema oscuro y claro")
        self._btn_theme.clicked.connect(self._toggle_theme)
        toolbar.addWidget(self._btn_theme)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Main splitter: left panel | center content
        self._h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: spectrum list
        self._spectrum_list = SpectrumListPanel()
        self._spectrum_list.setMinimumWidth(200)
        self._spectrum_list.setMaximumWidth(400)
        self._h_splitter.addWidget(self._spectrum_list)

        # Right side: stacked layout for spectra mode vs multivariate mode
        from PyQt6.QtWidgets import QStackedWidget
        self._right_stack = QStackedWidget()

        # --- Page 0: Spectra mode (canvas + tabs) ---
        spectra_page = QWidget()
        spectra_layout = QVBoxLayout(spectra_page)
        spectra_layout.setContentsMargins(0, 0, 0, 0)
        spectra_layout.setSpacing(0)

        self._v_splitter = QSplitter(Qt.Orientation.Vertical)

        self._canvas = CanvasWidget()
        self._v_splitter.addWidget(self._canvas)

        # Bottom tabs
        self._tabs = QTabWidget()
        self._vista_tab = VistaTab()
        self._processing_tab = ProcessingTab()
        self._annotations_tab = AnnotationsTab()
        self._export_tab = ExportTab()

        self._tabs.addTab(self._vista_tab, "Vista")
        self._tabs.addTab(self._processing_tab, "Procesamiento")
        self._tabs.addTab(self._annotations_tab, "Anotaciones")
        self._tabs.addTab(self._export_tab, "Exportar")
        self._tabs.setMinimumHeight(180)

        self._v_splitter.addWidget(self._tabs)
        self._v_splitter.setSizes([500, 250])
        self._v_splitter.setStretchFactor(0, 3)
        self._v_splitter.setStretchFactor(1, 1)

        spectra_layout.addWidget(self._v_splitter)
        self._right_stack.addWidget(spectra_page)

        # --- Page 1: Multivariate mode ---
        self._multivariate = MultivariatePanel()
        self._right_stack.addWidget(self._multivariate)

        self._right_stack.setCurrentIndex(0)
        self._h_splitter.addWidget(self._right_stack)
        self._h_splitter.setSizes([250, 900])

        main_layout.addWidget(self._h_splitter)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Listo — abrir una carpeta con archivos .SPA")

    def _connect_signals(self):
        # Spectrum list
        self._spectrum_list.spectrum_toggled.connect(self._on_spectrum_toggled)
        self._spectrum_list.batch_toggled.connect(self._on_batch_toggled)

        # Vista tab
        self._vista_tab.redraw_requested.connect(self._redraw)
        self._vista_tab.derivative_requested.connect(self._add_derivative)
        self._vista_tab.derivative_removed.connect(self._remove_derivative)
        self._vista_tab.move_requested.connect(self._on_move)

        # Processing tab
        self._processing_tab.settings_changed.connect(self._redraw)

        # Annotations tab
        self._annotations_tab.detect_requested.connect(self._detect_peaks)
        self._annotations_tab.vlines_changed.connect(self._redraw)
        self._annotations_tab.center_on_peak.connect(self._canvas.center_on_wavenumber)

        # Export tab
        self._export_tab.export_image.connect(self._do_export_image)
        self._export_tab.export_csv.connect(self._do_export_csv)

        # Canvas mouse
        self._canvas.mouse_moved.connect(self._on_mouse_coords)

        # Multivariate
        self._multivariate.loadings_to_vista.connect(self._add_loading_to_vista)
        self._multivariate.run_requested.connect(self._run_multivariate_analysis)
        self._multivariate.project_requested.connect(self._project_in_multivariate)

    # ─────────────────────────────────────────────────────────────────
    # Folder & Metadata
    # ─────────────────────────────────────────────────────────────────

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta con .SPA")
        if not folder:
            return

        self._status.showMessage(f"Cargando espectros de {folder}...")
        self._all_spectra = scan_folder(Path(folder))

        # Reset state
        self._stage_indices.clear()
        self._spectrum_configs.clear()
        self._derivative_configs.clear()
        self._color_counter = 0
        self._deriv_color_counter = 0
        self._color_memory.clear()

        self._spectrum_list.set_spectra(self._all_spectra)

        # Set wavenumber range from loaded spectra
        valid = [s for s in self._all_spectra if s.load_error is None and s.n_points > 0]
        if valid:
            wn_min = min(s.wn_min for s in valid)
            wn_max = max(s.wn_max for s in valid)
            self._canvas.set_wavenumber_range(wn_min, wn_max)
            self._multivariate.set_data_info(
                len(valid),
                valid[0].n_points,
                wn_min, wn_max,
            )

        errors = [s for s in self._all_spectra if s.load_error]
        msg = f"{len(self._all_spectra)} espectros cargados"
        if errors:
            error_names = [s.name for s in errors]
            msg += f" ({len(errors)} con error)"
            self._status.showMessage(msg)
            # Show error details in a dialog
            detail = "\n".join(f"• {s.name}: {s.load_error}" for s in errors[:10])
            if len(errors) > 10:
                detail += f"\n... y {len(errors) - 10} más"
            QMessageBox.warning(
                self, "Errores al cargar espectros",
                f"{len(errors)} de {len(self._all_spectra)} archivos no se pudieron leer:\n\n{detail}"
            )
        else:
            self._status.showMessage(msg)

    def _load_metadata(self):
        if not self._all_spectra:
            QMessageBox.warning(self, "Sin datos", "Primero abrí una carpeta con espectros.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Cargar metadatos", "",
            "CSV/Excel (*.csv *.xlsx *.xls *.tsv)"
        )
        if not path:
            return

        names = [s.name for s in self._all_spectra]
        dlg = MetadataDialog(path, names, self)
        if dlg.exec() and dlg.result:
            mapping = dlg.result
            categories = dlg.categories

            # Apply metadata to spectra
            matched = 0
            for sp in self._all_spectra:
                if sp.name in mapping:
                    sp.metadata = mapping[sp.name]
                    matched += 1

            self._metadata_categories = categories
            self._spectrum_list.update_metadata_keys(categories)  # also refreshes badges
            self._vista_tab.update_metadata_categories(categories)
            self._vista_tab.set_metadata_lookup(mapping)
            self._multivariate.set_categories(categories)

            self._status.showMessage(
                f"Metadatos importados: {matched}/{len(self._all_spectra)} matcheados, "
                f"categorías: {', '.join(categories)}"
            )

    # ─────────────────────────────────────────────────────────────────
    # Stage management
    # ─────────────────────────────────────────────────────────────────

    def _on_spectrum_toggled(self, index: int, checked: bool):
        if checked:
            self._add_to_stage(index)
        else:
            self._remove_from_stage(index)
        self._rebuild_vista()
        self._redraw()

    def _on_batch_toggled(self, changes: list[tuple[int, bool]]):
        for index, checked in changes:
            if checked:
                self._add_to_stage(index)
            else:
                self._remove_from_stage(index)
        self._rebuild_vista()
        self._redraw()

    def _add_to_stage(self, index: int):
        if index not in self._stage_indices:
            self._stage_indices.append(index)
            sp = self._all_spectra[index]
            # Reuse remembered color if one was assigned before
            remembered = self._color_memory.get(index)
            color = remembered if remembered else get_spectrum_color(self._color_counter)
            cfg = SpectrumViewConfig(
                name=sp.name,
                color=color,
                spectrum_index=index,
            )
            if not remembered:
                self._color_counter += 1
            self._spectrum_configs.append(cfg)
            # Update list swatch
            self._spectrum_list.set_spectrum_color(index, color)

    def _remove_from_stage(self, index: int):
        if index in self._stage_indices:
            pos = self._stage_indices.index(index)
            # Remember color before removing
            cfg = self._spectrum_configs[pos]
            self._color_memory[index] = cfg.color
            self._stage_indices.pop(pos)
            self._spectrum_configs.pop(pos)
            # Remove associated derivatives
            self._derivative_configs = [
                d for d in self._derivative_configs if d.parent_index != pos
            ]
            # Update parent indices for remaining derivatives
            for d in self._derivative_configs:
                if d.parent_index > pos:
                    d.parent_index -= 1

    def _add_derivative(self, stage_index: int):
        """Add a second derivative entry for a staged spectrum."""
        if stage_index >= len(self._spectrum_configs):
            return
        cfg = self._spectrum_configs[stage_index]
        dcfg = DerivativeViewConfig(
            name=cfg.name,
            parent_index=stage_index,
            color=get_derivative_color(self._deriv_color_counter),
        )
        self._deriv_color_counter += 1
        self._derivative_configs.append(dcfg)
        self._rebuild_vista()
        self._redraw()

    def _remove_derivative(self, deriv_index: int):
        if 0 <= deriv_index < len(self._derivative_configs):
            self._derivative_configs.pop(deriv_index)
            self._rebuild_vista()
            self._redraw()

    def _on_move(self, section: str, index: int, direction: int):
        if section == "spectrum":
            lst = self._spectrum_configs
            idx_lst = self._stage_indices
        else:
            lst = self._derivative_configs
            idx_lst = None

        new_idx = index + direction
        if 0 <= new_idx < len(lst):
            lst[index], lst[new_idx] = lst[new_idx], lst[index]
            if idx_lst and 0 <= new_idx < len(idx_lst):
                idx_lst[index], idx_lst[new_idx] = idx_lst[new_idx], idx_lst[index]
            # Keep derivative parent_index consistent after spectrum reorder
            if section == "spectrum":
                for d in self._derivative_configs:
                    if d.parent_index == index:
                        d.parent_index = new_idx
                    elif d.parent_index == new_idx:
                        d.parent_index = index
            self._rebuild_vista()
            self._redraw()

    def _rebuild_vista(self):
        """Rebuild the Vista tab rows from current configs."""
        self._vista_tab.rebuild(self._spectrum_configs, self._derivative_configs)
        # Update annotations spectrum selector
        visible = [c.name for c in self._spectrum_configs if c.visible]
        self._annotations_tab.set_visible_spectra(visible)
        # Update multivariate projection list (spectra NOT in stage)
        staged_names = {self._all_spectra[i].name for i in self._stage_indices}
        available = [
            s.name for s in self._all_spectra
            if s.name not in staged_names and s.load_error is None
        ]
        self._multivariate.set_projection_spectra(available)

    # ─────────────────────────────────────────────────────────────────
    # Drawing
    # ─────────────────────────────────────────────────────────────────

    def _get_processed_spectrum(self, sp: Spectrum) -> tuple[np.ndarray, np.ndarray]:
        """Get spectrum data with optional SG smoothing applied."""
        wn, y = sp.wavenumbers, sp.absorbance
        if self._processing_tab.smoothing_enabled and len(y) > 0:
            y = smooth_sg(y, self._processing_tab.window, self._processing_tab.polyorder)
        return wn, y

    def _get_derivative(self, sp: Spectrum) -> tuple[np.ndarray, np.ndarray]:
        """Get second derivative of a spectrum."""
        wn, y = sp.wavenumbers, sp.absorbance
        if len(y) == 0:
            return wn, np.array([])
        window = self._processing_tab.window if self._processing_tab.smoothing_enabled else 15
        polyorder = self._processing_tab.polyorder if self._processing_tab.smoothing_enabled else 3
        delta = abs(wn[1] - wn[0]) if len(wn) > 1 else 1.0
        d2 = second_derivative_sg(y, window=window, polyorder=polyorder, delta=delta)
        return wn, d2

    def _redraw(self):
        """Redraw the canvas with current configs."""
        if self._current_mode != "spectra":
            return

        # Sync color memory and list swatches
        for cfg in self._spectrum_configs:
            self._color_memory[cfg.spectrum_index] = cfg.color
            self._spectrum_list.set_spectrum_color(cfg.spectrum_index, cfg.color)

        spec_data = []
        for cfg in self._spectrum_configs:
            sp = self._all_spectra[cfg.spectrum_index]
            wn, y = self._get_processed_spectrum(sp)
            spec_data.append((wn, y))

        deriv_data = []
        for dcfg in self._derivative_configs:
            if dcfg.parent_index < len(self._stage_indices):
                sp_idx = self._stage_indices[dcfg.parent_index]
                sp = self._all_spectra[sp_idx]
                wn, d2 = self._get_derivative(sp)
                deriv_data.append((wn, d2))
            else:
                deriv_data.append((np.array([]), np.array([])))

        self._canvas.update_spectra(
            spec_data, self._spectrum_configs,
            deriv_data, self._derivative_configs,
            vlines=self._annotations_tab.vlines,
        )

    # ─────────────────────────────────────────────────────────────────
    # Mode switching
    # ─────────────────────────────────────────────────────────────────

    def _on_mode_changed(self, mode_text: str):
        if mode_text == "Espectros":
            self._current_mode = "spectra"
            self._right_stack.setCurrentIndex(0)
            self._redraw()
        else:
            self._current_mode = "multivariate"
            self._right_stack.setCurrentIndex(1)

    def _toggle_theme(self):
        """Switch between dark and light theme."""
        from PyQt6.QtWidgets import QApplication
        self._dark_theme = not self._dark_theme

        if self._dark_theme:
            QApplication.instance().setStyleSheet(QSS)
            apply_mpl_dark_theme()
            self._btn_theme.setText("Tema claro")
            # Update canvas colors
            self._canvas.plot_config.fig_bg = BG_DARK
            self._canvas.plot_config.plot_bg = BG_MID
            self._canvas.plot_config.text_color = FG_TEXT
            self._canvas.plot_config.tick_color = FG_DIM
            self._canvas.plot_config.spine_color = BORDER
            self._canvas.figure.set_facecolor(BG_DARK)
            self._canvas.canvas.setStyleSheet(f"background-color: {BG_DARK};")
            # Multivariate canvas
            self._multivariate.figure.set_facecolor(BG_DARK)
        else:
            QApplication.instance().setStyleSheet(QSS_LIGHT)
            apply_mpl_light_theme()
            self._btn_theme.setText("Tema oscuro")
            self._canvas.plot_config.fig_bg = LT_BG
            self._canvas.plot_config.plot_bg = LT_BG_MID
            self._canvas.plot_config.text_color = LT_FG_TEXT
            self._canvas.plot_config.tick_color = LT_FG_DIM
            self._canvas.plot_config.spine_color = LT_BORDER
            self._canvas.figure.set_facecolor(LT_BG)
            self._canvas.canvas.setStyleSheet(f"background-color: {LT_BG};")
            # Multivariate canvas
            self._multivariate.figure.set_facecolor(LT_BG)

        # Update range slider colors
        if self._dark_theme:
            self._canvas.range_slider.groove_color = "#3c3c3c"
            self._canvas.range_slider.handle_fill = "#ffffff"
        else:
            self._canvas.range_slider.groove_color = "#cccccc"
            self._canvas.range_slider.handle_fill = "#ffffff"
        self._canvas.range_slider.update()

        # Refresh both canvases
        self._canvas._style_axes()
        self._redraw()
        self._multivariate._style_ax()
        self._multivariate.canvas.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    # Peak detection
    # ─────────────────────────────────────────────────────────────────

    def _detect_peaks(self):
        name = self._annotations_tab.selected_spectrum_name
        cfg = next((c for c in self._spectrum_configs if c.name == name and c.visible), None)
        if not cfg:
            self._status.showMessage("Seleccioná un espectro visible para detectar picos.")
            return

        sp = self._all_spectra[cfg.spectrum_index]
        wn, y = self._get_processed_spectrum(sp)

        peaks = detect_peaks(
            wn, y,
            prominence=self._annotations_tab.prominence_spin.value(),
            distance=self._annotations_tab.distance_spin.value(),
        )

        self._annotations_tab.set_peaks(peaks)
        # Redraw with peaks
        self._canvas.update_spectra(
            [(self._get_processed_spectrum(self._all_spectra[c.spectrum_index]))
             for c in self._spectrum_configs],
            self._spectrum_configs,
            [(self._get_derivative(self._all_spectra[self._stage_indices[d.parent_index]])
              if d.parent_index < len(self._stage_indices) else (np.array([]), np.array([])))
             for d in self._derivative_configs],
            self._derivative_configs,
            vlines=self._annotations_tab.vlines,
            peaks=peaks,
        )
        self._status.showMessage(f"{len(peaks)} picos detectados en {name}")

    # ─────────────────────────────────────────────────────────────────
    # Mouse coords
    # ─────────────────────────────────────────────────────────────────

    def _on_mouse_coords(self, wn: float, y_val: float):
        if self._canvas.transmittance_mode:
            self._status.showMessage(
                f"Número de onda: {wn:.1f} cm⁻¹  |  Transmitancia: {y_val:.2f} %"
            )
        else:
            self._status.showMessage(
                f"Número de onda: {wn:.1f} cm⁻¹  |  Absorbancia: {y_val:.4f}"
            )

    # ─────────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────────

    def _do_export_image(self, params: dict):
        fmt = params["format"]
        dpi = params["dpi"]
        white_bg = params.get("white_bg", True)

        if params.get("batch"):
            folder = Path(params["folder"])
            # Get the visible config as template
            visible_cfgs = [c for c in self._spectrum_configs if c.visible]
            if len(visible_cfgs) != 1:
                QMessageBox.warning(
                    self, "Batch",
                    "Batch requiere exactamente 1 espectro visible como template."
                )
                return

            template_cfg = visible_cfgs[0]
            for cfg in self._spectrum_configs:
                # Temporarily make only this one visible
                old_vis = cfg.visible
                for c in self._spectrum_configs:
                    c.visible = False
                cfg.visible = True
                cfg.color = template_cfg.color
                cfg.linewidth = template_cfg.linewidth
                cfg.alpha = template_cfg.alpha

                self._redraw()
                path = folder / f"{cfg.name}.{fmt}"
                self._canvas.export_figure(
                    str(path), dpi=dpi, use_export_bg=white_bg,
                )
                cfg.visible = old_vis

            # Restore visibility
            self._rebuild_vista()
            self._redraw()
            self._status.showMessage(
                f"Batch exportado: {len(self._spectrum_configs)} imágenes en {folder}"
            )
        else:
            path = params["path"]
            self._canvas.export_figure(
                path, dpi=dpi, use_export_bg=white_bg,
            )
            self._status.showMessage(f"Imagen guardada: {path}")

    def _do_export_csv(self, params: dict):
        # Select spectra
        if params["source"] == "visible":
            cfgs = [c for c in self._spectrum_configs if c.visible]
        else:
            cfgs = self._spectrum_configs

        if not cfgs:
            self._status.showMessage("No hay espectros para exportar.")
            return

        spectra = [self._all_spectra[c.spectrum_index] for c in cfgs]

        # Build common wavenumber grid (from first spectrum)
        wn_ref = spectra[0].wavenumbers
        rows = []
        for sp in spectra:
            if params["data_type"] == "raw":
                y = sp.interpolate_to(wn_ref)
            elif params["data_type"] == "smoothed":
                _, y_proc = self._get_processed_spectrum(sp)
                # Interpolate processed onto reference grid
                y_proc_interp = np.interp(wn_ref[::-1], sp.wavenumbers[::-1], y_proc[::-1])[::-1]
                y = y_proc_interp
            else:  # derivative
                _, d2 = self._get_derivative(sp)
                d2_interp = np.interp(wn_ref[::-1], sp.wavenumbers[::-1], d2[::-1])[::-1]
                y = d2_interp

            row = {"Nombre": sp.name}
            if params["include_metadata"]:
                row.update(sp.metadata)
            for i, w in enumerate(wn_ref):
                row[f"{w:.1f}"] = y[i]
            rows.append(row)

        df = pd.DataFrame(rows)
        if params.get("transpose"):
            # Transpose: wavenumbers as rows
            meta_cols = ["Nombre"] + (list(spectra[0].metadata.keys()) if params["include_metadata"] else [])
            wn_cols = [c for c in df.columns if c not in meta_cols]
            df_t = df[wn_cols].T
            df_t.columns = df["Nombre"].values
            df_t.index.name = "Wavenumber"
            df_t.to_csv(params["path"])
        else:
            df.to_csv(params["path"], index=False)

        self._status.showMessage(f"CSV exportado: {params['path']}")

    # ─────────────────────────────────────────────────────────────────
    # Multivariate analysis
    # ─────────────────────────────────────────────────────────────────

    def _run_multivariate_analysis(self):
        if not self._stage_indices:
            QMessageBox.warning(self, "Sin datos", "Agregá espectros al stage primero.")
            return

        model = self._multivariate.model_combo.currentText()
        category = self._multivariate.category_combo.currentText()

        if model != "PCA" and not category:
            QMessageBox.warning(
                self, "Sin categoría",
                "Cargá metadatos y seleccioná una categoría para clasificación."
            )
            return

        # Build data matrix
        spectra = [self._all_spectra[i] for i in self._stage_indices]
        valid = [s for s in spectra if s.load_error is None and s.n_points > 0]
        if len(valid) < 2:
            QMessageBox.warning(self, "Datos insuficientes", "Necesitás al menos 2 espectros.")
            return

        # Common wavenumber grid
        wn_ref = valid[0].wavenumbers

        # Apply spectral range
        if not self._multivariate.full_range_cb.isChecked():
            wn_lo = self._multivariate.wn_min_spin.value()
            wn_hi = self._multivariate.wn_max_spin.value()
            mask = (wn_ref >= wn_lo) & (wn_ref <= wn_hi)
            wn_ref = wn_ref[mask]

        # Build matrix
        X = np.zeros((len(valid), len(wn_ref)))
        names = []
        for i, sp in enumerate(valid):
            X[i] = sp.interpolate_to(wn_ref)
            names.append(sp.name)

        # Apply pipeline
        steps = self._multivariate.get_pipeline_steps()
        X_proc, fit_params = apply_pipeline_to_matrix(wn_ref, X, steps)

        # Get labels if needed
        labels = None
        if category:
            labels = np.array([
                self._all_spectra[idx].metadata.get(category, "?")
                for idx in self._stage_indices
                if self._all_spectra[idx].load_error is None
            ])
            unique = np.unique(labels)
            if len(unique) < 2 and model != "PCA":
                QMessageBox.warning(
                    self, "Clases insuficientes",
                    f"Se necesitan al menos 2 clases. Encontradas: {list(unique)}"
                )
                return

        try:
            if model == "PCA":
                result = run_pca(
                    X_proc,
                    sample_names=names,
                    labels=labels,
                    wavenumbers=wn_ref,
                    fit_params=fit_params,
                    pipeline_steps=steps,
                )
                if labels is not None:
                    self._multivariate.plot_pca_scores(result, labels, category)
                else:
                    self._multivariate.plot_pca_scores(
                        result, np.array(names), "Nombre"
                    )

            elif model == "PLS-DA":
                result = run_plsda(
                    X_proc, labels,
                    test_size=self._multivariate.test_spin.value(),
                    random_state=self._multivariate.seed_spin.value(),
                    sample_names=names,
                )
                self._multivariate.plot_plsda_scores(result, labels, category)

            elif model == "Random Forest":
                result = run_random_forest(
                    X_proc, labels, wn_ref,
                    test_size=self._multivariate.test_spin.value(),
                    random_state=self._multivariate.seed_spin.value(),
                    sample_names=names,
                )
                self._multivariate.plot_rf_importances(result)

            self._status.showMessage(f"{model} ejecutado exitosamente")

        except Exception as e:
            QMessageBox.critical(self, "Error en análisis", str(e))
            log.exception("Multivariate analysis failed")

    def _project_in_multivariate(self):
        """Project a sample onto existing PCA axes."""
        if self._multivariate._pca_result is None:
            QMessageBox.warning(self, "Sin modelo", "Ejecutá PCA primero.")
            return

        name = self._multivariate.project_combo.currentText()
        if not name:
            return

        sp = next((s for s in self._all_spectra if s.name == name), None)
        if sp is None or sp.load_error:
            return

        result = self._multivariate._pca_result
        wn_ref = result.wavenumbers

        # Interpolate onto same grid
        y = sp.interpolate_to(wn_ref).reshape(1, -1)

        # Apply same pipeline
        from .core.processing import apply_pipeline_to_matrix
        y_proc, _ = apply_pipeline_to_matrix(wn_ref, y, result.pipeline_steps)

        # Project
        scores_new = result.project(y_proc)

        # Plot on existing axes
        self._multivariate.ax.scatter(
            scores_new[0, 0], scores_new[0, 1],
            marker="*", s=200, c="#ff6b6b", edgecolors="white",
            linewidth=1.5, zorder=20, label=f"→ {name}",
        )
        self._multivariate.ax.annotate(
            name, (scores_new[0, 0], scores_new[0, 1]),
            fontsize=8, color="#ff6b6b",
            xytext=(10, 10), textcoords="offset points",
        )
        self._multivariate.ax.legend(fontsize=8)
        self._multivariate.canvas.draw_idle()
        self._status.showMessage(f"Proyectado: {name}")

    # ─────────────────────────────────────────────────────────────────
    # Loadings → Vista
    # ─────────────────────────────────────────────────────────────────

    def _add_loading_to_vista(self, name: str, wavenumbers, loadings):
        """Add PCA loadings as a pseudo-spectrum in Vista."""
        # Create a Spectrum object for the loading
        sp = Spectrum(
            name=name,
            wavenumbers=np.asarray(wavenumbers),
            absorbance=np.asarray(loadings),
            metadata={"Tipo": "Loading"},
        )
        self._all_spectra.append(sp)
        idx = len(self._all_spectra) - 1
        self._stage_indices.append(idx)

        cfg = SpectrumViewConfig(
            name=name,
            color=get_spectrum_color(self._color_counter),
            spectrum_index=idx,
        )
        self._color_counter += 1
        self._spectrum_configs.append(cfg)

        self._rebuild_vista()
        # Switch to spectra mode to see them
        self._mode_combo.setCurrentText("Espectros")
        self._status.showMessage(f"Loading '{name}' agregado a Vista")

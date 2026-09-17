"""Multivariate analysis panel: preprocessing pipeline, PCA/PLS-DA/RF, projection."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit, QGroupBox,
    QFileDialog, QFrame, QMessageBox, QSplitter, QScrollArea,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from ..utils.theme import BG_DARK, FG_DIM, LT_BG
from ..utils.colors import SPECTRUM_COLORS


class PipelineStepWidget(QFrame):
    """One step in the preprocessing pipeline with enable toggle and params."""

    changed = pyqtSignal()

    def __init__(self, name: str, label: str, description: str,
                 params: dict | None = None, parent=None):
        super().__init__(parent)
        self.step_name = name
        self._params = params or {}
        self._param_widgets: dict = {}
        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self._btn_up = QPushButton("▲")
        self._btn_up.setFixedSize(20, 20)
        layout.addWidget(self._btn_up)
        self._btn_down = QPushButton("▼")
        self._btn_down.setFixedSize(20, 20)
        layout.addWidget(self._btn_down)

        self.enable_cb = QCheckBox()
        self.enable_cb.stateChanged.connect(lambda: self.changed.emit())
        layout.addWidget(self.enable_cb)

        info = QVBoxLayout()
        info.setSpacing(0)
        lbl = QLabel(f"<b>{label}</b>")
        info.addWidget(lbl)
        desc = QLabel(description)
        desc.setProperty("role", "subtle")
        desc.setWordWrap(True)
        info.addWidget(desc)
        layout.addLayout(info, 1)

        if "window" in self._params:
            layout.addWidget(QLabel("vent:"))
            w = QSpinBox()
            w.setRange(3, 101)
            w.setSingleStep(2)
            w.setValue(self._params["window"])
            w.setFixedWidth(52)
            w.valueChanged.connect(lambda: self.changed.emit())
            layout.addWidget(w)
            self._param_widgets["window"] = w

        if "polyorder" in self._params:
            layout.addWidget(QLabel("grado:"))
            w = QSpinBox()
            w.setRange(1, 10)
            w.setValue(self._params["polyorder"])
            w.setFixedWidth(42)
            w.valueChanged.connect(lambda: self.changed.emit())
            layout.addWidget(w)
            self._param_widgets["polyorder"] = w

        if "degree" in self._params:
            layout.addWidget(QLabel("grado:"))
            w = QSpinBox()
            w.setRange(1, 10)
            w.setValue(self._params["degree"])
            w.setFixedWidth(42)
            w.valueChanged.connect(lambda: self.changed.emit())
            layout.addWidget(w)
            self._param_widgets["degree"] = w

    def get_step_dict(self) -> dict:
        params = {}
        if "window" in self._param_widgets:
            v = self._param_widgets["window"].value()
            params["window"] = v if v % 2 == 1 else v + 1
        if "polyorder" in self._param_widgets:
            params["polyorder"] = self._param_widgets["polyorder"].value()
        if "degree" in self._param_widgets:
            params["degree"] = self._param_widgets["degree"].value()
        return {
            "name": self.step_name,
            "enabled": self.enable_cb.isChecked(),
            "params": params,
        }


class MultivariatePanel(QWidget):
    """Full multivariate analysis mode with pipeline, models, and projection."""

    loadings_to_vista = pyqtSignal(str, object, object)
    run_requested = pyqtSignal()
    project_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pca_result = None
        # Artists stored for toggle without re-running
        self._annotations: list = []
        self._biplot_texts: list = []
        self._setup_ui()

    def _setup_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Plot and controls share a draggable splitter so the control column
        # can be widened (or collapsed) instead of squeezing the plot.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        # ── Left: canvas + navigation toolbar + visibility toggles ──
        left = QVBoxLayout()
        left.setSpacing(2)
        left.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.figure.set_facecolor(BG_DARK)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self._style_ax()

        # Zoom/pan toolbar — essential for dense biplots
        self._nav_toolbar = NavigationToolbar2QT(self.canvas, self)
        left.addWidget(self._nav_toolbar)
        left.addWidget(self.canvas, 1)

        # Annotation toggles placed below the canvas, outside the plot area
        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(6, 2, 6, 2)

        self.show_names_cb = QCheckBox("Nombres de muestras")
        self.show_names_cb.setChecked(False)
        self.show_names_cb.setToolTip(
            "Mostrar/ocultar etiquetas con el nombre de cada muestra sobre su punto"
        )
        self.show_names_cb.stateChanged.connect(self._toggle_sample_labels)
        toggle_row.addWidget(self.show_names_cb)

        self.show_biplot_labels_cb = QCheckBox("Etiquetas biplot (cm⁻¹)")
        self.show_biplot_labels_cb.setChecked(True)
        self.show_biplot_labels_cb.setToolTip(
            "Mostrar/ocultar números de número de onda en las flechas del biplot"
        )
        self.show_biplot_labels_cb.stateChanged.connect(self._toggle_biplot_labels)
        toggle_row.addWidget(self.show_biplot_labels_cb)
        toggle_row.addStretch()
        left.addLayout(toggle_row)

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setMinimumWidth(320)
        self.canvas.setMinimumSize(280, 240)
        splitter.addWidget(left_w)

        # ── Right: controls ──
        right = QVBoxLayout()
        right.setSpacing(4)
        right.setContentsMargins(4, 4, 4, 4)

        # Preprocessing pipeline
        pipe_group = QGroupBox("Pipeline de preprocesamiento")
        pipe_layout = QVBoxLayout(pipe_group)
        pipe_layout.setSpacing(2)

        range_row = QHBoxLayout()
        range_row.setSpacing(4)
        range_row.addWidget(QLabel("Rango:"))
        self.wn_min_spin = QDoubleSpinBox()
        self.wn_min_spin.setRange(200, 5000)
        self.wn_min_spin.setValue(600)
        self.wn_min_spin.setDecimals(0)
        self.wn_min_spin.setMaximumWidth(70)
        range_row.addWidget(self.wn_min_spin, 1)
        range_row.addWidget(QLabel("–"))
        self.wn_max_spin = QDoubleSpinBox()
        self.wn_max_spin.setRange(200, 5000)
        self.wn_max_spin.setValue(1800)
        self.wn_max_spin.setDecimals(0)
        self.wn_max_spin.setMaximumWidth(70)
        range_row.addWidget(self.wn_max_spin, 1)
        range_row.addWidget(QLabel("cm⁻¹"))
        pipe_layout.addLayout(range_row)

        self.full_range_cb = QCheckBox("Rango completo")
        pipe_layout.addWidget(self.full_range_cb)

        self._pipeline_steps: list[PipelineStepWidget] = []
        self._pipe_container = QVBoxLayout()
        self._pipe_container.setSpacing(1)

        steps_def = [
            ("smooth_sg",            "Suavizado SG",
             "savgol_filter — ventana impar, grado polinomial",
             {"window": 15, "polyorder": 3}),
            ("second_derivative_sg", "2ª derivada SG",
             "savgol_filter deriv=2 — incluye suavizado",
             {"window": 15, "polyorder": 3}),
            ("baseline_polynomial",  "Línea base polinomial",
             "polyfit+polyval → X - baseline",
             {"degree": 2}),
            ("normalize_row_sum",    "Norm. row-sum",
             "X[i] / Σ X[i]",
             None),
            ("normalize_snv",        "SNV",
             "(X[i] − μ) / σ  por fila",
             None),
            ("transform_hellinger",  "Hellinger",
             "√(X[i] / Σ X[i])  — requiere valores ≥ 0",
             None),
            ("scale_standard",       "Escalado UV (StandardScaler)",
             "(X − μ) / σ  por columna",
             None),
            ("scale_pareto",         "Escalado Pareto",
             "(X − μ) / √σ  por columna",
             None),
        ]

        for name, label, desc, params in steps_def:
            step_w = PipelineStepWidget(name, label, desc, params)
            step_w._btn_up.clicked.connect(lambda _, w=step_w: self._move_step(w, -1))
            step_w._btn_down.clicked.connect(lambda _, w=step_w: self._move_step(w, 1))
            self._pipeline_steps.append(step_w)
            self._pipe_container.addWidget(step_w)

        pipe_layout.addLayout(self._pipe_container)

        self._data_info = QLabel("Datos: —")
        self._data_info.setProperty("role", "subtle")
        pipe_layout.addWidget(self._data_info)

        right.addWidget(pipe_group)

        # Model controls
        model_group = QGroupBox("Análisis")
        model_layout = QVBoxLayout(model_group)

        ctrl = QGridLayout()
        ctrl.setSpacing(4)
        ctrl.setColumnStretch(1, 1)
        ctrl.setColumnStretch(3, 1)

        ctrl.addWidget(QLabel("Modelo:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["PCA", "PLS-DA", "Random Forest"])
        ctrl.addWidget(self.model_combo, 0, 1, 1, 3)

        ctrl.addWidget(QLabel("Categoría:"), 1, 0)
        self.category_combo = QComboBox()
        ctrl.addWidget(self.category_combo, 1, 1, 1, 3)

        ctrl.addWidget(QLabel("Test:"), 2, 0)
        self.test_spin = QDoubleSpinBox()
        self.test_spin.setRange(0.1, 0.5)
        self.test_spin.setValue(0.30)
        self.test_spin.setSingleStep(0.05)
        ctrl.addWidget(self.test_spin, 2, 1)

        ctrl.addWidget(QLabel("Seed:"), 2, 2)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        ctrl.addWidget(self.seed_spin, 2, 3)
        model_layout.addLayout(ctrl)

        # PCA-specific controls
        pca = QGridLayout()
        pca.setSpacing(4)
        pca.setColumnStretch(1, 1)
        pca.setColumnStretch(3, 1)

        self.biplot_cb = QCheckBox("Biplot")
        pca.addWidget(self.biplot_cb, 0, 0, 1, 4)

        pca.addWidget(QLabel("Top:"), 1, 0)
        self.biplot_top = QSpinBox()
        self.biplot_top.setRange(1, 50)
        self.biplot_top.setValue(10)
        pca.addWidget(self.biplot_top, 1, 1)

        pca.addWidget(QLabel("Escala:"), 1, 2)
        self.biplot_scale = QDoubleSpinBox()
        self.biplot_scale.setRange(0.1, 100.0)
        self.biplot_scale.setValue(1.0)
        pca.addWidget(self.biplot_scale, 1, 3)

        btn_loadings = QPushButton("Loadings → Vista")
        btn_loadings.clicked.connect(self._send_loadings_to_vista)
        pca.addWidget(btn_loadings, 2, 0, 1, 4)
        model_layout.addLayout(pca)

        # Projection
        proj = QGridLayout()
        proj.setSpacing(4)
        proj.setColumnStretch(1, 1)
        proj.addWidget(QLabel("Proyectar:"), 0, 0)
        self.project_combo = QComboBox()
        self.project_combo.setToolTip(
            "Espectros fuera del stage para proyectar en ejes existentes"
        )
        proj.addWidget(self.project_combo, 0, 1)
        btn_project = QPushButton("Proyectar")
        btn_project.clicked.connect(self._project_sample)
        proj.addWidget(btn_project, 1, 0, 1, 2)
        model_layout.addLayout(proj)

        btn_run = QPushButton("Ejecutar análisis")
        btn_run.setProperty("role", "primary")
        btn_run.clicked.connect(self._run_analysis)
        model_layout.addWidget(btn_run)

        right.addWidget(model_group)

        # Results
        results_group = QGroupBox("Resultados")
        results_layout = QVBoxLayout(results_group)
        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setMinimumHeight(90)
        self._results_text.setMaximumHeight(220)
        self._results_text.setProperty("role", "report")
        results_layout.addWidget(self._results_text)

        btn_save_plot = QPushButton("Guardar gráfico...")
        btn_save_plot.clicked.connect(self._save_plot)
        results_layout.addWidget(btn_save_plot)

        btn_save_report = QPushButton("Guardar reporte...")
        btn_save_report.clicked.connect(self._save_results)
        results_layout.addWidget(btn_save_report)

        right.addWidget(results_group)

        right.addStretch()
        right_w = QWidget()
        right_w.setLayout(right)

        # The control stack is taller than a laptop screen — scroll it rather
        # than letting the bottom groups get clipped.
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setWidget(right_w)
        right_scroll.setMinimumWidth(260)
        splitter.addWidget(right_scroll)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([700, 420])

    # ──────────────────────────────────────────────────────────────────
    # Axis styling
    # ──────────────────────────────────────────────────────────────────

    def _style_ax(self):
        self.ax.set_facecolor(mpl.rcParams["axes.facecolor"])
        self.ax.tick_params(colors=mpl.rcParams["xtick.color"])
        for spine in self.ax.spines.values():
            spine.set_color(mpl.rcParams["axes.edgecolor"])

    def apply_theme(self, dark: bool):
        """Repaint the figure for the active theme."""
        self.figure.set_facecolor(BG_DARK if dark else LT_BG)
        self._style_ax()
        self.canvas.draw_idle()

    # ──────────────────────────────────────────────────────────────────
    # Pipeline step reordering
    # ──────────────────────────────────────────────────────────────────

    def _move_step(self, widget: PipelineStepWidget, direction: int):
        idx = self._pipeline_steps.index(widget)
        new_idx = idx + direction
        if 0 <= new_idx < len(self._pipeline_steps):
            self._pipeline_steps[idx], self._pipeline_steps[new_idx] = \
                self._pipeline_steps[new_idx], self._pipeline_steps[idx]
            while self._pipe_container.count():
                self._pipe_container.takeAt(0)
            for step in self._pipeline_steps:
                self._pipe_container.addWidget(step)

    def get_pipeline_steps(self) -> list[dict]:
        """Get current pipeline configuration, validating mutual exclusions."""
        steps = []
        has_smooth = has_deriv = False
        for step_w in self._pipeline_steps:
            d = step_w.get_step_dict()
            if d["enabled"]:
                if d["name"] == "smooth_sg":
                    has_smooth = True
                if d["name"] == "second_derivative_sg":
                    has_deriv = True
            steps.append(d)

        if has_smooth and has_deriv:
            QMessageBox.warning(
                self, "Conflicto en pipeline",
                "Suavizado SG y 2ª derivada SG están activos simultáneamente.\n"
                "La derivada SG ya incluye suavizado — se desactivará el paso de "
                "suavizado para este análisis."
            )
            for s in steps:
                if s["name"] == "smooth_sg":
                    s["enabled"] = False
        return steps

    # ──────────────────────────────────────────────────────────────────
    # Data / category updates
    # ──────────────────────────────────────────────────────────────────

    def set_data_info(self, n_spectra: int, n_wavenumbers: int,
                      wn_min: float, wn_max: float):
        self._data_info.setText(
            f"Datos: {n_spectra} espectros · {n_wavenumbers} puntos"
        )
        self.wn_min_spin.setRange(wn_min, wn_max)
        self.wn_max_spin.setRange(wn_min, wn_max)
        if self.wn_min_spin.value() < wn_min:
            self.wn_min_spin.setValue(wn_min)
        if self.wn_max_spin.value() > wn_max:
            self.wn_max_spin.setValue(wn_max)

    def set_categories(self, categories: list[str]):
        self.category_combo.clear()
        for cat in categories:
            self.category_combo.addItem(cat)

    def set_projection_spectra(self, names: list[str]):
        self.project_combo.clear()
        for name in names:
            self.project_combo.addItem(name)

    # ──────────────────────────────────────────────────────────────────
    # Signal forwarders
    # ──────────────────────────────────────────────────────────────────

    def _run_analysis(self):
        self.run_requested.emit()

    def _project_sample(self):
        self.project_requested.emit()

    def _send_loadings_to_vista(self):
        if self._pca_result is None:
            return
        result = self._pca_result
        for i in range(min(3, result.scores.shape[1])):
            var_pct = result.explained_variance_ratio[i] * 100
            name = f"PC{i+1} ({var_pct:.1f}%)"
            self.loadings_to_vista.emit(
                name, result.wavenumbers, result.loading_as_spectrum(i)
            )

    # ──────────────────────────────────────────────────────────────────
    # Annotation visibility toggles (no re-run needed)
    # ──────────────────────────────────────────────────────────────────

    def _toggle_sample_labels(self):
        visible = self.show_names_cb.isChecked()
        for ann in self._annotations:
            ann.set_visible(visible)
        self.canvas.draw_idle()

    def _toggle_biplot_labels(self):
        visible = self.show_biplot_labels_cb.isChecked()
        for txt in self._biplot_texts:
            txt.set_visible(visible)
        self.canvas.draw_idle()

    def _clear_figure_legends(self):
        """Remove all figure-level legends (called before each plot)."""
        for leg in self.figure.legends[:]:
            leg.remove()

    def _apply_legend_outside(self):
        """Place category legend outside the axes area, right side."""
        handles, labels = self.ax.get_legend_handles_labels()
        if not handles:
            self.figure.tight_layout()
            return
        # Remove any axes-internal legend matplotlib may have created
        leg = self.ax.get_legend()
        if leg:
            leg.remove()
        self.figure.legend(
            handles, labels,
            loc="upper left",
            bbox_to_anchor=(0.73, 0.97),
            fontsize=8,
            framealpha=0.75,
            labelcolor=mpl.rcParams["text.color"],
        )
        # Reserve right margin for the legend
        self.figure.subplots_adjust(left=0.09, right=0.71, bottom=0.09, top=0.93)

    # ──────────────────────────────────────────────────────────────────
    # Plotting
    # ──────────────────────────────────────────────────────────────────

    def plot_pca_scores(self, result, labels, category_name: str):
        """Plot PCA score plot colored by category."""
        self._pca_result = result
        self._annotations = []
        self._biplot_texts = []
        self.ax.cla()
        self._clear_figure_legends()
        self._style_ax()

        unique_labels = np.unique(labels)
        colors = {lbl: SPECTRUM_COLORS[i % len(SPECTRUM_COLORS)]
                  for i, lbl in enumerate(unique_labels)}

        var1 = result.explained_variance_ratio[0] * 100
        var2 = result.explained_variance_ratio[1] * 100

        for lbl in unique_labels:
            mask = labels == lbl
            self.ax.scatter(
                result.scores[mask, 0], result.scores[mask, 1],
                c=colors[lbl], label=str(lbl), s=60, alpha=0.85,
                edgecolors="white", linewidth=0.5,
            )

        # Sample name annotations — off by default to avoid clutter
        show_names = self.show_names_cb.isChecked()
        if result.sample_names:
            for i, name in enumerate(result.sample_names):
                ann = self.ax.annotate(
                    name,
                    (result.scores[i, 0], result.scores[i, 1]),
                    fontsize=7, color=FG_DIM, alpha=0.9,
                    xytext=(4, 4), textcoords="offset points",
                    visible=show_names,
                )
                self._annotations.append(ann)

        # Biplot arrows
        if self.biplot_cb.isChecked() and result.wavenumbers is not None:
            n_top = self.biplot_top.value()
            user_scale = self.biplot_scale.value()
            loadings = result.loadings[:2]

            score_range = max(np.ptp(result.scores[:, 0]),
                              np.ptp(result.scores[:, 1]))
            load_max = np.max(np.sqrt(loadings[0]**2 + loadings[1]**2))
            auto_scale = (
                (score_range * 0.4) / load_max
                if load_max > 0 and score_range > 0
                else 1.0
            )
            scale = auto_scale * user_scale

            magnitude = np.sqrt(loadings[0]**2 + loadings[1]**2)
            top_idx = np.argsort(magnitude)[-n_top:]

            show_bl = self.show_biplot_labels_cb.isChecked()
            for idx in top_idx:
                x, y = loadings[0, idx] * scale, loadings[1, idx] * scale
                wn = result.wavenumbers[idx]
                self.ax.annotate(
                    "", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(
                        arrowstyle="-|>", color="#ff6b6b", lw=1.5,
                        mutation_scale=12,
                    ),
                )
                txt = self.ax.text(
                    x * 1.1, y * 1.1, f"{wn:.0f}",
                    fontsize=8, color="#ff6b6b", fontweight="bold",
                    ha="center", va="center",
                    visible=show_bl,
                )
                self._biplot_texts.append(txt)

        tc = mpl.rcParams["text.color"]
        self.ax.set_xlabel(f"PC1 ({var1:.1f}%)", color=tc)
        self.ax.set_ylabel(f"PC2 ({var2:.1f}%)", color=tc)
        self.ax.set_title(f"PCA — {category_name}", color=tc)
        self._apply_legend_outside()
        self.canvas.draw_idle()

        var_text = "Varianza explicada:\n"
        cumulative = 0.0
        for i, v in enumerate(result.explained_variance_ratio):
            cumulative += v
            var_text += f"  PC{i+1}: {v*100:.1f}%  (acumulada: {cumulative*100:.1f}%)\n"
        self._results_text.setPlainText(var_text)

    def plot_plsda_scores(self, result, labels, category_name: str):
        """Plot PLS-DA score plot and classification results."""
        self._annotations = []
        self._biplot_texts = []
        self.ax.cla()
        self._clear_figure_legends()
        self._style_ax()

        unique_labels = np.unique(labels)
        colors = {lbl: SPECTRUM_COLORS[i % len(SPECTRUM_COLORS)]
                  for i, lbl in enumerate(unique_labels)}

        scores = result.scores
        for lbl in unique_labels:
            mask = labels == lbl
            self.ax.scatter(
                scores[mask, 0],
                scores[mask, 1] if scores.shape[1] > 1 else np.zeros(mask.sum()),
                c=colors[lbl], label=str(lbl), s=60, alpha=0.85,
                edgecolors="white", linewidth=0.5,
            )

        # Sample name annotations
        show_names = self.show_names_cb.isChecked()
        if result.sample_names:
            for i, name in enumerate(result.sample_names):
                y_val = scores[i, 1] if scores.shape[1] > 1 else 0.0
                ann = self.ax.annotate(
                    name, (scores[i, 0], y_val),
                    fontsize=7, color=FG_DIM, alpha=0.9,
                    xytext=(4, 4), textcoords="offset points",
                    visible=show_names,
                )
                self._annotations.append(ann)

        tc = mpl.rcParams["text.color"]
        self.ax.set_xlabel("LV1", color=tc)
        self.ax.set_ylabel("LV2", color=tc)
        self.ax.set_title(f"PLS-DA — {category_name}", color=tc)
        self._apply_legend_outside()
        self.canvas.draw_idle()

        text = f"Classification Report (test set):\n{result.report}\n\n"
        text += "Confusion Matrix:\n"
        text += str(result.confusion)
        self._results_text.setPlainText(text)

    def plot_rf_importances(self, result):
        """Plot Random Forest feature importances."""
        self._annotations = []
        self._biplot_texts = []
        self.ax.cla()
        self._clear_figure_legends()
        self._style_ax()

        importances = result.importances
        wn = result.wavenumbers
        top_n = min(20, len(importances))
        top_idx = np.argsort(importances)[-top_n:][::-1]

        self.ax.barh(range(top_n), importances[top_idx], color="#4a9eff", alpha=0.8)
        self.ax.set_yticks(range(top_n))
        self.ax.set_yticklabels([f"{wn[i]:.0f} cm⁻¹" for i in top_idx])
        tc = mpl.rcParams["text.color"]
        self.ax.set_xlabel("Feature Importance", color=tc)
        self.ax.set_title("Random Forest — Top Features", color=tc)
        self.ax.invert_yaxis()
        self.figure.tight_layout()
        self.canvas.draw_idle()

        text = f"Classification Report (test set):\n{result.report}\n\n"
        text += "Confusion Matrix:\n"
        text += str(result.confusion)
        self._results_text.setPlainText(text)

    def plot_projected_sample(self, name: str, x: float, y: float):
        """Mark a sample projected onto the existing PCA axes."""
        self.ax.scatter(
            x, y, marker="*", s=200, c="#ff6b6b", edgecolors="white",
            linewidth=1.5, zorder=20, label=f"→ {name}",
        )
        self.ax.annotate(
            name, (x, y), fontsize=8, color="#ff6b6b",
            xytext=(10, 10), textcoords="offset points",
        )
        self._apply_legend_outside()
        self.canvas.draw_idle()

    @property
    def pca_result(self):
        """Last fitted PCA, or None if no PCA has been run yet."""
        return self._pca_result

    # ──────────────────────────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────────────────────────

    def _save_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar gráfico", "analysis.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if path:
            self.figure.savefig(
                path, dpi=300,
                facecolor=self.figure.get_facecolor(),
                bbox_inches="tight",
            )

    def _save_results(self):
        # This is the formatted report shown in the panel, not tabular data.
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar reporte", "resultados.txt", "Texto (*.txt)"
        )
        if path:
            text = self._results_text.toPlainText()
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)

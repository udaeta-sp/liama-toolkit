"""Annotations tab: peak detection and vertical reference lines."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QGroupBox, QFileDialog,
    QFrame, QHeaderView, QGridLayout,
)

import pandas as pd
from ..widgets.vista_tab import ColorButton


class AnnotationsTab(QWidget):
    """Peak detection and manual vertical line annotations."""

    detect_requested = pyqtSignal()
    vlines_changed = pyqtSignal()
    center_on_peak = pyqtSignal(float)  # wavenumber to center on

    def __init__(self, parent=None):
        super().__init__(parent)
        self._peaks: list[dict] = []
        self._vlines: list[dict] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        # --- Left: Peak detection ---
        peak_group = QGroupBox("Detección de picos")
        peak_layout = QVBoxLayout(peak_group)
        peak_layout.setSpacing(6)

        # Controls grid
        ctrl = QGridLayout()
        ctrl.setSpacing(6)

        ctrl.addWidget(QLabel("Espectro:"), 0, 0)
        self.spectrum_combo = QComboBox()
        ctrl.addWidget(self.spectrum_combo, 0, 1, 1, 3)

        ctrl.addWidget(QLabel("Prominencia:"), 1, 0)
        self.prominence_spin = QDoubleSpinBox()
        self.prominence_spin.setRange(0.001, 10.0)
        self.prominence_spin.setValue(0.05)
        self.prominence_spin.setDecimals(3)
        self.prominence_spin.setSingleStep(0.01)
        ctrl.addWidget(self.prominence_spin, 1, 1)

        ctrl.addWidget(QLabel("Distancia:"), 1, 2)
        self.distance_spin = QSpinBox()
        self.distance_spin.setRange(1, 500)
        self.distance_spin.setValue(10)
        ctrl.addWidget(self.distance_spin, 1, 3)

        btn_detect = QPushButton("Detectar picos")
        btn_detect.setStyleSheet(
            "font-weight: bold; padding: 6px 16px;"
        )
        btn_detect.clicked.connect(self.detect_requested.emit)
        ctrl.addWidget(btn_detect, 2, 0, 1, 4)

        peak_layout.addLayout(ctrl)

        # Peak table — clean style
        self.peak_table = QTableWidget(0, 2)
        self.peak_table.setHorizontalHeaderLabels(["cm⁻¹", "Absorbancia"])
        self.peak_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.peak_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.peak_table.setAlternatingRowColors(True)
        self.peak_table.verticalHeader().setVisible(False)
        self.peak_table.cellClicked.connect(self._on_peak_clicked)
        peak_layout.addWidget(self.peak_table, 1)

        # Peak count + export
        bottom_row = QHBoxLayout()
        self._peak_count_label = QLabel("0 picos")
        self._peak_count_label.setStyleSheet("font-style: italic;")
        bottom_row.addWidget(self._peak_count_label)
        bottom_row.addStretch()
        btn_export = QPushButton("Exportar CSV")
        btn_export.clicked.connect(self._export_peaks)
        bottom_row.addWidget(btn_export)
        peak_layout.addLayout(bottom_row)

        layout.addWidget(peak_group, 2)

        # --- Right: Vertical lines ---
        vline_group = QGroupBox("Líneas verticales de referencia")
        vl_layout = QVBoxLayout(vline_group)
        vl_layout.setSpacing(6)

        # Input grid
        vl_grid = QGridLayout()
        vl_grid.setSpacing(6)

        vl_grid.addWidget(QLabel("cm⁻¹:"), 0, 0)
        self.vline_wn = QDoubleSpinBox()
        self.vline_wn.setRange(200, 5000)
        self.vline_wn.setValue(1000)
        self.vline_wn.setDecimals(1)
        vl_grid.addWidget(self.vline_wn, 0, 1)

        vl_grid.addWidget(QLabel("Color:"), 0, 2)
        self.vline_color = ColorButton("#ffffff")
        vl_grid.addWidget(self.vline_color, 0, 3)

        vl_grid.addWidget(QLabel("Grosor:"), 1, 0)
        self.vline_lw = QDoubleSpinBox()
        self.vline_lw.setRange(0.1, 20.0)
        self.vline_lw.setSingleStep(0.5)
        self.vline_lw.setValue(1.0)
        vl_grid.addWidget(self.vline_lw, 1, 1)

        vl_grid.addWidget(QLabel("Alpha:"), 1, 2)
        self.vline_alpha = QDoubleSpinBox()
        self.vline_alpha.setRange(0.0, 1.0)
        self.vline_alpha.setSingleStep(0.1)
        self.vline_alpha.setValue(0.8)
        vl_grid.addWidget(self.vline_alpha, 1, 3)

        self.vline_label_cb = QCheckBox("Mostrar etiqueta")
        self.vline_label_cb.setChecked(True)
        vl_grid.addWidget(self.vline_label_cb, 2, 0, 1, 2)

        btn_add = QPushButton("Agregar línea")
        btn_add.setStyleSheet("font-weight: bold; padding: 6px 16px;")
        btn_add.clicked.connect(self._add_vline)
        vl_grid.addWidget(btn_add, 2, 2, 1, 2)

        vl_layout.addLayout(vl_grid)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        vl_layout.addWidget(sep)

        # Vline list
        self._vline_list_layout = QVBoxLayout()
        self._vline_list_layout.setSpacing(2)
        vl_layout.addLayout(self._vline_list_layout)
        vl_layout.addStretch()

        layout.addWidget(vline_group, 1)

    def set_visible_spectra(self, names: list[str]):
        """Update the spectrum selector combo."""
        current = self.spectrum_combo.currentText()
        self.spectrum_combo.blockSignals(True)
        self.spectrum_combo.clear()
        for name in names:
            self.spectrum_combo.addItem(name)
        idx = self.spectrum_combo.findText(current)
        if idx >= 0:
            self.spectrum_combo.setCurrentIndex(idx)
        self.spectrum_combo.blockSignals(False)

    def set_peaks(self, peaks: list[dict]):
        """Display detected peaks in the table."""
        self._peaks = peaks
        self.peak_table.setRowCount(len(peaks))
        for i, p in enumerate(peaks):
            wn_item = QTableWidgetItem(f"{p['wavenumber']:.1f}")
            wn_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            abs_item = QTableWidgetItem(f"{p['absorbance']:.4f}")
            abs_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.peak_table.setItem(i, 0, wn_item)
            self.peak_table.setItem(i, 1, abs_item)
        self._peak_count_label.setText(f"{len(peaks)} picos")

    def _on_peak_clicked(self, row: int, _col: int):
        if row < len(self._peaks):
            wn = self._peaks[row]["wavenumber"]
            self.center_on_peak.emit(wn)

    def _export_peaks(self):
        if not self._peaks:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exportar picos", "", "CSV (*.csv)")
        if path:
            df = pd.DataFrame(self._peaks)
            df.to_csv(path, index=False)

    def _add_vline(self):
        vl = {
            "wavenumber": self.vline_wn.value(),
            "color": self.vline_color.color,
            "linewidth": self.vline_lw.value(),
            "alpha": self.vline_alpha.value(),
            "show_label": self.vline_label_cb.isChecked(),
        }
        self._vlines.append(vl)
        self._rebuild_vline_list()
        self.vlines_changed.emit()

    def _remove_vline(self, index: int):
        if 0 <= index < len(self._vlines):
            self._vlines.pop(index)
            self._rebuild_vline_list()
            self.vlines_changed.emit()

    def _rebuild_vline_list(self):
        # Clear
        while self._vline_list_layout.count():
            item = self._vline_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, vl in enumerate(self._vlines):
            row = QFrame()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(4, 2, 4, 2)
            rl.setSpacing(8)

            # Color swatch (small, non-interactive)
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            c = vl["color"]
            swatch.setStyleSheet(
                f"background-color: {c}; border-radius: 6px; border: none;"
            )
            rl.addWidget(swatch)

            lbl = QLabel(f"{vl['wavenumber']:.0f} cm⁻¹")
            rl.addWidget(lbl, 1)

            lw_lbl = QLabel(f"lw={vl['linewidth']:.1f}")
            lw_lbl.setStyleSheet("font-size: 11px;")
            rl.addWidget(lw_lbl)

            btn = QPushButton("×")
            btn.setFixedSize(20, 20)
            btn.clicked.connect(lambda _, idx=i: self._remove_vline(idx))
            rl.addWidget(btn)
            self._vline_list_layout.addWidget(row)

    @property
    def vlines(self) -> list[dict]:
        return self._vlines.copy()

    @property
    def selected_spectrum_name(self) -> str:
        return self.spectrum_combo.currentText()

"""Vista tab: per-spectrum and per-derivative visual controls with multi-select."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel,
    QCheckBox, QPushButton, QDoubleSpinBox, QComboBox,
    QColorDialog, QFrame, QGridLayout,
)

from ..utils.colors import get_spectrum_color, get_derivative_color
from .canvas_widget import SpectrumViewConfig, DerivativeViewConfig


class ColorButton(QPushButton):
    """Small button that shows a color swatch and opens a color picker."""
    color_changed = pyqtSignal(str)

    def __init__(self, color: str = "#4a9eff", parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(24, 24)
        self._update_style()
        self.clicked.connect(self._pick_color)

    def _update_style(self):
        self.setStyleSheet(
            f"background-color: {self._color}; border: 1px solid #555; border-radius: 3px;"
        )

    def _pick_color(self):
        c = QColorDialog.getColor(QColor(self._color), self)
        if c.isValid():
            self._color = c.name()
            self._update_style()
            self.color_changed.emit(self._color)

    @property
    def color(self) -> str:
        return self._color

    def set_color(self, color: str):
        self._color = color
        self._update_style()


class SpectrumRow(QFrame):
    """One row of spectrum controls in the Vista tab."""
    changed = pyqtSignal()

    def __init__(self, config: SpectrumViewConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.selected = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(4)

        # Selection checkbox (for multi-select operations)
        self.select_cb = QCheckBox()
        self.select_cb.setToolTip("Seleccionar para operación grupal")
        layout.addWidget(self.select_cb)

        # Visibility toggle
        self.vis_cb = QCheckBox()
        self.vis_cb.setChecked(self.config.visible)
        self.vis_cb.setToolTip("Visible")
        self.vis_cb.stateChanged.connect(self._on_change)
        layout.addWidget(self.vis_cb)

        # Name label
        name_lbl = QLabel(self.config.name)
        name_lbl.setMinimumWidth(120)
        name_lbl.setToolTip(self.config.name)
        layout.addWidget(name_lbl, 1)

        # Color button
        self.color_btn = ColorButton(self.config.color)
        self.color_btn.color_changed.connect(self._on_change)
        layout.addWidget(self.color_btn)

        # Linewidth
        self.lw_spin = QDoubleSpinBox()
        self.lw_spin.setRange(0.1, 10.0)
        self.lw_spin.setSingleStep(0.5)
        self.lw_spin.setValue(self.config.linewidth)
        self.lw_spin.setFixedWidth(60)
        self.lw_spin.setToolTip("Grosor")
        self.lw_spin.valueChanged.connect(self._on_change)
        layout.addWidget(self.lw_spin)

        # Alpha
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(self.config.alpha)
        self.alpha_spin.setFixedWidth(55)
        self.alpha_spin.setToolTip("Transparencia")
        self.alpha_spin.valueChanged.connect(self._on_change)
        layout.addWidget(self.alpha_spin)

        # Scale
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(-100.0, 100.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setValue(self.config.scale)
        self.scale_spin.setFixedWidth(65)
        self.scale_spin.setToolTip("Escala (1.0 = sin cambio)")
        self.scale_spin.valueChanged.connect(self._on_change)
        layout.addWidget(self.scale_spin)

        # Offset (absorbance units)
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-50.0, 50.0)
        self.offset_spin.setSingleStep(0.1)
        self.offset_spin.setDecimals(2)
        self.offset_spin.setValue(self.config.offset)
        self.offset_spin.setFixedWidth(65)
        self.offset_spin.setToolTip("Offset (unidades de absorbancia)")
        self.offset_spin.valueChanged.connect(self._on_change)
        layout.addWidget(self.offset_spin)

        # Move up/down
        btn_up = QPushButton("▲")
        btn_up.setFixedSize(22, 22)
        btn_up.setToolTip("Mover arriba")
        btn_up.clicked.connect(lambda: self._emit_move(-1))
        layout.addWidget(btn_up)

        btn_down = QPushButton("▼")
        btn_down.setFixedSize(22, 22)
        btn_down.setToolTip("Mover abajo")
        btn_down.clicked.connect(lambda: self._emit_move(1))
        layout.addWidget(btn_down)

        # Add derivative button
        btn_deriv = QPushButton("D²")
        btn_deriv.setFixedSize(28, 22)
        btn_deriv.setToolTip("Agregar derivada segunda")
        btn_deriv.clicked.connect(self._request_derivative)
        layout.addWidget(btn_deriv)

    def _on_change(self, *_):
        self.config.visible = self.vis_cb.isChecked()
        self.config.color = self.color_btn.color
        self.config.linewidth = self.lw_spin.value()
        self.config.alpha = self.alpha_spin.value()
        self.config.scale = self.scale_spin.value()
        self.config.offset = self.offset_spin.value()
        self.changed.emit()

    def _emit_move(self, direction: int):
        self.setProperty("move_direction", direction)
        self.changed.emit()

    def _request_derivative(self):
        self.setProperty("request_derivative", True)
        self.changed.emit()

    def update_from_config(self):
        """Sync widgets from config."""
        self.vis_cb.blockSignals(True)
        self.vis_cb.setChecked(self.config.visible)
        self.vis_cb.blockSignals(False)
        self.color_btn.set_color(self.config.color)
        self.lw_spin.blockSignals(True)
        self.lw_spin.setValue(self.config.linewidth)
        self.lw_spin.blockSignals(False)
        self.alpha_spin.blockSignals(True)
        self.alpha_spin.setValue(self.config.alpha)
        self.alpha_spin.blockSignals(False)
        self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(self.config.scale)
        self.scale_spin.blockSignals(False)
        self.offset_spin.blockSignals(True)
        self.offset_spin.setValue(self.config.offset)
        self.offset_spin.blockSignals(False)


class DerivativeRow(QFrame):
    """One row of derivative controls in the Vista tab."""
    changed = pyqtSignal()
    remove_requested = pyqtSignal()

    def __init__(self, config: DerivativeViewConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(4)

        # Selection checkbox
        self.select_cb = QCheckBox()
        layout.addWidget(self.select_cb)

        # Visibility
        self.vis_cb = QCheckBox()
        self.vis_cb.setChecked(self.config.visible)
        self.vis_cb.stateChanged.connect(self._on_change)
        layout.addWidget(self.vis_cb)

        # Name
        name_lbl = QLabel(f"D² {self.config.name}")
        name_lbl.setMinimumWidth(120)
        name_lbl.setStyleSheet("color: #a0a0a0; font-style: italic;")
        layout.addWidget(name_lbl, 1)

        # Color
        self.color_btn = ColorButton(self.config.color)
        self.color_btn.color_changed.connect(self._on_change)
        layout.addWidget(self.color_btn)

        # Linewidth
        self.lw_spin = QDoubleSpinBox()
        self.lw_spin.setRange(0.1, 10.0)
        self.lw_spin.setSingleStep(0.5)
        self.lw_spin.setValue(self.config.linewidth)
        self.lw_spin.setFixedWidth(60)
        self.lw_spin.valueChanged.connect(self._on_change)
        layout.addWidget(self.lw_spin)

        # Alpha
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(self.config.alpha)
        self.alpha_spin.setFixedWidth(55)
        self.alpha_spin.valueChanged.connect(self._on_change)
        layout.addWidget(self.alpha_spin)

        # Scale
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(-1e6, 1e6)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setValue(self.config.scale)
        self.scale_spin.setFixedWidth(65)
        self.scale_spin.setToolTip("Escala (1.0 = sin cambio)")
        self.scale_spin.valueChanged.connect(self._on_change)
        layout.addWidget(self.scale_spin)

        # Offset — derivative values are very small, need fine steps
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-1e6, 1e6)
        self.offset_spin.setSingleStep(0.0001)
        self.offset_spin.setDecimals(6)
        self.offset_spin.setValue(self.config.offset)
        self.offset_spin.setFixedWidth(85)
        self.offset_spin.setToolTip("Offset (unidades de derivada)")
        self.offset_spin.valueChanged.connect(self._on_change)
        layout.addWidget(self.offset_spin)

        # Move up/down
        btn_up = QPushButton("▲")
        btn_up.setFixedSize(22, 22)
        btn_up.clicked.connect(lambda: self._emit_move(-1))
        layout.addWidget(btn_up)
        btn_down = QPushButton("▼")
        btn_down.setFixedSize(22, 22)
        btn_down.clicked.connect(lambda: self._emit_move(1))
        layout.addWidget(btn_down)

        # Remove
        btn_remove = QPushButton("×")
        btn_remove.setFixedSize(22, 22)
        btn_remove.setToolTip("Eliminar derivada")
        btn_remove.clicked.connect(self.remove_requested.emit)
        layout.addWidget(btn_remove)

    def _on_change(self, *_):
        self.config.visible = self.vis_cb.isChecked()
        self.config.color = self.color_btn.color
        self.config.linewidth = self.lw_spin.value()
        self.config.alpha = self.alpha_spin.value()
        self.config.scale = self.scale_spin.value()
        self.config.offset = self.offset_spin.value()
        self.changed.emit()

    def _emit_move(self, direction: int):
        self.setProperty("move_direction", direction)
        self.changed.emit()


class VistaTab(QWidget):
    """Vista tab with spectrum rows, derivative rows, and batch controls."""

    redraw_requested = pyqtSignal()
    derivative_requested = pyqtSignal(int)  # spectrum stage index
    derivative_removed = pyqtSignal(int)    # derivative index
    move_requested = pyqtSignal(str, int, int)  # "spectrum"/"derivative", index, direction

    def __init__(self, parent=None):
        super().__init__(parent)
        self._spectrum_rows: list[SpectrumRow] = []
        self._derivative_rows: list[DerivativeRow] = []
        self._metadata_categories: list[str] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Top toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Colorear por:"))
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItem("Ninguno")
        self.color_by_combo.currentTextChanged.connect(self._on_color_by_changed)
        toolbar.addWidget(self.color_by_combo)

        btn_stack = QPushButton("Apilar auto")
        btn_stack.setToolTip("Distribuir offsets equidistantes")
        btn_stack.clicked.connect(self._auto_stack)
        toolbar.addWidget(btn_stack)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Scroll area for spectra and derivatives
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        scroll.setWidget(self._content)
        layout.addWidget(scroll, 1)

        # Batch apply bar
        apply_frame = QFrame()
        apply_frame.setStyleSheet("border-top: 1px solid #555;")
        apply_layout = QHBoxLayout(apply_frame)
        apply_layout.setContentsMargins(4, 4, 4, 4)
        apply_layout.addWidget(QLabel("Aplicar a selección:"))

        apply_layout.addWidget(QLabel("Color:"))
        self._batch_color = ColorButton("#7a7a7a")
        apply_layout.addWidget(self._batch_color)

        apply_layout.addWidget(QLabel("Grosor:"))
        self._batch_lw = QDoubleSpinBox()
        self._batch_lw.setRange(0.1, 10.0)
        self._batch_lw.setValue(1.5)
        self._batch_lw.setFixedWidth(60)
        apply_layout.addWidget(self._batch_lw)

        apply_layout.addWidget(QLabel("Alpha:"))
        self._batch_alpha = QDoubleSpinBox()
        self._batch_alpha.setRange(0.0, 1.0)
        self._batch_alpha.setSingleStep(0.1)
        self._batch_alpha.setValue(1.0)
        self._batch_alpha.setFixedWidth(55)
        apply_layout.addWidget(self._batch_alpha)

        apply_layout.addWidget(QLabel("Escala:"))
        self._batch_scale = QDoubleSpinBox()
        self._batch_scale.setRange(-100.0, 100.0)
        self._batch_scale.setValue(1.0)
        self._batch_scale.setFixedWidth(65)
        apply_layout.addWidget(self._batch_scale)

        btn_apply = QPushButton("Aplicar")
        btn_apply.clicked.connect(self._apply_to_selected)
        apply_layout.addWidget(btn_apply)

        apply_layout.addStretch()
        layout.addWidget(apply_frame)

    def rebuild(
        self,
        spectrum_configs: list[SpectrumViewConfig],
        derivative_configs: list[DerivativeViewConfig],
    ):
        """Rebuild all rows from configs."""
        # Clear old
        for row in self._spectrum_rows + self._derivative_rows:
            row.setParent(None)
            row.deleteLater()
        self._spectrum_rows.clear()
        self._derivative_rows.clear()

        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # Spectrum section header
        if spectrum_configs:
            hdr = QLabel("  Espectros")
            hdr.setStyleSheet("font-weight: bold; color: #e0e0e0; padding: 2px;")
            self._content_layout.addWidget(hdr)

        for i, cfg in enumerate(spectrum_configs):
            row = SpectrumRow(cfg)
            row.changed.connect(lambda idx=i: self._on_spectrum_row_changed(idx))
            self._spectrum_rows.append(row)
            self._content_layout.addWidget(row)

        # Derivative section header
        if derivative_configs:
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet("color: #555;")
            self._content_layout.addWidget(sep)
            hdr = QLabel("  Derivadas")
            hdr.setStyleSheet("font-weight: bold; color: #a0a0a0; padding: 2px;")
            self._content_layout.addWidget(hdr)

        for i, dcfg in enumerate(derivative_configs):
            row = DerivativeRow(dcfg)
            row.changed.connect(lambda idx=i: self._on_deriv_row_changed(idx))
            row.remove_requested.connect(lambda idx=i: self.derivative_removed.emit(idx))
            self._derivative_rows.append(row)
            self._content_layout.addWidget(row)

        self._content_layout.addStretch()

    def _on_spectrum_row_changed(self, index: int):
        row = self._spectrum_rows[index]
        # Check if this is a move request
        direction = row.property("move_direction")
        if direction is not None:
            row.setProperty("move_direction", None)
            # Check if multiple selected — move as group
            selected = [i for i, r in enumerate(self._spectrum_rows) if r.select_cb.isChecked()]
            if selected and index in selected:
                for idx in (selected if direction < 0 else reversed(selected)):
                    self.move_requested.emit("spectrum", idx, direction)
            else:
                self.move_requested.emit("spectrum", index, direction)
            return

        # Check derivative request
        if row.property("request_derivative"):
            row.setProperty("request_derivative", None)
            self.derivative_requested.emit(index)
            return

        self.redraw_requested.emit()

    def _on_deriv_row_changed(self, index: int):
        row = self._derivative_rows[index]
        direction = row.property("move_direction")
        if direction is not None:
            row.setProperty("move_direction", None)
            self.move_requested.emit("derivative", index, direction)
            return
        self.redraw_requested.emit()

    def _apply_to_selected(self):
        """Apply batch settings to all selected rows."""
        color = self._batch_color.color
        lw = self._batch_lw.value()
        alpha = self._batch_alpha.value()
        scale = self._batch_scale.value()

        for row in self._spectrum_rows:
            if row.select_cb.isChecked():
                row.config.color = color
                row.config.linewidth = lw
                row.config.alpha = alpha
                row.config.scale = scale
                row.update_from_config()

        for row in self._derivative_rows:
            if row.select_cb.isChecked():
                row.config.color = color
                row.config.linewidth = lw
                row.config.alpha = alpha
                row.config.scale = scale
                row.changed.emit()

        self.redraw_requested.emit()

    def _auto_stack(self):
        """Auto-distribute offsets for visible spectra."""
        visible_rows = [r for r in self._spectrum_rows if r.config.visible]
        if len(visible_rows) < 2:
            return
        # Estimate a good offset step from the first visible spectrum
        step = 0.5  # default offset step in absorbance units
        for i, row in enumerate(visible_rows):
            row.config.offset = i * step
            row.update_from_config()
        self.redraw_requested.emit()

    def set_metadata_lookup(self, lookup: dict[str, dict[str, str]]):
        """Set metadata lookup: spectrum_name -> {category: value}."""
        self._metadata_lookup = lookup

    def _on_color_by_changed(self, category: str):
        """Assign colors by metadata category."""
        if category == "Ninguno":
            for i, row in enumerate(self._spectrum_rows):
                row.config.color = get_spectrum_color(i)
                row.update_from_config()
        else:
            lookup = getattr(self, "_metadata_lookup", {})
            # Group by category value
            groups: dict[str, list[SpectrumRow]] = {}
            for row in self._spectrum_rows:
                meta = lookup.get(row.config.name, {})
                val = meta.get(category, "?")
                groups.setdefault(val, []).append(row)
            # Assign one color per group
            for i, (val, rows) in enumerate(groups.items()):
                color = get_spectrum_color(i)
                for row in rows:
                    row.config.color = color
                    row.update_from_config()
        self.redraw_requested.emit()

    def update_metadata_categories(self, categories: list[str]):
        """Update the 'color by' combo with available categories."""
        self._metadata_categories = categories
        self.color_by_combo.blockSignals(True)
        self.color_by_combo.clear()
        self.color_by_combo.addItem("Ninguno")
        for cat in categories:
            self.color_by_combo.addItem(cat)
        self.color_by_combo.blockSignals(False)

    def get_spectrum_configs(self) -> list[SpectrumViewConfig]:
        return [row.config for row in self._spectrum_rows]

    def get_derivative_configs(self) -> list[DerivativeViewConfig]:
        return [row.config for row in self._derivative_rows]

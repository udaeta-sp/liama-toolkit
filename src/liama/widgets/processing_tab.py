"""Processing tab: Savitzky-Golay smoothing controls."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QCheckBox, QLabel, QSpinBox,
)


class ProcessingTab(QWidget):
    """Smoothing controls that affect visualization globally."""

    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        self.sg_enabled = QCheckBox("Suavizado Savitzky-Golay")
        self.sg_enabled.stateChanged.connect(self._on_change)
        layout.addWidget(self.sg_enabled)

        layout.addWidget(QLabel("Ventana:"))
        self.sg_window = QSpinBox()
        self.sg_window.setRange(3, 101)
        self.sg_window.setSingleStep(2)
        self.sg_window.setValue(5)
        self.sg_window.setToolTip("Debe ser impar. Si es par, se ajusta al impar superior.")
        self.sg_window.valueChanged.connect(self._on_window_change)
        layout.addWidget(self.sg_window)

        layout.addWidget(QLabel("Grado:"))
        self.sg_poly = QSpinBox()
        self.sg_poly.setRange(1, 10)
        self.sg_poly.setValue(3)
        self.sg_poly.valueChanged.connect(self._on_change)
        layout.addWidget(self.sg_poly)

        self._info_label = QLabel(
            "Afecta: vista de espectros, derivada 2ª en vista, y detección de picos.\n"
            "NO afecta el pipeline de análisis multivariado (tiene sus propios parámetros)."
        )
        self._info_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)
        layout.addStretch()

    def _on_window_change(self, value: int):
        # Auto-correct to odd
        if value % 2 == 0:
            self.sg_window.blockSignals(True)
            self.sg_window.setValue(value + 1)
            self.sg_window.blockSignals(False)
        # Ensure polyorder < window
        max_poly = self.sg_window.value() - 1
        if self.sg_poly.value() > max_poly:
            self.sg_poly.setValue(max_poly)
        self._on_change()

    def _on_change(self, *_):
        self.settings_changed.emit()

    @property
    def smoothing_enabled(self) -> bool:
        return self.sg_enabled.isChecked()

    @property
    def window(self) -> int:
        v = self.sg_window.value()
        return v if v % 2 == 1 else v + 1

    @property
    def polyorder(self) -> int:
        return min(self.sg_poly.value(), self.window - 1)

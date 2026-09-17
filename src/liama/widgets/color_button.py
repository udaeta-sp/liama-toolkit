"""Small color swatch button shared by the Vista and Annotations tabs."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QPushButton, QColorDialog


class ColorButton(QPushButton):
    """Button that shows a color swatch and opens a color picker."""

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

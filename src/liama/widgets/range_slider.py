"""Double-handle range slider widget for wavenumber selection."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal, QRect
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QMouseEvent
from PyQt6.QtWidgets import QWidget


class RangeSlider(QWidget):
    """Horizontal slider with two handles defining a range.

    Dragging a handle moves that end.
    Dragging the bar between handles pans the window.
    """

    range_changed = pyqtSignal(float, float)

    HANDLE_W = 12
    BAR_H = 8
    MIN_SPAN = 10  # minimum range in value units

    def __init__(self, vmin: float = 400, vmax: float = 4000,
                 inverted: bool = False, parent=None):
        super().__init__(parent)
        self._vmin = vmin
        self._vmax = vmax
        self._low = vmin
        self._high = vmax
        self._inverted = inverted  # True: left=high value, right=low value
        self._dragging: str | None = None  # "low", "high", "bar"
        self._drag_offset = 0.0
        # Themeable colors
        self.groove_color = "#3c3c3c"
        self.range_color = "#4a9eff"
        self.handle_fill = "#ffffff"
        self.handle_border = "#4a9eff"
        self.setMinimumHeight(28)
        self.setMaximumHeight(28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_range(self, vmin: float, vmax: float):
        self._vmin = vmin
        self._vmax = vmax
        self._low = max(self._low, vmin)
        self._high = min(self._high, vmax)
        self.update()

    def set_values(self, low: float, high: float):
        self._low = max(self._vmin, min(low, self._vmax))
        self._high = max(self._vmin, min(high, self._vmax))
        if self._high - self._low < self.MIN_SPAN:
            self._high = self._low + self.MIN_SPAN
        self.update()

    @property
    def low(self) -> float:
        return self._low

    @property
    def high(self) -> float:
        return self._high

    def _val_to_x(self, val: float) -> int:
        w = self.width() - 2 * self.HANDLE_W
        if self._vmax == self._vmin:
            return self.HANDLE_W
        frac = (val - self._vmin) / (self._vmax - self._vmin)
        if self._inverted:
            frac = 1.0 - frac
        return int(self.HANDLE_W + frac * w)

    def _x_to_val(self, x: int) -> float:
        w = self.width() - 2 * self.HANDLE_W
        if w <= 0:
            return self._vmin
        frac = (x - self.HANDLE_W) / w
        frac = max(0.0, min(1.0, frac))
        if self._inverted:
            frac = 1.0 - frac
        return self._vmin + frac * (self._vmax - self._vmin)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        h = self.height()
        bar_y = (h - self.BAR_H) // 2

        # Background groove
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(self.groove_color)))
        p.drawRoundedRect(self.HANDLE_W, bar_y,
                          self.width() - 2 * self.HANDLE_W, self.BAR_H, 3, 3)

        # Selected range
        x_low = self._val_to_x(self._low)
        x_high = self._val_to_x(self._high)
        x_left = min(x_low, x_high)
        x_right = max(x_low, x_high)
        p.setBrush(QBrush(QColor(self.range_color)))
        p.drawRoundedRect(x_left, bar_y, x_right - x_left, self.BAR_H, 3, 3)

        # Handles
        for x in (x_low, x_high):
            p.setBrush(QBrush(QColor(self.handle_fill)))
            p.setPen(QPen(QColor(self.handle_border), 2))
            r = QRect(x - self.HANDLE_W // 2, 2, self.HANDLE_W, h - 4)
            p.drawRoundedRect(r, 4, 4)

        p.end()

    def mousePressEvent(self, ev: QMouseEvent):
        x = int(ev.position().x())
        x_low = self._val_to_x(self._low)
        x_high = self._val_to_x(self._high)

        if abs(x - x_low) <= self.HANDLE_W:
            self._dragging = "low"
        elif abs(x - x_high) <= self.HANDLE_W:
            self._dragging = "high"
        elif min(x_low, x_high) < x < max(x_low, x_high):
            self._dragging = "bar"
            self._drag_offset = self._x_to_val(x) - self._low

    def mouseMoveEvent(self, ev: QMouseEvent):
        if not self._dragging:
            return
        val = self._x_to_val(int(ev.position().x()))

        if self._dragging == "low":
            self._low = max(self._vmin, min(val, self._high - self.MIN_SPAN))
        elif self._dragging == "high":
            self._high = min(self._vmax, max(val, self._low + self.MIN_SPAN))
        elif self._dragging == "bar":
            new_low = val - self._drag_offset
            span = self._high - self._low
            if new_low < self._vmin:
                new_low = self._vmin
            if new_low + span > self._vmax:
                new_low = self._vmax - span
            self._low = new_low
            self._high = new_low + span

        self.update()
        self.range_changed.emit(self._low, self._high)

    def mouseReleaseEvent(self, ev: QMouseEvent):
        self._dragging = None

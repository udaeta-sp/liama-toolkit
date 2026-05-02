"""Side panel: spectrum list with search, metadata indicators, and color memory."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QScrollArea, QCheckBox, QLabel, QFrame, QToolTip,
)

from ..core.spectrum import Spectrum


class SpectrumItemWidget(QFrame):
    """Single spectrum row in the list with checkbox, color swatch, and metadata badge."""

    toggled = pyqtSignal(int, bool)  # index, checked

    def __init__(self, index: int, spectrum: Spectrum, parent=None):
        super().__init__(parent)
        self.index = index
        self.spectrum = spectrum
        self._color: str | None = None  # remembered color from stage
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(4)

        # Color swatch (shows last assigned color, or nothing)
        self._swatch = QLabel()
        self._swatch.setFixedSize(10, 10)
        self._swatch.setStyleSheet(
            "background-color: transparent; border-radius: 5px; border: none;"
        )
        layout.addWidget(self._swatch)

        # Checkbox
        self.checkbox = QCheckBox()
        self.checkbox.setEnabled(self.spectrum.load_error is None)
        self.checkbox.stateChanged.connect(self._on_toggle)
        layout.addWidget(self.checkbox)

        # Name
        name = self.spectrum.name
        if self.spectrum.load_error:
            name = f"{name} (error)"
        self._name_label = QLabel(name)
        self._name_label.setMinimumWidth(80)
        if self.spectrum.load_error:
            self._name_label.setStyleSheet("color: #ff6b6b;")
        layout.addWidget(self._name_label, 1)

        # Metadata badge
        self._meta_badge = QLabel()
        self._meta_badge.setFixedSize(16, 16)
        self._meta_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._meta_badge.setCursor(Qt.CursorShape.WhatsThisCursor)
        self._update_meta_badge()
        layout.addWidget(self._meta_badge)

    def _on_toggle(self, state: int):
        checked = state == Qt.CheckState.Checked.value
        self.toggled.emit(self.index, checked)

    def set_color(self, color: str | None):
        """Remember color from stage assignment."""
        self._color = color
        if color:
            self._swatch.setStyleSheet(
                f"background-color: {color}; border-radius: 5px; border: none;"
            )
        else:
            self._swatch.setStyleSheet(
                "background-color: transparent; border-radius: 5px; border: none;"
            )

    def _update_meta_badge(self):
        """Show metadata indicator."""
        if self.spectrum.metadata:
            self._meta_badge.setText("M")
            self._meta_badge.setStyleSheet(
                "color: #4a9eff; font-weight: bold; font-size: 10px; "
                "border: 1px solid #4a9eff; border-radius: 8px; background: transparent;"
            )
            # Build tooltip
            lines = [f"<b>{k}:</b> {v}" for k, v in self.spectrum.metadata.items()]
            self._meta_badge.setToolTip("<br>".join(lines))
        else:
            self._meta_badge.setText("")
            self._meta_badge.setStyleSheet("border: none; background: transparent;")
            self._meta_badge.setToolTip("")

    def refresh_metadata(self):
        """Refresh metadata badge after metadata import."""
        self._update_meta_badge()


class SpectrumListPanel(QWidget):
    """Panel lateral with searchable checkbox list of spectra."""

    # Emitted when a spectrum is checked/unchecked: (index, checked)
    spectrum_toggled = pyqtSignal(int, bool)
    # Emitted for batch selection changes: list of (index, checked)
    batch_toggled = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._spectra: list[Spectrum] = []
        self._items: list[SpectrumItemWidget] = []
        self._metadata_keys: list[str] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header
        self._header_label = QLabel("Espectros (0)")
        self._header_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self._header_label)

        # Search box
        self._search = QLineEdit()
        self._search.setPlaceholderText("Buscar... (AND por espacio, | para OR)")
        self._search.textChanged.connect(self._apply_filter)
        layout.addWidget(self._search)

        # Select/deselect buttons
        btn_row = QHBoxLayout()
        btn_select = QPushButton("Seleccionar filtrados")
        btn_select.clicked.connect(self._select_filtered)
        btn_none = QPushButton("Ninguno")
        btn_none.clicked.connect(self._deselect_all)
        btn_row.addWidget(btn_select)
        btn_row.addWidget(btn_none)
        layout.addLayout(btn_row)

        # Scrollable list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list_container = QWidget()
        self._list_layout = QVBoxLayout(self._list_container)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(0)
        self._list_layout.addStretch()
        scroll.setWidget(self._list_container)
        layout.addWidget(scroll, 1)

    def set_spectra(self, spectra: list[Spectrum]):
        """Replace the entire list with new spectra."""
        self._spectra = spectra

        # Clear old items
        for item in self._items:
            item.setParent(None)
            item.deleteLater()
        self._items.clear()

        # Remove stretch
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create item widgets
        for i, sp in enumerate(spectra):
            item = SpectrumItemWidget(i, sp)
            item.toggled.connect(self._on_toggle)
            self._list_layout.addWidget(item)
            self._items.append(item)

        self._list_layout.addStretch()
        self._header_label.setText(f"Espectros ({len(spectra)})")

    def _on_toggle(self, index: int, checked: bool):
        self.spectrum_toggled.emit(index, checked)

    def _apply_filter(self):
        text = self._search.text().strip().lower()
        if not text:
            for item in self._items:
                item.setVisible(True)
            return

        # Parse OR groups (separated by |), each group is AND (space-separated)
        or_groups = [g.strip().split() for g in text.split("|")]

        for item in self._items:
            sp = item.spectrum
            # Build searchable text: name + metadata values
            searchable = sp.name.lower()
            for v in sp.metadata.values():
                searchable += " " + v.lower()

            visible = False
            for group in or_groups:
                if all(term in searchable for term in group):
                    visible = True
                    break
            item.setVisible(visible)

    def _select_filtered(self):
        changes = []
        for item in self._items:
            if item.isVisible() and item.checkbox.isEnabled() and not item.checkbox.isChecked():
                item.checkbox.blockSignals(True)
                item.checkbox.setChecked(True)
                item.checkbox.blockSignals(False)
                changes.append((item.index, True))
        if changes:
            self.batch_toggled.emit(changes)

    def _deselect_all(self):
        changes = []
        for item in self._items:
            if item.checkbox.isChecked():
                item.checkbox.blockSignals(True)
                item.checkbox.setChecked(False)
                item.checkbox.blockSignals(False)
                changes.append((item.index, False))
        if changes:
            self.batch_toggled.emit(changes)

    def set_checked(self, index: int, checked: bool):
        """Programmatically set checkbox state."""
        if 0 <= index < len(self._items):
            self._items[index].checkbox.blockSignals(True)
            self._items[index].checkbox.setChecked(checked)
            self._items[index].checkbox.blockSignals(False)

    def set_spectrum_color(self, index: int, color: str | None):
        """Remember color assignment for a spectrum (persists after unstaging)."""
        if 0 <= index < len(self._items):
            self._items[index].set_color(color)

    def refresh_metadata(self):
        """Refresh all metadata badges after metadata import."""
        for item in self._items:
            item.refresh_metadata()

    def update_metadata_keys(self, keys: list[str]):
        """Update known metadata keys for search matching."""
        self._metadata_keys = keys
        self.refresh_metadata()

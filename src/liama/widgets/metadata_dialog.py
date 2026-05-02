"""Metadata preview dialog: load CSV/Excel, match to spectra, import."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QTableWidget, QTableWidgetItem, QDialogButtonBox,
)

from ..utils.theme import OK_BG, ERROR_BG


class MetadataDialog(QDialog):
    """Preview and import metadata from CSV/Excel before associating to spectra."""

    def __init__(self, file_path: str, spectrum_names: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Previsualización de metadatos")
        self.setMinimumSize(700, 500)

        self._spectrum_names = spectrum_names
        self._spectrum_names_lower = [n.lower() for n in spectrum_names]
        self._file_path = file_path
        self._df: pd.DataFrame | None = None
        self._match_col: str = ""
        self._categories: list[str] = []
        self._result: dict[str, dict[str, str]] | None = None

        self._load_file()
        self._setup_ui()
        self._update_preview()

    def _load_file(self):
        p = Path(self._file_path)
        if p.suffix.lower() in (".xlsx", ".xls"):
            self._df = pd.read_excel(self._file_path)
        else:
            # Try different separators
            for sep in (",", ";", "\t"):
                try:
                    df = pd.read_csv(self._file_path, sep=sep)
                    if len(df.columns) > 1:
                        self._df = df
                        break
                except Exception:
                    continue
            if self._df is None:
                self._df = pd.read_csv(self._file_path)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # File info
        info = QLabel(
            f"Archivo: {Path(self._file_path).name}  |  "
            f"Filas: {len(self._df)}  |  Columnas: {len(self._df.columns)}"
        )
        layout.addWidget(info)

        # Match column selector
        match_row = QHBoxLayout()
        match_row.addWidget(QLabel("Columna de match:"))
        self._match_combo = QComboBox()
        for col in self._df.columns:
            self._match_combo.addItem(str(col))
        self._match_combo.currentTextChanged.connect(self._update_preview)
        match_row.addWidget(self._match_combo, 1)
        layout.addLayout(match_row)

        # Preview table
        self._table = QTableWidget()
        layout.addWidget(self._table, 1)

        # Match count
        self._match_label = QLabel()
        layout.addWidget(self._match_label)

        # Categories
        cat_row = QHBoxLayout()
        cat_row.addWidget(QLabel("Categorías detectadas:"))
        self._cat_labels: list[QLabel] = []
        self._cat_layout = cat_row
        layout.addLayout(cat_row)

        # Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("Importar metadatos")
        btns.accepted.connect(self._do_import)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _update_preview(self):
        if self._df is None:
            return

        match_col = self._match_combo.currentText()
        self._match_col = match_col

        # Set up table
        cols = list(self._df.columns)
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels([str(c) for c in cols])
        self._table.setRowCount(len(self._df))

        match_count = 0
        match_col_idx = cols.index(match_col) if match_col in cols else 0

        for row_idx in range(len(self._df)):
            match_val = str(self._df.iloc[row_idx][match_col]).strip()
            matched = self._find_match(match_val) is not None
            if matched:
                match_count += 1

            for col_idx, col_name in enumerate(cols):
                val = str(self._df.iloc[row_idx][col_name])
                item = QTableWidgetItem(val)
                if col_idx == match_col_idx:
                    if matched:
                        item.setBackground(QColor(OK_BG))
                    else:
                        item.setBackground(QColor(ERROR_BG))
                self._table.setItem(row_idx, col_idx, item)

        self._table.resizeColumnsToContents()
        self._match_label.setText(
            f"Matcheados: {match_count}/{len(self._spectrum_names)} espectros"
        )

        # Detect categorical columns (exclude match column)
        self._categories = []
        for col in cols:
            if col == match_col:
                continue
            n_unique = self._df[col].nunique()
            # Consider categorical if few unique values relative to total
            if 1 < n_unique <= max(20, len(self._df) // 2):
                self._categories.append(str(col))

        # Update category labels
        for lbl in self._cat_labels:
            lbl.deleteLater()
        self._cat_labels.clear()
        for cat in self._categories:
            lbl = QLabel(f"[{cat}]")
            lbl.setStyleSheet(
                "background-color: #3c3c3c; padding: 2px 8px; border-radius: 3px;"
            )
            self._cat_layout.addWidget(lbl)
            self._cat_labels.append(lbl)

    def _find_match(self, value: str) -> int | None:
        """Find spectrum index matching this metadata value."""
        val_lower = value.lower().strip()
        # Exact match first
        for i, name in enumerate(self._spectrum_names_lower):
            if name == val_lower:
                return i
        # Substring match
        for i, name in enumerate(self._spectrum_names_lower):
            if val_lower in name or name in val_lower:
                return i
        return None

    def _do_import(self):
        """Build the metadata mapping and accept."""
        self._result = {}
        match_col = self._match_col

        for row_idx in range(len(self._df)):
            match_val = str(self._df.iloc[row_idx][match_col]).strip()
            sp_idx = self._find_match(match_val)
            if sp_idx is not None:
                sp_name = self._spectrum_names[sp_idx]
                meta = {}
                for cat in self._categories:
                    meta[cat] = str(self._df.iloc[row_idx][cat])
                self._result[sp_name] = meta

        self.accept()

    @property
    def result(self) -> dict[str, dict[str, str]] | None:
        """Mapping: spectrum_name -> {category: value}. None if cancelled."""
        return self._result

    @property
    def categories(self) -> list[str]:
        return self._categories

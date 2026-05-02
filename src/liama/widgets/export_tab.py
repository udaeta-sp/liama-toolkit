"""Export tab: image export (single/batch) and spectral data CSV export."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QSpinBox, QCheckBox,
    QGroupBox, QFileDialog, QComboBox,
)


class ExportTab(QWidget):
    """Image and CSV data export controls."""

    export_image = pyqtSignal(dict)   # params for image export
    export_csv = pyqtSignal(dict)     # params for CSV export

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(16)

        # --- Image export ---
        img_group = QGroupBox("Imagen")
        img_layout = QVBoxLayout(img_group)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Formato:"))
        self._fmt_group = QButtonGroup(self)
        for fmt in ("PNG", "PDF", "SVG"):
            rb = QRadioButton(fmt)
            self._fmt_group.addButton(rb)
            fmt_row.addWidget(rb)
            if fmt == "PNG":
                rb.setChecked(True)
        img_layout.addLayout(fmt_row)

        dpi_row = QHBoxLayout()
        dpi_row.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setValue(300)
        dpi_row.addWidget(self.dpi_spin)
        dpi_row.addStretch()
        img_layout.addLayout(dpi_row)

        self.white_bg_cb = QCheckBox("Fondo blanco (para publicación)")
        self.white_bg_cb.setChecked(True)
        img_layout.addWidget(self.white_bg_cb)

        self.batch_cb = QCheckBox(
            "Batch (genera 1 imagen por espectro en stage, misma configuración)"
        )
        img_layout.addWidget(self.batch_cb)

        btn_img = QPushButton("Exportar imagen...")
        btn_img.clicked.connect(self._on_export_image)
        img_layout.addWidget(btn_img)

        layout.addWidget(img_group)

        # --- Data export ---
        data_group = QGroupBox("Datos espectrales")
        data_layout = QVBoxLayout(data_group)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Fuente:"))
        self._src_group = QButtonGroup(self)
        rb_stage = QRadioButton("Espectros en stage")
        rb_stage.setChecked(True)
        rb_visible = QRadioButton("Solo visibles en vista")
        self._src_group.addButton(rb_stage, 0)
        self._src_group.addButton(rb_visible, 1)
        src_row.addWidget(rb_stage)
        src_row.addWidget(rb_visible)
        data_layout.addLayout(src_row)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Datos:"))
        self._data_group = QButtonGroup(self)
        for i, label in enumerate(["Absorbancia cruda", "Con suavizado actual", "Derivada 2ª"]):
            rb = QRadioButton(label)
            self._data_group.addButton(rb, i)
            type_row.addWidget(rb)
            if i == 0:
                rb.setChecked(True)
        data_layout.addLayout(type_row)

        self.include_meta_cb = QCheckBox("Incluir metadatos como columnas adicionales")
        self.include_meta_cb.setChecked(True)
        data_layout.addWidget(self.include_meta_cb)

        self.transpose_cb = QCheckBox("Transponer (filas=wavenumbers, columnas=muestras)")
        data_layout.addWidget(self.transpose_cb)

        btn_csv = QPushButton("Exportar CSV...")
        btn_csv.clicked.connect(self._on_export_csv)
        data_layout.addWidget(btn_csv)

        layout.addWidget(data_group)

    def _on_export_image(self):
        fmt_btn = self._fmt_group.checkedButton()
        fmt = fmt_btn.text().lower() if fmt_btn else "png"

        if self.batch_cb.isChecked():
            folder = QFileDialog.getExistingDirectory(self, "Carpeta de destino para batch")
            if not folder:
                return
            self.export_image.emit({
                "format": fmt,
                "dpi": self.dpi_spin.value(),
                "batch": True,
                "folder": folder,
                "white_bg": self.white_bg_cb.isChecked(),
            })
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Guardar imagen",
                f"espectro.{fmt}",
                f"{fmt.upper()} (*.{fmt})"
            )
            if path:
                self.export_image.emit({
                    "format": fmt,
                    "dpi": self.dpi_spin.value(),
                    "batch": False,
                    "path": path,
                    "white_bg": self.white_bg_cb.isChecked(),
                })

    def _on_export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar CSV", "spectra.csv", "CSV (*.csv)"
        )
        if not path:
            return

        self.export_csv.emit({
            "path": path,
            "source": "stage" if self._src_group.checkedId() == 0 else "visible",
            "data_type": ["raw", "smoothed", "derivative"][self._data_group.checkedId()],
            "include_metadata": self.include_meta_cb.isChecked(),
            "transpose": self.transpose_cb.isChecked(),
        })

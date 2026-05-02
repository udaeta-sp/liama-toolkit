"""Matplotlib canvas with dual Y axes and navigation sliders."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QSpinBox,
    QDoubleSpinBox, QLabel, QCheckBox,
)

from .range_slider import RangeSlider
from ..utils.theme import BG_DARK, BG_MID, BORDER, FG_TEXT, FG_DIM


@dataclass
class SpectrumViewConfig:
    """Visual settings for one spectrum in Vista."""
    name: str
    visible: bool = True
    color: str = "#4a9eff"
    linewidth: float = 1.5
    alpha: float = 1.0
    scale: float = 1.0      # multiplier (1.0 = no change)
    offset: float = 0.0     # absorbance units
    spectrum_index: int = 0  # index in the master spectra list


@dataclass
class DerivativeViewConfig:
    """Visual settings for one derivative in Vista."""
    name: str
    parent_index: int  # index in stage list, links to spectrum
    visible: bool = True
    color: str = "#87ceeb"
    linewidth: float = 1.0
    alpha: float = 0.7
    scale: float = 1.0
    offset: float = 0.0


@dataclass
class PlotConfig:
    """Graph appearance settings."""
    plot_bg: str = BG_MID       # plot area background
    fig_bg: str = BG_DARK       # figure background
    text_color: str = FG_TEXT
    tick_color: str = FG_DIM
    spine_color: str = BORDER
    legend_loc: str = "upper left"
    title: str = ""
    x_label: str = "Número de onda (cm⁻¹)"
    y_label: str = "Absorbancia"
    y2_label: str = "Segunda derivada"
    export_bg: str = "#ffffff"   # white bg for export


class CanvasWidget(QWidget):
    """Central visualization area with matplotlib canvas and sliders."""

    mouse_moved = pyqtSignal(float, float)  # wavenumber, absorbance

    DEBOUNCE_MS = 80

    def __init__(self, parent=None):
        super().__init__(parent)
        self._spectrum_lines: dict[str, Line2D] = {}
        self._derivative_lines: dict[str, Line2D] = {}
        self._vlines: list = []
        self._peak_scatter = None
        self.plot_config = PlotConfig()
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._do_redraw)
        self._pending_redraw = False
        self._hover_timer = QTimer()
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._do_hover_check)
        self._last_hover_event = None
        self.transmittance_mode = False
        self._last_plot_data = None
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left: toggle row + canvas + horizontal slider
        left = QVBoxLayout()
        left.setSpacing(2)

        # Toggle row
        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(10)
        toggle_row.setContentsMargins(4, 2, 4, 0)

        self.zero_axes_cb = QCheckBox("Ejes en 0")
        self.zero_axes_cb.setToolTip("Mostrar líneas horizontales y verticales en y=0")
        self.zero_axes_cb.stateChanged.connect(self._schedule_redraw)
        toggle_row.addWidget(self.zero_axes_cb)

        self.show_labels_cb = QCheckBox("Etiquetas")
        self.show_labels_cb.setChecked(True)
        self.show_labels_cb.setToolTip("Mostrar títulos de ejes y leyenda")
        self.show_labels_cb.stateChanged.connect(self._schedule_redraw)
        toggle_row.addWidget(self.show_labels_cb)

        self.transmittance_cb = QCheckBox("Transmitancia")
        self.transmittance_cb.setToolTip("Mostrar espectros en transmitancia (%) en lugar de absorbancia")
        self.transmittance_cb.stateChanged.connect(self._on_display_mode_changed)
        toggle_row.addWidget(self.transmittance_cb)

        toggle_row.addStretch()
        left.addLayout(toggle_row)

        # Matplotlib figure
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.figure.set_facecolor(self.plot_config.fig_bg)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setStyleSheet(f"background-color: {BG_DARK};")

        self.ax = self.figure.add_subplot(111)
        self.ax_right = self.ax.twinx()
        self._style_axes()

        # Mouse tracking
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        left.addWidget(self.canvas, 1)

        # Horizontal range slider row
        # FTIR convention: high wavenumber (4000) on LEFT, low (400) on RIGHT
        # So slider LEFT handle = high wn, RIGHT handle = low wn
        h_row = QHBoxLayout()
        h_row.setSpacing(6)

        # High wn spinbox (left side visually = high wavenumber)
        self.wn_high_spin = QDoubleSpinBox()
        self.wn_high_spin.setRange(200, 5000)
        self.wn_high_spin.setValue(4000)
        self.wn_high_spin.setDecimals(0)
        self.wn_high_spin.setFixedWidth(70)
        self.wn_high_spin.setToolTip("Límite superior (cm⁻¹)")
        self.wn_high_spin.valueChanged.connect(self._on_spin_range_changed)
        h_row.addWidget(self.wn_high_spin)

        self.range_slider = RangeSlider(400, 4000, inverted=True)
        self.range_slider.range_changed.connect(self._on_range_changed)
        h_row.addWidget(self.range_slider, 1)

        # Low wn spinbox (right side visually = low wavenumber)
        self.wn_low_spin = QDoubleSpinBox()
        self.wn_low_spin.setRange(200, 5000)
        self.wn_low_spin.setValue(400)
        self.wn_low_spin.setDecimals(0)
        self.wn_low_spin.setFixedWidth(70)
        self.wn_low_spin.setToolTip("Límite inferior (cm⁻¹)")
        self.wn_low_spin.valueChanged.connect(self._on_spin_range_changed)
        h_row.addWidget(self.wn_low_spin)

        h_row.addWidget(QLabel("Margen:"))
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 100)
        self.margin_spin.setValue(10)
        self.margin_spin.setSuffix(" %")
        self.margin_spin.valueChanged.connect(self._schedule_redraw)
        h_row.addWidget(self.margin_spin)

        left.addLayout(h_row)
        main_layout.addLayout(left, 1)

        # Right: vertical scroll slider
        self.v_slider = QSlider(Qt.Orientation.Vertical)
        self.v_slider.setRange(-500, 500)
        self.v_slider.setValue(0)
        self.v_slider.setInvertedAppearance(True)
        self.v_slider.valueChanged.connect(self._schedule_redraw)
        self.v_slider.setFixedWidth(20)
        main_layout.addWidget(self.v_slider)

    def _style_axes(self):
        pc = self.plot_config
        for ax in (self.ax, self.ax_right):
            ax.set_facecolor(pc.plot_bg)
            ax.tick_params(colors=pc.tick_color)
            for spine in ax.spines.values():
                spine.set_color(pc.spine_color)

        self.ax.set_xlabel(pc.x_label, color=pc.text_color)
        self.ax.set_ylabel(pc.y_label, color=pc.text_color)
        self.ax_right.set_ylabel(pc.y2_label, color=pc.tick_color)

        if pc.title:
            self.ax.set_title(pc.title, color=pc.text_color)

        # FTIR convention is handled by set_xlim(high, low) in _update_axes_limits

    def _on_mouse_move(self, event):
        if event.inaxes == self.ax and event.xdata is not None:
            self.mouse_moved.emit(event.xdata, event.ydata)
            # Throttle tooltip: only check every ~150ms
            self._last_hover_event = event
            if not self._hover_timer.isActive():
                self._hover_timer.start(150)

    def _do_hover_check(self):
        if self._last_hover_event:
            self._update_hover_label(self._last_hover_event)

    def _update_hover_label(self, event):
        """Show tooltip with spectrum name when mouse is near a line."""
        nearest_name = None

        for name, line in self._spectrum_lines.items():
            contains, _ = line.contains(event)
            if contains:
                nearest_name = name
                break

        if nearest_name is None:
            for name, line in self._derivative_lines.items():
                contains, _ = line.contains(event)
                if contains:
                    nearest_name = f"D² {name}"
                    break

        if nearest_name:
            self.canvas.setToolTip(nearest_name)
        else:
            self.canvas.setToolTip("")

    def _on_display_mode_changed(self):
        self.transmittance_mode = self.transmittance_cb.isChecked()
        if self.transmittance_mode:
            self.plot_config.y_label = "Transmitancia (%)"
        else:
            self.plot_config.y_label = "Absorbancia"
        if self._last_plot_data is not None:
            self.update_spectra(*self._last_plot_data)

    def _on_range_changed(self, low: float, high: float):
        # Update spinboxes without triggering loop
        self.wn_high_spin.blockSignals(True)
        self.wn_low_spin.blockSignals(True)
        self.wn_high_spin.setValue(high)
        self.wn_low_spin.setValue(low)
        self.wn_high_spin.blockSignals(False)
        self.wn_low_spin.blockSignals(False)
        self._schedule_redraw()

    def _on_spin_range_changed(self):
        low = self.wn_low_spin.value()
        high = self.wn_high_spin.value()
        if low >= high:
            return
        self.range_slider.set_values(low, high)
        self._schedule_redraw()

    def _schedule_redraw(self):
        """Debounced redraw — waits DEBOUNCE_MS after last change."""
        self._pending_redraw = True
        self._debounce_timer.start(self.DEBOUNCE_MS)

    def _do_redraw(self):
        if self._pending_redraw:
            self._pending_redraw = False
            self._update_axes_limits()
            self.canvas.draw()
            self.canvas.flush_events()

    def update_spectra(
        self,
        wavenumber_data: list[tuple[np.ndarray, np.ndarray]],
        configs: list[SpectrumViewConfig],
        derivative_data: list[tuple[np.ndarray, np.ndarray]],
        deriv_configs: list[DerivativeViewConfig],
        vlines: list[dict] | None = None,
        peaks: list[dict] | None = None,
    ):
        """Full redraw of all spectra and derivatives."""
        self._last_plot_data = (wavenumber_data, configs, derivative_data, deriv_configs, vlines, peaks)
        if self.transmittance_mode:
            wavenumber_data = [(wn, 10.0 ** (2.0 - y)) for wn, y in wavenumber_data]

        self.ax.cla()
        self.ax_right.cla()
        self._style_axes()
        self._spectrum_lines.clear()
        self._derivative_lines.clear()

        # Draw spectra on left axis
        for (wn, y), cfg in zip(wavenumber_data, configs):
            if not cfg.visible or len(wn) == 0:
                continue
            y_draw = y * cfg.scale + cfg.offset
            line, = self.ax.plot(
                wn, y_draw,
                color=cfg.color, linewidth=cfg.linewidth, alpha=cfg.alpha,
                label=cfg.name, picker=5,
            )
            self._spectrum_lines[cfg.name] = line

        # Draw derivatives on right axis
        has_derivatives = False
        for (wn, d2), dcfg in zip(derivative_data, deriv_configs):
            if not dcfg.visible or len(wn) == 0:
                continue
            d2_draw = d2 * dcfg.scale + dcfg.offset
            line, = self.ax_right.plot(
                wn, d2_draw,
                color=dcfg.color, linewidth=dcfg.linewidth, alpha=dcfg.alpha,
                label=f"D² {dcfg.name}",
                picker=5,
            )
            self._derivative_lines[dcfg.name] = line
            has_derivatives = True

        if not has_derivatives:
            self.ax_right.set_yticks([])
            self.ax_right.set_ylabel("")
        else:
            self.ax_right.set_ylabel(
                self.plot_config.y2_label, color=self.plot_config.tick_color
            )

        # Vertical lines
        self._vlines.clear()
        if vlines:
            for vl in vlines:
                line = self.ax.axvline(
                    vl["wavenumber"],
                    color=vl.get("color", "#ffffff"),
                    linewidth=vl.get("linewidth", 1.0),
                    alpha=vl.get("alpha", 0.8),
                )
                self._vlines.append(line)
                if vl.get("show_label", True):
                    self.ax.text(
                        vl["wavenumber"], self.ax.get_ylim()[1] * 0.98,
                        f" {vl['wavenumber']:.0f}",
                        color=vl.get("color", "#ffffff"),
                        fontsize=8, rotation=90,
                        verticalalignment="top",
                    )

        # Peak markers
        if peaks:
            wn_peaks = [p["wavenumber"] for p in peaks]
            abs_peaks = [p["absorbance"] for p in peaks]
            self.ax.scatter(
                wn_peaks, abs_peaks,
                marker="v", s=40, c="#ff6b6b", zorder=10, label="_peaks"
            )

        # Legend
        handles = list(self._spectrum_lines.values()) + list(self._derivative_lines.values())
        if handles:
            leg = self.ax.legend(
                loc=self.plot_config.legend_loc,
                fontsize=8, framealpha=0.85,
                facecolor=self.plot_config.plot_bg,
                edgecolor=self.plot_config.spine_color,
                labelcolor=self.plot_config.text_color,
            )
            if leg:
                leg.set_draggable(True)

        self._update_axes_limits()
        self.canvas.draw()
        self.canvas.flush_events()

    def _update_axes_limits(self):
        """Set axis limits based on slider positions and visible data."""
        wn_low = self.range_slider.low
        wn_high = self.range_slider.high
        margin_pct = self.margin_spin.value() / 100.0
        v_shift = self.v_slider.value() / 100.0  # normalized shift

        # X limits (inverted: high on left, low on right)
        self.ax.set_xlim(wn_high, wn_low)

        # Auto Y limits for left axis from visible lines
        y_mins, y_maxs = [], []
        for line in self._spectrum_lines.values():
            xd, yd = line.get_xdata(), line.get_ydata()
            mask = (xd >= wn_low) & (xd <= wn_high)
            if np.any(mask):
                y_mins.append(np.nanmin(yd[mask]))
                y_maxs.append(np.nanmax(yd[mask]))

        if y_mins:
            ymin, ymax = min(y_mins), max(y_maxs)
            amp = ymax - ymin if ymax > ymin else 0.1
            pad = amp * margin_pct
            ymin -= pad
            ymax += pad
            # Apply vertical shift
            shift = v_shift * amp
            self.ax.set_ylim(ymin + shift, ymax + shift)

        # Auto Y limits for right axis (derivatives)
        d_mins, d_maxs = [], []
        for line in self._derivative_lines.values():
            xd, yd = line.get_xdata(), line.get_ydata()
            mask = (xd >= wn_low) & (xd <= wn_high)
            if np.any(mask):
                d_mins.append(np.nanmin(yd[mask]))
                d_maxs.append(np.nanmax(yd[mask]))

        if d_mins:
            dmin, dmax = min(d_mins), max(d_maxs)
            damp = dmax - dmin if dmax > dmin else 0.001
            dpad = damp * margin_pct
            self.ax_right.set_ylim(dmin - dpad, dmax + dpad)

        # Zero-axes lines
        # Remove previous zero-axes if any
        for line in getattr(self, '_zero_lines', []):
            try:
                line.remove()
            except ValueError:
                pass
        self._zero_lines = []
        if self.zero_axes_cb.isChecked():
            zl = self.ax.axhline(0, color=self.plot_config.tick_color, linewidth=0.6,
                                  linestyle="-", alpha=0.5, zorder=0)
            self._zero_lines.append(zl)

        # Label visibility
        show = self.show_labels_cb.isChecked()
        self.ax.xaxis.label.set_visible(show)
        self.ax.yaxis.label.set_visible(show)
        self.ax_right.yaxis.label.set_visible(show)
        leg = self.ax.get_legend()
        if leg:
            leg.set_visible(show)
        if self.plot_config.title:
            self.ax.title.set_visible(show)

    def set_wavenumber_range(self, wn_min: float, wn_max: float):
        """Set the total range of the horizontal slider."""
        self.range_slider.set_range(wn_min, wn_max)
        self.range_slider.set_values(wn_min, wn_max)
        self.wn_high_spin.blockSignals(True)
        self.wn_low_spin.blockSignals(True)
        self.wn_high_spin.setValue(wn_max)
        self.wn_low_spin.setValue(wn_min)
        self.wn_high_spin.blockSignals(False)
        self.wn_low_spin.blockSignals(False)

    def center_on_wavenumber(self, wn: float, window_width: float = 200):
        """Center the view on a specific wavenumber."""
        half = window_width / 2
        self.range_slider.set_values(wn - half, wn + half)
        self._schedule_redraw()

    def export_figure(self, path: str, dpi: int = 300, use_export_bg: bool = True):
        """Save figure with optional white background for publication."""
        if use_export_bg:
            # Temporarily switch to export colors
            old_fig_bg = self.figure.get_facecolor()
            old_ax_bg = self.ax.get_facecolor()
            export_bg = self.plot_config.export_bg

            self.figure.set_facecolor(export_bg)
            self.ax.set_facecolor(export_bg)
            self.ax.set_xlabel(self.plot_config.x_label, color="black")
            self.ax.set_ylabel(self.plot_config.y_label, color="black")
            self.ax.tick_params(colors="black")
            for spine in self.ax.spines.values():
                spine.set_color("black")
            if self.plot_config.title:
                self.ax.set_title(self.plot_config.title, color="black")

            # Right axis
            self.ax_right.tick_params(colors="black")
            for spine in self.ax_right.spines.values():
                spine.set_color("black")
            if self.ax_right.get_ylabel():
                self.ax_right.set_ylabel(self.ax_right.get_ylabel(), color="black")

            # Legend
            leg = self.ax.get_legend()
            if leg:
                leg.get_frame().set_facecolor("white")
                leg.get_frame().set_edgecolor("black")
                for text in leg.get_texts():
                    text.set_color("black")

            self.figure.savefig(
                path, dpi=dpi, facecolor=export_bg, bbox_inches="tight"
            )

            # Restore dark theme
            self.figure.set_facecolor(old_fig_bg)
            self.ax.set_facecolor(old_ax_bg)
            self._style_axes()
            self.canvas.draw_idle()
        else:
            self.figure.savefig(
                path, dpi=dpi,
                facecolor=self.figure.get_facecolor(),
                bbox_inches="tight",
            )

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

def plot_spectrum_overlay(
    x, y_plot, d2, x_units, y_units, xmin, xmax,
    user_bands=(), user_band_colors=(), vlines=(),
    height_segments=(), fig_size=(13, 8), dpi=600
):
    """
    Exportación estilo 'overlay': un solo eje con twin y (ax2) para 2ª derivada si d2 no es None.
    Bandas, líneas, segmentos de altura y picos se dibujan sobre ax1.
    """
    fig, ax1 = plt.subplots(figsize=fig_size, dpi=dpi)

    # Derivada en twin-y si está disponible
    ax2 = None
    if d2 is not None:
        ax2 = ax1.twinx()
        ax2.plot(x, d2, lw=1, color='red', label='2ª derivada')
        ax2.set_ylabel('2ª derivada', color='black', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='black', labelsize=8)
        # límites de la derivada
        d2min, d2max = float(np.min(d2)), float(np.max(d2))
        ax2.set_ylim(d2min * 1.1, d2max * 1.8)

    # Espectro
    ax1.plot(x, y_plot, lw=1, color='blue', label='Espectro FTIR-ATR')
    ax1.set_xlabel(x_units, fontsize=10)
    ax1.set_ylabel(y_units, fontsize=10)
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=8)
    ax1.tick_params(axis='x', labelsize=8)
    ax1.set_xlim(xmin, xmax)
    y_min, y_max = float(np.min(y_plot)), float(np.max(y_plot))
    ax1.set_ylim(y_min * 0.95, y_max * 1.05)

    # Asegurar que el trazo azul queda 'encima' del fondo del twin
    fig.canvas.draw()
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)

    # Bandas y líneas verticales en ax1
    for idx, (c, w) in enumerate(user_bands):
        color = user_band_colors[idx] if idx < len(user_band_colors) else 'gray'
        ax1.axvspan(c - w/2, c + w/2, color=color, alpha=0.3)
    for xv in vlines:
        ax1.axvline(x=xv, color='green', linestyle='--', linewidth=1)

    # Segmentos de altura
    if height_segments:
        cap_w = max(1.0, abs(xmin - xmax) * 0.01)
        for xpk, ybl, h, color in height_segments:
            if not (np.isfinite(xpk) and np.isfinite(ybl) and np.isfinite(h)):
                continue
            if (xpk <= xmin) and (xpk >= xmax):
                y0, y1 = ybl, ybl + h
                ax1.vlines(xpk, y0, y1, colors=color, linewidth=2, alpha=0.95, zorder=4)
                ax1.hlines([y0, y1], xpk - cap_w, xpk + cap_w, colors=color, linewidth=1.0, alpha=0.95, zorder=4)
                ax1.annotate(f"{h:.3e}", xy=(xpk, y1), xytext=(0, 2),
                             textcoords="offset points", ha="center", va="bottom",
                             fontsize=9, color=color)

    plt.tight_layout()
    return fig, (ax1, ax2)

def plot_spectrum_only(
    x, y_plot, x_units, y_units, xmin, xmax,
    user_bands=(), user_band_colors=(), vlines=(),
    height_segments=(), fig_size=(13, 8), dpi=600
):
    """
    Exportación sin derivada: un solo panel con todos los overlays.
    """
    fig, ax1 = plt.subplots(figsize=fig_size, dpi=dpi)
    ax1.plot(x, y_plot, lw=1, color='blue')
    ax1.set_xlabel(x_units, fontsize=10)
    ax1.set_ylabel(y_units, fontsize=10)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(float(np.min(y_plot)) * 0.95, float(np.max(y_plot)) * 1.05)
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=8)
    ax1.tick_params(axis='x', labelsize=8)

    for idx, (c, w) in enumerate(user_bands):
        color = user_band_colors[idx] if idx < len(user_band_colors) else 'gray'
        ax1.axvspan(c - w/2, c + w/2, color=color, alpha=0.3)
    for xv in vlines:
        ax1.axvline(x=xv, color='green', linestyle='--', linewidth=1)

    if height_segments:
        cap_w = max(1.0, abs(xmin - xmax) * 0.01)
        for xpk, ybl, h, color in height_segments:
            if not (np.isfinite(xpk) and np.isfinite(ybl) and np.isfinite(h)):
                continue
            if (xpk <= xmin) and (xpk >= xmax):
                y0, y1 = ybl, ybl + h
                ax1.vlines(xpk, y0, y1, colors=color, linewidth=2, alpha=0.95, zorder=4)
                ax1.hlines([y0, y1], xpk - cap_w, xpk + cap_w, colors=color, linewidth=1.0, alpha=0.95, zorder=4)
                ax1.annotate(f"{h:.3e}", xy=(xpk, y1), xytext=(0, 2),
                             textcoords="offset points", ha="center", va="bottom",
                             fontsize=9, color=color)

    plt.tight_layout()
    return fig, ax1

def plot_spectrum_with_derivative(
    x: np.ndarray,
    y_plot: np.ndarray,
    d2: np.ndarray,
    x_units: str,
    y_units: str,
    xmin: float,
    xmax: float,
    user_bands: Sequence[Tuple[float, float]] = (),
    user_band_colors: Sequence[str] = (),
    vlines: Sequence[float] = (),
    height_segments: Sequence[Tuple[float, float, float, str]] = (),
    fig_size=(6,5),
    dpi=130
):
    """
    height_segments: iterable of (xpk, y_baseline_at_peak, height, color)
    All arrays must be already masked/cut to [xmin,xmax] if needed.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size, sharex=True, dpi=dpi)

    ax1.plot(x, y_plot, lw=1)
    ax1.set_ylabel(y_units, fontsize=10)
    ax1.set_title('Espectro FTIR-ATR', fontsize=11)
    ax1.tick_params(labelsize=8)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(np.min(y_plot)*0.95, np.max(y_plot)*1.05)

    # user-defined shaded bands
    for idx, (c, w) in enumerate(user_bands):
        color = user_band_colors[idx] if idx < len(user_band_colors) else "gray"
        ax1.axvspan(c - w/2, c + w/2, color=color, alpha=0.3)

    # optional vertical lines
    for xv in vlines:
        ax1.axvline(x=xv, color="green", linestyle="--", linewidth=1)

    # height segments
    if height_segments:
        cap_w = max(1.0, abs(xmin - xmax) * 0.01)
        for xpk, ybl, h, color in height_segments:
            if not (np.isfinite(xpk) and np.isfinite(ybl) and np.isfinite(h)):
                continue
            if (xpk <= xmin) and (xpk >= xmax):
                y0, y1 = ybl, ybl + h
                ax1.vlines(xpk, y0, y1, colors=color, linewidth=2, alpha=0.95, zorder=4)
                ax1.hlines([y0, y1], xpk - cap_w, xpk + cap_w, colors=color, linewidth=1.0, alpha=0.95, zorder=4)
                ax1.annotate(f"{h:.3e}", xy=(xpk, y1), xytext=(0, 2),
                             textcoords="offset points", ha="center", va="bottom",
                             fontsize=9, color=color)

    ax2.plot(x, d2, color='red', lw=1)
    ax2.set_xlabel(x_units, fontsize=10)
    ax2.set_ylabel('2ª derivada', fontsize=10)
    ax2.set_title('Segunda derivada', fontsize=11)
    ax2.tick_params(labelsize=8)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(np.min(d2)*1.1, np.max(d2)*1.1)

    plt.tight_layout()
    return fig, (ax1, ax2)

"""Dark theme for the application — QSS and matplotlib rcParams."""

from __future__ import annotations

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
BG_DARK = "#1e1e1e"
BG_MID = "#2d2d2d"
BG_LIGHT = "#3c3c3c"
FG_TEXT = "#e0e0e0"
FG_DIM = "#a0a0a0"
ACCENT = "#4a9eff"
BORDER = "#555555"
ERROR_BG = "#5c2020"
OK_BG = "#1e3a1e"

# ---------------------------------------------------------------------------
# Qt Style Sheet
# ---------------------------------------------------------------------------
QSS = f"""
QMainWindow, QDialog {{
    background-color: {BG_DARK};
    color: {FG_TEXT};
}}

QWidget {{
    background-color: {BG_DARK};
    color: {FG_TEXT};
    font-family: "Segoe UI", "DejaVu Sans", sans-serif;
    font-size: 13px;
}}

QLabel {{
    background: transparent;
    color: {FG_TEXT};
}}

QPushButton {{
    background-color: {BG_LIGHT};
    color: {FG_TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 14px;
    min-height: 22px;
}}
QPushButton:hover {{
    background-color: {ACCENT};
    color: #ffffff;
}}
QPushButton:pressed {{
    background-color: #3a7fd5;
}}

QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {BG_MID};
    color: {FG_TEXT};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 3px 6px;
    font-family: "Consolas", "DejaVu Sans Mono", monospace;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}

QComboBox {{
    background-color: {BG_MID};
    color: {FG_TEXT};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 3px 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_MID};
    color: {FG_TEXT};
    selection-background-color: {ACCENT};
}}

QCheckBox {{
    color: {FG_TEXT};
    spacing: 6px;
    background: transparent;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background-color: {BG_MID};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

QRadioButton {{
    color: {FG_TEXT};
    spacing: 6px;
    background: transparent;
}}

QTabWidget::pane {{
    border: 1px solid {BORDER};
    background-color: {BG_DARK};
}}
QTabBar::tab {{
    background-color: {BG_MID};
    color: {FG_DIM};
    border: 1px solid {BORDER};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 6px 16px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {BG_DARK};
    color: {FG_TEXT};
    border-bottom: 2px solid {ACCENT};
}}

QScrollArea {{
    border: none;
    background-color: {BG_DARK};
}}

QScrollBar:vertical {{
    background-color: {BG_DARK};
    width: 12px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background-color: {BG_LIGHT};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {BG_DARK};
    height: 12px;
}}
QScrollBar::handle:horizontal {{
    background-color: {BG_LIGHT};
    border-radius: 4px;
    min-width: 30px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QSplitter::handle {{
    background-color: {BORDER};
}}
QSplitter::handle:horizontal {{
    width: 3px;
}}
QSplitter::handle:vertical {{
    height: 3px;
}}

QTableWidget {{
    background-color: {BG_MID};
    color: {FG_TEXT};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    font-family: "Consolas", "DejaVu Sans Mono", monospace;
}}
QTableWidget::item:selected {{
    background-color: {ACCENT};
}}
QHeaderView::section {{
    background-color: {BG_LIGHT};
    color: {FG_TEXT};
    border: 1px solid {BORDER};
    padding: 4px;
}}

QStatusBar {{
    background-color: {BG_MID};
    color: {FG_DIM};
    font-family: "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 12px;
}}

QToolBar {{
    background-color: {BG_MID};
    border-bottom: 1px solid {BORDER};
    spacing: 6px;
    padding: 4px;
}}

QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 16px;
    color: {FG_TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    padding: 0 6px;
}}

QSlider::groove:horizontal {{
    background: {BG_LIGHT};
    height: 6px;
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}}
QSlider::groove:vertical {{
    background: {BG_LIGHT};
    width: 6px;
    border-radius: 3px;
}}
QSlider::handle:vertical {{
    background: {ACCENT};
    width: 14px;
    height: 14px;
    margin: 0 -4px;
    border-radius: 7px;
}}
"""


# ---------------------------------------------------------------------------
# Light theme constants
# ---------------------------------------------------------------------------
LT_BG = "#f5f5f5"
LT_BG_MID = "#ffffff"
LT_BG_LIGHT = "#e8e8e8"
LT_FG_TEXT = "#1e1e1e"
LT_FG_DIM = "#555555"
LT_ACCENT = "#2979ff"
LT_BORDER = "#cccccc"

# ---------------------------------------------------------------------------
# Light QSS
# ---------------------------------------------------------------------------
QSS_LIGHT = f"""
QMainWindow, QDialog {{
    background-color: {LT_BG};
    color: {LT_FG_TEXT};
}}

QWidget {{
    background-color: {LT_BG};
    color: {LT_FG_TEXT};
    font-family: "Segoe UI", "DejaVu Sans", sans-serif;
    font-size: 13px;
}}

QLabel {{
    background: transparent;
    color: {LT_FG_TEXT};
}}

QPushButton {{
    background-color: {LT_BG_LIGHT};
    color: {LT_FG_TEXT};
    border: 1px solid {LT_BORDER};
    border-radius: 4px;
    padding: 5px 14px;
    min-height: 22px;
}}
QPushButton:hover {{
    background-color: {LT_ACCENT};
    color: #ffffff;
}}
QPushButton:pressed {{
    background-color: #1565c0;
}}

QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {LT_BG_MID};
    color: {LT_FG_TEXT};
    border: 1px solid {LT_BORDER};
    border-radius: 3px;
    padding: 3px 6px;
    font-family: "Consolas", "DejaVu Sans Mono", monospace;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {LT_ACCENT};
}}

QComboBox {{
    background-color: {LT_BG_MID};
    color: {LT_FG_TEXT};
    border: 1px solid {LT_BORDER};
    border-radius: 3px;
    padding: 3px 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {LT_BG_MID};
    color: {LT_FG_TEXT};
    selection-background-color: {LT_ACCENT};
}}

QCheckBox {{
    color: {LT_FG_TEXT};
    spacing: 6px;
    background: transparent;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {LT_BORDER};
    border-radius: 3px;
    background-color: {LT_BG_MID};
}}
QCheckBox::indicator:checked {{
    background-color: {LT_ACCENT};
    border-color: {LT_ACCENT};
}}

QRadioButton {{
    color: {LT_FG_TEXT};
    spacing: 6px;
    background: transparent;
}}

QTabWidget::pane {{
    border: 1px solid {LT_BORDER};
    background-color: {LT_BG};
}}
QTabBar::tab {{
    background-color: {LT_BG_LIGHT};
    color: {LT_FG_DIM};
    border: 1px solid {LT_BORDER};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 6px 16px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {LT_BG};
    color: {LT_FG_TEXT};
    border-bottom: 2px solid {LT_ACCENT};
}}

QScrollArea {{
    border: none;
    background-color: {LT_BG};
}}

QScrollBar:vertical {{
    background-color: {LT_BG};
    width: 12px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background-color: {LT_BORDER};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {LT_BG};
    height: 12px;
}}
QScrollBar::handle:horizontal {{
    background-color: {LT_BORDER};
    border-radius: 4px;
    min-width: 30px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QSplitter::handle {{
    background-color: {LT_BORDER};
}}
QSplitter::handle:horizontal {{
    width: 3px;
}}
QSplitter::handle:vertical {{
    height: 3px;
}}

QTableWidget {{
    background-color: {LT_BG_MID};
    color: {LT_FG_TEXT};
    gridline-color: {LT_BORDER};
    border: 1px solid {LT_BORDER};
    font-family: "Consolas", "DejaVu Sans Mono", monospace;
}}
QTableWidget::item:selected {{
    background-color: {LT_ACCENT};
}}
QHeaderView::section {{
    background-color: {LT_BG_LIGHT};
    color: {LT_FG_TEXT};
    border: 1px solid {LT_BORDER};
    padding: 4px;
}}

QStatusBar {{
    background-color: {LT_BG_LIGHT};
    color: {LT_FG_DIM};
    font-family: "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 12px;
}}

QToolBar {{
    background-color: {LT_BG_LIGHT};
    border-bottom: 1px solid {LT_BORDER};
    spacing: 6px;
    padding: 4px;
}}

QGroupBox {{
    border: 1px solid {LT_BORDER};
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 16px;
    color: {LT_FG_TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    padding: 0 6px;
}}

QSlider::groove:horizontal {{
    background: {LT_BORDER};
    height: 6px;
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {LT_ACCENT};
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}}
QSlider::groove:vertical {{
    background: {LT_BORDER};
    width: 6px;
    border-radius: 3px;
}}
QSlider::handle:vertical {{
    background: {LT_ACCENT};
    width: 14px;
    height: 14px;
    margin: 0 -4px;
    border-radius: 7px;
}}
"""


def apply_mpl_dark_theme():
    """Configure matplotlib rcParams for dark theme."""
    mpl.rcParams.update({
        "figure.facecolor": BG_DARK,
        "axes.facecolor": BG_MID,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": FG_TEXT,
        "xtick.color": FG_DIM,
        "ytick.color": FG_DIM,
        "text.color": FG_TEXT,
        "legend.facecolor": BG_LIGHT,
        "legend.edgecolor": BORDER,
        "legend.labelcolor": FG_TEXT,
        "grid.color": BG_LIGHT,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
        "font.size": 10,
    })


def apply_mpl_light_theme():
    """Configure matplotlib rcParams for light theme."""
    mpl.rcParams.update({
        "figure.facecolor": LT_BG,
        "axes.facecolor": LT_BG_MID,
        "axes.edgecolor": LT_BORDER,
        "axes.labelcolor": LT_FG_TEXT,
        "xtick.color": LT_FG_DIM,
        "ytick.color": LT_FG_DIM,
        "text.color": LT_FG_TEXT,
        "legend.facecolor": LT_BG_LIGHT,
        "legend.edgecolor": LT_BORDER,
        "legend.labelcolor": LT_FG_TEXT,
        "grid.color": LT_BORDER,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
        "font.size": 10,
    })

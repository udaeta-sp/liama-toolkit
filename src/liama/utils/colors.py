"""Color cycle management for spectrum visualization."""

from __future__ import annotations

# Default color cycle — high contrast on dark background
SPECTRUM_COLORS = [
    "#4a9eff",  # blue
    "#ff6b6b",  # red
    "#51cf66",  # green
    "#ffd43b",  # yellow
    "#cc5de8",  # purple
    "#ff922b",  # orange
    "#20c997",  # teal
    "#f06595",  # pink
    "#a9e34b",  # lime
    "#74c0fc",  # light blue
    "#e599f7",  # lavender
    "#ffa94d",  # light orange
    "#63e6be",  # mint
    "#ff8787",  # salmon
    "#b197fc",  # violet
    "#ffe066",  # gold
]

DERIVATIVE_COLORS = [
    "#87ceeb",  # light sky blue
    "#ffb3b3",  # light red
    "#90ee90",  # light green
    "#fff68f",  # light yellow
    "#dda0dd",  # plum
    "#ffcc80",  # light orange
    "#80cbc4",  # light teal
    "#f8bbd0",  # light pink
]


def get_spectrum_color(index: int) -> str:
    """Get color for spectrum at given index (cycles through palette)."""
    return SPECTRUM_COLORS[index % len(SPECTRUM_COLORS)]


def get_derivative_color(index: int) -> str:
    """Get color for derivative at given index."""
    return DERIVATIVE_COLORS[index % len(DERIVATIVE_COLORS)]

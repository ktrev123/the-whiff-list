"""Shared blue → white → red styling for EDA and app visualizations."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.patches import Rectangle

from src.statcast_zones import (
    DEFAULT_SZ_BOT,
    DEFAULT_SZ_TOP,
    zone_rectangles,
)

VIBRANT_BLUE = "#2563eb"
NEUTRAL_WHITE = "#ffffff"
VIBRANT_RED = "#dc2626"

PROJECT_COLORSCALE = [
    [0.0, VIBRANT_BLUE],
    [0.5, NEUTRAL_WHITE],
    [1.0, VIBRANT_RED],
]

MATPLOTLIB_CMAP = LinearSegmentedColormap.from_list(
    "project_bwr",
    [VIBRANT_BLUE, NEUTRAL_WHITE, VIBRANT_RED],
)


def clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def normalize_to_unit(value: float, center: float, span: float) -> float:
    if span <= 0:
        return 0.5
    return clamp((value - center) / span + 0.5, 0.0, 1.0)


def value_to_hex(value: float, center: float, span: float) -> str:
    rgba = MATPLOTLIB_CMAP(normalize_to_unit(value, center, span))
    return to_hex(rgba, keep_alpha=False)


def value_to_rgba(value: float, center: float, span: float, alpha: float = 1.0) -> str:
    r, g, b, _ = MATPLOTLIB_CMAP(normalize_to_unit(value, center, span), alpha=alpha)
    return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{alpha:.3f})"


def plotly_coloraxis(
    values: Iterable[float],
    *,
    center: float,
    span: float,
    colorbar_title: str,
    ticksuffix: str = "",
) -> dict:
    """Plotly marker coloraxis kwargs for the project gradient."""
    values = np.asarray(list(values), dtype=float)
    return dict(
        color=values,
        colorscale=PROJECT_COLORSCALE,
        cmin=center - span,
        cmax=center + span,
        colorbar=dict(title=colorbar_title, ticksuffix=ticksuffix),
        showscale=True,
    )


def _zone_centroid(rects, zone_id: int) -> tuple[float, float]:
    parts = [rect for rect in rects if rect.zone == zone_id]
    areas = [(rect.x1 - rect.x0) * (rect.z1 - rect.z0) for rect in parts]
    total = sum(areas)
    cx = sum((rect.x0 + rect.x1) / 2 * area for rect, area in zip(parts, areas)) / total
    cz = sum((rect.z0 + rect.z1) / 2 * area for rect, area in zip(parts, areas)) / total
    return cx, cz


def plot_statcast_zone_heatmap(
    zone_values: dict[int, float],
    *,
    title: str,
    center: float,
    span: float,
    value_fmt: str = "{:.3f}",
    sz_top: float = DEFAULT_SZ_TOP,
    sz_bot: float = DEFAULT_SZ_BOT,
    x_lim: tuple[float, float] = (-1.15, 1.15),
    z_lim: tuple[float, float] = (1.05, 3.95),
):
    """Matplotlib 13-zone heatmap with numeric labels (MLB Statcast grid)."""
    rects = zone_rectangles(sz_top=sz_top, sz_bot=sz_bot)
    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor="white")
    ax.set_facecolor("#f8fafc")

    for rect in rects:
        value = zone_values.get(rect.zone, center)
        ax.add_patch(
            Rectangle(
                (rect.x0, rect.z0),
                rect.x1 - rect.x0,
                rect.z1 - rect.z0,
                facecolor=value_to_hex(value, center, span),
                edgecolor="#1a202c",
                linewidth=0.9,
                zorder=2,
            )
        )

    labeled: set[int] = set()
    for rect in rects:
        if rect.zone in labeled:
            continue
        labeled.add(rect.zone)
        cx, cz = _zone_centroid(rects, rect.zone)
        value = zone_values.get(rect.zone, center)
        ax.text(
            cx,
            cz,
            value_fmt.format(value),
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#0f172a",
            zorder=3,
        )

    norm = plt.Normalize(vmin=center - span, vmax=center + span)
    sm = plt.cm.ScalarMappable(cmap=MATPLOTLIB_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(title.split("—")[-1].strip() if "—" in title else "Value", rotation=270, labelpad=18)

    ax.set_xlim(*x_lim)
    ax.set_ylim(*z_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Horizontal Distance (Catcher Perspective) [ft]")
    ax.set_ylabel("Vertical Distance (Above Home Plate) [ft]")
    ax.set_title(title, fontsize=14, pad=12)
    ax.grid(False)
    plt.tight_layout()
    return fig, ax

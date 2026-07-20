"""Tests for shared plot styling."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from src.plot_style import (
    PROJECT_COLORSCALE,
    normalize_to_unit,
    plot_statcast_zone_heatmap,
    value_to_hex,
)


def test_project_colorscale_is_blue_white_red():
    assert PROJECT_COLORSCALE[0][1] == "#2563eb"
    assert PROJECT_COLORSCALE[1][1] == "#ffffff"
    assert PROJECT_COLORSCALE[2][1] == "#dc2626"


def test_value_to_hex_centers_on_white():
    assert value_to_hex(0.320, 0.320, 0.12).lower().startswith("#fff")


def test_statcast_zone_heatmap_renders():
    zone_values = {zone: 0.320 + (zone - 5) * 0.01 for zone in list(range(1, 10)) + [11, 12, 13, 14]}
    fig, ax = plot_statcast_zone_heatmap(
        zone_values,
        title="MLB xwOBAcon By Zone — test",
        center=0.320,
        span=0.12,
    )
    assert len(ax.patches) == 17
    assert normalize_to_unit(0.320, 0.320, 0.12) == 0.5

"""Statcast Gameday zones 1–14 geometry and assignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Rule-book plate half-width (ft): 8.5 in from center to edge.
PLATE_HALF_WIDTH_FT = 17.0 / 24.0

DEFAULT_SZ_TOP = 3.5
DEFAULT_SZ_BOT = 1.5

# Chart padding beyond the rule-book zone for outside regions (ft).
OUTSIDE_X_PAD_FT = 0.55
OUTSIDE_Z_PAD_FT = 0.55


@dataclass(frozen=True)
class ZoneRect:
    zone: int
    x0: float
    x1: float
    z0: float
    z1: float


def assign_statcast_zone(
    plate_x: float | np.ndarray,
    plate_z: float | np.ndarray,
    sz_top: float | np.ndarray,
    sz_bot: float | np.ndarray,
) -> int | np.ndarray:
    """
    Map plate crossing to Gameday zones 1–14.

    Inside 1–9: 3×3 grid within the rule-book strike zone (high→low rows,
    left→right columns, catcher's view).

    Outside 11–14: four shadow quadrants around the heart, split at plate center
    (x = 0) and strike-zone midpoint (z = (top + bot) / 2):
      11 top-left, 12 top-right, 13 bottom-left, 14 bottom-right.
    """
    scalar = np.isscalar(plate_x)
    px = np.asarray(plate_x, dtype=float)
    pz = np.asarray(plate_z, dtype=float)
    top = np.asarray(sz_top, dtype=float)
    bot = np.asarray(sz_bot, dtype=float)

    hw = PLATE_HALF_WIDTH_FT
    h = top - bot
    third_x = hw / 3.0
    z_low = bot + h / 3.0
    z_high = bot + 2.0 * h / 3.0
    z_mid = (top + bot) / 2.0

    inside_sz = (px >= -hw) & (px <= hw) & (pz >= bot) & (pz <= top)
    zone = np.full(px.shape, np.nan, dtype=float)

    if inside_sz.any():
        inside_mask = inside_sz
        col = np.select(
            [px[inside_mask] < -third_x, px[inside_mask] > third_x],
            [0, 2],
            default=1,
        )
        row = np.select(
            [pz[inside_mask] < z_low[inside_mask], pz[inside_mask] > z_high[inside_mask]],
            [2, 0],
            default=1,
        )
        zone[inside_mask] = row * 3 + col + 1

    outside = ~inside_sz
    if outside.any():
        px_o = px[outside]
        pz_o = pz[outside]
        top_o = np.broadcast_to(top, px.shape)[outside]
        bot_o = np.broadcast_to(bot, px.shape)[outside]
        z_mid_o = (top_o + bot_o) / 2.0
        left = px_o < 0.0
        upper = pz_o > z_mid_o
        zone[outside] = np.select(
            [left & upper, ~left & upper, left & ~upper],
            [11, 12, 13],
            default=14,
        )

    if scalar:
        return int(zone.item())
    return zone.astype(int)


def zone_rectangles(
    *,
    sz_top: float = DEFAULT_SZ_TOP,
    sz_bot: float = DEFAULT_SZ_BOT,
    x_pad: float = OUTSIDE_X_PAD_FT,
    z_pad: float = OUTSIDE_Z_PAD_FT,
    x_min: float | None = None,
    x_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
) -> list[ZoneRect]:
    """Axis-aligned rectangles for zones 1–14 on the strike-zone chart."""
    hw = PLATE_HALF_WIDTH_FT
    h = sz_top - sz_bot
    third_x = hw / 3.0
    z_low = sz_bot + h / 3.0
    z_high = sz_bot + 2.0 * h / 3.0

    z_floor = sz_bot - z_pad if z_min is None else z_min
    z_ceil = sz_top + z_pad if z_max is None else z_max
    x_left = -hw - x_pad
    x_right = hw + x_pad
    x_outer_left = x_min if x_min is not None else x_left
    x_outer_right = x_max if x_max is not None else x_right

    x_splits = [-hw, -third_x, third_x, hw]

    rects: list[ZoneRect] = []

    # Inside 3×3: rows high → mid → low map to zones 1–3, 4–6, 7–9.
    z_rows = [(z_high, sz_top), (z_low, z_high), (sz_bot, z_low)]
    for row_idx, (z_bottom, z_top_edge) in enumerate(z_rows):
        for col_idx in range(3):
            zone_id = row_idx * 3 + col_idx + 1
            rects.append(
                ZoneRect(
                    zone=zone_id,
                    x0=x_splits[col_idx],
                    x1=x_splits[col_idx + 1],
                    z0=z_bottom,
                    z1=z_top_edge,
                )
            )

    rects.extend(_outer_quadrant_zone_rects(
        hw=hw,
        sz_top=sz_top,
        sz_bot=sz_bot,
        x_outer_left=x_outer_left,
        x_outer_right=x_outer_right,
        z_floor=z_floor,
        z_ceil=z_ceil,
    ))
    return rects


def _outer_quadrant_zone_rects(
    *,
    hw: float,
    sz_top: float,
    sz_bot: float,
    x_outer_left: float,
    x_outer_right: float,
    z_floor: float,
    z_ceil: float,
) -> list[ZoneRect]:
    """
    Shadow zones 11–14 as four quadrants around the inner 3×3 (MLB Statcast grid).

    Split at plate center (x = 0) and strike-zone midpoint (z = (top + bot) / 2).
    Each quadrant is two axis-aligned tiles forming an L around the heart.
    """
    z_mid = (sz_top + sz_bot) / 2.0
    return [
        # 11 — top-left
        ZoneRect(zone=11, x0=x_outer_left, x1=-hw, z0=z_mid, z1=z_ceil),
        ZoneRect(zone=11, x0=-hw, x1=0.0, z0=sz_top, z1=z_ceil),
        # 12 — top-right
        ZoneRect(zone=12, x0=hw, x1=x_outer_right, z0=z_mid, z1=z_ceil),
        ZoneRect(zone=12, x0=0.0, x1=hw, z0=sz_top, z1=z_ceil),
        # 13 — bottom-left
        ZoneRect(zone=13, x0=x_outer_left, x1=-hw, z0=z_floor, z1=z_mid),
        ZoneRect(zone=13, x0=-hw, x1=0.0, z0=z_floor, z1=sz_bot),
        # 14 — bottom-right
        ZoneRect(zone=14, x0=hw, x1=x_outer_right, z0=z_floor, z1=z_mid),
        ZoneRect(zone=14, x0=0.0, x1=hw, z0=z_floor, z1=sz_bot),
    ]


def zone_values_from_frame(
    frame: pd.DataFrame,
    *,
    batter_col: str = "batter",
    value_col: str = "xwobacon",
    sz_top: float = DEFAULT_SZ_TOP,
    sz_bot: float = DEFAULT_SZ_BOT,
) -> pd.Series:
    """Mean contact value by Statcast zone for one batter sample."""
    required = {batter_col, value_col, "plate_x", "plate_z"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Frame missing columns: {sorted(missing)}")

    bip = frame.loc[frame[value_col].notna()].copy()
    if bip.empty:
        return pd.Series(dtype=float)

    top_arr = np.full(len(bip), sz_top)
    bot_arr = np.full(len(bip), sz_bot)
    bip["zone"] = assign_statcast_zone(
        bip["plate_x"].to_numpy(),
        bip["plate_z"].to_numpy(),
        top_arr,
        bot_arr,
    )
    grouped = bip.groupby("zone", as_index=True)[value_col].mean()
    return grouped.sort_index()

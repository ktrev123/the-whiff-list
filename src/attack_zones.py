"""FanGraphs-style attack zones (Heart / Shadow / Chase / Waste)."""

from __future__ import annotations

import numpy as np
import pandas as pd

ZONE_ORDER = ["Heart", "Shadow", "Chase", "Waste"]
VERTICAL_ORDER = ["Low", "Middle", "High"]
HORIZONTAL_ORDER = ["Inside", "Middle", "Outside"]

# Light fills for strike-zone chart backgrounds (catcher's view).
ZONE_FILL = {
    "Heart": "rgba(32, 144, 141, 0.32)",
    "Shadow": "rgba(212, 169, 55, 0.26)",
    "Chase": "rgba(230, 57, 70, 0.20)",
    "Waste": "rgba(100, 116, 139, 0.14)",
}

ZONE_BORDER = {
    "Heart": "rgba(32, 144, 141, 0.55)",
    "Shadow": "rgba(212, 169, 55, 0.45)",
    "Chase": "rgba(230, 57, 70, 0.40)",
    "Waste": "rgba(100, 116, 139, 0.30)",
}

ZONE_CODE = {"Waste": 0, "Chase": 1, "Shadow": 2, "Heart": 3}


def boundary_signed_inches(plate_x: float, plate_z: float, sz_top: float, sz_bot: float) -> float:
    """Inches to nearest zone boundary; positive = inside, negative = outside."""
    half_w_ft = 17.0 / 24.0
    inside_x = half_w_ft - abs(plate_x)
    inside_z_top = sz_top - plate_z
    inside_z_bot = plate_z - sz_bot
    inside_min_ft = min(inside_x, inside_z_top, inside_z_bot)
    return inside_min_ft * 12.0


def assign_attack_zone(
    plate_x: float,
    plate_z: float,
    sz_top: float,
    sz_bot: float,
) -> str:
    signed_in = boundary_signed_inches(plate_x, plate_z, sz_top, sz_bot)
    if signed_in > 2.0:
        return "Heart"
    if signed_in >= -2.0:
        return "Shadow"
    if signed_in >= -4.0:
        return "Chase"
    return "Waste"


def attack_zone_for_row(row) -> str:
    px, pz = row.get("plate_x"), row.get("plate_z")
    if pd.isna(px) or pd.isna(pz):
        return "Unknown"
    return assign_attack_zone(
        float(px),
        float(pz),
        float(row.get("sz_top", 3.5)),
        float(row.get("sz_bot", 1.5)),
    )


def zone_description(zone: str) -> str:
    return {
        "Heart": "Middle of the zone — hitters swing here most often.",
        "Shadow": "Edge of the zone — competitive but not middle-middle.",
        "Chase": "Just off the plate — swing rate drops; discipline test.",
        "Waste": "Nowhere near the zone — rarely swung at.",
    }.get(zone, "")


def _inside_sign(bats: str) -> float:
    """Deprecated helper — prefer explicit L/R branches in matches_horizontal."""
    return -1.0 if bats != "L" else 1.0


def matches_horizontal(plate_x: float, horizontal: str, bats: str) -> bool:
    """Inside/outside from hitter perspective; catcher-view plate_x (L = 3B, R = 1B)."""
    if horizontal == "Middle":
        return abs(plate_x) <= 0.32
    if bats == "L":
        if horizontal == "Inside":
            return plate_x >= 0.12
        if horizontal == "Outside":
            return plate_x <= -0.12
    else:
        if horizontal == "Inside":
            return plate_x <= -0.12
        if horizontal == "Outside":
            return plate_x >= 0.12
    return True


def _fallback_location(
    attack_zone: str,
    vertical: str,
    horizontal: str,
    sz_top: float,
    sz_bot: float,
    bats: str,
) -> tuple[float, float]:
    zone_h = sz_top - sz_bot
    mid_z = (sz_bot + sz_top) / 2.0
    if horizontal == "Middle":
        x = 0.0
    elif bats == "L":
        x = 0.55 if horizontal == "Inside" else -0.55
    else:
        x = -0.55 if horizontal == "Inside" else 0.55

    if vertical == "Low":
        z = sz_bot + 0.25 * zone_h
    elif vertical == "High":
        z = sz_bot + 0.75 * zone_h
    else:
        z = mid_z

    for dx in (0.0, 0.12, -0.12, 0.24, -0.24, 0.36):
        for dz in (0.0, 0.12, -0.12, 0.24, -0.24):
            tx, tz = x + dx, z + dz
            if assign_attack_zone(tx, tz, sz_top, sz_bot) != attack_zone:
                continue
            if not matches_vertical(tz, vertical, sz_top, sz_bot):
                continue
            if not matches_horizontal(tx, horizontal, bats):
                continue
            return round(tx, 2), round(tz, 2)

    for tx in np.linspace(-1.6, 1.6, 33):
        for tz in np.linspace(sz_bot - 0.85, sz_top + 0.55, 40):
            if assign_attack_zone(tx, tz, sz_top, sz_bot) != attack_zone:
                continue
            if not matches_vertical(tz, vertical, sz_top, sz_bot):
                continue
            if not matches_horizontal(tx, horizontal, bats):
                continue
            return round(float(tx), 2), round(float(tz), 2)

    for tx in np.linspace(-1.6, 1.6, 33):
        for tz in np.linspace(sz_bot - 0.85, sz_top + 0.55, 40):
            if assign_attack_zone(tx, tz, sz_top, sz_bot) == attack_zone:
                return round(float(tx), 2), round(float(tz), 2)

    return 0.0, round(mid_z, 2)


def matches_vertical(plate_z: float, vertical: str, sz_top: float, sz_bot: float) -> bool:
    zone_h = sz_top - sz_bot
    if vertical == "Low":
        return plate_z <= sz_bot + 0.40 * zone_h
    if vertical == "High":
        return plate_z >= sz_bot + 0.60 * zone_h
    if vertical == "Middle":
        return sz_bot + 0.35 * zone_h <= plate_z <= sz_bot + 0.65 * zone_h
    return True


def random_pitch_location(
    attack_zone: str,
    vertical: str,
    horizontal: str,
    sz_top: float,
    sz_bot: float,
    bats: str,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Sample a plate location in the requested attack zone + vertical + horizontal buckets."""
    rng = rng or np.random.default_rng()

    for _ in range(1500):
        x = float(rng.uniform(-2.0, 2.0))
        z = float(rng.uniform(0.55, 4.35))
        if assign_attack_zone(x, z, sz_top, sz_bot) != attack_zone:
            continue
        if not matches_vertical(z, vertical, sz_top, sz_bot):
            continue
        if not matches_horizontal(x, horizontal, bats):
            continue
        return round(x, 2), round(z, 2)

    # Some zone + placement combos are geometrically empty (e.g. Shadow + middle-middle).
    for _ in range(500):
        x = float(rng.uniform(-2.0, 2.0))
        z = float(rng.uniform(0.55, 4.35))
        if assign_attack_zone(x, z, sz_top, sz_bot) == attack_zone:
            return round(x, 2), round(z, 2)

    return _fallback_location(attack_zone, vertical, horizontal, sz_top, sz_bot, bats)

"""FanGraphs-style attack zones (Heart / Shadow / Chase / Waste)."""

from __future__ import annotations

import numpy as np
import pandas as pd

ZONE_HALF_WIDTH_FT = 17.0 / 24.0

ZONE_ORDER = ["Heart", "Shadow", "Chase", "Waste"]
VERTICAL_ORDER = ["Low", "Middle", "High"]
HORIZONTAL_ORDER = ["Inside", "Middle", "Outside"]


def boundary_signed_inches(
    plate_x: float,
    plate_z: float,
    sz_top: float,
    sz_bot: float,
) -> float:
    """Inches to nearest strike-zone edge; positive = inside, negative = outside."""
    inside_x = ZONE_HALF_WIDTH_FT - abs(plate_x)
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


def attack_zone_for_row(row: pd.Series) -> str:
    px, pz = row.get("plate_x"), row.get("plate_z")
    if pd.isna(px) or pd.isna(pz):
        return "Unknown"
    return assign_attack_zone(
        float(px),
        float(pz),
        float(row.get("sz_top", 3.5)),
        float(row.get("sz_bot", 1.5)),
    )


def assign_vertical(plate_z: float, sz_top: float, sz_bot: float) -> str:
    zone_h = sz_top - sz_bot
    if plate_z <= sz_bot + 0.40 * zone_h:
        return "Low"
    if plate_z >= sz_bot + 0.60 * zone_h:
        return "High"
    return "Middle"


def vertical_for_row(row: pd.Series) -> str:
    pz = row.get("plate_z")
    if pd.isna(pz):
        return "Unknown"
    return assign_vertical(float(pz), float(row.get("sz_top", 3.5)), float(row.get("sz_bot", 1.5)))


def assign_horizontal(plate_x: float, stand: str) -> str:
    if abs(plate_x) <= 0.32:
        return "Middle"
    if stand == "L":
        return "Inside" if plate_x >= 0.12 else "Outside"
    return "Inside" if plate_x <= -0.12 else "Outside"


def horizontal_for_row(row: pd.Series) -> str:
    px, stand = row.get("plate_x"), row.get("stand")
    if pd.isna(px) or stand not in {"L", "R"}:
        return "Unknown"
    return assign_horizontal(float(px), str(stand))


def add_attack_zones(frame: pd.DataFrame) -> pd.DataFrame:
    """Add attack_zone, vertical_tier, horizontal_tier columns (vectorized)."""
    out = frame.copy()
    valid = out[["plate_x", "plate_z", "sz_top", "sz_bot"]].notna().all(axis=1)

    px = out.loc[valid, "plate_x"].astype(float)
    pz = out.loc[valid, "plate_z"].astype(float)
    sz_top = out.loc[valid, "sz_top"].astype(float)
    sz_bot = out.loc[valid, "sz_bot"].astype(float)

    signed_in = boundary_signed_inches_array(px.to_numpy(), pz.to_numpy(), sz_top.to_numpy(), sz_bot.to_numpy())
    attack = np.full(len(out), "Unknown", dtype=object)
    attack[valid.to_numpy()] = np.select(
        [signed_in > 2.0, signed_in >= -2.0, signed_in >= -4.0],
        ["Heart", "Shadow", "Chase"],
        default="Waste",
    )
    out["attack_zone"] = attack

    zone_h = sz_top - sz_bot
    vertical = np.full(len(out), "Unknown", dtype=object)
    pz_v = pz.to_numpy()
    bot_v = sz_bot.to_numpy()
    h_v = zone_h.to_numpy()
    vertical[valid.to_numpy()] = np.select(
        [pz_v <= bot_v + 0.40 * h_v, pz_v >= bot_v + 0.60 * h_v],
        ["Low", "High"],
        default="Middle",
    )
    out["vertical_tier"] = vertical

    horizontal = np.full(len(out), "Unknown", dtype=object)
    h_valid = valid & out["stand"].isin(["L", "R"])
    px_h = out.loc[h_valid, "plate_x"].astype(float).to_numpy()
    stand_h = out.loc[h_valid, "stand"].astype(str).to_numpy()
    abs_x = np.abs(px_h)
    is_middle = abs_x <= 0.32
    is_l = stand_h == "L"
    is_inside = np.where(is_l, px_h >= 0.12, px_h <= -0.12)
    h_labels = np.where(is_middle, "Middle", np.where(is_inside, "Inside", "Outside"))
    horizontal[h_valid.to_numpy()] = h_labels
    out["horizontal_tier"] = horizontal

    return out


def boundary_signed_inches_array(
    plate_x: np.ndarray,
    plate_z: np.ndarray,
    sz_top: np.ndarray,
    sz_bot: np.ndarray,
) -> np.ndarray:
    inside_x = ZONE_HALF_WIDTH_FT - np.abs(plate_x)
    inside_z_top = sz_top - plate_z
    inside_z_bot = plate_z - sz_bot
    return np.minimum.reduce([inside_x, inside_z_top, inside_z_bot]) * 12.0

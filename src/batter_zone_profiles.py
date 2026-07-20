"""Batter contact-quality profiles by Statcast zone for app heatmaps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.statcast_zones import (
    DEFAULT_SZ_BOT,
    DEFAULT_SZ_TOP,
    assign_statcast_zone,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELING_PATH = ROOT / "data" / "modeling_frame_2025.parquet"
DEFAULT_CACHE_PATH = ROOT / "data" / "batter_zone_xwobacon.parquet"

LEAGUE_ZONE_XWOBAcon = 0.320
XWOBAcon_COLOR_MIN = 0.220
XWOBAcon_COLOR_MAX = 0.460

VALID_STATCAST_ZONES = set(range(1, 10)) | {11, 12, 13, 14}


def attach_statcast_zone(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Assign Statcast zones 1–9 and 11–14 from plate location.

    Always computed from ``plate_x``, ``plate_z``, and per-pitch ``sz_top`` /
    ``sz_bot`` so outer shadow quadrants match the MLB 13-zone grid used in the app.
    """
    if not {"plate_x", "plate_z"}.issubset(frame.columns):
        raise ValueError("frame must include plate_x and plate_z")

    out = frame.copy()
    if "sz_top" in out.columns and "sz_bot" in out.columns:
        top = out["sz_top"].fillna(DEFAULT_SZ_TOP).to_numpy(dtype=float)
        bot = out["sz_bot"].fillna(DEFAULT_SZ_BOT).to_numpy(dtype=float)
    else:
        top = np.full(len(out), DEFAULT_SZ_TOP)
        bot = np.full(len(out), DEFAULT_SZ_BOT)

    out["zone"] = assign_statcast_zone(
        out["plate_x"].to_numpy(),
        out["plate_z"].to_numpy(),
        top,
        bot,
    )
    return out


def build_batter_zone_cache(
    modeling_path: Path | str = DEFAULT_MODELING_PATH,
    *,
    output_path: Path | str = DEFAULT_CACHE_PATH,
    batter_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Aggregate mean xwOBAcon by batter and Statcast zone (BIP only)."""
    frame = pd.read_parquet(modeling_path)

    keep = [
        col
        for col in ["batter", "plate_x", "plate_z", "xwobacon", "sz_top", "sz_bot"]
        if col in frame.columns
    ]
    bip = frame.loc[frame["xwobacon"].notna(), keep].copy()
    if batter_ids is not None:
        bip = bip.loc[bip["batter"].isin(batter_ids)]

    bip = attach_statcast_zone(bip)

    grouped = (
        bip.groupby(["batter", "zone"], as_index=False)
        .agg(mean_xwobacon=("xwobacon", "mean"), n_bip=("xwobacon", "size"))
        .sort_values(["batter", "zone"])
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_parquet(output_path, index=False)
    return grouped


def load_batter_zone_cache(path: Path | str = DEFAULT_CACHE_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/build_batter_zone_profiles.py first."
        )
    return pd.read_parquet(path)


def batter_zone_lookup(
    batter_id: int,
    cache: pd.DataFrame | None = None,
    *,
    cache_path: Path | str = DEFAULT_CACHE_PATH,
    fallback: float = LEAGUE_ZONE_XWOBAcon,
) -> dict[int, float]:
    """Zone → mean xwOBAcon for one batter; missing zones use ``fallback``."""
    if cache is None:
        cache = load_batter_zone_cache(cache_path)

    subset = cache.loc[cache["batter"] == batter_id]
    values = {int(z): float(v) for z, v in zip(subset["zone"], subset["mean_xwobacon"])}

    out = {zone: fallback for zone in sorted(VALID_STATCAST_ZONES)}
    out.update(values)
    return out

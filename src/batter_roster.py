"""Qualified batter roster for the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import (
    BAT_TRACKING_COLUMNS,
    add_targets,
    league_default_batter_profile,
    rule_book_in_zone,
)
from src.player_lookup import MIN_QUALIFIED_AB, PLAYERS_FILE, qualified_batter_ids

ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_FILE = ROOT / "data" / "preprocessed_2025.parquet"
ROSTER_FILE = ROOT / "data" / "batter_roster.parquet"

DISPLAY_COLUMNS = [
    "name",
    "mlb_id",
    "stand",
    "swing_pct",
    "whiff_pct",
    "o_zone_pct",
    "z_zone_pct",
    "o_whiff_pct",
    "z_whiff_pct",
    "bat_speed",
    "attack_angle",
    "squared_up_rate",
]


def _stand_mode(series: pd.Series) -> str:
    mode = series.mode()
    if mode.empty:
        return "R"
    return str(mode.iloc[0])


def _unique_display_names(names: pd.Series, mlb_ids: pd.Series) -> pd.Series:
    counts: dict[str, int] = {}
    out: list[str] = []
    for name, mlb_id in zip(names, mlb_ids, strict=True):
        base = str(name).strip()
        if not base:
            base = f"Player {int(mlb_id)}"
        if base not in counts:
            counts[base] = 0
            out.append(base)
            continue
        counts[base] += 1
        out.append(f"{base} ({int(mlb_id)})")
    return pd.Series(out, index=names.index, dtype="object")


def aggregate_qualified_batter_stats(
    frame: pd.DataFrame,
    *,
    qualified_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Season-level scouting rates for qualified batters."""
    if "events" not in frame.columns:
        raise ValueError("preprocessed frame must include an events column")

    qualified = qualified_ids or qualified_batter_ids(frame)
    work = frame.loc[frame["batter"].isin(qualified)].copy()
    if work.empty:
        raise ValueError("No qualified batters found in the supplied frame.")

    work = add_targets(work)
    work["is_in_zone"] = rule_book_in_zone(
        work["plate_x"],
        work["plate_z"],
        work["sz_bot"],
        work["sz_top"],
    ).astype(np.int8)

    in_zone = work["is_in_zone"].astype(bool)
    is_swing = work["is_swing"].astype(bool)
    is_whiff = work["is_whiff"].astype(bool)

    base = work.groupby("batter", as_index=False).agg(
        stand=("stand", _stand_mode),
        swing_pct=("is_swing", "mean"),
    )
    iz = work.loc[in_zone]
    ooz = work.loc[~in_zone]
    iz_swings = work.loc[in_zone & is_swing]
    ooz_swings = work.loc[~in_zone & is_swing]

    z_swing = (
        iz.groupby("batter")["is_swing"]
        .mean()
        .rename("z_zone_pct")
        .reset_index()
    )
    o_swing = (
        ooz.groupby("batter")["is_swing"]
        .mean()
        .rename("o_zone_pct")
        .reset_index()
    )
    z_whiff = (
        iz_swings.groupby("batter")["is_whiff"]
        .mean()
        .rename("z_whiff_pct")
        .reset_index()
    )
    o_whiff = (
        ooz_swings.groupby("batter")["is_whiff"]
        .mean()
        .rename("o_whiff_pct")
        .reset_index()
    )

    stats = (
        base.merge(z_swing, on="batter", how="left")
        .merge(o_swing, on="batter", how="left")
        .merge(z_whiff, on="batter", how="left")
        .merge(o_whiff, on="batter", how="left")
    )

    swings = work.loc[is_swing].copy()
    whiff_on_swings = (
        swings.groupby("batter")["is_whiff"]
        .mean()
        .rename("whiff_pct")
        .reset_index()
    )
    stats = stats.drop(columns=["whiff_pct"], errors="ignore").merge(
        whiff_on_swings, on="batter", how="left"
    )

    defaults = league_default_batter_profile(work)
    league_fill = {
        "swing_pct": defaults["swing_pct"],
        "whiff_pct": defaults["whiff_pct"],
        "o_zone_pct": defaults["o_swing_pct"],
        "z_zone_pct": defaults["z_swing_pct"],
        "o_whiff_pct": defaults["o_whiff_pct"],
        "z_whiff_pct": defaults["z_whiff_pct"],
    }
    for col, default in league_fill.items():
        stats[col] = stats[col].fillna(default)

    for col in league_fill:
        stats[col] = stats[col] * 100.0

    tracking_cols = [col for col in BAT_TRACKING_COLUMNS if col in swings.columns]
    if tracking_cols:
        tracked = swings.dropna(subset=tracking_cols[:2], how="any")
        if not tracked.empty:
            tracking = tracked.groupby("batter")[tracking_cols].mean().reset_index()
            stats = stats.merge(tracking, on="batter", how="left")

    sq_default = defaults["squared_up_rate"]
    if sq_default <= 1.0:
        sq_default *= 100.0
    for col, default in (
        ("bat_speed", defaults["bat_speed"]),
        ("attack_angle", defaults["attack_angle"]),
        ("squared_up_rate", sq_default),
    ):
        if col not in stats.columns:
            stats[col] = default
        else:
            stats[col] = stats[col].fillna(default)
    if stats["squared_up_rate"].max(skipna=True) <= 1.0:
        stats["squared_up_rate"] = stats["squared_up_rate"] * 100.0

    stats = stats.rename(columns={"batter": "mlb_id"})
    stats["mlb_id"] = stats["mlb_id"].astype(int)
    return stats


def attach_player_names(stats: pd.DataFrame, players_path: Path = PLAYERS_FILE) -> pd.DataFrame:
    if not players_path.exists():
        raise FileNotFoundError(
            f"Missing {players_path}. Run notebooks/00_data_pull.ipynb or "
            "scripts that build data/players.parquet first."
        )
    players = pd.read_parquet(players_path)
    merged = stats.merge(
        players.rename(columns={"mlbam_id": "mlb_id"}),
        on="mlb_id",
        how="left",
    )
    merged["player_name"] = merged["player_name"].fillna(
        merged["mlb_id"].astype(str).radd("Player ")
    )
    merged["name"] = _unique_display_names(merged["player_name"], merged["mlb_id"])
    return merged


def build_batter_roster_table(
    preprocessed_path: Path = PREPROCESSED_FILE,
    *,
    players_path: Path = PLAYERS_FILE,
    min_ab: int = MIN_QUALIFIED_AB,
) -> pd.DataFrame:
    if not preprocessed_path.exists():
        raise FileNotFoundError(
            f"Missing {preprocessed_path}. Run 02_preprocessing.ipynb first."
        )
    frame = pd.read_parquet(preprocessed_path)
    qualified = qualified_batter_ids(frame, min_ab=min_ab)
    stats = aggregate_qualified_batter_stats(frame, qualified_ids=qualified)
    roster = attach_player_names(stats, players_path=players_path)
    roster = roster[DISPLAY_COLUMNS].sort_values("name").reset_index(drop=True)
    return roster


def save_batter_roster(
    roster: pd.DataFrame,
    out_path: Path = ROSTER_FILE,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    roster.to_parquet(out_path, index=False)
    return out_path


def load_batter_roster_table(path: Path = ROSTER_FILE) -> pd.DataFrame:
    if not path.exists():
        roster = build_batter_roster_table()
        save_batter_roster(roster, path)
    table = pd.read_parquet(path)
    missing = [col for col in DISPLAY_COLUMNS if col not in table.columns]
    if missing:
        raise ValueError(f"batter roster cache missing columns: {missing}")
    return table.sort_values("name").reset_index(drop=True)


def roster_table_to_dict(table: pd.DataFrame) -> dict[str, dict]:
    roster: dict[str, dict] = {}
    for row in table.to_dict(orient="records"):
        name = str(row.pop("name"))
        roster[name] = row
    return roster


def load_batter_roster(path: Path = ROSTER_FILE) -> dict[str, dict]:
    return roster_table_to_dict(load_batter_roster_table(path))


def qualified_batter_mlb_ids(path: Path = ROSTER_FILE) -> list[int]:
    table = load_batter_roster_table(path)
    return table["mlb_id"].astype(int).tolist()

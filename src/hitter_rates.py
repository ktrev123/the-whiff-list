"""Hitter swing / out-of-zone swing rates for Whiff Lab display."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.whiff_features import TRAIN_CUTOFF, engineer_features
from src.statcast_schema import filter_competitive_pitches

ROOT = Path(__file__).resolve().parents[1]
RATES_FILE = ROOT / "data" / "model" / "pitch_lab_hitter_rates.csv"
LEAGUE_FILE = ROOT / "data" / "model" / "pitch_lab_league_rates.json"
STATCAST_FILE = ROOT / "data" / "statcast_2025.parquet"

COLOR_CLOSE = "#d4a937"
COLOR_BETTER = "#20908d"
COLOR_WORSE = "#e63946"
COMPARE_TOLERANCE = 0.02


def compute_hitter_rates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Mar–Aug train-window rates aligned with the modeling pipeline."""
    frame = engineer_features(filter_competitive_pitches(df))
    if "game_date" in frame.columns:
        frame["game_date"] = pd.to_datetime(frame["game_date"])
        frame = frame[frame["game_date"] < TRAIN_CUTOFF]
    return _rates_from_frame(frame)


def compute_rates_from_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Rates from an already-filtered modeling frame (e.g. train split)."""
    return _rates_from_frame(frame)


def _rates_from_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    in_zone = frame[frame["miss_dist_in"] <= 0]
    out_of_zone = frame[frame["miss_dist_in"] > 0]

    batter = frame.groupby("batter", as_index=False).agg(
        swing_rate=("is_swing", "mean"),
        pitches=("is_swing", "size"),
    )
    in_zone_rates = in_zone.groupby("batter", as_index=False).agg(
        in_zone_swing_pct=("is_swing", "mean"),
        in_zone_pitches=("is_swing", "size"),
    )
    ozone = out_of_zone.groupby("batter", as_index=False).agg(
        o_zone_swing_pct=("is_swing", "mean"),
        o_zone_pitches=("is_swing", "size"),
    )
    rates = batter.merge(in_zone_rates, on="batter", how="left").merge(ozone, on="batter", how="left")
    rates["in_zone_swing_pct"] = rates["in_zone_swing_pct"].fillna(rates["swing_rate"])
    rates["o_zone_swing_pct"] = rates["o_zone_swing_pct"].fillna(0.0)

    league = {
        "swing_rate": float(frame["is_swing"].mean()),
        "in_zone_swing_pct": float(in_zone["is_swing"].mean()) if len(in_zone) else 0.0,
        "o_zone_swing_pct": float(out_of_zone["is_swing"].mean()) if len(out_of_zone) else 0.0,
    }
    return rates, league


def export_hitter_rates_from_frame(frame: pd.DataFrame, rates_path: Path = RATES_FILE, league_path: Path = LEAGUE_FILE) -> None:
    rates, league = compute_rates_from_frame(frame)
    rates_path.parent.mkdir(parents=True, exist_ok=True)
    rates.to_csv(rates_path, index=False, float_format="%.6f")
    league_path.write_text(json.dumps(league, indent=2), encoding="utf-8")


def export_hitter_rates(df: pd.DataFrame, rates_path: Path = RATES_FILE, league_path: Path = LEAGUE_FILE) -> None:
    rates, league = compute_hitter_rates(df)
    rates_path.parent.mkdir(parents=True, exist_ok=True)
    rates.to_csv(rates_path, index=False, float_format="%.6f")
    league_path.write_text(json.dumps(league, indent=2), encoding="utf-8")


@st.cache_data
def load_hitter_rates() -> pd.DataFrame:
    if RATES_FILE.exists():
        rates = pd.read_csv(RATES_FILE)
        if "in_zone_swing_pct" not in rates.columns and STATCAST_FILE.exists():
            computed, _ = compute_hitter_rates(pd.read_parquet(STATCAST_FILE))
            rates = rates.merge(
                computed[["batter", "in_zone_swing_pct", "in_zone_pitches"]],
                on="batter",
                how="left",
            )
            rates["in_zone_swing_pct"] = rates["in_zone_swing_pct"].fillna(rates["swing_rate"])
            rates.to_csv(RATES_FILE, index=False, float_format="%.6f")
        return rates

    if not STATCAST_FILE.exists():
        return pd.DataFrame(
            columns=[
                "batter",
                "swing_rate",
                "pitches",
                "in_zone_swing_pct",
                "in_zone_pitches",
                "o_zone_swing_pct",
                "o_zone_pitches",
            ]
        )

    df = pd.read_parquet(STATCAST_FILE)
    rates, league = compute_hitter_rates(df)
    export_hitter_rates(df)
    return rates


@st.cache_data
def load_league_rates() -> dict[str, float]:
    defaults = {
        "swing_rate": 0.47,
        "in_zone_swing_pct": 0.68,
        "o_zone_swing_pct": 0.30,
    }
    if LEAGUE_FILE.exists():
        league = json.loads(LEAGUE_FILE.read_text(encoding="utf-8"))
        if "in_zone_swing_pct" not in league and STATCAST_FILE.exists():
            _, computed = compute_hitter_rates(pd.read_parquet(STATCAST_FILE))
            league["in_zone_swing_pct"] = computed["in_zone_swing_pct"]
            LEAGUE_FILE.write_text(json.dumps(league, indent=2), encoding="utf-8")
        return {**defaults, **league}

    if not STATCAST_FILE.exists():
        return defaults

    df = pd.read_parquet(STATCAST_FILE)
    _, league = compute_hitter_rates(df)
    LEAGUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEAGUE_FILE.write_text(json.dumps(league, indent=2), encoding="utf-8")
    return league


def compare_color(player: float, league: float, *, lower_is_better: bool = False) -> str:
    diff = player - league
    if abs(diff) <= COMPARE_TOLERANCE:
        return COLOR_CLOSE
    if lower_is_better:
        return COLOR_BETTER if diff < 0 else COLOR_WORSE
    return COLOR_BETTER if diff < 0 else COLOR_WORSE


def lookup_hitter_rates(batter_id: int) -> dict[str, float] | None:
    rates = load_hitter_rates()
    if rates.empty:
        return None
    match = rates.loc[rates["batter"] == batter_id]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "swing_rate": float(row["swing_rate"]),
        "in_zone_swing_pct": float(row.get("in_zone_swing_pct", row["swing_rate"])),
        "o_zone_swing_pct": float(row["o_zone_swing_pct"]),
    }


def format_stat_line(
    label: str,
    league_label: str,
    player: float,
    league: float,
    *,
    lower_is_better: bool = False,
) -> str:
    color = compare_color(player, league, lower_is_better=lower_is_better)
    return (
        f'<p class="whiff-hitter-stat">{label}: '
        f'<span style="color:{color}">{player:.1%}</span></p>'
        f'<p class="whiff-hitter-stat-league">{league_label}: {league:.1%}</p>'
    )


def hitter_stats_html(batter_id: int) -> str:
    league = load_league_rates()
    player = lookup_hitter_rates(batter_id)
    if player is None:
        return (
            '<p class="whiff-hitter-stat-muted">Rate stats unavailable for this hitter.</p>'
        )

    swing = format_stat_line(
        "In-zone swing %",
        "League in-zone swing %",
        player["in_zone_swing_pct"],
        league.get("in_zone_swing_pct", league["swing_rate"]),
    )
    ozone = format_stat_line(
        "O-zone swing %",
        "League O-zone swing %",
        player["o_zone_swing_pct"],
        league["o_zone_swing_pct"],
        lower_is_better=True,
    )
    return swing + ozone

"""Statcast pull schema: season bounds, columns, and shared filters."""

from __future__ import annotations

import pandas as pd

# 2025 MLB regular season (Opening Day Mar 27 – final day Sep 28)
SEASON_START = "2025-03-27"
SEASON_END = "2025-09-28"
REGULAR_SEASON_GAME_TYPE = "R"

# Competitive pitch types only — exclude position-player / junk / misreads
EXCLUDED_PITCH_TYPES = frozenset({"PO", "FO", "UN", "KN", "CS", "FA", "EP"})

PITCH_TYPE_GROUPS: dict[str, list[str]] = {
    "Fastballs": ["FF", "SI", "FC"],
    "Breaking Balls": ["SL", "KC", "ST", "SV", "CU"],
    "Off-Speed": ["CH", "FS"],
}

ALLOWED_PITCH_TYPES = frozenset(code for codes in PITCH_TYPE_GROUPS.values() for code in codes)


def pitch_type_group(pitch_type: str) -> str | None:
    for group, codes in PITCH_TYPE_GROUPS.items():
        if pitch_type in codes:
            return group
    return None


def filter_competitive_pitches(df: pd.DataFrame) -> pd.DataFrame:
    """Drop excluded pitch types and keep competitive MLB pitch codes."""
    out = df.copy()
    if "pitch_type" not in out.columns:
        return out
    out["pitch_type"] = out["pitch_type"].astype(str)
    return out[out["pitch_type"].isin(ALLOWED_PITCH_TYPES)].copy()

# Columns persisted to statcast_2025.parquet
# Model inputs + derived inputs + targets + minimal app/leaderboard fields
STATCAST_KEEP_COLS = [
    # keys
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    # players (player_name = pitcher on each pitch row in Statcast)
    "batter",
    "pitcher",
    "player_name",
    # targets / leaderboard
    "description",
    "events",
    # app display (not model features)
    "pitch_name",
    "stand",
    # model: location + derived miss distance inputs
    "pitch_type",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    # model: context
    "balls",
    "strikes",
    "on_1b",
    "on_2b",
    "on_3b",
    # model: pitch physics (see PITCH_METRIC_COLS in whiff_features)
    "release_speed",
    "effective_speed",
    "pfx_x",
    "pfx_z",
    "release_spin_rate",
    "spin_axis",
    "release_extension",
    "estimated_woba_using_speedangle",
]

INT_COLS = [
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "batter",
    "pitcher",
    "balls",
    "strikes",
    "on_1b",
    "on_2b",
    "on_3b",
]

FLOAT_COLS = [
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "release_speed",
    "effective_speed",
    "pfx_x",
    "pfx_z",
    "release_spin_rate",
    "spin_axis",
    "release_extension",
    "estimated_woba_using_speedangle",
]

STRING_COLS = [
    "player_name",
    "description",
    "events",
    "pitch_type",
    "pitch_name",
    "stand",
]


def regular_season_month_ranges(
    season_start: str = SEASON_START,
    season_end: str = SEASON_END,
) -> list[tuple[str, str]]:
    """Monthly API chunks from season start through season end."""
    start = pd.Timestamp(season_start)
    end = pd.Timestamp(season_end)
    ranges: list[tuple[str, str]] = []
    current = start
    while current <= end:
        month_end = min(current + pd.offsets.MonthEnd(0), end)
        ranges.append((current.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")))
        current = month_end + pd.Timedelta(days=1)
    return ranges


def filter_regular_season(df: pd.DataFrame) -> pd.DataFrame:
    """Keep regular-season rows only; drop spring training / exhibitions."""
    out = df.copy()
    if "game_type" in out.columns:
        out = out[out["game_type"] == REGULAR_SEASON_GAME_TYPE]
    dates = pd.to_datetime(out["game_date"])
    start = pd.Timestamp(SEASON_START)
    end = pd.Timestamp(SEASON_END)
    return out[(dates >= start) & (dates <= end)].copy()


def filter_season_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Date-only season window (used on cached files that lack game_type)."""
    dates = pd.to_datetime(df["game_date"])
    start = pd.Timestamp(SEASON_START)
    end = pd.Timestamp(SEASON_END)
    return df[(dates >= start) & (dates <= end)].copy()

"""2025 Statcast pull: season bounds, raw columns, dtypes. No feature engineering."""

from __future__ import annotations

import pandas as pd

SEASON_START = "2025-03-27"
SEASON_END = "2025-09-28"
REGULAR_SEASON_GAME_TYPE = "R"

SCHEMA_VERSION = 4

DEDUP_KEYS = ["game_pk", "at_bat_number", "pitch_number"]

# Competitive pitch types (matches EDA Section 3 / 01_eda.ipynb)
PITCH_CATEGORIES: dict[str, list[str]] = {
    "Fastballs": ["FF", "SI", "FC"],
    "Vertical Breaking Balls": ["CU", "SV", "KC"],
    "Horizontal Breaking Balls": ["SL", "ST"],
    "Offspeed": ["CH", "FS"],
}

COMPETITIVE_PITCH_TYPES = frozenset(
    code for codes in PITCH_CATEGORIES.values() for code in codes
)

# Columns the downstream pipeline requires. Pull fails if any are absent from Savant.
REQUIRED_STATCAST_COLS = frozenset(
    [
        "game_date",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "batter",
        "pitcher",
        "plate_x",
        "plate_z",
        "sz_top",
        "sz_bot",
        "balls",
        "strikes",
        "p_throws",
        "stand",
        "release_speed",
        "pfx_x",
        "pfx_z",
        "release_spin_rate",
        "release_extension",
        "release_pos_x",
        "release_pos_y",
        "release_pos_z",
        "pitch_type",
        "description",
        "estimated_woba_using_speedangle",
    ]
)

STATCAST_KEEP_COLS = [
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "batter",
    "pitcher",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "balls",
    "strikes",
    "zone",
    "on_1b",
    "on_2b",
    "on_3b",
    "outs_when_up",
    "p_throws",
    "stand",
    "pitch_name",
    "release_speed",
    "effective_speed",
    "pfx_x",
    "pfx_z",
    "release_spin_rate",
    "spin_axis",
    "release_extension",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "pitch_type",
    "description",
    "events",
    "estimated_woba_using_speedangle",
    "bat_speed",
    "attack_angle",
]

# Optional bat-tracking columns (2024+ Statcast); not required for pull validation.
BAT_TRACKING_RAW_COLS = frozenset(["bat_speed", "attack_angle"])

INT_COLS = [
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "batter",
    "pitcher",
    "balls",
    "strikes",
    "zone",
    "on_1b",
    "on_2b",
    "on_3b",
    "outs_when_up",
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
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "estimated_woba_using_speedangle",
    "bat_speed",
    "attack_angle",
]

STRING_COLS = [
    "description",
    "events",
    "pitch_type",
    "pitch_name",
    "p_throws",
    "stand",
]


def regular_season_month_ranges(
    season_start: str = SEASON_START,
    season_end: str = SEASON_END,
) -> list[tuple[str, str]]:
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
    out = df.copy()
    if "game_type" in out.columns:
        out = out[out["game_type"] == REGULAR_SEASON_GAME_TYPE]
    dates = pd.to_datetime(out["game_date"])
    start = pd.Timestamp(SEASON_START)
    end = pd.Timestamp(SEASON_END)
    return out[(dates >= start) & (dates <= end)].copy()


def filter_season_dates(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["game_date"])
    start = pd.Timestamp(SEASON_START)
    end = pd.Timestamp(SEASON_END)
    return df[(dates >= start) & (dates <= end)].copy()


def filter_competitive_pitches(df: pd.DataFrame) -> pd.DataFrame:
    """Keep competitive MLB pitch codes only (Section 3 categories)."""
    out = df.copy()
    if "pitch_type" not in out.columns:
        return out
    out["pitch_type"] = out["pitch_type"].astype(str)
    return out[out["pitch_type"].isin(COMPETITIVE_PITCH_TYPES)].copy()

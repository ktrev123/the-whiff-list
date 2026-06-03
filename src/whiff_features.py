"""Shared feature engineering for swing and whiff probability modeling."""

import numpy as np
import pandas as pd

from src.statcast_schema import SEASON_END, SEASON_START, filter_competitive_pitches

WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}

SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
    "foul",
    "foul_tip",
    "foul_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}

PITCH_METRIC_COLS = [
    "release_speed",
    "speed_diff",
    "pfx_x",
    "pfx_z",
    "release_spin_rate",
    "spin_axis",
    "release_extension",
]

def count_leverage_label(balls: int, strikes: int) -> str:
    if strikes == 2:
        return "two_strike"
    if balls > strikes:
        return "hitter_ahead"
    if strikes > balls:
        return "pitcher_ahead"
    return "even"


def count_state_label(balls: int, strikes: int) -> str:
    """Categorical count leverage for tree models: Hitter Ahead, Pitcher Ahead, Even, Full (3-2)."""
    if balls == 3 and strikes == 2:
        return "Full"
    if balls > strikes:
        return "Hitter Ahead"
    if strikes > balls:
        return "Pitcher Ahead"
    return "Even"


CATEGORICAL_FEATURE_COLS = ["pitch_type", "count_state"]

NUMERIC_FEATURE_COLS = [
    "plate_x",
    "plate_z",
    "miss_dist_in",
    "is_two_strike",
    "runners_on",
    *PITCH_METRIC_COLS,
]

MODEL_INPUT_COLS = NUMERIC_FEATURE_COLS + CATEGORICAL_FEATURE_COLS

# Backward-compatible alias used by training scripts and saved artifacts.
FEATURE_COLS = MODEL_INPUT_COLS

TRAIN_CUTOFF = pd.Timestamp("2025-09-01")


def calculate_miss_distance(row):
    x, z = row["plate_x"], row["plate_z"]
    left, right = -0.708, 0.708
    bot, top = row["sz_bot"], row["sz_top"]
    x_out = max(0, left - x) if x < left else max(0, x - right)
    z_out = max(0, bot - z) if z < bot else max(0, z - top)
    return np.sqrt((x_out ** 2) + (z_out ** 2))


def engineer_features(df):
    out = df.copy()
    out["miss_dist_in"] = (out.apply(calculate_miss_distance, axis=1) * 12).astype("float32")
    out["runners_on"] = out[["on_1b", "on_2b", "on_3b"]].notna().sum(axis=1).astype("int8")
    out["balls"] = out["balls"].fillna(0).astype("int8")
    out["strikes"] = out["strikes"].fillna(0).astype("int8")
    out["count_leverage"] = out.apply(
        lambda r: count_leverage_label(int(r["balls"]), int(r["strikes"])), axis=1
    )
    out["count_state"] = out.apply(
        lambda r: count_state_label(int(r["balls"]), int(r["strikes"])), axis=1
    )
    out["is_two_strike"] = (out["strikes"] == 2).astype("int8")
    if "effective_speed" in out.columns and "release_speed" in out.columns:
        out["speed_diff"] = (
            pd.to_numeric(out["effective_speed"], errors="coerce")
            - pd.to_numeric(out["release_speed"], errors="coerce")
        ).astype("float32")
    out["is_whiff"] = out["description"].isin(WHIFF_DESCRIPTIONS).astype("int8")
    out["is_swing"] = out["description"].isin(SWING_DESCRIPTIONS).astype("int8")
    if "pitch_type" in out.columns:
        out["pitch_type"] = out["pitch_type"].fillna("UNK").astype(str)
    for col in PITCH_METRIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out


def compute_pitch_medians(train_df: pd.DataFrame) -> pd.Series:
    return train_df[PITCH_METRIC_COLS].median(numeric_only=True)


def pitch_profile_defaults(train_df: pd.DataFrame, pitch_type: str) -> dict:
    subset = train_df[train_df["pitch_type"] == pitch_type]
    if subset.empty:
        subset = train_df
    return subset[PITCH_METRIC_COLS].median(numeric_only=True).to_dict()


def apply_pitch_imputation(df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    out = df.copy()
    if "pitch_type" in out.columns:
        out["pitch_type"] = out["pitch_type"].fillna("UNK").astype(str)
    for col in PITCH_METRIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
        out[col] = out[col].fillna(float(medians[col]))
    return out


def filter_modeling_frame(df, qualified_batters):
    frame = engineer_features(df)
    frame = filter_competitive_pitches(frame)
    frame = frame[frame["batter"].isin(qualified_batters)]
    frame = frame.dropna(subset=["batter", "plate_x", "plate_z", "sz_top", "sz_bot"])
    return frame


def chronological_split(df):
    dates = pd.to_datetime(df["game_date"])
    train = df[dates < TRAIN_CUTOFF].copy()
    test = df[dates >= TRAIN_CUTOFF].copy()
    return train, test

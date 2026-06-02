"""Shared feature engineering for whiff probability modeling."""

import numpy as np
import pandas as pd

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

FEATURE_COLS = [
    "plate_x",
    "plate_z",
    "miss_dist_in",
    "balls",
    "strikes",
    "runners_on",
]

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
    out["is_whiff"] = out["description"].isin(WHIFF_DESCRIPTIONS).astype("int8")
    out["is_swing"] = out["description"].isin(SWING_DESCRIPTIONS).astype("int8")
    return out


def filter_modeling_frame(df, qualified_batters):
    frame = engineer_features(df)
    frame = frame[frame["batter"].isin(qualified_batters)]
    frame = frame.dropna(subset=["batter", "plate_x", "plate_z", "sz_top", "sz_bot"])
    return frame


def chronological_split(df):
    dates = pd.to_datetime(df["game_date"])
    train = df[dates < TRAIN_CUTOFF].copy()
    test = df[dates >= TRAIN_CUTOFF].copy()
    return train, test

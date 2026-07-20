"""Chronological train/validation/test splits for 2025 Statcast modeling."""

from __future__ import annotations

import pandas as pd

TRAIN_END = pd.Timestamp("2025-08-01")
VAL_END = pd.Timestamp("2025-09-01")


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by ``game_date``: Mar–Jul train, Aug val, Sep+ test."""
    dates = pd.to_datetime(frame["game_date"])
    train = frame.loc[dates < TRAIN_END].copy()
    val = frame.loc[(dates >= TRAIN_END) & (dates < VAL_END)].copy()
    test = frame.loc[dates >= VAL_END].copy()
    return train, val, test

"""Tests for chronological modeling splits."""

from __future__ import annotations

import pandas as pd

from src.model_splits import TRAIN_END, VAL_END, chronological_split


def test_chronological_split_boundaries():
    frame = pd.DataFrame(
        {
            "game_date": [
                "2025-03-15",
                "2025-07-31",
                "2025-08-01",
                "2025-08-31",
                "2025-09-01",
            ]
        }
    )
    train, val, test = chronological_split(frame)

    assert len(train) == 2
    assert len(val) == 2
    assert len(test) == 1
    assert train["game_date"].max() < str(TRAIN_END.date())
    assert val["game_date"].min() >= str(TRAIN_END.date())
    assert val["game_date"].max() < str(VAL_END.date())
    assert test["game_date"].min() >= str(VAL_END.date())

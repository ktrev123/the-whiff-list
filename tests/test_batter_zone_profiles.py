"""Tests for batter zone profile aggregation."""

from __future__ import annotations

import pandas as pd

from src.batter_zone_profiles import attach_statcast_zone


def test_attach_statcast_zone_overwrites_raw_zone_column():
    frame = pd.DataFrame(
        {
            "plate_x": [0.0, -1.0],
            "plate_z": [4.0, 1.25],
            "sz_top": [3.5, 3.5],
            "sz_bot": [1.5, 1.5],
            "zone": [11, 12],
        }
    )
    out = attach_statcast_zone(frame)
    assert out["zone"].tolist() == [12, 13]

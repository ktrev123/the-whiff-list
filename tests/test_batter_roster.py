"""Tests for qualified batter roster aggregation."""

from __future__ import annotations

import pandas as pd

from src.batter_roster import aggregate_qualified_batter_stats, roster_table_to_dict
from src.player_lookup import MIN_QUALIFIED_AB


def _sample_preprocessed_frame() -> pd.DataFrame:
    rows = []
    for batter, stand, n_ab in ((101, "R", MIN_QUALIFIED_AB), (202, "L", MIN_QUALIFIED_AB)):
        for i in range(n_ab):
            rows.append(
                {
                    "batter": batter,
                    "stand": stand,
                    "events": "field_out",
                    "description": "swinging_strike" if i % 3 == 0 else "called_strike",
                    "plate_x": 0.2 if i % 2 == 0 else 0.8,
                    "plate_z": 2.5 if i % 2 == 0 else 1.2,
                    "sz_top": 3.5,
                    "sz_bot": 1.5,
                    "estimated_woba_using_speedangle": 0.320,
                }
            )
    return pd.DataFrame(rows)


def test_aggregate_qualified_batter_stats_returns_percentages():
    frame = _sample_preprocessed_frame()
    stats = aggregate_qualified_batter_stats(frame, qualified_ids={101, 202})

    assert len(stats) == 2
    assert set(stats["mlb_id"]) == {101, 202}
    assert stats["swing_pct"].between(0, 100).all()
    assert stats["whiff_pct"].between(0, 100).all()


def test_roster_table_to_dict_uses_display_names():
    table = pd.DataFrame(
        {
            "name": ["Aaron Judge", "Juan Soto"],
            "mlb_id": [592450, 665742],
            "stand": ["R", "L"],
            "swing_pct": [45.0, 41.0],
            "whiff_pct": [24.0, 21.0],
            "o_zone_pct": [27.0, 18.0],
            "z_zone_pct": [72.0, 68.0],
            "o_whiff_pct": [35.0, 38.0],
            "z_whiff_pct": [28.0, 22.0],
            "bat_speed": [72.0, 71.0],
            "attack_angle": [12.0, 11.0],
            "squared_up_rate": [28.0, 27.0],
        }
    )
    roster = roster_table_to_dict(table)
    assert roster["Aaron Judge"]["mlb_id"] == 592450
    assert roster["Juan Soto"]["stand"] == "L"

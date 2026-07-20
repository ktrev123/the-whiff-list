"""Tests for rolling batter metrics in feature engineering."""

from __future__ import annotations

from src.feature_engineering import (
    BATTER_ROLLING_COLUMNS,
    add_rolling_batter_metrics,
    add_targets,
    engineer_features,
)


def _mini_frame():
    import pandas as pd

    return pd.DataFrame(
        {
            "game_date": ["2025-04-01"] * 4,
            "game_pk": [1, 1, 2, 2],
            "at_bat_number": [1, 1, 1, 1],
            "pitch_number": [1, 2, 1, 2],
            "batter": [101, 101, 101, 101],
            "pitcher": [201, 201, 201, 201],
            "pitch_type": ["FF", "FF", "SL", "SL"],
            "p_throws": ["R", "R", "R", "R"],
            "stand": ["R", "R", "R", "R"],
            "plate_x": [0.0, 0.1, 0.2, 0.3],
            "plate_z": [2.5, 2.5, 2.5, 2.5],
            "sz_top": [3.5, 3.5, 3.5, 3.5],
            "sz_bot": [1.5, 1.5, 1.5, 1.5],
            "balls": [0, 0, 0, 0],
            "strikes": [0, 0, 0, 0],
            "on_1b": [None, None, None, None],
            "on_2b": [None, None, None, None],
            "on_3b": [None, None, None, None],
            "pfx_x": [0.0, 0.0, 0.0, 0.0],
            "pfx_z": [0.0, 0.0, 0.0, 0.0],
            "release_speed": [95.0, 95.0, 85.0, 85.0],
            "release_spin_rate": [2400, 2400, 2400, 2400],
            "release_extension": [6.0, 6.0, 6.0, 6.0],
            "release_pos_x": [-2.0, -2.0, -2.0, -2.0],
            "release_pos_y": [50.0, 50.0, 50.0, 50.0],
            "release_pos_z": [5.8, 5.8, 5.8, 5.8],
            "description": [
                "ball",
                "swinging_strike",
                "foul",
                "hit_into_play",
            ],
            "estimated_woba_using_speedangle": [None, None, None, 0.450],
        }
    )


def test_rolling_metrics_include_zone_whiff_columns():
    assert "z_whiff_pct" in BATTER_ROLLING_COLUMNS
    assert "o_whiff_pct" in BATTER_ROLLING_COLUMNS


def test_rolling_metrics_use_only_prior_pitches():
    frame = engineer_features(add_targets(_mini_frame()))
    rolled = add_rolling_batter_metrics(frame)

    second = rolled.iloc[1]
    assert second["swing_pct"] == 0.0


def test_bunt_attempts_are_not_swings_or_whiffs():
    from src.feature_engineering import BUNT_DESCRIPTIONS

    for desc in BUNT_DESCRIPTIONS:
        frame = _mini_frame()
        frame.loc[0, "description"] = desc
        out = add_targets(frame)
        assert out.loc[0, "is_swing"] == 0
        assert out.loc[0, "is_whiff"] == 0


def test_z_whiff_pct_uses_prior_in_zone_swings():
    frame = engineer_features(add_targets(_mini_frame()))
    rolled = add_rolling_batter_metrics(frame)

    # Row 2: one prior in-zone swing (whiff on pitch 2) -> z_whiff_pct = 1.0
    third = rolled.iloc[2]
    assert third["z_whiff_pct"] == 1.0


def test_center_dist_in_at_zone_center():
    import numpy as np

    from src.feature_engineering import center_dist_inches

    dist = center_dist_inches(
        np.array([0.0]),
        np.array([2.5]),
        np.array([1.5]),
        np.array([3.5]),
    )
    assert dist[0] == 0.0


def test_engineer_features_includes_center_dist_in():
    frame = engineer_features(add_targets(_mini_frame()))
    assert "center_dist_in" in frame.columns
    assert frame.loc[0, "center_dist_in"] >= 0.0


def test_bat_tracking_defaults_without_raw_columns():
    from src.feature_engineering import (
        BAT_TRACKING_COLUMNS,
        add_rolling_bat_tracking_metrics,
    )

    frame = engineer_features(add_targets(_mini_frame()))
    rolled = add_rolling_bat_tracking_metrics(frame)
    first = rolled.iloc[0]
    for col in BAT_TRACKING_COLUMNS:
        assert col in rolled.columns
        assert first[col] == rolled[col].iloc[0]


def test_bat_tracking_rolling_uses_prior_tracked_swings():
    import numpy as np
    import pandas as pd

    from src.feature_engineering import add_rolling_bat_tracking_metrics

    frame = engineer_features(add_targets(_mini_frame()))
    frame["bat_speed"] = pd.Series([np.nan, 70.0, np.nan, 75.0], index=frame.index)
    frame["attack_angle"] = pd.Series([np.nan, 10.0, np.nan, 15.0], index=frame.index)
    rolled = add_rolling_bat_tracking_metrics(frame)

    # Row 4: one prior tracked swing (70 mph, 10 deg) before the BIP swing
    fourth = rolled.iloc[3]
    assert fourth["bat_speed"] == 70.0
    assert fourth["attack_angle"] == 10.0

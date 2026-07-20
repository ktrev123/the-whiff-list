"""Shared pytest fixtures for the Portfolio backend."""

from __future__ import annotations

import pytest

from src.ev_calculator import load_ev_artifacts
from src.feature_engineering import BATTER_ROLLING_COLUMNS, BAT_TRACKING_COLUMNS, BASELINE_FEATURES
from src.pitch_recommender import load_league_average_physics
from src.statcast_schema import COMPETITIVE_PITCH_TYPES
from tests.sample_profiles import SAMPLE_BATTER_PROFILE


@pytest.fixture(scope="session")
def ev_artifacts():
    """Load production RF artifacts once per test session."""
    return load_ev_artifacts()


@pytest.fixture(scope="session")
def league_physics():
    lookup = load_league_average_physics()
    required = {
        "release_speed",
        "release_spin_rate",
        "norm_hb",
        "norm_ivb",
        "release_pos_x",
        "release_pos_z",
        "release_extension",
        "p_throws_L",
    }
    if required.issubset(lookup.columns):
        return lookup

    # Fallback until league_average_physics.csv is rebuilt from the new frame.
    rows = []
    for pitch_type in sorted(COMPETITIVE_PITCH_TYPES):
        rows.append(
            {
                "pitch_type": pitch_type,
                "release_speed": 93.0,
                "release_spin_rate": 2300.0,
                "norm_hb": 0.0,
                "norm_ivb": 15.0,
                "release_pos_x": -2.0,
                "release_pos_y": 50.0,
                "release_pos_z": 5.8,
                "release_extension": 6.2,
                "p_throws_L": 0.0,
            }
        )
    return __import__("pandas").DataFrame(rows)


@pytest.fixture
def sample_batter_profile() -> dict[str, float]:
    return dict(SAMPLE_BATTER_PROFILE)


def artifacts_use_rolling_metrics(artifacts) -> bool:
    swing, whiff, xwobacon = artifacts
    for art in (swing, whiff, xwobacon):
        features = set(art.get("features", []))
        if "batter_te" in features:
            return False
        if not all(col in features for col in BASELINE_FEATURES):
            return False
    swing_features = set(swing.get("features", []))
    whiff_features = set(whiff.get("features", []))
    xwobacon_features = set(xwobacon.get("features", []))
    return (
        BATTER_ROLLING_COLUMNS[0] in swing_features
        and "z_whiff_pct" in swing_features
        and BATTER_ROLLING_COLUMNS[0] in whiff_features
        and all(col in xwobacon_features for col in BAT_TRACKING_COLUMNS)
    )

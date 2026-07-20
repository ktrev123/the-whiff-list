"""Unit tests for the AI Catcher pitch recommender."""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering import BASE_STATE_COLUMNS, COUNT_STATE_COLUMNS
from src.pitch_recommender import (
    RECOMMENDATION_COLUMNS,
    build_base_state,
    build_count_state,
    simulate_plate_appearance,
)
from tests.conftest import artifacts_use_rolling_metrics
from tests.sample_profiles import SAMPLE_BATTER_PROFILE


def _requires_new_artifacts(ev_artifacts):
    if not artifacts_use_rolling_metrics(ev_artifacts):
        pytest.skip(
            "Production joblib artifacts still use batter_te. "
            "Re-run notebooks 03, 05, 07, 09 and rebuild league_average_physics.csv."
        )


def test_simulation_contract(ev_artifacts, league_physics, sample_batter_profile):
    _requires_new_artifacts(ev_artifacts)
    result = simulate_plate_appearance(
        stand="R",
        count_state=build_count_state(0, 0),
        base_state=build_base_state("state_empty"),
        batter_profile=sample_batter_profile,
        p_throws="R",
        artifacts=ev_artifacts,
        league_physics=league_physics,
        profile=False,
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == RECOMMENDATION_COLUMNS
    assert len(result) == 2_210
    assert result["Rank"].tolist() == list(range(1, len(result) + 1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "stand": "X",
            "count_state": build_count_state(0, 0),
            "base_state": build_base_state("state_empty"),
            "batter_profile": dict(SAMPLE_BATTER_PROFILE),
        },
        {
            "stand": "R",
            "count_state": {col: 0 for col in COUNT_STATE_COLUMNS},
            "base_state": build_base_state("state_empty"),
            "batter_profile": dict(SAMPLE_BATTER_PROFILE),
        },
        {
            "stand": "R",
            "count_state": build_count_state(0, 0),
            "base_state": {col: 0 for col in BASE_STATE_COLUMNS},
            "batter_profile": dict(SAMPLE_BATTER_PROFILE),
        },
        {
            "stand": "R",
            "count_state": build_count_state(0, 0),
            "base_state": build_base_state("state_empty"),
            "batter_profile": dict(SAMPLE_BATTER_PROFILE),
            "pitch_types": ["FF", "INVALID"],
        },
    ],
    ids=["invalid_stand", "invalid_count", "invalid_base", "invalid_pitch_type"],
)
def test_invalid_inputs_raise_errors(kwargs):
    with pytest.raises(ValueError):
        simulate_plate_appearance(**kwargs, profile=False)


def test_mathematical_sanity_check(ev_artifacts, league_physics, sample_batter_profile):
    _requires_new_artifacts(ev_artifacts)
    result = simulate_plate_appearance(
        stand="R",
        count_state=build_count_state(0, 0),
        base_state=build_base_state("state_empty"),
        batter_profile=sample_batter_profile,
        p_throws="R",
        artifacts=ev_artifacts,
        league_physics=league_physics,
        profile=False,
    )

    erv = result["ERV"]
    assert erv.is_monotonic_increasing
    assert result.iloc[0]["ERV"] == pytest.approx(erv.min())
    assert result.iloc[0]["Rank"] == 1


def test_spatial_smoothing_changes_erv_and_drives_rank(
    ev_artifacts, league_physics, sample_batter_profile
):
    _requires_new_artifacts(ev_artifacts)
    result = simulate_plate_appearance(
        stand="R",
        count_state=build_count_state(0, 0),
        base_state=build_base_state("state_empty"),
        batter_profile=sample_batter_profile,
        p_throws="R",
        artifacts=ev_artifacts,
        league_physics=league_physics,
        profile=False,
    )

    assert not result["ERV"].equals(result["ERV_raw"])
    assert result.iloc[0]["ERV"] == pytest.approx(result["ERV"].min())
    raw_best_idx = result["ERV_raw"].idxmin()
    if raw_best_idx != 0:
        assert result.iloc[0]["ERV_raw"] >= result.iloc[raw_best_idx]["ERV_raw"]

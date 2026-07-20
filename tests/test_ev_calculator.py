"""Unit tests for the expected value calculator."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ev_calculator import (
    BASELINE_FEATURES,
    calculate_expected_run_value,
    describe_ev_artifacts,
)
from src.feature_engineering import BAT_TRACKING_COLUMNS, BATTER_ROLLING_COLUMNS
from tests.conftest import artifacts_use_rolling_metrics


def _requires_new_artifacts(ev_artifacts):
    if not artifacts_use_rolling_metrics(ev_artifacts):
        pytest.skip(
            "Production joblib artifacts still use batter_te. "
            "Re-run scripts/retrain_production_models.py after notebooks 05/07/09."
        )


def test_describe_ev_artifacts(ev_artifacts):
    _requires_new_artifacts(ev_artifacts)
    summary = describe_ev_artifacts(ev_artifacts)

    assert list(summary["model"]) == ["swing", "whiff", "xwobacon"]
    assert summary["baseline_ok"].all()
    assert not summary["uses_batter_te"].any()
    assert not summary.loc[summary["model"] == "swing", "uses_bat_tracking"].iloc[0]
    assert summary.loc[summary["model"] == "xwobacon", "uses_bat_tracking"].iloc[0]


def test_calculate_expected_run_value_on_sample(modeling_frame_sample, ev_artifacts):
    _requires_new_artifacts(ev_artifacts)
    scored = calculate_expected_run_value(modeling_frame_sample, artifacts=ev_artifacts)

    for col in [
        "p_swing",
        "p_take",
        "p_whiff_given_swing",
        "p_contact_given_swing",
        "pred_xwobacon",
        "expected_run_value",
    ]:
        assert col in scored.columns

    assert scored["p_swing"].between(0, 1).all()
    assert scored["p_whiff_given_swing"].between(0, 1).all()
    assert scored["p_take"].equals(1.0 - scored["p_swing"])
    assert scored["p_contact_given_swing"].equals(1.0 - scored["p_whiff_given_swing"])


@pytest.fixture(scope="session")
def modeling_frame_sample():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "modeling_frame_2025.parquet"
    if not path.exists():
        pytest.skip("modeling_frame_2025.parquet not found — run 03_feature_engineering first")
    frame = pd.read_parquet(path)
    required = BASELINE_FEATURES + BATTER_ROLLING_COLUMNS + BAT_TRACKING_COLUMNS
    missing = [col for col in required if col not in frame.columns]
    if missing:
        pytest.skip(f"Modeling frame missing columns: {missing}")
    return frame.head(500).copy()

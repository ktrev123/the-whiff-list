"""Vectorized batch prediction for the pitch recommendation grid."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from src.attack_zones import assign_attack_zone, boundary_signed_inches
from src.pitch_lab_inputs import (
    PITCH_FAMILY_DEFAULT_CODE,
    VELOCITY_INTENT_TO_TIER,
    apply_physics_tiers,
)
from src.pitch_outcomes import is_in_zone, mock_pitch_probs, mock_xwoba_con
from src.pitch_prescription import (
    HORIZONTAL_OPTIONS,
    PITCH_FAMILIES,
    VERTICAL_OPTIONS,
    location_for_placement,
)
from src.whiff_features import (
    MODEL_INPUT_COLS,
    PITCH_METRIC_COLS,
    _median_or_fallback,
    apply_pitch_imputation,
    engineer_features,
    model_feature_frame,
)

RECOMMENDATION_VELOCITY_INTENT = "Average"


def _runner_columns(runners_on: int) -> dict[str, Any]:
    return {
        "on_1b": 1 if runners_on >= 1 else pd.NA,
        "on_2b": 2 if runners_on >= 2 else pd.NA,
        "on_3b": 3 if runners_on >= 3 else pd.NA,
    }


def _physics_profile(
    pitch_family: str,
    velocity_intent: str,
    profile_lookup: dict[str, Any],
    swing_medians: pd.Series,
) -> dict[str, float]:
    pitch_type = PITCH_FAMILY_DEFAULT_CODE[pitch_family]
    speed_tier = VELOCITY_INTENT_TO_TIER[velocity_intent]
    pitch_profile = apply_physics_tiers(
        pitch_family,
        release_speed=speed_tier,
        vertical_break="Average",
        spin_rate="Average",
        horizontal_break="Average",
    )
    for col in PITCH_METRIC_COLS:
        if col not in pitch_profile:
            pitch_profile[col] = _median_or_fallback(swing_medians, col)
    lab_profile = profile_lookup.get(pitch_type, {})
    if lab_profile:
        for col in PITCH_METRIC_COLS:
            if col in lab_profile:
                pitch_profile[col] = float(lab_profile[col])
    return pitch_profile


def build_recommendation_grid_df(
    *,
    balls: int,
    strikes: int,
    runners_on: int,
    sz_top: float,
    sz_bot: float,
    bats: str,
    profile_lookup: dict[str, Any],
    swing_medians: pd.Series,
) -> pd.DataFrame:
    runner_cols = _runner_columns(runners_on)
    pitch_profile_by_family = {
        family: _physics_profile(family, RECOMMENDATION_VELOCITY_INTENT, profile_lookup, swing_medians)
        for family in PITCH_FAMILIES
    }

    rows: list[dict[str, Any]] = []
    for family, vertical, horizontal in itertools.product(
        PITCH_FAMILIES, VERTICAL_OPTIONS, HORIZONTAL_OPTIONS
    ):
        px, pz = location_for_placement(vertical, horizontal, sz_top, sz_bot, bats)
        rows.append(
            {
                "pitch_family": family,
                "vertical": vertical,
                "horizontal": horizontal,
                "velocity_intent": RECOMMENDATION_VELOCITY_INTENT,
                "plate_x": px,
                "plate_z": pz,
                "sz_bot": sz_bot,
                "sz_top": sz_top,
                "balls": balls,
                "strikes": strikes,
                "pitch_type": PITCH_FAMILY_DEFAULT_CODE[family],
                "description": "placeholder",
                "in_zone": is_in_zone(px, pz, sz_bot, sz_top),
                "attack_zone": assign_attack_zone(px, pz, sz_top, sz_bot),
                **runner_cols,
                **pitch_profile_by_family[family],
            }
        )
    return pd.DataFrame(rows)


def build_thrown_pitch_df(
    *,
    balls: int,
    strikes: int,
    runners_on: int,
    plate_x: float,
    plate_z: float,
    pitch_family: str,
    velocity_intent: str,
    sz_top: float,
    sz_bot: float,
    profile_lookup: dict[str, Any],
    swing_medians: pd.Series,
) -> pd.DataFrame:
    profile = _physics_profile(pitch_family, velocity_intent, profile_lookup, swing_medians)
    return pd.DataFrame(
        [
            {
                "pitch_family": pitch_family,
                "velocity_intent": velocity_intent,
                "plate_x": plate_x,
                "plate_z": plate_z,
                "sz_bot": sz_bot,
                "sz_top": sz_top,
                "balls": balls,
                "strikes": strikes,
                "pitch_type": PITCH_FAMILY_DEFAULT_CODE[pitch_family],
                "description": "placeholder",
                "in_zone": is_in_zone(plate_x, plate_z, sz_bot, sz_top),
                "attack_zone": assign_attack_zone(plate_x, plate_z, sz_top, sz_bot),
                **_runner_columns(runners_on),
                **profile,
            }
        ]
    )


def attach_outcome_columns(
    grid_df: pd.DataFrame,
    swing_p: np.ndarray,
    whiff_p: np.ndarray,
) -> pd.DataFrame:
    in_zone = grid_df["in_zone"].astype(bool).to_numpy()
    p_take = 1.0 - swing_p
    p_whiff_total = swing_p * whiff_p
    p_contact = swing_p * (1.0 - whiff_p)
    xstrike = np.where(in_zone, p_take + p_whiff_total, p_whiff_total)

    p_called = np.where(in_zone, p_take, 0.0)
    p_zone_whiff = np.where(in_zone, p_whiff_total, 0.0)
    p_chase_whiff = np.where(~in_zone, p_whiff_total, 0.0)
    p_favorable = p_called + p_zone_whiff + p_chase_whiff
    p_ball = np.where(~in_zone, p_take, 0.0)

    out = grid_df.copy()
    out["swing"] = swing_p
    out["whiff_if_swing"] = whiff_p
    out["swing_whiff"] = swing_p * whiff_p
    out["p_take"] = p_take
    out["p_whiff"] = p_whiff_total
    out["p_contact"] = p_contact
    out["xstrike"] = xstrike
    out["p_called_strike"] = p_called
    out["p_zone_whiff"] = p_zone_whiff
    out["p_chase_whiff"] = p_chase_whiff
    out["p_favorable"] = p_favorable
    out["p_ball"] = p_ball
    out["p_swing"] = swing_p
    out["p_whiff_if_swing"] = whiff_p
    return out


def batch_predict_outcomes_ml(
    swing_bundle: dict,
    whiff_bundle: dict,
    xwoba_bundle: dict | None,
    grid_df: pd.DataFrame,
) -> pd.DataFrame:
    medians = pd.Series(swing_bundle["pitch_medians"])
    engineered = apply_pitch_imputation(engineer_features(grid_df.copy()), medians)
    swing_cols = list(swing_bundle.get("features", MODEL_INPUT_COLS))
    whiff_cols = list(whiff_bundle.get("features", MODEL_INPUT_COLS))
    swing_frame = model_feature_frame(engineered, swing_cols, medians)
    whiff_frame = model_feature_frame(engineered, whiff_cols, medians)

    swing_p = swing_bundle["model"].predict_proba(swing_frame)[:, 1]
    whiff_p = whiff_bundle["model"].predict_proba(whiff_frame)[:, 1]
    out = attach_outcome_columns(grid_df, swing_p, whiff_p)

    if xwoba_bundle is not None:
        xwoba_cols = list(xwoba_bundle.get("features", MODEL_INPUT_COLS))
        xwoba_frame = model_feature_frame(engineered, xwoba_cols, medians)
        xwoba_pred = np.clip(xwoba_bundle["model"].predict(xwoba_frame), 0.0, 1.25)
        contact_mask = out["p_contact"].to_numpy() > 0
        out["xwoba_con"] = np.where(contact_mask, xwoba_pred, np.nan)
    else:
        out["xwoba_con"] = np.nan

    return out


def batch_predict_outcomes_mock(
    grid_df: pd.DataFrame,
    *,
    balls: int,
    strikes: int,
    hitter_rates: dict[str, float] | None,
) -> pd.DataFrame:
    rates = hitter_rates or {}
    swing_ps = np.empty(len(grid_df), dtype=float)
    whiff_ps = np.empty(len(grid_df), dtype=float)

    for i, row in enumerate(grid_df.itertuples(index=False)):
        miss_dist = max(
            0.0,
            -boundary_signed_inches(row.plate_x, row.plate_z, row.sz_top, row.sz_bot),
        )
        swing_ps[i], whiff_ps[i] = mock_pitch_probs(
            balls=balls,
            strikes=strikes,
            in_zone=bool(row.in_zone),
            pitch_family=row.pitch_family,
            velocity_intent=row.velocity_intent,
            miss_dist_in=miss_dist,
            attack_zone=row.attack_zone,
            in_zone_swing_pct=rates.get("in_zone_swing_pct"),
            o_zone_swing_pct=rates.get("o_zone_swing_pct"),
        )

    out = attach_outcome_columns(grid_df, swing_ps, whiff_ps)
    contact_mask = out["p_contact"].to_numpy() > 0
    xwoba = np.full(len(out), np.nan)
    if contact_mask.any():
        for idx in np.flatnonzero(contact_mask):
            row = out.iloc[idx]
            xwoba[idx] = mock_xwoba_con(
                pitch_family=row["pitch_family"],
                in_zone=bool(row["in_zone"]),
                attack_zone=row["attack_zone"],
            )
    out["xwoba_con"] = xwoba
    return out


def scored_row_to_dict(row: pd.Series) -> dict[str, Any]:
    return row.to_dict()

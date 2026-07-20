"""Incremental whiff-model training utilities (is_whiff target, swings only)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.feature_engineering import (
    BASE_STATE_COLUMNS,
    BASELINE_FEATURES,
    BAT_TRACKING_COLUMNS,
    BATTER_ROLLING_COLUMNS,
    COUNT_STATE_COLUMNS,
    PITCH_MOVEMENT_COLUMNS,
    PITCH_QUALITY_COLUMNS,
    PITCH_TYPE_COLUMNS,
    PITCHER_DELIVERY_COLUMNS,
    PLATOON_COLUMNS,
    WHIFF_DESCRIPTIONS,
    ZONE_COLUMNS,
    build_feature_groups,
)
from src.model_splits import TRAIN_END, VAL_END, chronological_split
from src.swing_modeling import (
    MIN_GAIN,
    SwingModelResult,
    calibration_bins,
    coefficient_frame,
    encoded_feature_names,
    make_swing_pipeline,
    results_table,
    roc_curve_data,
)

TARGET = "is_whiff"
SWING_FLAG = "is_swing"

FEATURE_GROUPS, FEATURE_GROUP_ORDER = build_feature_groups(include_bat_tracking=True)


@dataclass
class ForwardSelectReport:
    baseline: SwingModelResult
    steps: list[dict]
    selected_groups: list[str]
    final: SwingModelResult
    kept_results: list[SwingModelResult] = field(default_factory=list)


def whiff_target_definition() -> str:
    return (
        "is_whiff = 1 when Statcast description is a whiff "
        f"({', '.join(sorted(WHIFF_DESCRIPTIONS))}); else 0. "
        "Evaluated on **swings only** (`is_swing = 1`). Bunts are not swings."
    )


def swings_only(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame[SWING_FLAG] == 1].copy()


def train_and_evaluate(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    *,
    name: str,
) -> SwingModelResult:
    """Fit on train swings; score on validation swings (August holdout)."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    missing = [col for col in feature_cols + [TARGET, SWING_FLAG] if col not in train.columns]
    if missing:
        raise ValueError(f"Missing columns in training frame: {missing}")

    train_sw = swings_only(train)
    val_sw = swings_only(val)

    x_train = train_sw[feature_cols]
    y_train = train_sw[TARGET].astype(int)
    x_val = val_sw[feature_cols]
    y_val = val_sw[TARGET].astype(int)

    model = make_swing_pipeline(feature_cols)
    model.fit(x_train, y_train)

    train_prob = model.predict_proba(x_train)[:, 1]
    val_prob = model.predict_proba(x_val)[:, 1]

    return SwingModelResult(
        name=name,
        features=feature_cols,
        train_log_loss=float(log_loss(y_train, train_prob)),
        val_log_loss=float(log_loss(y_val, val_prob)),
        train_roc_auc=float(roc_auc_score(y_train, train_prob)),
        val_roc_auc=float(roc_auc_score(y_val, val_prob)),
        train_brier=float(brier_score_loss(y_train, train_prob)),
        val_brier=float(brier_score_loss(y_val, val_prob)),
        model=model,
        y_val=y_val.to_numpy(),
        y_prob_val=val_prob,
    )


def train_master_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    *,
    name: str = "master",
) -> SwingModelResult:
    """Train on Mar–Jul swings; evaluate on locked September holdout."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    missing = [col for col in feature_cols + [TARGET, SWING_FLAG] if col not in train.columns]
    if missing:
        raise ValueError(f"Missing columns in training frame: {missing}")

    train_sw = swings_only(train)
    test_sw = swings_only(test)

    x_train = train_sw[feature_cols]
    y_train = train_sw[TARGET].astype(int)
    x_test = test_sw[feature_cols]
    y_test = test_sw[TARGET].astype(int)

    model = make_swing_pipeline(feature_cols)
    model.fit(x_train, y_train)

    train_prob = model.predict_proba(x_train)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    return SwingModelResult(
        name=name,
        features=feature_cols,
        train_log_loss=float(log_loss(y_train, train_prob)),
        val_log_loss=float("nan"),
        train_roc_auc=float(roc_auc_score(y_train, train_prob)),
        val_roc_auc=float("nan"),
        train_brier=float(brier_score_loss(y_train, train_prob)),
        val_brier=float("nan"),
        test_log_loss=float(log_loss(y_test, test_prob)),
        test_roc_auc=float(roc_auc_score(y_test, test_prob)),
        test_brier=float(brier_score_loss(y_test, test_prob)),
        model=model,
        y_val=np.array([], dtype=int),
        y_prob_val=np.array([], dtype=float),
        y_test=y_test.to_numpy(),
        y_prob_test=test_prob,
    )


def forward_select_feature_groups(
    train: pd.DataFrame,
    val: pd.DataFrame,
    *,
    min_log_loss_gain: float = MIN_GAIN,
    verbose: bool = False,
) -> ForwardSelectReport:
    """Forward-select feature groups for whiff probability (swings only)."""
    current_features = BASELINE_FEATURES.copy()
    baseline = train_and_evaluate(train, val, current_features, name="baseline")
    final = baseline
    best_log_loss = baseline.val_log_loss
    kept_results = [baseline]

    if verbose:
        print(
            f"Baseline — val log-loss: {baseline.val_log_loss:.4f} "
            f"({len(current_features)} features)"
        )

    steps: list[dict] = [
        {
            "step": "baseline",
            "added_group": None,
            "kept": True,
            "n_features": len(current_features),
            "val_log_loss": baseline.val_log_loss,
            "val_roc_auc": baseline.val_roc_auc,
            "delta_log_loss": 0.0,
        }
    ]
    selected_groups: list[str] = []

    for group_name in FEATURE_GROUP_ORDER:
        candidate_features = current_features + FEATURE_GROUPS[group_name]
        result = train_and_evaluate(
            train,
            val,
            candidate_features,
            name=f"+ {group_name}",
        )
        delta = best_log_loss - result.val_log_loss
        kept = round(delta, 4) >= min_log_loss_gain

        if verbose:
            if kept:
                print(
                    f"  {group_name}: Kept (log-loss gain = {delta:+.4f}, "
                    f"val log-loss = {result.val_log_loss:.4f}, "
                    f"{len(candidate_features)} features)"
                )
            else:
                print(
                    f"  {group_name}: Dropped due to insufficient gain "
                    f"(< {min_log_loss_gain:.4f}) (log-loss gain = {delta:+.4f})"
                )

        steps.append(
            {
                "step": group_name,
                "added_group": group_name,
                "kept": kept,
                "n_features": len(candidate_features),
                "val_log_loss": result.val_log_loss,
                "val_roc_auc": result.val_roc_auc,
                "delta_log_loss": delta,
            }
        )

        if kept:
            current_features = candidate_features
            best_log_loss = result.val_log_loss
            final = result
            selected_groups.append(group_name)
            kept_results.append(result)

    return ForwardSelectReport(
        baseline=baseline,
        steps=steps,
        selected_groups=selected_groups,
        final=final,
        kept_results=kept_results,
    )


__all__ = [
    "BASELINE_FEATURES",
    "FEATURE_GROUP_ORDER",
    "FEATURE_GROUPS",
    "MIN_GAIN",
    "TARGET",
    "TRAIN_END",
    "VAL_END",
    "ForwardSelectReport",
    "SwingModelResult",
    "calibration_bins",
    "coefficient_frame",
    "chronological_split",
    "encoded_feature_names",
    "forward_select_feature_groups",
    "results_table",
    "roc_curve_data",
    "swings_only",
    "train_and_evaluate",
    "train_master_model",
    "whiff_target_definition",
]

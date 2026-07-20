"""Random Forest swing-model training utilities (is_swing target)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

from src.feature_engineering import (
    BUNT_DESCRIPTIONS,
    SWING_DESCRIPTIONS,
)
from src.model_splits import TRAIN_END, VAL_END, chronological_split
from src.swing_modeling import (
    BASELINE_FEATURES,
    FEATURE_GROUP_ORDER,
    FEATURE_GROUPS,
    MIN_GAIN,
    calibration_bins,
)

TARGET = "is_swing"

# Swing RF intentionally excludes bat_tracking (pitch/count/hitter tendency only).


@dataclass
class RFModelResult:
    name: str
    features: list[str]
    train_log_loss: float
    val_log_loss: float
    train_roc_auc: float
    val_roc_auc: float
    train_brier: float
    val_brier: float
    model: Pipeline
    y_val: np.ndarray = field(repr=False)
    y_prob_val: np.ndarray = field(repr=False)
    # Final holdout metrics (September) populated by train_master_model only.
    test_log_loss: float | None = None
    test_roc_auc: float | None = None
    test_brier: float | None = None
    y_test: np.ndarray | None = field(default=None, repr=False)
    y_prob_test: np.ndarray | None = field(default=None, repr=False)


@dataclass
class ForwardSelectReport:
    baseline: RFModelResult
    steps: list[dict]
    selected_groups: list[str]
    final: RFModelResult
    kept_results: list[RFModelResult] = field(default_factory=list)


def swing_target_definition() -> str:
    return (
        "is_swing = 1 when Statcast description is a full swing "
        f"({', '.join(sorted(SWING_DESCRIPTIONS))}); else 0. "
        f"Bunt attempts ({', '.join(sorted(BUNT_DESCRIPTIONS))}) are **not** swings."
    )


def make_rf_pipeline() -> Pipeline:
    """Shallow trees for fast forward-selection loops."""
    return Pipeline(
        [
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=12,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def make_master_rf_pipeline() -> Pipeline:
    """Deeper trees for the production swing model."""
    return Pipeline(
        [
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_leaf=4,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def train_and_evaluate(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    *,
    name: str,
) -> RFModelResult:
    """Fit on train; score on validation (August holdout)."""
    missing = [col for col in feature_cols + [TARGET] if col not in train.columns]
    if missing:
        raise ValueError(f"Missing columns in training frame: {missing}")

    x_train = train[feature_cols]
    y_train = train[TARGET].astype(int)
    x_val = val[feature_cols]
    y_val = val[TARGET].astype(int)

    model = make_rf_pipeline()
    model.fit(x_train, y_train)

    train_prob = model.predict_proba(x_train)[:, 1]
    val_prob = model.predict_proba(x_val)[:, 1]

    return RFModelResult(
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
) -> RFModelResult:
    """Train the production RF on Mar–Jul; evaluate on locked September holdout."""
    missing = [col for col in feature_cols + [TARGET] if col not in train.columns]
    if missing:
        raise ValueError(f"Missing columns in training frame: {missing}")

    x_train = train[feature_cols]
    y_train = train[TARGET].astype(int)
    x_test = test[feature_cols]
    y_test = test[TARGET].astype(int)

    model = make_master_rf_pipeline()
    model.fit(x_train, y_train)

    train_prob = model.predict_proba(x_train)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    return RFModelResult(
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


def feature_importance_from_model(model: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    step = model.named_steps.get("clf") or model.named_steps.get("reg")
    if step is None:
        raise KeyError("Expected pipeline step 'clf' or 'reg'")
    importances = step.feature_importances_
    out = pd.DataFrame({"feature": feature_cols, "importance": importances})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def forward_select_feature_groups(
    train: pd.DataFrame,
    val: pd.DataFrame,
    *,
    min_log_loss_gain: float = MIN_GAIN,
    verbose: bool = False,
) -> ForwardSelectReport:
    """Greedy group selection: fit on train, keep/drop by August validation log-loss."""
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


def kept_features_by_group(report: ForwardSelectReport) -> dict[str, list[str]]:
    """Return final model columns grouped by baseline + each kept candidate group."""
    grouped: dict[str, list[str]] = {"baseline": BASELINE_FEATURES.copy()}
    for group_name in report.selected_groups:
        grouped[group_name] = FEATURE_GROUPS[group_name]
    return grouped


def kept_features_table(report: ForwardSelectReport) -> pd.DataFrame:
    """One row per kept feature with its group label."""
    rows: list[dict[str, str]] = []
    for group_name, cols in kept_features_by_group(report).items():
        for col in cols:
            rows.append({"group": group_name, "feature": col})
    return pd.DataFrame(rows)


def roc_curve_data(result: RFModelResult) -> tuple[np.ndarray, np.ndarray]:
    y = result.y_test if result.y_test is not None and len(result.y_test) else result.y_val
    prob = (
        result.y_prob_test
        if result.y_prob_test is not None and len(result.y_prob_test)
        else result.y_prob_val
    )
    fpr, tpr, _ = roc_curve(y, prob)
    return fpr, tpr


def feature_importance_frame(result: RFModelResult) -> pd.DataFrame:
    importances = result.model.named_steps["clf"].feature_importances_
    out = pd.DataFrame({"feature": result.features, "importance": importances})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def selection_results_table(report: ForwardSelectReport) -> pd.DataFrame:
    return pd.DataFrame(report.steps).assign(
        kept=lambda d: d["kept"].map({True: "yes", False: "no"}),
        val_log_loss=lambda d: d["val_log_loss"].map(lambda x: f"{x:.4f}"),
        val_roc_auc=lambda d: d["val_roc_auc"].map(lambda x: f"{x:.3f}"),
        delta_log_loss=lambda d: d["delta_log_loss"].map(lambda x: f"{x:+.4f}"),
    )[
        [
            "step",
            "added_group",
            "kept",
            "n_features",
            "val_log_loss",
            "val_roc_auc",
            "delta_log_loss",
        ]
    ]


# Re-export split helpers for notebooks.
__all__ = [
    "BASELINE_FEATURES",
    "FEATURE_GROUP_ORDER",
    "FEATURE_GROUPS",
    "MIN_GAIN",
    "TARGET",
    "TRAIN_END",
    "VAL_END",
    "ForwardSelectReport",
    "RFModelResult",
    "chronological_split",
    "feature_importance_frame",
    "feature_importance_from_model",
    "forward_select_feature_groups",
    "kept_features_by_group",
    "kept_features_table",
    "make_master_rf_pipeline",
    "make_rf_pipeline",
    "roc_curve_data",
    "selection_results_table",
    "swing_target_definition",
    "train_and_evaluate",
    "train_master_model",
]

results_table = selection_results_table

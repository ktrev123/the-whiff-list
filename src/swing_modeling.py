"""Incremental swing-model training utilities (is_swing target)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import (
    BASE_STATE_COLUMNS,
    BASELINE_FEATURES,
    BUNT_DESCRIPTIONS,
    COUNT_STATE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    PITCH_MOVEMENT_COLUMNS,
    PITCH_QUALITY_COLUMNS,
    PITCH_TYPE_COLUMNS,
    PITCHER_DELIVERY_COLUMNS,
    PLATOON_COLUMNS,
    SWING_DESCRIPTIONS,
    ZONE_COLUMNS,
    build_feature_groups,
)
from src.model_splits import TRAIN_END, VAL_END, chronological_split

TARGET = "is_swing"

FEATURE_GROUPS, FEATURE_GROUP_ORDER = build_feature_groups(include_bat_tracking=False)

MIN_GAIN = 0.0010


@dataclass
class SwingModelResult:
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
    test_log_loss: float | None = None
    test_roc_auc: float | None = None
    test_brier: float | None = None
    y_test: np.ndarray | None = field(default=None, repr=False)
    y_prob_test: np.ndarray | None = field(default=None, repr=False)


@dataclass
class ForwardSelectReport:
    baseline: SwingModelResult
    steps: list[dict]
    selected_groups: list[str]
    final: SwingModelResult
    kept_results: list[SwingModelResult] = field(default_factory=list)


def swing_target_definition() -> str:
    return (
        "is_swing = 1 when Statcast description is a full swing "
        f"({', '.join(sorted(SWING_DESCRIPTIONS))}); else 0. "
        f"Bunt attempts ({', '.join(sorted(BUNT_DESCRIPTIONS))}) are **not** swings."
    )


def _numeric_columns(feature_cols: list[str]) -> list[str]:
    return [col for col in feature_cols if col in NUMERIC_FEATURE_COLUMNS]


def make_swing_pipeline(feature_cols: list[str]) -> Pipeline:
    numeric_cols = _numeric_columns(feature_cols)
    passthrough_cols = [col for col in feature_cols if col not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if passthrough_cols:
        transformers.append(("pass", "passthrough", passthrough_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    return Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
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
) -> SwingModelResult:
    """Fit on train; score on validation (August holdout)."""
    missing = [col for col in feature_cols + [TARGET] if col not in train.columns]
    if missing:
        raise ValueError(f"Missing columns in training frame: {missing}")

    x_train = train[feature_cols]
    y_train = train[TARGET].astype(int)
    x_val = val[feature_cols]
    y_val = val[TARGET].astype(int)

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
    """Train on Mar–Jul; evaluate on locked September holdout."""
    missing = [col for col in feature_cols + [TARGET] if col not in train.columns]
    if missing:
        raise ValueError(f"Missing columns in training frame: {missing}")

    x_train = train[feature_cols]
    y_train = train[TARGET].astype(int)
    x_test = test[feature_cols]
    y_test = test[TARGET].astype(int)

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


def results_table(report: ForwardSelectReport) -> pd.DataFrame:
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


def roc_curve_data(result: SwingModelResult) -> tuple[np.ndarray, np.ndarray]:
    y = result.y_test if result.y_test is not None and len(result.y_test) else result.y_val
    prob = (
        result.y_prob_test
        if result.y_prob_test is not None and len(result.y_prob_test)
        else result.y_prob_val
    )
    fpr, tpr, _ = roc_curve(y, prob)
    return fpr, tpr


def calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    frame = pd.DataFrame({"y": y_true, "p": y_prob})
    frame["bin"] = pd.cut(frame["p"], bins=n_bins, labels=False, include_lowest=True)
    grouped = frame.groupby("bin", observed=True).agg(
        pred_mean=("p", "mean"),
        obs_rate=("y", "mean"),
        n=("y", "size"),
    )
    return grouped.reset_index(drop=True)


def encoded_feature_names(model: Pipeline, feature_cols: list[str]) -> list[str]:
    prep = model.named_steps["prep"]
    try:
        raw_names = prep.get_feature_names_out()
    except AttributeError:
        return feature_cols
    cleaned = []
    for name in raw_names:
        if name.startswith("num__"):
            cleaned.append(name.removeprefix("num__"))
        elif name.startswith("pass__"):
            cleaned.append(name.removeprefix("pass__"))
        else:
            cleaned.append(name)
    return cleaned


def coefficient_frame(result: SwingModelResult) -> pd.DataFrame:
    names = encoded_feature_names(result.model, result.features)
    coefs = result.model.named_steps["clf"].coef_.ravel()
    out = pd.DataFrame({"feature": names, "coefficient": coefs})
    out["abs_coef"] = out["coefficient"].abs()
    return out.sort_values("abs_coef", ascending=False).reset_index(drop=True)


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
    "chronological_split",
    "coefficient_frame",
    "encoded_feature_names",
    "forward_select_feature_groups",
    "make_swing_pipeline",
    "results_table",
    "roc_curve_data",
    "swing_target_definition",
    "train_and_evaluate",
    "train_master_model",
]

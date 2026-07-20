"""Random Forest xwOBAcon modeling utilities (BIP / contact only)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.model_splits import TRAIN_END, VAL_END, chronological_split
from src.xwobacon_modeling import (
    BASELINE_FEATURES,
    DESCRIPTION_COL,
    FEATURE_GROUP_ORDER,
    FEATURE_GROUPS,
    MIN_GAIN_RMSE,
    TARGET,
    contact_only,
)

__all__ = [
    "BASELINE_FEATURES",
    "FEATURE_GROUP_ORDER",
    "FEATURE_GROUPS",
    "MIN_GAIN_RMSE",
    "TARGET",
    "TRAIN_END",
    "VAL_END",
    "ForwardSelectReport",
    "RFXwOBAconModelResult",
    "chronological_split",
    "combined_feature_columns",
    "contact_only",
    "feature_groups_summary",
    "feature_importance_from_model",
    "forward_select_feature_groups",
    "kept_features_by_group",
    "kept_features_table",
    "results_table",
    "train_master_model",
    "xwobacon_target_definition",
]


@dataclass
class RFXwOBAconModelResult:
    name: str
    features: list[str]
    train_rmse: float
    val_rmse: float
    train_r2: float
    val_r2: float
    model: Pipeline
    y_val: np.ndarray = field(repr=False)
    y_pred_val: np.ndarray = field(repr=False)
    test_rmse: float | None = None
    test_r2: float | None = None
    y_test: np.ndarray | None = field(default=None, repr=False)
    y_pred_test: np.ndarray | None = field(default=None, repr=False)


@dataclass
class ForwardSelectReport:
    baseline: RFXwOBAconModelResult
    steps: list[dict]
    selected_groups: list[str]
    final: RFXwOBAconModelResult
    kept_results: list[RFXwOBAconModelResult] = field(default_factory=list)


def xwobacon_target_definition() -> str:
    from src.xwobacon_modeling import xwobacon_target_definition as _def

    return _def()


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_rf_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=12,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def make_master_rf_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=50,
                    max_features="sqrt",
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
) -> RFXwOBAconModelResult:
    missing = [col for col in feature_cols + [TARGET, DESCRIPTION_COL] if col not in train.columns]
    if missing:
        raise ValueError(f"Missing columns in training frame: {missing}")

    train_bip = contact_only(train)
    val_bip = contact_only(val)

    x_train = train_bip[feature_cols]
    y_train = train_bip[TARGET].astype(float)
    x_val = val_bip[feature_cols]
    y_val = val_bip[TARGET].astype(float)

    model = make_rf_pipeline()
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)

    return RFXwOBAconModelResult(
        name=name,
        features=feature_cols,
        train_rmse=_rmse(y_train.to_numpy(), train_pred),
        val_rmse=_rmse(y_val.to_numpy(), val_pred),
        train_r2=float(r2_score(y_train, train_pred)),
        val_r2=float(r2_score(y_val, val_pred)),
        model=model,
        y_val=y_val.to_numpy(),
        y_pred_val=val_pred,
    )


def train_master_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    *,
    name: str = "master",
) -> RFXwOBAconModelResult:
    missing = [col for col in feature_cols + [TARGET, DESCRIPTION_COL] if col not in train.columns]
    if missing:
        raise ValueError(f"Missing columns in training frame: {missing}")

    train_bip = contact_only(train)
    test_bip = contact_only(test)

    x_train = train_bip[feature_cols]
    y_train = train_bip[TARGET].astype(float)
    x_test = test_bip[feature_cols]
    y_test = test_bip[TARGET].astype(float)

    model = make_master_rf_pipeline()
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    return RFXwOBAconModelResult(
        name=name,
        features=feature_cols,
        train_rmse=_rmse(y_train.to_numpy(), train_pred),
        val_rmse=float("nan"),
        train_r2=float(r2_score(y_train, train_pred)),
        val_r2=float("nan"),
        test_rmse=_rmse(y_test.to_numpy(), test_pred),
        test_r2=float(r2_score(y_test, test_pred)),
        model=model,
        y_val=np.array([], dtype=float),
        y_pred_val=np.array([], dtype=float),
        y_test=y_test.to_numpy(),
        y_pred_test=test_pred,
    )


def forward_select_feature_groups(
    train: pd.DataFrame,
    val: pd.DataFrame,
    *,
    min_rmse_gain: float = MIN_GAIN_RMSE,
    verbose: bool = False,
) -> ForwardSelectReport:
    current_features = BASELINE_FEATURES.copy()
    baseline = train_and_evaluate(train, val, current_features, name="baseline")
    final = baseline
    best_rmse = baseline.val_rmse
    kept_results = [baseline]

    if verbose:
        print(
            f"Baseline — val RMSE: {baseline.val_rmse:.4f} "
            f"({len(current_features)} features)"
        )

    steps: list[dict] = [
        {
            "step": "baseline",
            "added_group": None,
            "kept": True,
            "n_features": len(current_features),
            "val_rmse": baseline.val_rmse,
            "val_r2": baseline.val_r2,
            "delta_rmse": 0.0,
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
        delta = best_rmse - result.val_rmse
        kept = round(delta, 4) >= min_rmse_gain

        if verbose:
            if kept:
                print(
                    f"  {group_name}: Kept (RMSE gain = {delta:+.4f}, "
                    f"val RMSE = {result.val_rmse:.4f}, "
                    f"{len(candidate_features)} features)"
                )
            else:
                print(
                    f"  {group_name}: Dropped due to insufficient gain "
                    f"(< {min_rmse_gain:.4f}) (RMSE gain = {delta:+.4f})"
                )

        steps.append(
            {
                "step": group_name,
                "added_group": group_name,
                "kept": kept,
                "n_features": len(candidate_features),
                "val_rmse": result.val_rmse,
                "val_r2": result.val_r2,
                "delta_rmse": delta,
            }
        )

        if kept:
            current_features = candidate_features
            best_rmse = result.val_rmse
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
    grouped: dict[str, list[str]] = {"baseline": BASELINE_FEATURES.copy()}
    for group_name in report.selected_groups:
        grouped[group_name] = FEATURE_GROUPS[group_name]
    return grouped


def kept_features_table(report: ForwardSelectReport) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for group_name, cols in kept_features_by_group(report).items():
        for col in cols:
            rows.append({"group": group_name, "feature": col})
    return pd.DataFrame(rows)


def results_table(report: ForwardSelectReport) -> pd.DataFrame:
    return pd.DataFrame(report.steps).assign(
        kept=lambda d: d["kept"].map({True: "yes", False: "no"}),
        val_rmse=lambda d: d["val_rmse"].map(lambda x: f"{x:.4f}"),
        val_r2=lambda d: d["val_r2"].map(lambda x: f"{x:.3f}"),
        delta_rmse=lambda d: d["delta_rmse"].map(lambda x: f"{x:+.4f}"),
    )[
        [
            "step",
            "added_group",
            "kept",
            "n_features",
            "val_rmse",
            "val_r2",
            "delta_rmse",
        ]
    ]


def combined_feature_columns() -> list[str]:
    """Baseline location features plus every candidate group (no forward selection)."""
    features = BASELINE_FEATURES.copy()
    for group_name in FEATURE_GROUP_ORDER:
        features.extend(FEATURE_GROUPS[group_name])
    return features


def feature_groups_summary() -> pd.DataFrame:
    rows = [{"group": "baseline", "n_columns": len(BASELINE_FEATURES)}]
    for group_name in FEATURE_GROUP_ORDER:
        rows.append({"group": group_name, "n_columns": len(FEATURE_GROUPS[group_name])})
    rows.append({"group": "combined", "n_columns": len(combined_feature_columns())})
    return pd.DataFrame(rows)


def feature_importance_from_model(model: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    importances = model.named_steps["reg"].feature_importances_
    out = pd.DataFrame({"feature": feature_cols, "importance": importances})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)

"""Linear regression xwOBAcon modeling utilities (contact / BIP only)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import (
    BASE_STATE_COLUMNS,
    BASELINE_FEATURES,
    BIP_DESCRIPTIONS,
    COUNT_STATE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    PITCH_TYPE_COLUMNS,
    build_feature_groups,
)
from src.model_splits import TRAIN_END, VAL_END, chronological_split

TARGET = "xwobacon"
DESCRIPTION_COL = "description"

FEATURE_GROUPS, FEATURE_GROUP_ORDER = build_feature_groups(include_bat_tracking=True)

MIN_GAIN_RMSE = 0.002


@dataclass
class XwOBAconModelResult:
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
    baseline: XwOBAconModelResult
    steps: list[dict]
    selected_groups: list[str]
    final: XwOBAconModelResult
    kept_results: list[XwOBAconModelResult] = field(default_factory=list)


def xwobacon_target_definition() -> str:
    return (
        f"Target `{TARGET}` = Statcast `estimated_woba_using_speedangle` on balls in play "
        f"({', '.join(sorted(BIP_DESCRIPTIONS))}); NaN otherwise. "
        "Training rows are **contact only** with a non-null target."
    )


def contact_only(frame: pd.DataFrame) -> pd.DataFrame:
    if DESCRIPTION_COL not in frame.columns:
        raise ValueError(f"Missing {DESCRIPTION_COL} column")
    bip = frame[frame[DESCRIPTION_COL].isin(BIP_DESCRIPTIONS)].copy()
    return bip.loc[bip[TARGET].notna()].copy()


def _numeric_columns(feature_cols: list[str]) -> list[str]:
    return [col for col in feature_cols if col in NUMERIC_FEATURE_COLUMNS]


def make_xwobacon_pipeline(feature_cols: list[str]) -> Pipeline:
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
            ("reg", LinearRegression()),
        ]
    )


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_and_evaluate(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    *,
    name: str,
) -> XwOBAconModelResult:
    """Fit on train BIP; score on validation BIP (August holdout)."""
    missing = [col for col in feature_cols + [TARGET, DESCRIPTION_COL] if col not in train.columns]
    if missing:
        raise ValueError(f"Missing columns in training frame: {missing}")

    train_bip = contact_only(train)
    val_bip = contact_only(val)

    x_train = train_bip[feature_cols]
    y_train = train_bip[TARGET].astype(float)
    x_val = val_bip[feature_cols]
    y_val = val_bip[TARGET].astype(float)

    model = make_xwobacon_pipeline(feature_cols)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)

    return XwOBAconModelResult(
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
) -> XwOBAconModelResult:
    """Train on Mar–Jul BIP; evaluate on locked September holdout."""
    missing = [col for col in feature_cols + [TARGET, DESCRIPTION_COL] if col not in train.columns]
    if missing:
        raise ValueError(f"Missing columns in training frame: {missing}")

    train_bip = contact_only(train)
    test_bip = contact_only(test)

    x_train = train_bip[feature_cols]
    y_train = train_bip[TARGET].astype(float)
    x_test = test_bip[feature_cols]
    y_test = test_bip[TARGET].astype(float)

    model = make_xwobacon_pipeline(feature_cols)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    return XwOBAconModelResult(
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
    """Greedy group selection: fit on train BIP, keep/drop by August validation RMSE."""
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


def coefficient_frame(result: XwOBAconModelResult) -> pd.DataFrame:
    prep = result.model.named_steps["prep"]
    reg = result.model.named_steps["reg"]
    try:
        raw_names = prep.get_feature_names_out()
    except AttributeError:
        raw_names = result.features
    cleaned = []
    for name in raw_names:
        if name.startswith("num__"):
            cleaned.append(name.removeprefix("num__"))
        elif name.startswith("pass__"):
            cleaned.append(name.removeprefix("pass__"))
        else:
            cleaned.append(name)
    coefs = reg.coef_.ravel()
    out = pd.DataFrame({"feature": cleaned, "coefficient": coefs})
    out["abs_coef"] = out["coefficient"].abs()
    return out.sort_values("abs_coef", ascending=False).reset_index(drop=True)


__all__ = [
    "BASELINE_FEATURES",
    "FEATURE_GROUP_ORDER",
    "FEATURE_GROUPS",
    "MIN_GAIN_RMSE",
    "TARGET",
    "TRAIN_END",
    "VAL_END",
    "ForwardSelectReport",
    "XwOBAconModelResult",
    "chronological_split",
    "coefficient_frame",
    "contact_only",
    "forward_select_feature_groups",
    "make_xwobacon_pipeline",
    "results_table",
    "train_and_evaluate",
    "train_master_model",
    "xwobacon_target_definition",
]

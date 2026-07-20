"""Rebuild modeling frame and retrain production RF models (05 / 07 / 09)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import BASELINE_FEATURES, build_modeling_frame, feature_groups_present
from src.model_splits import TRAIN_END, VAL_END, chronological_split
from src.rf_modeling import (
    feature_importance_from_model,
    forward_select_feature_groups as swing_forward_select,
    kept_features_table as swing_kept_table,
    results_table as swing_results_table,
    train_master_model as train_swing_master,
)
from src.rf_whiff_modeling import (
    forward_select_feature_groups as whiff_forward_select,
    kept_features_table as whiff_kept_table,
    results_table as whiff_results_table,
    swings_only,
    train_master_model as train_whiff_master,
)
from src.rf_xwobacon_modeling import (
    combined_feature_columns,
    contact_only,
    feature_importance_from_model as xwobacon_feature_importance,
    train_master_model as train_xwobacon_master,
)


def rebuild_modeling_frame() -> pd.DataFrame:
    pre = pd.read_parquet(ROOT / "data" / "preprocessed_2025.parquet")
    frame, _ = build_modeling_frame(pre)
    out = ROOT / "data" / "modeling_frame_2025.parquet"
    frame.to_parquet(out, index=False)
    print(f"saved modeling frame -> {out} ({len(frame):,} rows)")
    return frame


def save_top_importance(model, features, path: Path, title: str) -> None:
    step = model.named_steps.get("reg") or model.named_steps.get("clf")
    if step is None:
        raise KeyError("Expected pipeline step 'reg' or 'clf'")
    importances = step.feature_importances_
    top15 = pd.DataFrame({"feature": features, "importance": importances}).sort_values(
        "importance", ascending=False
    ).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top15["feature"][::-1], top15["importance"][::-1], color="#6366f1")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved plot -> {path}")


def train_swing(frame: pd.DataFrame) -> dict:
    train, val, test = chronological_split(frame)
    report = swing_forward_select(train, val, verbose=True)
    selected = report.final.features
    baseline = train_swing_master(train, test, BASELINE_FEATURES, name="baseline")
    master = train_swing_master(train, test, selected, name="master")
    print(
        f"swing baseline test AUC={baseline.test_roc_auc:.3f} | "
        f"master test AUC={master.test_roc_auc:.3f} | features={len(selected)}"
    )
    print(f"selected groups: {report.selected_groups}")
    print(swing_results_table(report).to_string(index=False))
    artifact = {
        "model": master.model,
        "features": selected,
        "target": "is_swing",
        "train_end": str(TRAIN_END.date()),
        "val_end": str(VAL_END.date()),
        "test_log_loss": master.test_log_loss,
        "test_roc_auc": master.test_roc_auc,
        "test_brier": master.test_brier,
        "baseline_features": BASELINE_FEATURES,
        "selected_groups": report.selected_groups,
    }
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    joblib.dump(artifact, models / "swing_rf_master.joblib")
    save_top_importance(
        master.model,
        selected,
        ROOT / "data" / "rolling_metric_plots" / "swing_rf_feature_importance.png",
        "Top 15 feature importances — swing RF",
    )
    return artifact


def train_whiff(frame: pd.DataFrame) -> dict:
    train, val, test = chronological_split(frame)
    report = whiff_forward_select(train, val, verbose=True)
    selected = report.final.features
    baseline = train_whiff_master(train, test, BASELINE_FEATURES, name="baseline")
    master = train_whiff_master(train, test, selected, name="master")
    print(
        f"whiff baseline test AUC={baseline.test_roc_auc:.3f} | "
        f"master test AUC={master.test_roc_auc:.3f} | features={len(selected)}"
    )
    print(f"selected groups: {report.selected_groups}")
    print(whiff_results_table(report).to_string(index=False))
    artifact = {
        "model": master.model,
        "features": selected,
        "target": "is_whiff",
        "train_end": str(TRAIN_END.date()),
        "val_end": str(VAL_END.date()),
        "test_log_loss": master.test_log_loss,
        "test_roc_auc": master.test_roc_auc,
        "test_brier": master.test_brier,
        "baseline_features": BASELINE_FEATURES,
        "selected_groups": report.selected_groups,
    }
    models = ROOT / "models"
    joblib.dump(artifact, models / "whiff_rf_master.joblib")
    save_top_importance(
        master.model,
        selected,
        ROOT / "data" / "rolling_metric_plots" / "whiff_rf_feature_importance.png",
        "Top 15 feature importances — whiff RF",
    )
    return artifact


def train_xwobacon(frame: pd.DataFrame) -> dict:
    train, _, test = chronological_split(frame)
    all_features = combined_feature_columns()
    master = train_xwobacon_master(train, test, all_features, name="master")
    print(
        f"xwOBAcon train R2={master.train_r2:.3f} | test R2={master.test_r2:.3f} | "
        f"test RMSE={master.test_rmse:.4f} | features={len(all_features)}"
    )
    artifact = {
        "model": master.model,
        "features": all_features,
        "target": "xwobacon",
        "train_end": str(TRAIN_END.date()),
        "val_end": str(VAL_END.date()),
        "train_r2": master.train_r2,
        "test_rmse": master.test_rmse,
        "test_r2": master.test_r2,
        "feature_selection": "combined_all_groups",
        "baseline_features": BASELINE_FEATURES,
        "selected_groups": feature_groups_present(all_features, include_bat_tracking=True),
    }
    models = ROOT / "models"
    joblib.dump(artifact, models / "xwobacon_rf_master.joblib")
    save_top_importance(
        master.model,
        all_features,
        ROOT / "data" / "rolling_metric_plots" / "xwobacon_rf_feature_importance.png",
        "Top 15 feature importances — xwOBAcon RF",
    )
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(master.y_test, master.y_pred_test, alpha=0.15, s=8, color="#6366f1")
    lims = [0, max(master.y_test.max(), master.y_pred_test.max())]
    ax.plot(lims, lims, "--", color="gray", linewidth=1)
    ax.set_title(f"Predicted vs actual xwOBAcon — Sep test (R2={master.test_r2:.3f})")
    ax.set_xlabel("Actual xwOBAcon")
    ax.set_ylabel("Predicted xwOBAcon")
    fig.tight_layout()
    scatter_path = ROOT / "data" / "rolling_metric_plots" / "xwobacon_rf_pred_vs_actual.png"
    fig.savefig(scatter_path, dpi=120)
    plt.close(fig)
    print(f"saved scatter -> {scatter_path}")
    top = xwobacon_feature_importance(master.model, all_features)
    center_rank = int(top.index[top["feature"] == "center_dist_in"][0] + 1) if "center_dist_in" in top["feature"].values else -1
    print(f"center_dist_in importance rank: {center_rank} / {len(all_features)}")
    return artifact


if __name__ == "__main__":
    frame = rebuild_modeling_frame()
    print("baseline features:", BASELINE_FEATURES)
    print("\n=== SWING (05) ===")
    train_swing(frame)
    print("\n=== WHIFF (07) ===")
    train_whiff(frame)
    print("\n=== xwOBAcon (09) ===")
    train_xwobacon(frame)

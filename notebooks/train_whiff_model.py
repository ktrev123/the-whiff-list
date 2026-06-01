"""Train pitch-level whiff probability classifiers on 2025 Statcast data."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.whiff_features import FEATURE_COLS, chronological_split, engineer_features, filter_modeling_frame

DATA_FILE = ROOT / "data" / "statcast_2025.parquet"
LEADERBOARD_FILE = ROOT / "data" / "whiff_leaderboard_2025.csv"
MODEL_DIR = ROOT / "data" / "model"
MODEL_FILE = MODEL_DIR / "whiff_model.joblib"
METRICS_FILE = MODEL_DIR / "model_metrics.json"
BATTER_PRED_FILE = MODEL_DIR / "batter_predictions.csv"
PITCH_PRED_FILE = MODEL_DIR / "pitch_predictions_test.parquet"
GRID_PRED_FILE = MODEL_DIR / "league_whiff_grid.parquet"

MIN_ABS = 502
TRAIN_END = "August 2025"
TEST_START = "September 2025"


def load_qualified_batters():
    lb = pd.read_csv(LEADERBOARD_FILE)
    return lb.loc[lb["ab"] >= MIN_ABS, "batter"].astype(int).tolist()


def build_models():
    return {
        "logistic_regression": Pipeline(
            [
                (
                    "scale",
                    ColumnTransformer(
                        [("num", StandardScaler(), FEATURE_COLS)],
                        remainder="drop",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=50,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }


def feature_importance(model, model_name):
    if model_name == "random_forest":
        values = model.feature_importances_
    else:
        values = np.abs(model.named_steps["clf"].coef_[0])
    return dict(sorted(zip(FEATURE_COLS, values.round(4).tolist()), key=lambda x: x[1], reverse=True))


def build_prediction_grid(train_df):
    avg_bot = train_df["sz_bot"].mean()
    avg_top = train_df["sz_top"].mean()
    xs = np.linspace(-2.5, 2.5, 50)
    zs = np.linspace(0.5, 4.5, 50)
    rows = []
    for x in xs:
        for z in zs:
            row = pd.Series(
                {
                    "plate_x": x,
                    "plate_z": z,
                    "sz_bot": avg_bot,
                    "sz_top": avg_top,
                    "balls": 2,
                    "strikes": 2,
                    "on_1b": pd.NA,
                    "on_2b": pd.NA,
                    "on_3b": pd.NA,
                    "description": "placeholder",
                }
            )
            rows.append(row)
    return engineer_features(pd.DataFrame(rows))


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing {DATA_FILE}. Run notebooks/statcast_test_pull.py first."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(DATA_FILE)
    raw["game_date"] = pd.to_datetime(raw["game_date"])
    raw = raw[(raw["game_date"] >= "2025-03-23") & (raw["game_date"] <= "2025-09-27")]

    qualified = load_qualified_batters()
    frame = filter_modeling_frame(raw, qualified)
    train_df, test_df = chronological_split(frame)

    x_train = train_df[FEATURE_COLS]
    y_train = train_df["is_whiff"]
    x_test = test_df[FEATURE_COLS]
    y_test = test_df["is_whiff"]

    print(f"Training pitches (Apr–Aug): {len(train_df):,}")
    print(f"Test pitches (Sep):         {len(test_df):,}")
    print(f"Train whiff rate:           {y_train.mean():.3%}")
    print(f"Test whiff rate:            {y_test.mean():.3%}")

    results = {}
    best_name = None
    best_auc = -1.0
    best_model = None

    for name, model in build_models().items():
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        loss = log_loss(y_test, probs)
        results[name] = {
            "roc_auc": round(float(auc), 4),
            "log_loss": round(float(loss), 4),
            "feature_importance": feature_importance(model, name),
        }
        print(f"{name}: ROC-AUC={auc:.4f}, Log Loss={loss:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = model

    joblib.dump({"model": best_model, "model_name": best_name, "features": FEATURE_COLS}, MODEL_FILE)

    metrics_payload = {
        "statistical_question": (
            "Given an incoming pitch's location, count, and situational context, "
            "can we estimate the probability of a hitter swinging and missing?"
        ),
        "target": "Y=1 for swinging strike / missed bunt; Y=0 for all other pitch outcomes",
        "validation": {
            "strategy": "chronological split",
            "train_period": TRAIN_END,
            "test_period": TEST_START,
            "qualified_batters_min_ab": MIN_ABS,
        },
        "selected_model": best_name,
        "models": results,
    }
    METRICS_FILE.write_text(json.dumps(metrics_payload, indent=2))

    test_scored = test_df.copy()
    test_scored["pred_whiff_prob"] = best_model.predict_proba(x_test)[:, 1]
    test_scored.to_parquet(PITCH_PRED_FILE, index=False)

    batter_preds = (
        test_scored.groupby("batter", as_index=False)
        .agg(
            mean_pred_whiff=("pred_whiff_prob", "mean"),
            pitches=("pred_whiff_prob", "size"),
            actual_whiff_rate=("is_whiff", "mean"),
        )
        .sort_values("mean_pred_whiff", ascending=False)
    )
    batter_preds.to_csv(BATTER_PRED_FILE, index=False)

    grid = build_prediction_grid(train_df)
    grid["pred_whiff_prob"] = best_model.predict_proba(grid[FEATURE_COLS])[:, 1]
    grid.to_parquet(GRID_PRED_FILE, index=False)

    print(f"\nSelected model: {best_name}")
    print(f"Saved model to {MODEL_FILE}")
    print(f"Saved metrics to {METRICS_FILE}")


if __name__ == "__main__":
    main()

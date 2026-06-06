"""Train pitch-level swing and whiff (given swing) classifiers on 2025 Statcast data."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_insights import (
    EXAMPLE_SCENARIOS,
    combined_output_summary,
    friendly_feature_name,
    interpret_log_loss,
    interpret_roc_auc,
    interpret_swing_probability_plain,
    interpret_whiff_probability_plain,
    swing_output_summary,
    validation_summary,
    whiff_output_summary,
)
from src.model_viz import export_training_report
from src.whiff_features import (
    MODEL_INPUT_COLS,
    NUMERIC_FEATURE_COLS,
    CATEGORICAL_FEATURE_COLS,
    PITCH_METRIC_COLS,
    SEASON_END,
    SEASON_START,
    apply_pitch_imputation,
    chronological_split,
    compute_pitch_medians,
    engineer_features,
    filter_modeling_frame,
    pitch_profile_defaults,
)

DATA_FILE = ROOT / "data" / "statcast_2025.parquet"
LEADERBOARD_FILE = ROOT / "data" / "whiff_leaderboard_2025.csv"
MODEL_DIR = ROOT / "data" / "model"
SWING_MODEL_FILE = MODEL_DIR / "swing_model.joblib"
WHIFF_MODEL_FILE = MODEL_DIR / "whiff_model.joblib"
METRICS_FILE = MODEL_DIR / "model_metrics.json"
INSIGHTS_FILE = MODEL_DIR / "model_insights.json"
BATTER_PRED_FILE = MODEL_DIR / "batter_predictions.csv"
PITCH_PRED_FILE = MODEL_DIR / "pitch_predictions_test.parquet"
SWING_GRID_FILE = MODEL_DIR / "league_swing_grid.parquet"
WHIFF_GRID_FILE = MODEL_DIR / "league_whiff_grid.parquet"
PITCH_LAB_SWING_FILE = MODEL_DIR / "pitch_lab_swing.joblib"
PITCH_LAB_WHIFF_FILE = MODEL_DIR / "pitch_lab_whiff.joblib"
PITCH_LAB_PROFILES_FILE = MODEL_DIR / "pitch_lab_profiles.json"
PITCH_LAB_ZONES_FILE = MODEL_DIR / "pitch_lab_hitter_zones.csv"
PITCH_LAB_RATES_FILE = MODEL_DIR / "pitch_lab_hitter_rates.csv"
PITCH_LAB_LEAGUE_RATES_FILE = MODEL_DIR / "pitch_lab_league_rates.json"

MIN_ABS = 502
TRAIN_END = "April–August 2025"
TEST_START = "September 2025"


def load_qualified_batters():
    lb = pd.read_csv(LEADERBOARD_FILE)
    return lb.loc[lb["ab"] >= MIN_ABS, "batter"].astype(int).tolist()


def build_preprocessor():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURE_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURE_COLS),
        ]
    )


def build_models():
    preprocessor = build_preprocessor()
    return {
        "logistic_regression": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=12,
                        min_samples_leaf=50,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def encoded_feature_names(model) -> list[str]:
    return model.named_steps["prep"].get_feature_names_out().tolist()


def feature_importance(model, model_name):
    names = encoded_feature_names(model)
    if model_name == "random_forest":
        values = model.named_steps["clf"].feature_importances_
    else:
        values = np.abs(model.named_steps["clf"].coef_[0])
    ranked = sorted(zip(names, values), key=lambda x: x[1], reverse=True)
    out = {}
    for name, val in ranked:
        label = friendly_feature_name(name)
        if "effective velocity" in label.lower() or "effective_speed" in name.lower():
            continue
        out[label] = round(float(val), 4)
    return out


def build_prediction_grid(train_df, pitch_type="FF"):
    avg_bot = float(train_df["sz_bot"].mean())
    avg_top = float(train_df["sz_top"].mean())
    pitch_defaults = pitch_profile_defaults(train_df, pitch_type)
    xs = np.linspace(-2.5, 2.5, 50)
    zs = np.linspace(0.5, 4.5, 50)
    rows = []
    for x in xs:
        for z in zs:
            rows.append(
                {
                    "plate_x": x,
                    "plate_z": z,
                    "sz_bot": avg_bot,
                    "sz_top": avg_top,
                    "balls": 2,
                    "strikes": 2,
                    "pitch_type": pitch_type,
                    "on_1b": pd.NA,
                    "on_2b": pd.NA,
                    "on_3b": pd.NA,
                    "description": "placeholder",
                    **pitch_defaults,
                }
            )
    grid = engineer_features(pd.DataFrame(rows))
    medians = compute_pitch_medians(train_df)
    return apply_pitch_imputation(grid, medians)


def build_calibration_bins(scored: pd.DataFrame, prob_col: str, target_col: str, n_bins: int = 10) -> list[dict]:
    bins = scored.copy()
    bins["prob_bin"] = pd.qcut(bins[prob_col], n_bins, duplicates="drop")
    cal = (
        bins.groupby("prob_bin", observed=True)
        .agg(
            pred_mean=(prob_col, "mean"),
            actual_rate=(target_col, "mean"),
            count=(target_col, "size"),
        )
        .reset_index(drop=True)
    )
    return [
        {
            "pred_mean": round(float(row.pred_mean), 4),
            "actual_rate": round(float(row.actual_rate), 4),
            "count": int(row.count),
        }
        for row in cal.itertuples()
    ]


def export_pitch_lab_bundle(model, model_name, target, pitch_medians, path):
    joblib.dump(
        {
            "model": model,
            "model_name": model_name,
            "features": MODEL_INPUT_COLS,
            "pitch_medians": pitch_medians.to_dict(),
            "target": target,
        },
        path,
    )


def fit_logistic_model(x_train, y_train):
    preprocessor = build_preprocessor()
    model = Pipeline(
        [
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    return model


def export_pitch_lab_artifacts(train_df, train_swings, pitch_medians, qualified):
    swing_lr = fit_logistic_model(train_df[MODEL_INPUT_COLS], train_df["is_swing"])
    whiff_lr = fit_logistic_model(
        train_swings[MODEL_INPUT_COLS],
        train_swings["is_whiff"],
    )
    export_pitch_lab_bundle(
        swing_lr,
        "logistic_regression",
        "is_swing",
        pitch_medians,
        PITCH_LAB_SWING_FILE,
    )
    export_pitch_lab_bundle(
        whiff_lr,
        "logistic_regression",
        "is_whiff_given_swing",
        pitch_medians,
        PITCH_LAB_WHIFF_FILE,
    )

    profiles = (
        train_df.groupby("pitch_type", as_index=False)[PITCH_METRIC_COLS]
        .median(numeric_only=True)
        .set_index("pitch_type")
        .to_dict(orient="index")
    )
    profiles_payload = {
        "league_sz_bot": round(float(train_df["sz_bot"].median()), 4),
        "league_sz_top": round(float(train_df["sz_top"].median()), 4),
        "pitch_types": {
            pitch_type: {col: round(float(val), 4) for col, val in metrics.items()}
            for pitch_type, metrics in profiles.items()
        },
    }
    PITCH_LAB_PROFILES_FILE.write_text(
        json.dumps(profiles_payload, indent=2),
        encoding="utf-8",
    )

    zones = (
        train_df[train_df["batter"].isin(qualified)]
        .groupby("batter", as_index=False)
        .agg(sz_bot=("sz_bot", "median"), sz_top=("sz_top", "median"))
    )
    zones["sz_bot"] = zones["sz_bot"].round(4)
    zones["sz_top"] = zones["sz_top"].round(4)
    zones.to_csv(PITCH_LAB_ZONES_FILE, index=False)

    from src.hitter_rates import export_hitter_rates_from_frame

    export_hitter_rates_from_frame(
        train_df[train_df["batter"].isin(qualified)],
        PITCH_LAB_RATES_FILE,
        PITCH_LAB_LEAGUE_RATES_FILE,
    )


def train_best_model(x_train, y_train, x_test, y_test):
    results = {}
    best_name = None
    best_auc = -1.0
    best_model = None
    best_probs = None

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
        print(f"  {name}: ROC-AUC={auc:.4f}, Log Loss={loss:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = model
            best_probs = probs

    return best_name, best_model, results, best_probs


def build_model_insights(
    target_key: str,
    outcome_label: str,
    what_it_outputs: str,
    best_name: str,
    results: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_test: pd.Series,
    probs: np.ndarray,
    prob_col: str,
    target_col: str,
) -> dict:
    base_rate = float(y_test.mean())
    auc = results[best_name]["roc_auc"]
    loss = results[best_name]["log_loss"]
    fpr, tpr, _ = roc_curve(y_test, probs)
    importance = results[best_name]["feature_importance"]

    return {
        "target": target_key,
        "outcome_label": outcome_label,
        "selected_model": best_name,
        "n_train_pitches": int(len(train_df)),
        "n_test_pitches": int(len(test_df)),
        "test_positive_rate": round(base_rate, 4),
        "roc_auc": auc,
        "log_loss": loss,
        "layman": {
            "what_it_outputs": what_it_outputs,
            "roc_auc": interpret_roc_auc(auc, outcome_label),
            "log_loss": interpret_log_loss(loss, base_rate, outcome_label),
        },
        "feature_importance": [
            {"feature": feat, "label": feat, "importance": float(val)}
            for feat, val in importance.items()
        ],
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(x), 4) for x in tpr],
        },
        "calibration_bins": build_calibration_bins(
            test_df.assign(**{prob_col: probs}),
            prob_col=prob_col,
            target_col=target_col,
        ),
    }


def build_example_scenarios(swing_model, whiff_model, train_df, medians) -> list[dict]:
    avg_bot = float(train_df["sz_bot"].mean())
    avg_top = float(train_df["sz_top"].mean())
    rows = []
    for scenario in EXAMPLE_SCENARIOS:
        on_1b = 1 if scenario["runners_on"] >= 1 else pd.NA
        on_2b = 2 if scenario["runners_on"] >= 2 else pd.NA
        on_3b = 3 if scenario["runners_on"] >= 3 else pd.NA
        pitch_defaults = pitch_profile_defaults(train_df, scenario["pitch_type"])
        rows.append(
            {
                "plate_x": scenario["plate_x"],
                "plate_z": scenario["plate_z"],
                "balls": scenario["balls"],
                "strikes": scenario["strikes"],
                "pitch_type": scenario["pitch_type"],
                "sz_bot": avg_bot,
                "sz_top": avg_top,
                "on_1b": on_1b,
                "on_2b": on_2b,
                "on_3b": on_3b,
                "description": "placeholder",
                **pitch_defaults,
            }
        )
    frame = apply_pitch_imputation(engineer_features(pd.DataFrame(rows)), medians)
    swing_probs = swing_model.predict_proba(frame[MODEL_INPUT_COLS])[:, 1]
    whiff_probs = whiff_model.predict_proba(frame[MODEL_INPUT_COLS])[:, 1]

    examples = []
    for scenario, p_swing, p_whiff in zip(EXAMPLE_SCENARIOS, swing_probs, whiff_probs):
        p_combined = float(p_swing * p_whiff)
        examples.append(
            {
                "label": scenario["label"],
                "pitch_type": scenario["pitch_type"],
                "count": f"{scenario['balls']}-{scenario['strikes']}",
                "runners_on": scenario["runners_on"],
                "swing_prob_pct": round(float(p_swing) * 100, 1),
                "whiff_if_swing_pct": round(float(p_whiff) * 100, 1),
                "swing_whiff_pct": round(p_combined * 100, 1),
                "swing_takeaway": interpret_swing_probability_plain(float(p_swing)),
                "whiff_takeaway": interpret_whiff_probability_plain(float(p_whiff)),
            }
        )
    return examples


def main():
    parser = argparse.ArgumentParser(description="Train swing and whiff models.")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Save HTML report without opening a browser tab.",
    )
    args = parser.parse_args()

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing {DATA_FILE}. Run notebooks/statcast_pull.py first.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(DATA_FILE)
    raw["game_date"] = pd.to_datetime(raw["game_date"])
    raw = raw[(raw["game_date"] >= SEASON_START) & (raw["game_date"] <= SEASON_END)]

    qualified = load_qualified_batters()
    frame = filter_modeling_frame(raw, qualified)
    train_df, test_df = chronological_split(frame)
    pitch_medians = compute_pitch_medians(train_df)
    train_df = apply_pitch_imputation(train_df, pitch_medians)
    test_df = apply_pitch_imputation(test_df, pitch_medians)

    train_swings = train_df[train_df["is_swing"] == 1].copy()
    test_swings = test_df[test_df["is_swing"] == 1].copy()

    print(f"Features: {MODEL_INPUT_COLS}")

    print(f"All pitches — train: {len(train_df):,} | test: {len(test_df):,}")
    print(f"Swing rate — train: {train_df['is_swing'].mean():.3%} | test: {test_df['is_swing'].mean():.3%}")
    print(
        f"Whiff rate (if swung) — train: {train_swings['is_whiff'].mean():.3%} | "
        f"test: {test_swings['is_whiff'].mean():.3%}"
    )

    print("\n=== Model A: Swing (all pitches) ===")
    swing_name, swing_model, swing_results, swing_probs = train_best_model(
        train_df[MODEL_INPUT_COLS],
        train_df["is_swing"],
        test_df[MODEL_INPUT_COLS],
        test_df["is_swing"],
    )

    print("\n=== Model B: Whiff (swings only) ===")
    whiff_name, whiff_model, whiff_results, whiff_probs = train_best_model(
        train_swings[MODEL_INPUT_COLS],
        train_swings["is_whiff"],
        test_swings[MODEL_INPUT_COLS],
        test_swings["is_whiff"],
    )

    joblib.dump(
        {
            "model": swing_model,
            "model_name": swing_name,
            "features": MODEL_INPUT_COLS,
            "pitch_medians": pitch_medians.to_dict(),
            "target": "is_swing",
        },
        SWING_MODEL_FILE,
    )
    joblib.dump(
        {
            "model": whiff_model,
            "model_name": whiff_name,
            "features": MODEL_INPUT_COLS,
            "pitch_medians": pitch_medians.to_dict(),
            "target": "is_whiff_given_swing",
        },
        WHIFF_MODEL_FILE,
    )

    swing_insights = build_model_insights(
        "swing",
        "swing",
        swing_output_summary(),
        swing_name,
        swing_results,
        train_df,
        test_df,
        test_df["is_swing"],
        swing_probs,
        "pred_swing_prob",
        "is_swing",
    )
    whiff_insights = build_model_insights(
        "whiff",
        "whiff",
        whiff_output_summary(),
        whiff_name,
        whiff_results,
        train_swings,
        test_swings,
        test_swings["is_whiff"],
        whiff_probs,
        "pred_whiff_prob",
        "is_whiff",
    )
    whiff_insights["training_note"] = "Trained only on pitches where the batter swung."

    insights_payload = {
        "train_period": TRAIN_END,
        "test_period": TEST_START,
        "layman": {
            "validation": validation_summary(TRAIN_END, TEST_START, len(train_df), len(test_df)),
            "combined": combined_output_summary(),
        },
        "swing": swing_insights,
        "whiff": whiff_insights,
        "example_pitches": build_example_scenarios(swing_model, whiff_model, train_df, pitch_medians),
    }
    insights_payload["feature_columns"] = MODEL_INPUT_COLS
    INSIGHTS_FILE.write_text(json.dumps(insights_payload, indent=2), encoding="utf-8")

    metrics_payload = {
        "statistical_question": (
            "Given pitch characteristics and situational context, estimate "
            "(A) swing probability and (B) whiff probability given a swing."
        ),
        "validation": {
            "strategy": "chronological split",
            "train_period": TRAIN_END,
            "test_period": TEST_START,
            "qualified_batters_min_ab": MIN_ABS,
        },
        "swing_model": {
            "selected": swing_name,
            "candidates": swing_results,
            "features": MODEL_INPUT_COLS,
        },
        "whiff_model": {
            "selected": whiff_name,
            "candidates": whiff_results,
            "trained_on": "swings_only",
            "features": MODEL_INPUT_COLS,
        },
    }
    METRICS_FILE.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    test_scored = test_df.copy()
    test_scored["pred_swing_prob"] = swing_probs
    test_scored["pred_whiff_prob"] = whiff_model.predict_proba(test_df[MODEL_INPUT_COLS])[:, 1]
    test_scored["pred_swing_whiff_prob"] = (
        test_scored["pred_swing_prob"] * test_scored["pred_whiff_prob"]
    )
    test_scored.to_parquet(PITCH_PRED_FILE, index=False)

    batter_preds = (
        test_scored.groupby("batter", as_index=False)
        .agg(
            mean_pred_swing=("pred_swing_prob", "mean"),
            mean_pred_whiff=("pred_whiff_prob", "mean"),
            mean_pred_swing_whiff=("pred_swing_whiff_prob", "mean"),
            pitches=("pred_swing_prob", "size"),
            actual_swing_rate=("is_swing", "mean"),
            actual_whiff_rate=("is_whiff", "mean"),
        )
        .sort_values("mean_pred_swing_whiff", ascending=False)
    )
    batter_preds.to_csv(BATTER_PRED_FILE, index=False)

    export_pitch_lab_artifacts(train_df, train_swings, pitch_medians, qualified)
    print(f"Saved Pitch Lab deploy bundle: {PITCH_LAB_SWING_FILE.name}, {PITCH_LAB_WHIFF_FILE.name}")

    grid = build_prediction_grid(train_df, pitch_type="FF")
    grid["pred_swing_prob"] = swing_model.predict_proba(grid[MODEL_INPUT_COLS])[:, 1]
    grid["pred_whiff_prob"] = whiff_model.predict_proba(grid[MODEL_INPUT_COLS])[:, 1]
    grid["pred_swing_whiff_prob"] = grid["pred_swing_prob"] * grid["pred_whiff_prob"]
    grid.to_parquet(SWING_GRID_FILE, index=False)
    grid.to_parquet(WHIFF_GRID_FILE, index=False)

    name_lookup = pd.read_csv(LEADERBOARD_FILE)[["batter", "player_name"]].drop_duplicates()
    report_path = export_training_report(
        insights_payload,
        batter_preds,
        grid,
        grid,
        name_lookup,
        MODEL_DIR,
        open_browser=not args.no_browser,
    )

    print(f"\nSelected swing model: {swing_name}")
    print(f"Selected whiff model: {whiff_name}")
    print(f"Saved insights to {INSIGHTS_FILE}")
    print(f"Opened model report: {report_path}")
    print(f"Individual charts: {MODEL_DIR / 'charts'}")


if __name__ == "__main__":
    main()

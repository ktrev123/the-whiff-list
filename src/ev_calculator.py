"""Expected run value (ERV) from the RF probability chain (swing → whiff → xwOBAcon)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import (
    BASELINE_FEATURES,
    BAT_TRACKING_COLUMNS,
    BATTER_ROLLING_COLUMNS,
    CONTACT_FEATURE_GROUP_ORDER,
    PITCH_TYPE_COLUMNS,
    SWING_FEATURE_GROUP_ORDER,
    feature_groups_present,
)

# Run-value constants (runs above average per pitch outcome).
RV_STRIKE = -0.065
RV_BALL = 0.035
RV_WHIFF = -0.095

LEAGUE_WOBA = 0.320
WOBA_SCALE = 1.15

DEFAULT_MODELS_DIR = ROOT / "models"
DEFAULT_INPUT = ROOT / "data" / "modeling_frame_2025.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "ev_scored_pitches.parquet"

SWING_ARTIFACT_NAME = "swing_rf_master.joblib"
WHIFF_ARTIFACT_NAME = "whiff_rf_master.joblib"
XWOBA_ARTIFACT_NAME = "xwobacon_rf_master.joblib"

EV_MODEL_LABELS = ("swing", "whiff", "xwobacon")

COUNT_TWO_STRIKE_COLUMNS = [
    "count_0_2",
    "count_1_2",
    "count_2_2",
    "count_3_2",
]
BREAKING_BALL_COLUMNS = [
    "pitch_CU",
    "pitch_SL",
    "pitch_ST",
    "pitch_KC",
    "pitch_SV",
]


def load_ev_artifacts(models_dir: Path | str | None = None) -> tuple[dict, dict, dict]:
    """Load Model A (swing), B (whiff), and C (xwOBAcon) master artifacts."""
    base = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    swing = joblib.load(base / SWING_ARTIFACT_NAME)
    whiff = joblib.load(base / WHIFF_ARTIFACT_NAME)
    xwobacon = joblib.load(base / XWOBA_ARTIFACT_NAME)
    return swing, whiff, xwobacon


def describe_ev_artifacts(
    artifacts: tuple[dict, dict, dict] | None = None,
    *,
    models_dir: Path | str | None = None,
) -> pd.DataFrame:
    """
    Summarize the three production RF artifacts used by the EV chain.

    Returns one row per model with feature counts, baseline overlap, and
    inferred forward-selected / kitchen-sink feature groups.
    """
    if artifacts is None:
        artifacts = load_ev_artifacts(models_dir)
    swing, whiff, xwobacon = artifacts

    rows: list[dict] = []
    for label, art, contact in [
        ("swing", swing, False),
        ("whiff", whiff, True),
        ("xwobacon", xwobacon, True),
    ]:
        features = list(art.get("features", []))
        saved_groups = art.get("selected_groups")
        if saved_groups is None and art.get("feature_selection") == "combined_all_groups":
            saved_groups = feature_groups_present(features, include_bat_tracking=True)
        elif saved_groups is None:
            saved_groups = feature_groups_present(features, include_bat_tracking=contact)

        rows.append(
            {
                "model": label,
                "target": art.get("target", "?"),
                "n_features": len(features),
                "baseline_ok": all(col in features for col in BASELINE_FEATURES),
                "uses_batter_te": "batter_te" in features,
                "uses_rolling": all(col in features for col in BATTER_ROLLING_COLUMNS),
                "uses_bat_tracking": all(col in features for col in BAT_TRACKING_COLUMNS),
                "feature_groups": " → ".join(saved_groups) if saved_groups else "(baseline only)",
                "train_end": art.get("train_end"),
                "val_end": art.get("val_end"),
            }
        )
    return pd.DataFrame(rows)


def _feature_matrix(frame: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Build the model feature matrix from pre-engineered columns."""
    features = list(artifact["features"])
    missing = [col for col in features if col not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing feature columns for {artifact.get('target', 'model')}: {missing}"
        )

    # Legacy artifacts may still list batter_te and require a saved encoder.
    batter_col = artifact.get("batter_feature_col")
    encoder = artifact.get("batter_encoder")
    pooled_col = artifact.get("pooled_batter_col")
    if batter_col and batter_col in features and encoder is not None and pooled_col:
        non_batter = [col for col in features if col != batter_col]
        x = frame[non_batter].copy()
        x[batter_col] = encoder.transform(frame[[pooled_col]]).values.ravel().astype(np.float32)
        return x[features]

    return frame[features]


def _predict_classifier(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    x = _feature_matrix(frame, artifact)
    return artifact["model"].predict_proba(x)[:, 1]


def _predict_regressor(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    x = _feature_matrix(frame, artifact)
    return artifact["model"].predict(x)


def calculate_expected_run_value(
    frame: pd.DataFrame,
    artifacts: tuple[dict, dict, dict] | None = None,
    *,
    models_dir: Path | str | None = None,
) -> pd.DataFrame:
    """
    Score every pitch with the swing → whiff → xwOBAcon chain and compute ERV.

    Each artifact expects the same pre-engineered columns as notebooks 05/07/09:
    baseline location features plus forward-selected (or kitchen-sink) groups
    with season-to-date rolling batter metrics — no ``batter_te`` at inference.

    Returns a copy of ``frame`` with probability columns and ``expected_run_value``.
    """
    if artifacts is None:
        artifacts = load_ev_artifacts(models_dir)
    swing_art, whiff_art, xwobacon_art = artifacts

    out = frame.copy()

    out["p_swing"] = _predict_classifier(swing_art, out)
    out["p_take"] = 1.0 - out["p_swing"]
    out["p_whiff_given_swing"] = _predict_classifier(whiff_art, out)
    out["p_contact_given_swing"] = 1.0 - out["p_whiff_given_swing"]
    out["pred_xwobacon"] = _predict_regressor(xwobacon_art, out)

    take_value = np.where(out["is_in_zone"].eq(1), RV_STRIKE, RV_BALL)
    contact_run_value = (out["pred_xwobacon"] - LEAGUE_WOBA) / WOBA_SCALE
    swing_value = (
        out["p_whiff_given_swing"] * RV_WHIFF
        + out["p_contact_given_swing"] * contact_run_value
    )
    out["expected_run_value"] = out["p_take"] * take_value + out["p_swing"] * swing_value

    return out


def _pitch_type_label(row: pd.Series) -> str:
    if "pitch_type" in row.index and pd.notna(row["pitch_type"]):
        return str(row["pitch_type"])
    for col in PITCH_TYPE_COLUMNS:
        if col in row.index and row[col] == 1:
            return col.removeprefix("pitch_")
    return "?"


def _count_label(row: pd.Series) -> str:
    for balls in range(4):
        for strikes in range(3):
            col = f"count_{balls}_{strikes}"
            if col in row.index and row[col] == 1:
                return f"{balls}-{strikes}"
    return "?"


def _format_profile_snippet(row: pd.Series) -> str:
    parts: list[str] = []
    for col in BATTER_ROLLING_COLUMNS:
        if col in row.index and pd.notna(row[col]):
            parts.append(f"{col}={row[col]:.3f}")
    for col in BAT_TRACKING_COLUMNS:
        if col in row.index and pd.notna(row[col]):
            if col == "squared_up_rate":
                parts.append(f"{col}={row[col]:.3f}")
            else:
                parts.append(f"{col}={row[col]:.1f}")
    return ", ".join(parts) if parts else "(not in frame)"


def _print_pitch_example(label: str, row: pd.Series) -> None:
    in_zone = int(row["is_in_zone"]) == 1
    take_rv = RV_STRIKE if in_zone else RV_BALL
    take_name = "called strike" if in_zone else "called ball"
    contact_rv = (row["pred_xwobacon"] - LEAGUE_WOBA) / WOBA_SCALE
    swing_rv = (
        row["p_whiff_given_swing"] * RV_WHIFF
        + row["p_contact_given_swing"] * contact_rv
    )
    erv = row["expected_run_value"]

    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"{'=' * 72}")
    print(f"  Location   plate_x={row['plate_x']:+.3f} ft  plate_z={row['plate_z']:+.3f} ft")
    print(
        f"             in_zone={int(row['is_in_zone'])}  "
        f"miss_dist_in={row['miss_dist_in']:.2f} in  "
        f"center_dist_in={row.get('center_dist_in', float('nan')):.2f} in  "
        f"count={_count_label(row)}"
    )
    print(f"  Pitch      type={_pitch_type_label(row)}")
    print(f"  Batter     {_format_profile_snippet(row)}")
    print(f"  Model A    p_swing              = {row['p_swing']:.4f}")
    print(f"  Model B    p_whiff_given_swing  = {row['p_whiff_given_swing']:.4f}")
    print(f"             p_contact_given_swing= {row['p_contact_given_swing']:.4f}")
    print(f"  Model C    pred_xwobacon        = {row['pred_xwobacon']:.4f}")
    print(f"  Take RV    {take_name:14s}     = {take_rv:+.4f}")
    print(f"  Contact RV (pred_xwobacon - {LEAGUE_WOBA}) / {WOBA_SCALE} = {contact_rv:+.4f}")
    print(f"  Swing RV   p_whiff*{RV_WHIFF} + p_contact*contact_rv = {swing_rv:+.4f}")
    print(f"  ERV        p_take*{take_rv:+.4f} + p_swing*{swing_rv:+.4f} = {erv:+.4f}")


def print_verification_examples(
    frame: pd.DataFrame,
    *,
    random_state: int = 42,
) -> None:
    """Print three distinct pitches illustrating the ERV calculation."""
    used: set[int] = set()

    def pick_one(mask: pd.Series, sort_col: str, ascending: bool) -> pd.Series:
        candidates = frame.loc[mask].sort_values(sort_col, ascending=ascending)
        for idx in candidates.index:
            if idx not in used:
                used.add(idx)
                return frame.loc[idx]
        raise ValueError("Could not find a distinct example row for verification.")

    middle = pick_one(frame["is_in_zone"].eq(1), "miss_dist_in", ascending=True)
    outside = pick_one(frame["is_in_zone"].eq(0), "miss_dist_in", ascending=False)

    two_strike = frame[COUNT_TWO_STRIKE_COLUMNS].eq(1).any(axis=1)
    breaking = frame[BREAKING_BALL_COLUMNS].eq(1).any(axis=1)
    breaking_pool = frame.loc[two_strike & breaking]
    if breaking_pool.empty:
        raise ValueError("No breaking-ball two-strike pitches found for verification.")
    shuffled = breaking_pool.sample(frac=1.0, random_state=random_state)
    breaking_row = None
    for idx in shuffled.index:
        if idx not in used:
            used.add(idx)
            breaking_row = frame.loc[idx]
            break
    if breaking_row is None:
        raise ValueError("Could not find a distinct breaking-ball example.")

    print("\nEV verification - three example pitches")
    _print_pitch_example("Example 1 - middle of the zone (low miss_dist_in)", middle)
    _print_pitch_example("Example 2 - far outside the zone (high miss_dist_in)", outside)
    _print_pitch_example("Example 3 - breaking ball in a two-strike count", breaking_row)


def ev_architecture_markdown() -> str:
    """Architecture note shared by notebooks 10 and 11."""
    swing_order = " → ".join(f"`{g}`" for g in SWING_FEATURE_GROUP_ORDER)
    contact_order = " → ".join(f"`{g}`" for g in CONTACT_FEATURE_GROUP_ORDER)
    baseline = ", ".join(f"`{c}`" for c in BASELINE_FEATURES)
    return (
        "## Model chain architecture\n\n"
        "| Model | Artifact | Training sample | Feature selection |\n"
        "|-------|----------|-----------------|-------------------|\n"
        "| **A — Swing** | `swing_rf_master.joblib` | All pitches | Forward selection (Aug val log-loss, MIN_GAIN = 0.0010) |\n"
        "| **B — Whiff** | `whiff_rf_master.joblib` | Swings only | Forward selection (same threshold) |\n"
        "| **C — xwOBAcon** | `xwobacon_rf_master.joblib` | BIP only | Kitchen-sink (all groups combined) |\n\n"
        f"**Baseline (all three):** {baseline}\n\n"
        "**Batter signal:** season-to-date rolling rates in the modeling frame "
        "(`swing_pct`, `whiff_pct`, zone splits). **No `batter_te`.** "
        "Whiff and xwOBAcon also use rolling bat-tracking (`bat_speed`, `attack_angle`, `squared_up_rate`).\n\n"
        f"**Swing group order:** {swing_order}\n\n"
        f"**Whiff / xwOBAcon group order:** {contact_order}\n"
    )


if __name__ == "__main__":
    print(f"Loading {DEFAULT_INPUT} ...")
    pitches = pd.read_parquet(DEFAULT_INPUT)
    print(f"Scoring {len(pitches):,} pitches ...")

    artifacts = load_ev_artifacts()
    print("\nArtifact summary:")
    print(describe_ev_artifacts(artifacts).to_string(index=False))

    scored = calculate_expected_run_value(pitches, artifacts=artifacts)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(DEFAULT_OUTPUT, index=False)
    print(f"\nSaved -> {DEFAULT_OUTPUT}")

    print_verification_examples(scored)

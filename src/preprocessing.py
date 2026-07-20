"""Pitch-level filtering and cleaning before feature engineering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.statcast_schema import filter_competitive_pitches, filter_regular_season

from src.statcast_schema import REQUIRED_STATCAST_COLS, SCHEMA_VERSION

VELOCITY_IQR_MULTIPLIER = 1.5

REQUIRED_RAW_COLUMNS = [
    "game_date",
    "batter",
    "pitcher",
    "pitch_type",
    "p_throws",
    "stand",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "balls",
    "strikes",
    "release_speed",
    "pfx_x",
    "pfx_z",
    "release_spin_rate",
    "release_extension",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "description",
]

PREPROCESS_KEEP_COLS = [
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "batter",
    "pitcher",
    "pitch_type",
    "pitch_name",
    "p_throws",
    "stand",
    "description",
    "events",
    "estimated_woba_using_speedangle",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "balls",
    "strikes",
    "on_1b",
    "on_2b",
    "on_3b",
    "zone",
    "release_speed",
    "effective_speed",
    "pfx_x",
    "pfx_z",
    "release_spin_rate",
    "spin_axis",
    "release_extension",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "bat_speed",
    "attack_angle",
]


@dataclass
class PreprocessReport:
    n_raw: int = 0
    n_regular_season: int = 0
    n_competitive: int = 0
    n_unique_batters: int = 0
    n_velocity_outliers_removed: int = 0
    velocity_outlier_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_invalid_base_state: int = 0
    n_missing_required: int = 0
    n_final: int = 0


def assign_base_state(on_1b: bool, on_2b: bool, on_3b: bool) -> str | None:
    if not on_1b and not on_2b and not on_3b:
        return "state_empty"
    if on_1b and not on_2b and not on_3b:
        return "state_1b"
    if on_1b and on_2b and not on_3b:
        return "state_1b_2b"
    if on_1b and not on_2b and on_3b:
        return "state_1b_3b"
    if not on_1b and on_2b and not on_3b:
        return "state_2b"
    if not on_1b and on_2b and on_3b:
        return "state_2b_3b"
    if not on_1b and not on_2b and on_3b:
        return "state_3b"
    if on_1b and on_2b and on_3b:
        return "state_loaded"
    return None


def base_state_column(frame: pd.DataFrame) -> pd.Series:
    on_1b = frame["on_1b"].notna().to_numpy()
    on_2b = frame["on_2b"].notna().to_numpy()
    on_3b = frame["on_3b"].notna().to_numpy()
    labels = [
        assign_base_state(b1, b2, b3) for b1, b2, b3 in zip(on_1b, on_2b, on_3b, strict=True)
    ]
    return pd.Series(labels, index=frame.index, dtype="object")


def velocity_outlier_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pitch_type, group in frame.groupby("pitch_type", sort=True):
        speeds = pd.to_numeric(group["release_speed"], errors="coerce").dropna()
        if speeds.empty:
            continue
        q1 = speeds.quantile(0.25)
        q3 = speeds.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - VELOCITY_IQR_MULTIPLIER * iqr
        removed = int((speeds < lower_fence).sum())
        min_removed = float(speeds[speeds < lower_fence].min()) if removed else np.nan
        rows.append(
            {
                "pitch_type": pitch_type,
                "n_pitches": len(speeds),
                "q1_mph": round(q1, 2),
                "q3_mph": round(q3, 2),
                "lower_fence_mph": round(lower_fence, 2),
                "removed": removed,
                "min_removed_mph": round(min_removed, 1) if removed else np.nan,
            }
        )
    return pd.DataFrame(rows)


def remove_low_velocity_outliers(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop pitches below the Tukey lower fence on release_speed (per pitch type)."""
    fences: dict[str, float] = {}
    for pitch_type, group in frame.groupby("pitch_type", sort=True):
        speeds = pd.to_numeric(group["release_speed"], errors="coerce").dropna()
        if speeds.empty:
            continue
        q1 = speeds.quantile(0.25)
        q3 = speeds.quantile(0.75)
        fences[pitch_type] = float(q1 - VELOCITY_IQR_MULTIPLIER * (q3 - q1))

    speeds = pd.to_numeric(frame["release_speed"], errors="coerce")
    keep = pd.Series(True, index=frame.index)
    for pitch_type, fence in fences.items():
        mask = frame["pitch_type"] == pitch_type
        keep.loc[mask] = speeds.loc[mask].isna() | (speeds.loc[mask] >= fence)

    summary = velocity_outlier_summary(frame)
    return frame.loc[keep].copy(), summary


def build_preprocessed_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessReport]:
    """Filter and clean pitches for modeling. All batters are kept."""
    report = PreprocessReport(n_raw=len(raw))

    missing_cols = [col for col in REQUIRED_RAW_COLUMNS if col not in raw.columns]
    if missing_cols:
        release_missing = [c for c in missing_cols if c.startswith("release_pos")]
        if release_missing:
            raise ValueError(
                f"Missing required Statcast columns: {release_missing}. "
                f"Your `statcast_2025.parquet` was built with an old pull schema "
                f"(expected schema v{SCHEMA_VERSION} with 3D release point columns). "
                "Re-run the data pull first:\n"
                "  .\\.venv\\Scripts\\python.exe -m src.statcast_pull --force\n"
                "Or in `00_data_pull.ipynb`: set FORCE_PULL = True and run the pull cell."
            ) from None
        raise ValueError(f"Missing required Statcast columns: {missing_cols}")

    season = filter_regular_season(raw)
    report.n_regular_season = len(season)

    competitive = filter_competitive_pitches(season)
    report.n_competitive = len(competitive)
    report.n_unique_batters = int(competitive["batter"].nunique())

    before_velocity = len(competitive)
    after_velocity, velocity_summary = remove_low_velocity_outliers(competitive)
    report.velocity_outlier_summary = velocity_summary
    report.n_velocity_outliers_removed = before_velocity - len(after_velocity)

    base_labels = base_state_column(after_velocity)
    invalid_base = base_labels.isna()
    report.n_invalid_base_state = int(invalid_base.sum())
    after_base = after_velocity.loc[~invalid_base].copy()

    required_mask = after_base[REQUIRED_RAW_COLUMNS].notna().all(axis=1)
    report.n_missing_required = int((~required_mask).sum())
    after_required = after_base.loc[required_mask].copy()

    keep_cols = [col for col in PREPROCESS_KEEP_COLS if col in after_required.columns]
    report.n_final = len(after_required)
    return after_required[keep_cols].reset_index(drop=True), report

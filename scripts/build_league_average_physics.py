"""Build league-average pitch physics lookup by pitch type."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.model_splits import TRAIN_END

PHYSICS_COLS = [
    "release_speed",
    "release_spin_rate",
    "norm_hb",
    "norm_ivb",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "release_extension",
    "p_throws_L",
]
DEFAULT_OUTPUT = Path("data/league_average_physics.csv")


def build_league_average_physics(
    modeling_frame: pd.DataFrame,
    *,
    output_path: Path | str = DEFAULT_OUTPUT,
    train_only: bool = True,
) -> pd.DataFrame:
    """Mean physics / release metrics by pitch_type for pitch simulation in the app."""
    frame = modeling_frame.copy()
    if train_only:
        dates = pd.to_datetime(frame["game_date"])
        frame = frame.loc[dates < TRAIN_END]

    missing = [col for col in ["pitch_type", *PHYSICS_COLS] if col not in frame.columns]
    if missing:
        raise ValueError(f"Modeling frame missing columns: {missing}")

    lookup = (
        frame.groupby("pitch_type", as_index=False)[PHYSICS_COLS]
        .mean(numeric_only=True)
        .sort_values("pitch_type")
        .reset_index(drop=True)
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(output_path, index=False)
    return lookup


if __name__ == "__main__":
    frame = pd.read_parquet(ROOT / "data" / "modeling_frame_2025.parquet")
    lookup = build_league_average_physics(frame, output_path=ROOT / DEFAULT_OUTPUT)
    print(f"Saved {len(lookup)} pitch types -> {ROOT / DEFAULT_OUTPUT}")
    print(lookup.to_string(index=False))

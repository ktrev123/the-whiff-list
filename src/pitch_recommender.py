"""Grid-search pitch recommender using the EV probability chain."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attack_zones import add_attack_zones
from src.ev_calculator import calculate_expected_run_value, load_ev_artifacts
from src.feature_engineering import (
    BASE_STATE_COLUMNS,
    BATTER_PROFILE_COLUMNS,
    COUNT_STATE_COLUMNS,
    PITCH_TYPE_COLUMNS,
    ZONE_COLUMNS,
    center_dist_inches,
    league_default_batter_profile,
    miss_distance_inches,
    rule_book_in_zone,
)
from src.statcast_schema import COMPETITIVE_PITCH_TYPES

# Re-export for app / UI consumers.
__all__ = [
    "COMPETITIVE_PITCH_TYPES",
    "CORE_PITCH_TYPES",
    "RECOMMENDATION_COLUMNS",
    "build_base_state",
    "build_count_state",
    "generate_strike_zone_grid",
    "load_league_average_physics",
    "simulate_plate_appearance",
    "spatial_smooth_erv",
]

CORE_PITCH_TYPES = sorted(COMPETITIVE_PITCH_TYPES)

RECOMMENDATION_COLUMNS = [
    "Rank",
    "Type",
    "plate_x",
    "plate_z",
    "InZ",
    "Swing Prob",
    "Whiff Prob",
    "xwobacon",
    "ERV_raw",
    "ERV",
]

DEFAULT_PHYSICS_PATH = ROOT / "data" / "league_average_physics.csv"

# Standard rule-book plate width; vertical bounds use league-average batter height.
DEFAULT_SZ_TOP = 3.5
DEFAULT_SZ_BOT = 1.5

PLATE_X_MIN = -1.5
PLATE_X_MAX = 1.5
PLATE_Z_MIN = 0.5
PLATE_Z_MAX = 4.5
GRID_STEP_FT = 0.25
SMOOTH_RADIUS_IN = 3.0


def load_league_average_physics(path: Path | str | None = None) -> pd.DataFrame:
    """Load league-average pitch physics by pitch type."""
    return pd.read_csv(path or DEFAULT_PHYSICS_PATH)


def build_count_state(balls: int, strikes: int) -> dict[str, int]:
    """One-hot count dict for ``simulate_plate_appearance``."""
    active = f"count_{balls}_{strikes}"
    if active not in COUNT_STATE_COLUMNS:
        raise ValueError(
            f"Invalid count {balls}-{strikes}. "
            f"balls must be 0-3 and strikes 0-2 for a legal count."
        )
    return {col: int(col == active) for col in COUNT_STATE_COLUMNS}


def build_base_state(active_state: str) -> dict[str, int]:
    """One-hot base-state dict for ``simulate_plate_appearance``."""
    if active_state not in BASE_STATE_COLUMNS:
        raise ValueError(
            f"Unknown base state {active_state!r}. "
            f"Expected one of: {', '.join(BASE_STATE_COLUMNS)}"
        )
    return {col: int(col == active_state) for col in BASE_STATE_COLUMNS}


def validate_stand(stand: str) -> None:
    if stand not in ("R", "L"):
        raise ValueError(
            f"Invalid stand {stand!r}. Must be exactly 'R' (right) or 'L' (left)."
        )


def validate_one_hot_state(
    state: dict[str, int],
    valid_keys: list[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(valid_keys) - set(state.keys()))
    extra = sorted(set(state.keys()) - set(valid_keys))
    if missing:
        raise ValueError(
            f"{label} is missing required keys: {missing}. "
            f"Expected all of: {valid_keys}"
        )
    if extra:
        raise ValueError(
            f"{label} contains unknown keys: {extra}. "
            f"Expected only: {valid_keys}"
        )
    values = list(state.values())
    if not all(v in (0, 1) for v in values):
        raise ValueError(f"{label} values must be 0 or 1; got {state}")
    active = sum(values)
    if active != 1:
        raise ValueError(
            f"{label} must have exactly one key set to 1; got {active} active states."
        )


def validate_count_state(count_state: dict[str, int]) -> None:
    validate_one_hot_state(count_state, COUNT_STATE_COLUMNS, label="count_state")


def validate_base_state(base_state: dict[str, int]) -> None:
    validate_one_hot_state(base_state, BASE_STATE_COLUMNS, label="base_state")


def validate_pitch_types(pitch_types: list[str] | None) -> list[str]:
    types = CORE_PITCH_TYPES if pitch_types is None else pitch_types
    if not types:
        raise ValueError("pitch_types must be a non-empty list of competitive pitch codes.")
    invalid = sorted(set(types) - set(COMPETITIVE_PITCH_TYPES))
    if invalid:
        raise ValueError(
            f"Invalid pitch type(s): {invalid}. "
            f"Must be subset of competitive types: {sorted(COMPETITIVE_PITCH_TYPES)}"
        )
    return types


def generate_strike_zone_grid(
    *,
    sz_top: float = DEFAULT_SZ_TOP,
    sz_bot: float = DEFAULT_SZ_BOT,
) -> pd.DataFrame:
    """
    Build a synthetic plate-location grid with rule-book zone flags.

    ``plate_x`` spans -1.5 to +1.5 ft; ``plate_z`` spans 0.5 to 4.5 ft
    (0.25 ft steps). ``is_in_zone`` and ``miss_dist_in`` use MLB plate width
    and the supplied vertical strike-zone bounds.
    """
    plate_x = np.round(np.arange(PLATE_X_MIN, PLATE_X_MAX + GRID_STEP_FT / 2, GRID_STEP_FT), 4)
    plate_z = np.round(np.arange(PLATE_Z_MIN, PLATE_Z_MAX + GRID_STEP_FT / 2, GRID_STEP_FT), 4)
    grid = pd.MultiIndex.from_product(
        [plate_x, plate_z], names=["plate_x", "plate_z"]
    ).to_frame(index=False)

    sz_top_arr = np.full(len(grid), sz_top, dtype=float)
    sz_bot_arr = np.full(len(grid), sz_bot, dtype=float)
    grid["is_in_zone"] = rule_book_in_zone(
        grid["plate_x"], grid["plate_z"], sz_bot_arr, sz_top_arr
    ).astype(np.int8)
    grid["miss_dist_in"] = miss_distance_inches(
        grid["plate_x"], grid["plate_z"], sz_bot_arr, sz_top_arr
    ).astype(np.float32)
    grid["center_dist_in"] = center_dist_inches(
        grid["plate_x"], grid["plate_z"], sz_bot_arr, sz_top_arr
    ).astype(np.float32)
    grid["sz_top"] = sz_top
    grid["sz_bot"] = sz_bot
    return grid


def validate_batter_profile(profile: dict[str, float]) -> dict[str, float]:
    missing = [col for col in BATTER_PROFILE_COLUMNS if col not in profile]
    if missing:
        raise ValueError(
            f"batter_profile missing keys: {missing}. "
            f"Expected: {BATTER_PROFILE_COLUMNS}"
        )
    return {col: float(profile[col]) for col in BATTER_PROFILE_COLUMNS}


def validate_p_throws(p_throws: str) -> None:
    if p_throws not in ("R", "L"):
        raise ValueError(f"Invalid p_throws {p_throws!r}. Must be 'R' or 'L'.")


def _build_synthetic_frame(
    grid: pd.DataFrame,
    pitch_types: list[str],
    league_physics: pd.DataFrame,
    *,
    stand: str,
    p_throws: str,
    batter_profile: dict[str, float],
    count_state: dict[str, int],
    base_state: dict[str, int],
) -> pd.DataFrame:
    missing_types = sorted(set(pitch_types) - set(league_physics["pitch_type"]))
    if missing_types:
        raise ValueError(f"Missing league physics for pitch types: {missing_types}")

    locations = grid.copy()
    types = pd.DataFrame({"pitch_type": pitch_types})
    frame = locations.merge(types, how="cross")

    physics = league_physics.loc[league_physics["pitch_type"].isin(pitch_types)].copy()
    frame = frame.merge(physics, on="pitch_type", how="left", validate="many_to_one")

    for col in COUNT_STATE_COLUMNS:
        frame[col] = np.int8(count_state[col])
    for col in BASE_STATE_COLUMNS:
        frame[col] = np.int8(base_state[col])
    for col in PITCH_TYPE_COLUMNS:
        code = col.removeprefix("pitch_")
        frame[col] = np.int8(frame["pitch_type"].eq(code))

    frame["stand"] = stand
    frame["p_throws"] = p_throws
    frame["same_handed"] = np.int8(p_throws == stand)
    frame["p_throws_L"] = np.int8(p_throws == "L")
    for col, value in batter_profile.items():
        frame[col] = np.float32(value)

    frame = add_attack_zones(frame)
    zone_map = {
        "zone_heart": "Heart",
        "zone_shadow": "Shadow",
        "zone_chase": "Chase",
        "zone_waste": "Waste",
    }
    for col, label in zone_map.items():
        frame[col] = np.int8(frame["attack_zone"].eq(label))

    return frame


def _annotate_outcome_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add readable outcome columns used in recommender output."""
    out = frame.copy()
    out["swing_probability"] = out["p_swing"]
    out["whiff_probability"] = out["p_swing"] * out["p_whiff_given_swing"]
    out["xwobacon"] = out["pred_xwobacon"]
    return out


def spatial_smooth_erv(
    frame: pd.DataFrame,
    *,
    radius_in: float = SMOOTH_RADIUS_IN,
    sigma_in: float | None = None,
    erv_col: str = "expected_run_value",
    pitch_col: str = "pitch_type",
) -> pd.DataFrame:
    """
    Per pitch type, apply a 2D Gaussian weighted average of ERV within ``radius_in``.

    Distance is Euclidean in inches on the (plate_x, plate_z) plane. Each row
    receives ``erv_raw`` (point estimate) and ``erv_smoothed`` (spatial average).
    """
    if erv_col not in frame.columns:
        raise ValueError(f"Missing {erv_col} column for spatial smoothing.")

    sigma = radius_in / 2.0 if sigma_in is None else sigma_in
    out = frame.copy()
    out["erv_raw"] = out[erv_col].astype(float)
    smoothed = np.empty(len(out), dtype=float)

    for _, group in out.groupby(pitch_col, sort=False):
        idx = group.index.to_numpy()
        xs = group["plate_x"].to_numpy(dtype=float)
        zs = group["plate_z"].to_numpy(dtype=float)
        erv = group["erv_raw"].to_numpy(dtype=float)

        for local_i in range(len(group)):
            dx_in = (xs - xs[local_i]) * 12.0
            dz_in = (zs - zs[local_i]) * 12.0
            dist_in = np.sqrt(dx_in * dx_in + dz_in * dz_in)
            mask = dist_in <= radius_in
            weights = np.exp(-0.5 * (dist_in[mask] / sigma) ** 2)
            smoothed[idx[local_i]] = float(np.average(erv[mask], weights=weights))

    out["erv_smoothed"] = smoothed
    return out


def _format_recommendation_output(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the public recommendation contract ranked by spatially smoothed ERV."""
    ranked = frame.sort_values("erv_smoothed", ascending=True).reset_index(drop=True)
    return pd.DataFrame(
        {
            "Rank": np.arange(1, len(ranked) + 1, dtype=np.int32),
            "Type": ranked["pitch_type"],
            "plate_x": ranked["plate_x"],
            "plate_z": ranked["plate_z"],
            "InZ": ranked["is_in_zone"].astype(np.int8),
            "Swing Prob": ranked["swing_probability"],
            "Whiff Prob": ranked["whiff_probability"],
            "xwobacon": ranked["xwobacon"],
            "ERV_raw": ranked["erv_raw"],
            "ERV": ranked["erv_smoothed"],
        }
    )


def simulate_plate_appearance(
    stand: str,
    count_state: dict[str, int],
    base_state: dict[str, int],
    batter_profile: dict[str, float],
    *,
    p_throws: str = "R",
    pitch_types: list[str] | None = None,
    artifacts: tuple[dict, dict, dict] | None = None,
    league_physics: pd.DataFrame | None = None,
    sz_top: float = DEFAULT_SZ_TOP,
    sz_bot: float = DEFAULT_SZ_BOT,
    profile: bool = True,
    batter_id: str | None = None,
) -> pd.DataFrame:
    """
    Simulate every pitch type at every grid location and score with ERV.

    ``batter_profile`` supplies season-to-date rolling rates
    (``swing_pct``, ``whiff_pct``, ``z_swing_pct``, ``o_swing_pct``,
    ``z_whiff_pct``, ``o_whiff_pct``, ``bat_speed``, ``attack_angle``,
    ``squared_up_rate``) as decimals in [0, 1] for rates and raw units for
    bat-tracking averages. ``p_throws`` sets pitcher handedness for platoon /
    release features.
    """
    validate_stand(stand)
    validate_p_throws(p_throws)
    validate_count_state(count_state)
    validate_base_state(base_state)
    batter_profile = validate_batter_profile(batter_profile)
    types = validate_pitch_types(pitch_types)

    t_start = time.perf_counter()

    physics = league_physics if league_physics is not None else load_league_average_physics()
    grid = generate_strike_zone_grid(sz_top=sz_top, sz_bot=sz_bot)

    synthetic = _build_synthetic_frame(
        grid,
        types,
        physics,
        stand=stand,
        p_throws=p_throws,
        batter_profile=batter_profile,
        count_state=count_state,
        base_state=base_state,
    )
    if batter_id is not None:
        synthetic["batter"] = batter_id
    t_after_matrix = time.perf_counter()

    scored = calculate_expected_run_value(synthetic, artifacts=artifacts)
    t_after_inference = time.perf_counter()

    scored = _annotate_outcome_columns(scored)
    scored = spatial_smooth_erv(scored)
    result = _format_recommendation_output(scored)
    t_end = time.perf_counter()

    if profile:
        total = t_end - t_start
        print(
            f"[PERF] simulate_plate_appearance executed in {total:.4f} seconds "
            f"({len(result):,} matrix scenarios)."
        )
        print(
            f"       cross-join: {t_after_matrix - t_start:.4f}s | "
            f"model inference: {t_after_inference - t_after_matrix:.4f}s | "
            f"ERV formatting: {t_end - t_after_inference:.4f}s"
        )

    return result


def print_top_recommendations(
    frame: pd.DataFrame,
    *,
    n: int = 5,
    title: str | None = None,
) -> None:
    """Print the top-N lowest-ERV pitch/location combinations."""
    if title:
        print(f"\n{title}")
    print("-" * 96)
    print(
        f"{'Rank':>4}  {'Type':<4}  {'plate_x':>7}  {'plate_z':>7}  {'InZ':>3}  "
        f"{'Swing Prob':>10}  {'Whiff Prob':>10}  {'xwOBAcon':>8}  {'ERV':>8}"
    )
    print("-" * 104)

    top = frame.head(n)
    for _, row in top.iterrows():
        print(
            f"{int(row['Rank']):>4}  {row['Type']:<4}  {row['plate_x']:+7.2f}  {row['plate_z']:+7.2f}  "
            f"{int(row['InZ']):>3}  {row['Swing Prob']:10.3f}  {row['Whiff Prob']:10.3f}  "
            f"{row['xwobacon']:8.3f}  {row['ERV']:+8.4f}"
        )


if __name__ == "__main__":
    print("Loading EV models ...")
    artifacts = load_ev_artifacts()
    physics = load_league_average_physics()

    print(
        "Simulating 0-0 count, bases empty, vs Generic_R "
        f"({len(CORE_PITCH_TYPES)} pitch types x strike-zone grid) ..."
    )
    recommendations = simulate_plate_appearance(
        "R",
        build_count_state(0, 0),
        build_base_state("state_empty"),
        league_default_batter_profile(),
        p_throws="R",
        artifacts=artifacts,
        league_physics=physics,
    )

    print_top_recommendations(
        recommendations,
        n=5,
        title="Top 5 optimal pitches (lowest ERV = best for pitcher)",
    )

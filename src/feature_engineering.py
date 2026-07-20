"""Pitch-level feature and target construction for swing / whiff / xwOBAcon models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.attack_zones import add_attack_zones
from src.batter_zone_profiles import attach_statcast_zone
from src.preprocessing import REQUIRED_RAW_COLUMNS, base_state_column
from src.statcast_schema import COMPETITIVE_PITCH_TYPES

PLATE_HALF_WIDTH_FT = 17.0 / 24.0  # 0.708 ft

# Bunt attempts are treated as takes (not full swings) across the pipeline.
BUNT_DESCRIPTIONS = frozenset({"foul_bunt", "missed_bunt"})

SWING_DESCRIPTIONS = frozenset(
    {
        "swinging_strike",
        "swinging_strike_blocked",
        "foul",
        "foul_tip",
        "hit_into_play",
        "hit_into_play_no_out",
        "hit_into_play_score",
    }
)

WHIFF_DESCRIPTIONS = frozenset({"swinging_strike", "swinging_strike_blocked"})

BIP_DESCRIPTIONS = frozenset(
    {"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}
)

BASE_STATE_COLUMNS = [
    "state_empty",
    "state_1b",
    "state_1b_2b",
    "state_1b_3b",
    "state_2b",
    "state_2b_3b",
    "state_3b",
    "state_loaded",
]

COUNT_STATE_COLUMNS = [f"count_{balls}_{strikes}" for balls in range(4) for strikes in range(3)]

ZONE_COLUMNS = ["zone_heart", "zone_shadow", "zone_chase", "zone_waste"]

PITCH_TYPE_COLUMNS = [f"pitch_{code}" for code in sorted(COMPETITIVE_PITCH_TYPES)]

BATTER_ROLLING_COLUMNS = [
    "swing_pct",
    "whiff_pct",
    "z_swing_pct",
    "o_swing_pct",
    "z_whiff_pct",
    "o_whiff_pct",
]

# Season-to-date rolling bat-tracking profile (prior tracked swings only; no leakage).
BAT_TRACKING_COLUMNS = [
    "bat_speed",
    "attack_angle",
    "squared_up_rate",
]

BATTER_PROFILE_COLUMNS = BATTER_ROLLING_COLUMNS + BAT_TRACKING_COLUMNS

SQUARED_UP_ATTACK_ANGLE_MIN = 8.0
SQUARED_UP_ATTACK_ANGLE_MAX = 32.0
SQUARED_UP_BAT_SPEED_PCT = 0.90

PLATOON_COLUMNS = ["same_handed"]

PITCH_MOVEMENT_COLUMNS = ["norm_hb", "norm_ivb"]

PITCHER_DELIVERY_COLUMNS = [
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "release_extension",
    "p_throws_L",
]

PITCH_QUALITY_COLUMNS = ["release_speed", "release_spin_rate"]

# Foundational location baseline used by all modeling notebooks (04–09).
BASELINE_FEATURES = ["is_in_zone", "miss_dist_in", "center_dist_in"]

PHYSICS_COLUMNS = [
    "center_dist_in",
    *PITCH_MOVEMENT_COLUMNS,
    *PITCHER_DELIVERY_COLUMNS,
    *PITCH_QUALITY_COLUMNS,
]

SWING_FEATURE_GROUP_ORDER = [
    "pitch_type",
    "pitch_quality",
    "count",
    "batter_rolling",
    "pitch_movement",
    "pitcher_delivery",
    "attack_zone",
    "base_state",
    "platoon",
]

CONTACT_FEATURE_GROUP_ORDER = [
    "pitch_type",
    "pitch_quality",
    "count",
    "batter_rolling",
    "bat_tracking",
    "pitch_movement",
    "pitcher_delivery",
    "attack_zone",
    "base_state",
    "platoon",
]


def build_feature_groups(*, include_bat_tracking: bool = False) -> tuple[dict[str, list[str]], list[str]]:
    """Candidate feature groups for forward selection / kitchen-sink models."""
    order = CONTACT_FEATURE_GROUP_ORDER if include_bat_tracking else SWING_FEATURE_GROUP_ORDER
    groups: dict[str, list[str]] = {
        "count": COUNT_STATE_COLUMNS,
        "base_state": BASE_STATE_COLUMNS,
        "attack_zone": ZONE_COLUMNS,
        "pitch_type": PITCH_TYPE_COLUMNS,
        "batter_rolling": BATTER_ROLLING_COLUMNS,
        "platoon": PLATOON_COLUMNS,
        "pitch_movement": PITCH_MOVEMENT_COLUMNS,
        "pitcher_delivery": PITCHER_DELIVERY_COLUMNS,
        "pitch_quality": PITCH_QUALITY_COLUMNS,
    }
    if include_bat_tracking:
        groups["bat_tracking"] = BAT_TRACKING_COLUMNS
    return groups, order


def feature_groups_present(
    features: list[str] | set[str],
    *,
    include_bat_tracking: bool = False,
) -> list[str]:
    """Return feature-group names whose columns are all present in ``features``."""
    groups, order = build_feature_groups(include_bat_tracking=include_bat_tracking)
    feat = set(features)
    return [name for name in order if all(col in feat for col in groups[name])]


NUMERIC_FEATURE_COLUMNS = [
    "plate_x",
    "plate_z",
    "miss_dist_in",
    *PHYSICS_COLUMNS,
    *BATTER_ROLLING_COLUMNS,
    *BAT_TRACKING_COLUMNS,
]

BINARY_FEATURE_COLUMNS = ["is_in_zone", *PLATOON_COLUMNS]

CATEGORICAL_FEATURE_COLUMNS = (
    BASE_STATE_COLUMNS + COUNT_STATE_COLUMNS + ZONE_COLUMNS + PITCH_TYPE_COLUMNS
)

TARGET_COLUMNS = ["is_swing", "is_whiff", "xwobacon"]

BATTER_PITCH_THRESHOLD = 1000
POOLED_BATTER_COL = "pooled_batter_id"

MODELING_FEATURE_COLUMNS = (
    NUMERIC_FEATURE_COLUMNS + BINARY_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
)

KEY_COLUMNS = [
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "batter",
    "pitcher",
    "pitch_type",
    "p_throws",
    "stand",
    "description",
    POOLED_BATTER_COL,
    "zone",
]

SORT_COLUMNS = ["batter", "game_date", "game_pk", "at_bat_number", "pitch_number"]


@dataclass
class FeatureEngineerReport:
    n_input: int = 0
    n_unknown_attack_zone: int = 0
    n_final: int = 0
    n_qualified_batters: int = 0
    n_pooled_batters: int = 0
    league_swing_pct: float = 0.0
    league_whiff_pct: float = 0.0
    league_z_swing_pct: float = 0.0
    league_o_swing_pct: float = 0.0
    league_z_whiff_pct: float = 0.0
    league_o_whiff_pct: float = 0.0
    league_bat_speed: float = 0.0
    league_attack_angle: float = 0.0
    league_squared_up_rate: float = 0.0


def league_default_bat_tracking_profile(frame: pd.DataFrame | None = None) -> dict[str, float]:
    """Season-level bat-tracking defaults for cold-start inference."""
    if frame is None or frame.empty:
        return {
            "bat_speed": 72.0,
            "attack_angle": 12.0,
            "squared_up_rate": 0.28,
        }

    swings = frame["is_swing"].astype(bool)
    if "bat_speed" in frame.columns:
        raw_speed = pd.to_numeric(frame["bat_speed"], errors="coerce")
    else:
        raw_speed = pd.Series(np.nan, index=frame.index)
    if "attack_angle" in frame.columns:
        raw_angle = pd.to_numeric(frame["attack_angle"], errors="coerce")
    else:
        raw_angle = pd.Series(np.nan, index=frame.index)
    tracked = swings & raw_speed.notna() & raw_angle.notna()
    if not tracked.any():
        return {
            "bat_speed": 72.0,
            "attack_angle": 12.0,
            "squared_up_rate": 0.28,
        }

    speed = float(raw_speed[tracked].mean())
    angle = float(raw_angle[tracked].mean())
    max_speed = float(raw_speed[tracked].max())
    speed_ok = raw_speed >= (SQUARED_UP_BAT_SPEED_PCT * max_speed)
    angle_ok = raw_angle.between(SQUARED_UP_ATTACK_ANGLE_MIN, SQUARED_UP_ATTACK_ANGLE_MAX)
    rate = float((tracked & speed_ok & angle_ok).mean()) if tracked.any() else 0.28
    return {
        "bat_speed": speed,
        "attack_angle": angle,
        "squared_up_rate": rate,
    }


def league_default_batter_profile(frame: pd.DataFrame | None = None) -> dict[str, float]:
    """Season-level batter profile defaults for cold-start inference."""
    profile = {
        "swing_pct": 0.478,
        "whiff_pct": 0.229,
        "z_swing_pct": 0.685,
        "o_swing_pct": 0.310,
        "z_whiff_pct": 0.260,
        "o_whiff_pct": 0.320,
    }
    if frame is None or frame.empty:
        profile.update(league_default_bat_tracking_profile(None))
        return profile
    swings = frame["is_swing"].astype(bool)
    whiffs = frame["is_whiff"].astype(bool)
    in_zone = frame["is_in_zone"].astype(bool)
    league_swing = float(frame["is_swing"].mean())
    league_whiff = float(whiffs[swings].mean()) if swings.any() else 0.229
    z_mask = in_zone
    o_mask = ~in_zone
    iz_swings = swings & z_mask
    ooz_swings = swings & o_mask
    league_z = float(swings[z_mask].mean()) if z_mask.any() else 0.685
    league_o = float(swings[o_mask].mean()) if o_mask.any() else 0.310
    league_z_whiff = float(whiffs[iz_swings].mean()) if iz_swings.any() else 0.260
    league_o_whiff = float(whiffs[ooz_swings].mean()) if ooz_swings.any() else 0.320
    profile = {
        "swing_pct": league_swing,
        "whiff_pct": league_whiff,
        "z_swing_pct": league_z,
        "o_swing_pct": league_o,
        "z_whiff_pct": league_z_whiff,
        "o_whiff_pct": league_o_whiff,
    }
    profile.update(league_default_bat_tracking_profile(frame))
    return profile


def _squared_up_mask(
    bat_speed: pd.Series,
    attack_angle: pd.Series,
    prior_max_bat_speed: pd.Series,
) -> pd.Series:
    """Approximate Statcast squared-up flag using bat speed and attack angle."""
    valid = bat_speed.notna() & attack_angle.notna() & prior_max_bat_speed.notna() & (prior_max_bat_speed > 0)
    speed_ok = bat_speed >= (SQUARED_UP_BAT_SPEED_PCT * prior_max_bat_speed)
    angle_ok = attack_angle.between(SQUARED_UP_ATTACK_ANGLE_MIN, SQUARED_UP_ATTACK_ANGLE_MAX)
    return (valid & speed_ok & angle_ok).astype(np.float32)


def add_pooled_batter_id(
    frame: pd.DataFrame,
    *,
    min_pitches: int = BATTER_PITCH_THRESHOLD,
) -> pd.DataFrame:
    """Pool low-sample hitters to Generic_L / Generic_R by stand; keep all rows."""
    out = frame.copy()
    pitch_counts = out.groupby("batter").size()
    qualified = out["batter"].map(pitch_counts) >= min_pitches
    generic = np.where(out["stand"].eq("L"), "Generic_L", "Generic_R")
    out[POOLED_BATTER_COL] = np.where(qualified, out["batter"].astype(str), generic)
    return out


def rule_book_in_zone(
    plate_x: pd.Series | np.ndarray,
    plate_z: pd.Series | np.ndarray,
    sz_bot: pd.Series | np.ndarray,
    sz_top: pd.Series | np.ndarray,
) -> np.ndarray:
    """1 if pitch is inside the rule-book zone, else 0."""
    px = np.asarray(plate_x, dtype=float)
    pz = np.asarray(plate_z, dtype=float)
    bot = np.asarray(sz_bot, dtype=float)
    top = np.asarray(sz_top, dtype=float)
    in_x = (px >= -PLATE_HALF_WIDTH_FT) & (px <= PLATE_HALF_WIDTH_FT)
    in_z = (pz >= bot) & (pz <= top)
    return (in_x & in_z).astype(np.int8)


def miss_distance_inches(
    plate_x: pd.Series | np.ndarray,
    plate_z: pd.Series | np.ndarray,
    sz_bot: pd.Series | np.ndarray,
    sz_top: pd.Series | np.ndarray,
) -> np.ndarray:
    """Distance in inches from pitch to nearest rule-book zone edge (0 if inside)."""
    px = np.asarray(plate_x, dtype=float)
    pz = np.asarray(plate_z, dtype=float)
    bot = np.asarray(sz_bot, dtype=float)
    top = np.asarray(sz_top, dtype=float)
    left, right = -PLATE_HALF_WIDTH_FT, PLATE_HALF_WIDTH_FT
    x_out = np.where(px < left, left - px, np.where(px > right, px - right, 0.0))
    z_out = np.where(pz < bot, bot - pz, np.where(pz > top, pz - top, 0.0))
    return np.sqrt(x_out**2 + z_out**2) * 12.0


def center_dist_inches(
    plate_x: pd.Series | np.ndarray,
    plate_z: pd.Series | np.ndarray,
    sz_bot: pd.Series | np.ndarray,
    sz_top: pd.Series | np.ndarray,
) -> np.ndarray:
    """Euclidean distance in inches from pitch to rule-book zone center (plate_x=0, z mid)."""
    px = np.asarray(plate_x, dtype=float)
    pz = np.asarray(plate_z, dtype=float)
    bot = np.asarray(sz_bot, dtype=float)
    top = np.asarray(sz_top, dtype=float)
    center_z = (top + bot) / 2.0
    return np.sqrt(px**2 + (pz - center_z) ** 2) * 12.0


def normalize_movement(
    pfx_x: pd.Series | np.ndarray,
    pfx_z: pd.Series | np.ndarray,
    p_throws: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip horizontal break for LHP; vertical break unchanged."""
    hb = np.asarray(pfx_x, dtype=float).copy()
    ivb = np.asarray(pfx_z, dtype=float)
    is_left = np.asarray(p_throws, dtype=object) == "L"
    hb[is_left] *= -1.0
    return hb, ivb


def count_state_column(frame: pd.DataFrame) -> pd.Series:
    balls = frame["balls"].fillna(0).astype(int)
    strikes = frame["strikes"].fillna(0).astype(int)
    return pd.Series(
        [f"count_{b}_{s}" for b, s in zip(balls, strikes, strict=True)],
        index=frame.index,
        dtype="object",
    )


def one_hot_from_labels(labels: pd.Series, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(0, index=labels.index, columns=columns, dtype=np.int8)
    for col in columns:
        out.loc[labels.eq(col), col] = 1
    return out


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["is_swing"] = out["description"].isin(SWING_DESCRIPTIONS).astype(np.int8)
    out["is_whiff"] = out["description"].isin(WHIFF_DESCRIPTIONS).astype(np.int8)
    out["xwobacon"] = np.where(
        out["description"].isin(BIP_DESCRIPTIONS),
        pd.to_numeric(out["estimated_woba_using_speedangle"], errors="coerce"),
        np.nan,
    )
    return out


def add_platoon_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["same_handed"] = out["p_throws"].eq(out["stand"]).astype(np.int8)
    out["p_throws_L"] = out["p_throws"].eq("L").astype(np.int8)
    return out


def add_rolling_batter_metrics(
    frame: pd.DataFrame,
    *,
    defaults: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Season-to-date batter rates using only prior pitches (no leakage).

    Each row's metrics exclude the current pitch via ``shift(1)`` within batter.
    """
    out = frame.sort_values(SORT_COLUMNS).copy()
    defaults = defaults or league_default_batter_profile(out)

    in_zone = out["is_in_zone"].astype(np.int8)
    out["_iz"] = in_zone
    out["_ooz"] = (1 - in_zone).astype(np.int8)
    out["_swing"] = out["is_swing"].astype(np.int8)
    out["_whiff"] = out["is_whiff"].astype(np.int8)
    out["_iz_sw"] = (in_zone.astype(bool) & out["_swing"].astype(bool)).astype(np.int8)
    out["_ooz_sw"] = ((~in_zone.astype(bool)) & out["_swing"].astype(bool)).astype(np.int8)
    out["_iz_wh"] = (in_zone.astype(bool) & out["_whiff"].astype(bool)).astype(np.int8)
    out["_ooz_wh"] = ((~in_zone.astype(bool)) & out["_whiff"].astype(bool)).astype(np.int8)

    grouped = out.groupby("batter", sort=False)
    prior_pitches = grouped.cumcount()
    prior_swings = grouped["_swing"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    prior_whiffs = grouped["_whiff"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    prior_iz_p = grouped["_iz"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    prior_ooz_p = grouped["_ooz"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    prior_iz_sw = grouped["_iz_sw"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    prior_ooz_sw = grouped["_ooz_sw"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    prior_iz_wh = grouped["_iz_wh"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    prior_ooz_wh = grouped["_ooz_wh"].transform(lambda s: s.shift(1).cumsum()).fillna(0)

    out["swing_pct"] = prior_swings / prior_pitches.replace(0, np.nan)
    out["whiff_pct"] = prior_whiffs / prior_swings.replace(0, np.nan)
    out["z_swing_pct"] = prior_iz_sw / prior_iz_p.replace(0, np.nan)
    out["o_swing_pct"] = prior_ooz_sw / prior_ooz_p.replace(0, np.nan)
    out["z_whiff_pct"] = prior_iz_wh / prior_iz_sw.replace(0, np.nan)
    out["o_whiff_pct"] = prior_ooz_wh / prior_ooz_sw.replace(0, np.nan)

    for col in BATTER_ROLLING_COLUMNS:
        out[col] = out[col].fillna(defaults[col]).astype(np.float32)

    out = out.drop(
        columns=["_iz", "_ooz", "_swing", "_whiff", "_iz_sw", "_ooz_sw", "_iz_wh", "_ooz_wh"],
        errors="ignore",
    )
    return out


def add_rolling_bat_tracking_metrics(
    frame: pd.DataFrame,
    *,
    defaults: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Season-to-date rolling bat-tracking profile using only prior tracked swings.

    Raw Statcast ``bat_speed`` / ``attack_angle`` (when present) are consumed and
    replaced with rolling averages plus ``squared_up_rate``.
    """
    out = frame.sort_values(SORT_COLUMNS).copy()
    defaults = defaults or league_default_batter_profile(out)

    raw_speed = (
        pd.to_numeric(out["bat_speed"], errors="coerce")
        if "bat_speed" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    raw_angle = (
        pd.to_numeric(out["attack_angle"], errors="coerce")
        if "attack_angle" in out.columns
        else pd.Series(np.nan, index=out.index)
    )

    is_swing = out["is_swing"].astype(bool)
    has_tracking = is_swing & raw_speed.notna() & raw_angle.notna()
    out["_track_speed"] = np.where(has_tracking, raw_speed, np.nan)
    out["_track_angle"] = np.where(has_tracking, raw_angle, np.nan)

    grouped = out.groupby("batter", sort=False)
    prior_speed = grouped["_track_speed"].transform(lambda s: s.shift(1))
    prior_angle = grouped["_track_angle"].transform(lambda s: s.shift(1))
    prior_max_speed = grouped["_track_speed"].transform(lambda s: s.shift(1).cummax())
    prior_sq = _squared_up_mask(prior_speed, prior_angle, prior_max_speed)

    prior_track_count = grouped["_track_speed"].transform(
        lambda s: s.shift(1).notna().astype(int).cumsum()
    )
    speed_sum = grouped["_track_speed"].transform(lambda s: s.shift(1).fillna(0).cumsum())
    angle_sum = grouped["_track_angle"].transform(lambda s: s.shift(1).fillna(0).cumsum())
    sq_sum = prior_sq.fillna(0).groupby(out["batter"]).cumsum()

    out["bat_speed"] = speed_sum / prior_track_count.replace(0, np.nan)
    out["attack_angle"] = angle_sum / prior_track_count.replace(0, np.nan)
    out["squared_up_rate"] = sq_sum / prior_track_count.replace(0, np.nan)

    for col in BAT_TRACKING_COLUMNS:
        out[col] = out[col].fillna(defaults[col]).astype(np.float32)

    return out.drop(columns=["_track_speed", "_track_angle"], errors="ignore")


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_attack_zones(frame.copy())
    out = attach_statcast_zone(out)
    out["is_in_zone"] = rule_book_in_zone(out["plate_x"], out["plate_z"], out["sz_bot"], out["sz_top"])
    out["miss_dist_in"] = miss_distance_inches(
        out["plate_x"], out["plate_z"], out["sz_bot"], out["sz_top"]
    ).astype(np.float32)
    out["center_dist_in"] = center_dist_inches(
        out["plate_x"], out["plate_z"], out["sz_bot"], out["sz_top"]
    ).astype(np.float32)
    norm_hb, norm_ivb = normalize_movement(out["pfx_x"], out["pfx_z"], out["p_throws"])
    out["norm_hb"] = norm_hb.astype(np.float32)
    out["norm_ivb"] = norm_ivb.astype(np.float32)
    out["release_speed"] = pd.to_numeric(out["release_speed"], errors="coerce").astype(np.float32)
    out["release_spin_rate"] = pd.to_numeric(out["release_spin_rate"], errors="coerce").astype(
        np.float32
    )
    out["release_extension"] = pd.to_numeric(out["release_extension"], errors="coerce").astype(
        np.float32
    )
    out["release_pos_x"] = pd.to_numeric(out["release_pos_x"], errors="coerce").astype(np.float32)
    out["release_pos_y"] = pd.to_numeric(out["release_pos_y"], errors="coerce").astype(np.float32)
    out["release_pos_z"] = pd.to_numeric(out["release_pos_z"], errors="coerce").astype(np.float32)

    out = add_platoon_features(out)

    base_labels = base_state_column(out)
    count_labels = count_state_column(out)
    zone_labels = out["attack_zone"].map(
        {
            "Heart": "zone_heart",
            "Shadow": "zone_shadow",
            "Chase": "zone_chase",
            "Waste": "zone_waste",
        }
    )
    pitch_labels = out["pitch_type"].astype(str).map(lambda code: f"pitch_{code}")

    base_hot = one_hot_from_labels(base_labels, BASE_STATE_COLUMNS)
    count_hot = one_hot_from_labels(count_labels, COUNT_STATE_COLUMNS)
    zone_hot = one_hot_from_labels(zone_labels, ZONE_COLUMNS)
    pitch_hot = one_hot_from_labels(pitch_labels, PITCH_TYPE_COLUMNS)

    return pd.concat([out, base_hot, count_hot, zone_hot, pitch_hot], axis=1)


def build_modeling_frame(preprocessed: pd.DataFrame) -> tuple[pd.DataFrame, FeatureEngineerReport]:
    report = FeatureEngineerReport(n_input=len(preprocessed))

    missing_cols = [col for col in REQUIRED_RAW_COLUMNS if col not in preprocessed.columns]
    if missing_cols:
        raise ValueError(f"Preprocessed frame missing columns: {missing_cols}")
    for col in (
        "release_pos_x",
        "release_pos_y",
        "release_pos_z",
        "release_extension",
        "release_spin_rate",
        "stand",
    ):
        if col not in preprocessed.columns:
            raise ValueError(
                f"Preprocessed frame missing {col}. Re-run 02_preprocessing.ipynb after schema update."
            )
    if "estimated_woba_using_speedangle" not in preprocessed.columns:
        raise ValueError("Preprocessed frame missing estimated_woba_using_speedangle")

    with_targets = add_targets(preprocessed)
    modeled = engineer_features(with_targets)
    defaults = league_default_batter_profile(modeled)
    modeled = add_rolling_batter_metrics(modeled, defaults=defaults)
    modeled = add_rolling_bat_tracking_metrics(modeled, defaults=defaults)
    modeled = add_pooled_batter_id(modeled)

    report.league_swing_pct = defaults["swing_pct"]
    report.league_whiff_pct = defaults["whiff_pct"]
    report.league_z_swing_pct = defaults["z_swing_pct"]
    report.league_o_swing_pct = defaults["o_swing_pct"]
    report.league_z_whiff_pct = defaults["z_whiff_pct"]
    report.league_o_whiff_pct = defaults["o_whiff_pct"]
    report.league_bat_speed = defaults["bat_speed"]
    report.league_attack_angle = defaults["attack_angle"]
    report.league_squared_up_rate = defaults["squared_up_rate"]

    pitch_counts = modeled.groupby("batter").size()
    report.n_qualified_batters = int((pitch_counts >= BATTER_PITCH_THRESHOLD).sum())
    report.n_pooled_batters = int((pitch_counts < BATTER_PITCH_THRESHOLD).sum())

    unknown_zone = modeled["attack_zone"].eq("Unknown")
    report.n_unknown_attack_zone = int(unknown_zone.sum())
    modeled = modeled.loc[~unknown_zone].copy()

    keep_cols = KEY_COLUMNS + MODELING_FEATURE_COLUMNS + TARGET_COLUMNS
    keep_cols = [col for col in keep_cols if col in modeled.columns]
    report.n_final = len(modeled)
    return modeled[keep_cols].reset_index(drop=True), report

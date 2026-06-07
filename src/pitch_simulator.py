"""Interactive pitch simulator: user guesses vs model swing / whiff probabilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.attack_zones import (
    HORIZONTAL_ORDER,
    VERTICAL_ORDER,
    ZONE_BORDER,
    ZONE_FILL,
    ZONE_ORDER,
    assign_attack_zone,
    boundary_signed_inches,
    infer_horizontal,
    infer_vertical,
    random_pitch_location,
    zone_description,
)
from src.hitter_rates import hitter_stats_html, load_league_rates, lookup_hitter_rates
from src.pitch_lab_inputs import (
    PITCH_FAMILIES,
    PITCH_FAMILY_DEFAULT_CODE,
    VELOCITY_INTENT_TO_TIER,
    apply_physics_tiers,
    format_count_display,
)
from src.pitch_outcomes import (
    OUTCOME_LABELS,
    enrich_probs_dict,
    format_xwoba_display,
    is_in_zone,
    mock_pitch_probs,
    mock_xwoba_con,
    outcome_is_favorable,
    simulate_outcome,
    xwoba_damage_tier,
)
from src.pitch_models import load_pitch_lab_models
from src.pitch_prescription import (
    HORIZONTAL_OPTIONS,
    VERTICAL_OPTIONS,
    VELOCITY_INTENTS,
    compare_strategy_guess,
    recommend_best_pitch,
    score_thrown_pitch,
)
from src.whiff_features import (
    MODEL_INPUT_COLS,
    PITCH_METRIC_COLS,
    _median_or_fallback,
    apply_pitch_imputation,
    engineer_features,
    model_feature_frame,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "model"
PITCH_LAB_PROFILES_FILE = MODEL_DIR / "pitch_lab_profiles.json"
PITCH_LAB_ZONES_FILE = MODEL_DIR / "pitch_lab_hitter_zones.csv"
STATCAST_FILE = ROOT / "data" / "statcast_2025.parquet"

PITCH_CATEGORY_DEFAULT = {
    "Fastball": "FF",
    "Breaking": "SL",
    "Offspeed": "CH",
    **{
        "Fastballs": "FF",
        "Breaking Balls": "SL",
        "Off-Speed": "CH",
    },
}

CATEGORY_PITCH_LABELS = {
    "Fastball": {
        "4-Seam Fastball": "FF",
        "Sinker": "SI",
        "Cutter": "FC",
    },
    "Breaking": {
        "Slider": "SL",
        "Curveball": "CU",
        "Sweeper": "SV",
        "Knuckle Curve": "KC",
        "Sweeper (ST)": "ST",
    },
    "Offspeed": {
        "Changeup": "CH",
        "Splitter": "FS",
    },
    **{
        "Fastballs": {
            "4-Seam Fastball": "FF",
            "Sinker": "SI",
            "Cutter": "FC",
        },
        "Breaking Balls": {
            "Slider": "SL",
            "Curveball": "CU",
            "Sweeper": "SV",
            "Knuckle Curve": "KC",
            "Sweeper (ST)": "ST",
        },
        "Off-Speed": {
            "Changeup": "CH",
            "Splitter": "FS",
        },
    },
}

PITCH_NAME_TO_TYPE = {
    label: code
    for labels in CATEGORY_PITCH_LABELS.values()
    for label, code in labels.items()
}


def _capture_chart_click_pending() -> None:
    """Remember chart click as pending location — custom mode only."""
    if st.session_state.get("lab_location_mode") != "Customize Exact Pitch":
        return
    chart_state = st.session_state.get("lab_zone_chart")
    if chart_state is None or not getattr(chart_state, "selection", None):
        return
    points = chart_state.selection.points
    if not points:
        return
    pt = points[-1]
    if pt.get("x") is None or pt.get("y") is None:
        return

    sel_sig = (
        round(float(pt["x"]), 2),
        round(float(pt["y"]), 2),
        pt.get("pointIndex"),
        pt.get("curveNumber"),
    )
    if sel_sig == st.session_state.get("lab_last_sel_sig"):
        return

    st.session_state.lab_last_sel_sig = sel_sig
    st.session_state.lab_pending_px = round(float(np.clip(float(pt["x"]), -2.0, 2.0)), 2)
    st.session_state.lab_pending_pz = round(float(np.clip(float(pt["y"]), 0.8, 4.2)), 2)


def _apply_plot_guess_location(
    plate_x: float,
    plate_z: float,
    sz_bot: float,
    sz_top: float,
    bats: str,
) -> None:
    """Derive vertical / horizontal guess buckets from a map click."""
    st.session_state.lab_guess_vert = infer_vertical(plate_z, sz_top, sz_bot)
    st.session_state.lab_guess_horiz = infer_horizontal(plate_x, bats)


def _commit_pitch(sz_bot: float, sz_top: float, platoon_bats: str) -> None:
    """Apply chart click (custom) or randomize from placement picks (random mode)."""
    mode = st.session_state.get("lab_location_mode", "Pick a Random Pitch")
    pending_x = st.session_state.get("lab_pending_px")
    pending_z = st.session_state.get("lab_pending_pz")

    if mode == "Customize Exact Pitch":
        if pending_x is None or pending_z is None:
            st.session_state.lab_need_map_click = True
            return
        st.session_state.lab_px = float(pending_x)
        st.session_state.lab_pz = float(pending_z)
        _apply_plot_guess_location(
            st.session_state.lab_px, st.session_state.lab_pz, sz_bot, sz_top, platoon_bats
        )
    else:
        if "lab_rng" not in st.session_state:
            st.session_state.lab_rng = np.random.default_rng()
        px, pz = random_pitch_location(
            st.session_state.lab_zone_pick,
            st.session_state.lab_vert_pick,
            st.session_state.lab_horiz_pick,
            sz_top,
            sz_bot,
            platoon_bats,
            st.session_state.lab_rng,
        )
        st.session_state.lab_px = px
        st.session_state.lab_pz = pz

    st.session_state.lab_pending_px = None
    st.session_state.lab_pending_pz = None
    st.session_state.lab_last_sel_sig = None
    st.session_state.lab_need_map_click = False
    st.session_state.lab_has_pitched = True
    st.session_state.lab_sim_outcome = None
    st.session_state.lab_run_predictions = True


def _refresh_pitch_results(
    *,
    balls: int,
    strikes: int,
    runners_on: int,
    sz_top: float,
    sz_bot: float,
    bats: str,
    plate_x: float,
    plate_z: float,
    guess_family: str,
    guess_vertical: str,
    guess_horizontal: str,
    guess_velocity: str,
    swing_bundle,
    whiff_bundle,
    xwoba_bundle,
    profile_lookup: dict,
    hitter_rates: dict[str, float] | None,
    models_live: bool,
) -> None:
    predict_kwargs = {"hitter_rates": hitter_rates, "runners_on": runners_on}
    use_ml = models_live and swing_bundle is not None and whiff_bundle is not None

    rx = recommend_best_pitch(
        balls=balls,
        strikes=strikes,
        runners_on=runners_on,
        sz_top=sz_top,
        sz_bot=sz_bot,
        bats=bats,
        swing_bundle=swing_bundle if use_ml else None,
        whiff_bundle=whiff_bundle if use_ml else None,
        xwoba_bundle=xwoba_bundle if use_ml else None,
        profile_lookup=profile_lookup,
        predict_kwargs=predict_kwargs,
        use_ml=use_ml,
    )

    guess_vert = guess_vertical if guess_vertical != "—" else rx["vertical"]
    guess_horiz = guess_horizontal if guess_horizontal != "—" else rx["horizontal"]

    thrown = score_thrown_pitch(
        balls=balls,
        strikes=strikes,
        plate_x=plate_x,
        plate_z=plate_z,
        pitch_family=guess_family,
        velocity_intent=guess_velocity,
        sz_top=sz_top,
        sz_bot=sz_bot,
        swing_bundle=swing_bundle if use_ml else None,
        whiff_bundle=whiff_bundle if use_ml else None,
        xwoba_bundle=xwoba_bundle if use_ml else None,
        profile_lookup=profile_lookup,
        predict_kwargs=predict_kwargs,
        use_ml=use_ml,
    )

    cmp = compare_strategy_guess(
        guess_family=guess_family,
        guess_vertical=guess_vert,
        guess_horizontal=guess_horiz,
        guess_velocity=guess_velocity,
        recommendation=rx,
    )

    st.session_state.lab_pitch_rx = rx
    st.session_state.lab_pitch_thrown = thrown
    st.session_state.lab_pitch_cmp = cmp
    st.session_state.lab_run_predictions = False


MLB_HEADSHOT_URL = (
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/{id}/headshot/67/current"
)


def mlb_headshot_url(batter_id: int) -> str:
    return MLB_HEADSHOT_URL.format(id=batter_id)


def resolve_bats_for_platoon(raw_bats: str) -> str:
    """Map switch hitters to a batting side for inside/outside labels."""
    if raw_bats == "S":
        return str(st.session_state.get("lab_bat_side", "R"))
    return raw_bats


def batter_display_label(raw_bats: str) -> str:
    if raw_bats == "S":
        side = resolve_bats_for_platoon(raw_bats)
        handed = "Left" if side == "L" else "Right"
        return f"Switch hitter · batting {handed} ({side})"
    if raw_bats == "L":
        return "Left-handed"
    return "Right-handed"


def batter_handedness(batter_stand: pd.DataFrame | None, batter_id: int) -> str:
    if batter_stand is not None and not batter_stand.empty:
        match = batter_stand.loc[batter_stand["batter"] == batter_id, "bats"]
        if not match.empty:
            return str(match.iloc[0])
    return "R"


def plate_side_labels(bats: str) -> tuple[str, str]:
    """Catcher's view: left = 3B (negative x), right = 1B (positive x)."""
    if bats == "L":
        return "Outside", "Inside"
    return "Inside", "Outside"


def _patch_model_bundle(bundle: dict | None) -> dict | None:
    """Fix sklearn 1.6 loading artifacts saved under 1.8 (missing multi_class on LogisticRegression)."""
    if bundle is None:
        return None
    model = bundle.get("model")
    if model is None or not hasattr(model, "named_steps"):
        return bundle
    clf = model.named_steps.get("clf")
    if clf is None and getattr(model, "steps", None):
        clf = model.steps[-1][1]
    if clf is not None and not hasattr(clf, "multi_class"):
        clf.multi_class = "auto"
    return bundle


def pitch_models_predict_available(swing_bundle, whiff_bundle) -> bool:
    """True when saved sklearn pipelines can score a probe pitch (not mock fallback)."""
    if swing_bundle is None or whiff_bundle is None:
        return False
    profile = {
        "release_speed": 94.0,
        "speed_diff": 0.0,
        "pfx_x": -4.0,
        "pfx_z": 8.0,
        "release_spin_rate": 2280.0,
        "spin_axis": 180.0,
        "release_extension": 6.0,
    }
    row = build_pitch_row(0.35, 2.5, 1.5, 3.5, 1, 1, 0, "FF", profile)
    try:
        medians = pd.Series(swing_bundle["pitch_medians"])
        engineered = apply_pitch_imputation(engineer_features(pd.DataFrame([row])), medians)
        swing_frame = model_feature_frame(
            engineered, list(swing_bundle.get("features", MODEL_INPUT_COLS)), medians
        )
        whiff_frame = model_feature_frame(
            engineered, list(whiff_bundle.get("features", MODEL_INPUT_COLS)), medians
        )
        swing_bundle["model"].predict_proba(swing_frame)
        whiff_bundle["model"].predict_proba(whiff_frame)
        return True
    except Exception:
        return False


def predict_xwoba(xwoba_bundle, engineered_frame, feature_cols, medians) -> float | None:
    if xwoba_bundle is None:
        return None
    try:
        frame = model_feature_frame(engineered_frame, feature_cols, medians)
        return float(xwoba_bundle["model"].predict(frame)[0])
    except Exception:
        return None


@st.cache_data
def load_pitch_lab_profiles() -> dict:
    if PITCH_LAB_PROFILES_FILE.exists():
        return json.loads(PITCH_LAB_PROFILES_FILE.read_text(encoding="utf-8"))

    if not STATCAST_FILE.exists():
        return {"league_sz_bot": 1.5, "league_sz_top": 3.5, "pitch_types": {}}

    df = pd.read_parquet(STATCAST_FILE, columns=["pitch_type", "sz_bot", "sz_top", *PITCH_METRIC_COLS])
    grouped = (
        df.groupby("pitch_type", as_index=False)[PITCH_METRIC_COLS]
        .median(numeric_only=True)
        .set_index("pitch_type")
        .to_dict(orient="index")
    )
    return {
        "league_sz_bot": float(df["sz_bot"].median()),
        "league_sz_top": float(df["sz_top"].median()),
        "pitch_types": grouped,
    }


@st.cache_data
def load_hitter_strike_zones() -> pd.DataFrame:
    if PITCH_LAB_ZONES_FILE.exists():
        return pd.read_csv(PITCH_LAB_ZONES_FILE)

    if not STATCAST_FILE.exists():
        return pd.DataFrame(columns=["batter", "sz_bot", "sz_top"])

    df = pd.read_parquet(STATCAST_FILE, columns=["batter", "sz_bot", "sz_top"])
    return (
        df.groupby("batter", as_index=False)
        .agg(sz_bot=("sz_bot", "median"), sz_top=("sz_top", "median"))
    )


def hitter_strike_zone(zones_df: pd.DataFrame, profiles: dict, batter_id: int) -> tuple[float, float]:
    match = zones_df.loc[zones_df["batter"] == batter_id]
    if not match.empty:
        return float(match.iloc[0]["sz_bot"]), float(match.iloc[0]["sz_top"])
    return float(profiles["league_sz_bot"]), float(profiles["league_sz_top"])


def build_pitch_row(
    plate_x: float,
    plate_z: float,
    sz_bot: float,
    sz_top: float,
    balls: int,
    strikes: int,
    runners_on: int,
    pitch_type: str,
    pitch_profile: dict,
) -> pd.Series:
    on_1b = 1 if runners_on >= 1 else pd.NA
    on_2b = 2 if runners_on >= 2 else pd.NA
    on_3b = 3 if runners_on >= 3 else pd.NA
    row = {
        "plate_x": plate_x,
        "plate_z": plate_z,
        "sz_bot": sz_bot,
        "sz_top": sz_top,
        "balls": balls,
        "strikes": strikes,
        "pitch_type": pitch_type,
        "on_1b": on_1b,
        "on_2b": on_2b,
        "on_3b": on_3b,
        "description": "placeholder",
        **pitch_profile,
    }
    return pd.Series(row)


def predict_probs(
    swing_bundle,
    whiff_bundle,
    row: pd.Series,
    *,
    xwoba_bundle=None,
    balls: int | None = None,
    strikes: int | None = None,
    pitch_family: str = "Fastball",
    velocity_intent: str = "Average",
    hitter_rates: dict[str, float] | None = None,
) -> dict[str, float]:
    plate_x = float(row["plate_x"])
    plate_z = float(row["plate_z"])
    sz_bot = float(row["sz_bot"])
    sz_top = float(row["sz_top"])
    in_zone = bool(is_in_zone(plate_x, plate_z, sz_bot, sz_top))
    attack_zone = assign_attack_zone(plate_x, plate_z, sz_top, sz_bot)
    try:
        medians = pd.Series(swing_bundle["pitch_medians"])
        engineered = apply_pitch_imputation(engineer_features(pd.DataFrame([row])), medians)
        swing_cols = list(swing_bundle.get("features", MODEL_INPUT_COLS))
        whiff_cols = list(whiff_bundle.get("features", MODEL_INPUT_COLS))
        swing_frame = model_feature_frame(engineered, swing_cols, medians)
        whiff_frame = model_feature_frame(engineered, whiff_cols, medians)
        swing_p = float(swing_bundle["model"].predict_proba(swing_frame)[:, 1][0])
        whiff_p = float(whiff_bundle["model"].predict_proba(whiff_frame)[:, 1][0])
    except Exception:
        b = int(balls if balls is not None else row.get("balls", 0))
        s = int(strikes if strikes is not None else row.get("strikes", 0))
        miss_dist_in = max(0.0, -boundary_signed_inches(plate_x, plate_z, sz_top, sz_bot))
        rates = hitter_rates or {}
        swing_p, whiff_p = mock_pitch_probs(
            balls=b,
            strikes=s,
            in_zone=in_zone,
            pitch_family=pitch_family,
            velocity_intent=velocity_intent,
            miss_dist_in=miss_dist_in,
            attack_zone=attack_zone,
            in_zone_swing_pct=rates.get("in_zone_swing_pct"),
            o_zone_swing_pct=rates.get("o_zone_swing_pct"),
        )
    enriched = enrich_probs_dict(
        {
            "swing": swing_p,
            "whiff_if_swing": whiff_p,
            "swing_whiff": swing_p * whiff_p,
            "miss_dist_in": float(row.get("miss_dist_in", 0)),
            "in_zone": in_zone,
        }
    )
    if enriched["p_contact"] > 0:
        xwoba_val = None
        if xwoba_bundle is not None:
            try:
                medians_x = pd.Series(xwoba_bundle["pitch_medians"])
                engineered_x = apply_pitch_imputation(engineer_features(pd.DataFrame([row])), medians_x)
                xwoba_val = predict_xwoba(
                    xwoba_bundle,
                    engineered_x,
                    list(xwoba_bundle.get("features", MODEL_INPUT_COLS)),
                    medians_x,
                )
            except Exception:
                xwoba_val = None
        if xwoba_val is None:
            xwoba_val = mock_xwoba_con(
                pitch_family=pitch_family,
                in_zone=in_zone,
                attack_zone=attack_zone,
            )
        enriched["xwoba_con"] = float(np.clip(xwoba_val, 0.0, 1.25))
    else:
        enriched["xwoba_con"] = None
    return enriched


@st.cache_data
def load_batter_stand() -> pd.DataFrame:
    zones_path = PITCH_LAB_ZONES_FILE
    if zones_path.exists():
        zones = pd.read_csv(zones_path)
        if "bats" in zones.columns:
            return zones[["batter", "bats"]].drop_duplicates()

    if not STATCAST_FILE.exists():
        return pd.DataFrame(columns=["batter", "bats"])

    df = pd.read_parquet(STATCAST_FILE, columns=["batter", "stand"])
    return (
        df.dropna(subset=["stand"])
        .groupby("batter", as_index=False)["stand"]
        .first()
        .rename(columns={"stand": "bats"})
    )


@st.cache_data
def load_qualified_hitters() -> pd.DataFrame:
    lb_path = ROOT / "data" / "whiff_leaderboard_2025.csv"
    if not lb_path.exists():
        return pd.DataFrame(columns=["batter", "player_name"])

    leaderboard = pd.read_csv(lb_path)
    zones = load_hitter_strike_zones()
    if zones.empty:
        return leaderboard[["batter", "player_name"]].drop_duplicates()

    qualified = set(zones["batter"].astype(int))
    out = leaderboard.loc[leaderboard["batter"].isin(qualified), ["batter", "player_name"]]
    return out.drop_duplicates("batter").sort_values("player_name").reset_index(drop=True)


def _zone_map_points(sz_bot: float, sz_top: float) -> tuple[list[float], list[float], list[str]]:
    step = 0.05
    xs = np.arange(-2.2, 2.21, step)
    zs = np.arange(0.45, 4.51, step)
    x_pts: list[float] = []
    z_pts: list[float] = []
    colors: list[str] = []
    for z in zs:
        for x in xs:
            zone = assign_attack_zone(float(x), float(z), sz_top, sz_bot)
            x_pts.append(float(x))
            z_pts.append(float(z))
            colors.append(ZONE_FILL[zone])
    return x_pts, z_pts, colors


def render_hitter_card(batter_id: int, player_name: str, raw_bats: str) -> None:
    """Hitter headshot + stats row (handedness-aware), for display above the zone map."""
    bats_label = batter_display_label(raw_bats)
    stats_html = hitter_stats_html(batter_id)
    card_html = f"""
    <div class="whiff-hitter-card">
        <img src="{mlb_headshot_url(batter_id)}" alt="{player_name}" class="whiff-hitter-photo"/>
        <div class="whiff-hitter-meta">
            <div class="whiff-hitter-name">{player_name}</div>
            <div class="whiff-hitter-bats">{bats_label}</div>
            {stats_html}
        </div>
    </div>
    """
    platoon_bats = resolve_bats_for_platoon(raw_bats)
    if platoon_bats == "L":
        spacer, card_col = st.columns([0.55, 1.45])
        with card_col:
            st.markdown(card_html, unsafe_allow_html=True)
    else:
        card_col, spacer = st.columns([1.45, 0.55])
        with card_col:
            st.markdown(card_html, unsafe_allow_html=True)


def render_hitter_banner(
    batter_id: int,
    player_name: str,
    raw_bats: str,
    attack_zone: str,
    plate_x: float,
    plate_z: float,
) -> None:
    bats_label = batter_display_label(raw_bats)
    zone_color = ZONE_BORDER.get(attack_zone, "#d4a937")
    stats_html = hitter_stats_html(batter_id)
    card_html = f"""
    <div class="whiff-hitter-card">
        <img src="{mlb_headshot_url(batter_id)}" alt="{player_name}" class="whiff-hitter-photo"/>
        <div class="whiff-hitter-meta">
            <div class="whiff-hitter-name">{player_name}</div>
            <div class="whiff-hitter-bats">{bats_label}</div>
            {stats_html}
        </div>
    </div>
    """
    zone_html = f"""
    <div class="whiff-zone-panel">
        <div class="whiff-zone-label">Attack zone</div>
        <div class="whiff-zone-name" style="color:{zone_color}">{attack_zone}</div>
        <div class="whiff-zone-desc">{zone_description(attack_zone)}</div>
        <div class="whiff-zone-coords">Location: {plate_x:+.2f} ft H · {plate_z:.2f} ft V</div>
    </div>
    """

    left_col, center_col, right_col = st.columns([1.35, 1.8, 1.35])
    platoon_bats = resolve_bats_for_platoon(raw_bats)
    # Catcher's view: RHB stands on the 3B (left) side; LHB on the 1B (right) side.
    if platoon_bats == "L":
        with right_col:
            st.markdown(card_html, unsafe_allow_html=True)
        with center_col:
            st.markdown(zone_html, unsafe_allow_html=True)
    else:
        with left_col:
            st.markdown(card_html, unsafe_allow_html=True)
        with center_col:
            st.markdown(zone_html, unsafe_allow_html=True)


def build_location_figure(
    plate_x: float,
    plate_z: float,
    sz_bot: float,
    sz_top: float,
    bats: str = "R",
    attack_zone: str | None = None,
    pending_x: float | None = None,
    pending_z: float | None = None,
) -> go.Figure:
    left_label, right_label = plate_side_labels(bats)
    x_pts, z_pts, zone_colors = _zone_map_points(sz_bot, sz_top)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_pts,
            y=z_pts,
            mode="markers",
            marker=dict(size=16, color=zone_colors, line=dict(width=0)),
            hovertemplate="Horizontal: %{x:.2f} ft<br>Vertical: %{y:.2f} ft<extra></extra>",
            name="Zone map",
        )
    )

    half_w = 17.0 / 24.0
    fig.add_shape(
        type="rect",
        x0=-half_w,
        x1=half_w,
        y0=sz_bot,
        y1=sz_top,
        line=dict(color="#f5efe3", width=3),
        fillcolor="rgba(245, 239, 227, 0.04)",
        layer="above",
    )

    zone_name = attack_zone or assign_attack_zone(plate_x, plate_z, sz_top, sz_bot)
    dot_color = ZONE_BORDER.get(zone_name, "#fde725")
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=plate_x - 0.11,
        x1=plate_x + 0.11,
        y0=plate_z - 0.11,
        y1=plate_z + 0.11,
        fillcolor="#fde725",
        line=dict(color="#0f172a", width=2),
        layer="above",
    )
    fig.add_annotation(
        x=plate_x,
        y=plate_z + 0.2,
        text=f"<b>{zone_name}</b>",
        showarrow=False,
        font=dict(color=dot_color, size=12),
    )

    if pending_x is not None and pending_z is not None:
        if (round(pending_x, 2), round(pending_z, 2)) != (round(plate_x, 2), round(plate_z, 2)):
            fig.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=pending_x - 0.11,
                x1=pending_x + 0.11,
                y0=pending_z - 0.11,
                y1=pending_z + 0.11,
                fillcolor="rgba(253, 231, 37, 0.15)",
                line=dict(color="#d4a937", width=2, dash="dash"),
                layer="above",
            )
            fig.add_annotation(
                x=pending_x,
                y=pending_z - 0.22,
                text="Pending",
                showarrow=False,
                font=dict(color="#d4a937", size=10),
            )

    label_y = sz_top + 0.28
    fig.add_annotation(
        x=-2.15,
        y=label_y,
        text=f"<b>L</b><br>3B<br><span style='font-size:11px'>{left_label}</span>",
        showarrow=False,
        font=dict(color="#f5efe3", size=12),
        xanchor="center",
    )
    fig.add_annotation(
        x=2.15,
        y=label_y,
        text=f"<b>R</b><br>1B<br><span style='font-size:11px'>{right_label}</span>",
        showarrow=False,
        font=dict(color="#f5efe3", size=12),
        xanchor="center",
    )
    fig.add_annotation(
        x=0,
        y=0.52,
        text=(
            f"Bats: <b>{'Left' if bats == 'L' else 'Right'}</b> · Catcher's view · "
            "Heart · Shadow · Chase · Waste"
        ),
        showarrow=False,
        font=dict(color="#cbbfa8", size=11),
    )

    legend_y = 4.35
    for zone, x_pos in zip(["Waste", "Chase", "Shadow", "Heart"], [-1.85, -0.65, 0.65, 1.85]):
        fig.add_shape(
            type="rect",
            x0=x_pos - 0.22,
            x1=x_pos + 0.22,
            y0=legend_y - 0.06,
            y1=legend_y + 0.06,
            fillcolor=ZONE_FILL[zone],
            line=dict(color=ZONE_BORDER[zone], width=1),
            layer="above",
        )
        fig.add_annotation(
            x=x_pos,
            y=legend_y - 0.16,
            text=zone,
            showarrow=False,
            font=dict(color="#cbbfa8", size=10),
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[-2.5, 2.5],
            title="Horizontal (catcher's view: L = 3B side, R = 1B side)",
            scaleanchor="y",
            scaleratio=1,
            constrain="domain",
        ),
        yaxis=dict(range=[0.4, 4.55], title="Vertical (ft)", constrain="domain"),
        height=520,
        width=520,
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False,
        dragmode=False,
        clickmode="event+select",
        uirevision="whiff_lab_zone",
    )
    return fig


def miss_distance_inches(plate_x: float, plate_z: float, sz_bot: float, sz_top: float) -> float:
    left, right = -0.708, 0.708
    x_out = max(0, left - plate_x) if plate_x < left else max(0, plate_x - right)
    z_out = max(0, sz_bot - plate_z) if plate_z < sz_bot else max(0, plate_z - sz_top)
    return float(np.sqrt(x_out ** 2 + z_out ** 2) * 12)


def guess_label(prob: float, guess_yes: bool) -> str:
    model_says_yes = prob >= 0.5
    if guess_yes == model_says_yes:
        return "✓ Nice read"
    return "✗ Model disagrees"


def _render_strategy_feedback(
    *,
    has_pitched: bool,
    guess_family: str,
    guess_vertical: str,
    guess_horizontal: str,
    guess_velocity: str,
    rx: dict | None,
    thrown: dict | None,
    cmp: dict | None,
    simulated_outcome: str | None,
    models_live: bool = True,
) -> None:
    """Show best pitch for situation, thrown-pitch outcomes, and strategy comparison."""
    st.markdown("---")
    st.markdown("#### 4 · Results")

    if not has_pitched:
        st.info("Lock in your strategy above, then click **Pitch It!** to see the best pitch for this situation.")
        return

    if rx is None or thrown is None or cmp is None:
        st.info("Lock in your strategy above, then click **Pitch It!** to see the best pitch for this situation.")
        return

    if not models_live:
        st.caption(
            "Using count + hitter zone-rate estimates (ML bundles unavailable or incompatible). "
            "Run `pip install -r requirements.txt` and restart, or retrain with `python notebooks/train_whiff_model.py`."
        )

    st.markdown(
        f"""
        <div class="whiff-prescription-banner">
            <div class="whiff-prescription-label">Best pitch for this situation</div>
            <div class="whiff-prescription-headline">{rx["headline"]}</div>
            <div class="whiff-prescription-meta">
                Count: <b>{rx["count"]}</b> · {rx["velocity_intent"]} intent ·
                xStrike <b>{rx.get("xstrike", rx.get("p_favorable", 0)):.0%}</b>
            </div>
            <div class="whiff-prescription-rationale">{rx["rationale"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Optimal pitch — outcome split**")
    o1, o2, o3, o4, o5 = st.columns(5)
    o1.metric("Take %", f"{rx.get('p_take', 0):.0%}")
    o2.metric("Whiff %", f"{rx.get('p_whiff', rx.get('p_zone_whiff', 0) + rx.get('p_chase_whiff', 0)):.0%}")
    o3.metric("Contact %", f"{rx.get('p_contact', 0):.0%}")
    o4.metric("xStrike %", f"{rx.get('xstrike', 0):.0%}")
    if rx.get("xwoba_con") is not None:
        o5.metric("xwOBAcon", format_xwoba_display(rx["xwoba_con"]))
    else:
        o5.metric("xwOBAcon", "—")

    st.markdown("**Your pitch — outcome split**")
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Take %", f"{thrown.get('p_take', 0):.0%}")
    t2.metric("Whiff %", f"{thrown.get('p_whiff', thrown.get('p_zone_whiff', 0) + thrown.get('p_chase_whiff', 0)):.0%}")
    t3.metric("Contact %", f"{thrown.get('p_contact', 0):.0%}")
    t4.metric("xStrike %", f"{thrown.get('xstrike', 0):.0%}")
    if thrown.get("xwoba_con") is not None:
        t5.metric("xwOBAcon", format_xwoba_display(thrown["xwoba_con"]))
    else:
        t5.metric("xwOBAcon", "—")

    if thrown.get("xwoba_con") is not None:
        tier_key, tier_label, tier_color = xwoba_damage_tier(thrown["xwoba_con"])
        st.markdown(
            f"""
            <div style="margin: 12px 0 16px;">
                <div style="color: var(--whiff-cream-muted); font-size: 0.82rem; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
                    Expected damage on contact (xwOBAcon)
                </div>
                <div class="whiff-xwoba-box whiff-xwoba-{tier_key}"
                     style="background: {tier_color};">
                    {format_xwoba_display(thrown["xwoba_con"])}
                </div>
                <div style="color: var(--whiff-cream-muted); font-size: 0.88rem; margin-top: 8px;">
                    {tier_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if simulated_outcome is None and has_pitched:
        if "lab_rng" not in st.session_state:
            st.session_state.lab_rng = np.random.default_rng()
        simulated_outcome = simulate_outcome(thrown, st.session_state.lab_rng)
        st.session_state.lab_sim_outcome = simulated_outcome

    if simulated_outcome:
        label = OUTCOME_LABELS.get(simulated_outcome, simulated_outcome)
        if outcome_is_favorable(simulated_outcome):
            st.success(f"Simulated outcome: **{label}** ✅ — favorable for the pitcher.")
        else:
            st.error(f"Simulated outcome: **{label}** ❌ — unfavorable for the pitcher.")

    st.markdown("**Your strategy vs. optimal**")
    guess_vert = guess_vertical if guess_vertical != "—" else rx["vertical"]
    guess_horiz = guess_horizontal if guess_horizontal != "—" else rx["horizontal"]
    for label, matched, yours, optimal in (
        ("Pitch family", bool(cmp["family_match"]), guess_family, rx["pitch_family"]),
        ("Location", bool(cmp["location_match"]), f"{guess_horiz} / {guess_vert}", f"{rx['horizontal']} / {rx['vertical']}"),
        ("Velocity intent", bool(cmp["velocity_match"]), guess_velocity, rx["velocity_intent"]),
    ):
        icon = "✅" if matched else "❌"
        st.markdown(f"{icon} **{label}:** {yours} · Best: **{optimal}**")

    score = int(cmp["score"])
    if score == 3:
        st.success(f"Perfect read — {score}/3 strategy elements matched the optimal call.")
    elif score >= 2:
        st.info(f"Strong plan — {score}/3 strategy elements matched.")
    else:
        st.warning(f"Room to improve — {score}/3 strategy elements matched.")


def render_whiff_lab(
    qualified_df: pd.DataFrame | None = None,
    batter_stand: pd.DataFrame | None = None,
) -> None:
    """Interactive Whiff Lab: zone map, pitch setup, optional model reveal."""
    if qualified_df is None or qualified_df.empty:
        qualified_df = load_qualified_hitters()
    if qualified_df.empty:
        st.warning("No qualified hitters found. Run training to export Pitch Lab bundles and hitter zones.")
        return

    if batter_stand is None:
        batter_stand = load_batter_stand()

    swing_bundle, whiff_bundle, xwoba_bundle = load_pitch_lab_models()
    models_ready = swing_bundle is not None and whiff_bundle is not None
    models_live = models_ready and pitch_models_predict_available(swing_bundle, whiff_bundle)

    profiles = load_pitch_lab_profiles()
    zones_df = load_hitter_strike_zones()
    profile_lookup = profiles.get("pitch_types", {})

    if "lab_px" not in st.session_state:
        st.session_state.lab_px = 0.0
        st.session_state.lab_pz = 2.5
    if "lab_zone_pick" not in st.session_state:
        st.session_state.lab_zone_pick = "Heart"
    if "lab_vert_pick" not in st.session_state:
        st.session_state.lab_vert_pick = "Middle"
    if "lab_horiz_pick" not in st.session_state:
        st.session_state.lab_horiz_pick = "Middle"
    if "lab_bat_side" not in st.session_state:
        st.session_state.lab_bat_side = "R"
    if "lab_pending_px" not in st.session_state:
        st.session_state.lab_pending_px = None
        st.session_state.lab_pending_pz = None
    if "lab_has_pitched" not in st.session_state:
        st.session_state.lab_has_pitched = False
    if "lab_run_predictions" not in st.session_state:
        st.session_state.lab_run_predictions = False
    if "lab_location_mode" not in st.session_state:
        st.session_state.lab_location_mode = "Pick a Random Pitch"
    if "lab_guess_family" not in st.session_state:
        st.session_state.lab_guess_family = "Fastball"
    if "lab_guess_vert" not in st.session_state:
        st.session_state.lab_guess_vert = "Middle"
    if "lab_guess_horiz" not in st.session_state:
        st.session_state.lab_guess_horiz = "Middle"
    if "lab_guess_velocity" not in st.session_state:
        st.session_state.lab_guess_velocity = "Average"

    _capture_chart_click_pending()

    st.markdown(
        """
        <div class="methodology-box">
            <h4>Game-planning flow</h4>
            <p>Define the situation, set up location, lock your strategy guess, then click <b>Pitch It!</b>
            The engine finds the <b>best pitch for favorable outcomes</b>: called strikes, in-zone whiffs,
            and chase whiffs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hitters = qualified_df.sort_values("player_name")["player_name"].tolist()
    default_idx = hitters.index("Shohei Ohtani") if "Shohei Ohtani" in hitters else 0

    # --- 1. Define the Situation ---
    st.markdown('<div class="whiff-input-section"><h4>1 · Define the Situation</h4></div>', unsafe_allow_html=True)

    sit_hitter, sit_balls, sit_strikes, sit_runners = st.columns([2.2, 0.7, 0.7, 1.0])
    with sit_hitter:
        hitter = st.selectbox("Hitter", hitters, index=default_idx, key="lab_hitter")
    with sit_balls:
        balls = st.selectbox("Balls", [0, 1, 2, 3], index=2, key="lab_balls")
    with sit_strikes:
        strikes = st.selectbox("Strikes", [0, 1, 2], index=2, key="lab_strikes")
    with sit_runners:
        runners_on = st.selectbox("Runners on", [0, 1, 2, 3], key="lab_runners")

    st.markdown(
        f'<div class="whiff-count-display">{format_count_display(balls, strikes)}</div>',
        unsafe_allow_html=True,
    )

    batter_id = int(qualified_df.loc[qualified_df["player_name"] == hitter, "batter"].iloc[0])
    league_rates = load_league_rates()
    hitter_rates = lookup_hitter_rates(batter_id) or league_rates
    hz1, hz2, hz3 = st.columns(3)
    hz1.metric(
        "In-zone swing %",
        f"{hitter_rates['in_zone_swing_pct']:.1%}",
        delta=f"{(hitter_rates['in_zone_swing_pct'] - league_rates['in_zone_swing_pct']) * 100:+.1f} vs league",
        delta_color="off",
    )
    hz2.metric(
        "O-zone swing %",
        f"{hitter_rates['o_zone_swing_pct']:.1%}",
        delta=f"{(hitter_rates['o_zone_swing_pct'] - league_rates['o_zone_swing_pct']) * 100:+.1f} vs league",
        delta_color="off",
    )
    hz3.caption(
        f"League norms: **{league_rates['in_zone_swing_pct']:.1%}** in-zone · "
        f"**{league_rates['o_zone_swing_pct']:.1%}** O-zone — used to personalize swing decisions."
    )

    raw_bats = batter_handedness(batter_stand, batter_id)
    platoon_bats = resolve_bats_for_platoon(raw_bats)
    sz_bot, sz_top = hitter_strike_zone(zones_df, profiles, batter_id)

    if raw_bats == "S":
        st.info("**Switch hitter** — choose batting side for inside/outside labels.")
        st.radio(
            "Batting side (matchup)",
            options=["R", "L"],
            format_func=lambda s: "Right-handed" if s == "R" else "Left-handed",
            horizontal=True,
            key="lab_bat_side",
        )
        platoon_bats = resolve_bats_for_platoon(raw_bats)

    inside_label, outside_label = plate_side_labels(platoon_bats)

    # --- 2. Setup the Pitch ---
    st.markdown('<div class="whiff-input-section"><h4>2 · Setup the Pitch</h4></div>', unsafe_allow_html=True)

    location_mode = st.radio(
        "Setup method",
        ["Pick a Random Pitch", "Customize Exact Pitch"],
        horizontal=True,
        label_visibility="collapsed",
        key="lab_location_mode",
    )

    if location_mode == "Pick a Random Pitch":
        st.markdown(
            '<div class="whiff-location-mode whiff-location-mode-active">'
            "<b>Random pitch</b> — choose attack zone and placement; location randomizes on <b>Pitch It!</b></div>",
            unsafe_allow_html=True,
        )
        loc_zone, loc_vert, loc_horiz = st.columns(3)
        with loc_zone:
            st.selectbox("Attack zone", ZONE_ORDER, key="lab_zone_pick")
        with loc_vert:
            st.selectbox("Vertical placement", VERTICAL_ORDER, key="lab_vert_pick")
        with loc_horiz:
            st.selectbox(
                f"Horizontal placement ({inside_label} / Middle / {outside_label})",
                HORIZONTAL_ORDER,
                key="lab_horiz_pick",
            )
    else:
        st.markdown(
            '<div class="whiff-location-mode whiff-location-mode-active">'
            "<b>Custom pitch</b> — click the zone map to pin an exact location.</div>",
            unsafe_allow_html=True,
        )

    render_hitter_card(batter_id, hitter, raw_bats)

    pending_x = st.session_state.lab_pending_px
    pending_z = st.session_state.lab_pending_pz

    st.markdown(
        """
        <div class="whiff-location-panel">
            <div class="whiff-location-panel-label">Interactive location selector</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    plate_x = float(st.session_state.lab_px)
    plate_z = float(st.session_state.lab_pz)

    zone_fig = build_location_figure(
        plate_x,
        plate_z,
        sz_bot,
        sz_top,
        bats=platoon_bats,
        attack_zone=assign_attack_zone(plate_x, plate_z, sz_top, sz_bot),
        pending_x=pending_x if location_mode == "Customize Exact Pitch" else None,
        pending_z=pending_z if location_mode == "Customize Exact Pitch" else None,
    )
    st.plotly_chart(
        zone_fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="lab_zone_chart",
    )

    if location_mode == "Customize Exact Pitch":
        if pending_x is not None and pending_z is not None:
            st.caption(f"Selected: **{pending_x:+.2f} ft H · {pending_z:.2f} ft V**")
        else:
            st.caption("Click the zone map to select your exact pitch location.")
    else:
        st.caption("Preview only — final location randomizes within your placement bucket.")

    # --- 3. Formulate Your Strategy (Guess) — lock before pitching ---
    st.markdown(
        '<div class="whiff-input-section"><h4>3 · Formulate Your Strategy (Guess)</h4></div>',
        unsafe_allow_html=True,
    )

    custom_mode = location_mode == "Customize Exact Pitch"
    plot_px = pending_x if pending_x is not None else (
        float(st.session_state.lab_px) if st.session_state.get("lab_has_pitched") else None
    )
    plot_pz = pending_z if pending_z is not None else (
        float(st.session_state.lab_pz) if st.session_state.get("lab_has_pitched") else None
    )
    if custom_mode and plot_px is not None and plot_pz is not None:
        _apply_plot_guess_location(plot_px, plot_pz, sz_bot, sz_top, platoon_bats)

    if custom_mode:
        st.markdown(
            "Lock in pitch family and velocity intent. **Vertical and horizontal location are set by your map click.**"
        )
    else:
        st.markdown("Lock in your full pitch call **before** clicking **Pitch It!**")

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        guess_family = st.selectbox("Pitch family", PITCH_FAMILIES, key="lab_guess_family")
    with g2:
        if custom_mode:
            vert_value = (
                st.session_state.lab_guess_vert
                if plot_px is not None
                else "Click the map"
            )
            st.text_input("Vertical location", value=vert_value, disabled=True)
            guess_vertical = st.session_state.lab_guess_vert if plot_px is not None else vert_value
        else:
            guess_vertical = st.selectbox("Vertical location", VERTICAL_OPTIONS, key="lab_guess_vert")
    with g3:
        if custom_mode:
            horiz_value = (
                st.session_state.lab_guess_horiz
                if plot_px is not None
                else "Click the map"
            )
            st.text_input("Horizontal location", value=horiz_value, disabled=True)
            guess_horizontal = st.session_state.lab_guess_horiz if plot_px is not None else horiz_value
        else:
            guess_horizontal = st.selectbox("Horizontal location", HORIZONTAL_OPTIONS, key="lab_guess_horiz")
    with g4:
        guess_velocity = st.selectbox("Velocity intent", VELOCITY_INTENTS, index=1, key="lab_guess_velocity")

    st.button(
        "Pitch It!",
        type="primary",
        key="lab_pitch_it",
        use_container_width=True,
        on_click=_commit_pitch,
        args=(sz_bot, sz_top, platoon_bats),
    )

    if st.session_state.get("lab_need_map_click"):
        st.warning("Custom pitch mode — click the zone map to select a location before pitching.")

    plate_x = float(st.session_state.lab_px)
    plate_z = float(st.session_state.lab_pz)
    attack_zone = assign_attack_zone(plate_x, plate_z, sz_top, sz_bot)

    if st.session_state.get("lab_run_predictions") and st.session_state.get("lab_has_pitched"):
        _refresh_pitch_results(
            balls=balls,
            strikes=strikes,
            runners_on=runners_on,
            sz_top=sz_top,
            sz_bot=sz_bot,
            bats=platoon_bats,
            plate_x=plate_x,
            plate_z=plate_z,
            guess_family=guess_family,
            guess_vertical=guess_vertical if plot_px is not None or not custom_mode else "—",
            guess_horizontal=guess_horizontal if plot_px is not None or not custom_mode else "—",
            guess_velocity=guess_velocity,
            swing_bundle=swing_bundle,
            whiff_bundle=whiff_bundle,
            xwoba_bundle=xwoba_bundle,
            profile_lookup=profile_lookup,
            hitter_rates=hitter_rates,
            models_live=models_live,
        )

    _render_strategy_feedback(
        has_pitched=bool(st.session_state.get("lab_has_pitched")),
        guess_family=guess_family,
        guess_vertical=guess_vertical if plot_px is not None or not custom_mode else "—",
        guess_horizontal=guess_horizontal if plot_px is not None or not custom_mode else "—",
        guess_velocity=guess_velocity,
        rx=st.session_state.get("lab_pitch_rx"),
        thrown=st.session_state.get("lab_pitch_thrown"),
        cmp=st.session_state.get("lab_pitch_cmp"),
        simulated_outcome=st.session_state.get("lab_sim_outcome"),
        models_live=models_live,
    )

    if models_ready and st.session_state.get("lab_has_pitched"):
        thrown_cached = st.session_state.get("lab_pitch_thrown")
        with st.expander("Model probabilities (optional)"):
            if thrown_cached:
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Take %", f"{thrown_cached.get('p_take', 0) * 100:.1f}%")
                m2.metric("Whiff %", f"{thrown_cached.get('p_whiff', 0) * 100:.1f}%")
                m3.metric("Contact %", f"{thrown_cached.get('p_contact', 0) * 100:.1f}%")
                m4.metric("xStrike %", f"{thrown_cached.get('xstrike', 0) * 100:.1f}%")
                if thrown_cached.get("xwoba_con") is not None:
                    m5.metric("xwOBAcon", format_xwoba_display(thrown_cached["xwoba_con"]))
                else:
                    m5.metric("xwOBAcon", "—")
                st.caption(
                    f"Thrown to {attack_zone} @ {plate_x:+.2f}/{plate_z:.2f} ft · "
                    f"{guess_family} ({guess_velocity}) · in zone: {thrown_cached.get('in_zone', False)}"
                )


def render_pitch_lab(qualified_df: pd.DataFrame, batter_stand: pd.DataFrame | None = None) -> None:
    """Backward-compatible alias."""
    render_whiff_lab(qualified_df, batter_stand)

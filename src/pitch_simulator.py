"""Interactive pitch simulator: user guesses vs model swing / whiff probabilities."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
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
    infer_horizontal,
    infer_vertical,
    random_pitch_location,
    zone_description,
)
from src.hitter_rates import hitter_stats_html
from src.pitch_lab_inputs import (
    PITCH_FAMILIES,
    PITCH_FAMILY_DEFAULT_CODE,
    PITCH_FAMILY_DISPLAY_PITCH,
    VELOCITY_INTENT_TO_TIER,
    apply_physics_tiers,
    count_leverage_label,
    format_count_display,
)
from src.pitch_prescription import (
    HORIZONTAL_OPTIONS,
    VERTICAL_OPTIONS,
    VELOCITY_INTENTS,
    build_target_heatmap_placeholder,
    compare_strategy_guess,
    match_row,
    mock_count_prescription,
)
from src.whiff_features import (
    MODEL_INPUT_COLS,
    PITCH_METRIC_COLS,
    apply_pitch_imputation,
    engineer_features,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "model"
PITCH_LAB_SWING_FILE = MODEL_DIR / "pitch_lab_swing.joblib"
PITCH_LAB_WHIFF_FILE = MODEL_DIR / "pitch_lab_whiff.joblib"
SWING_MODEL_FILE = MODEL_DIR / "swing_model.joblib"
WHIFF_MODEL_FILE = MODEL_DIR / "whiff_model.joblib"
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


@st.cache_resource
def load_model_bundle(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_pitch_lab_models():
    """Prefer lightweight deploy bundles; fall back to full RF models locally."""
    for swing_path, whiff_path in (
        (PITCH_LAB_SWING_FILE, PITCH_LAB_WHIFF_FILE),
        (SWING_MODEL_FILE, WHIFF_MODEL_FILE),
    ):
        swing_bundle = load_model_bundle(swing_path)
        whiff_bundle = load_model_bundle(whiff_path)
        if swing_bundle is not None and whiff_bundle is not None:
            return swing_bundle, whiff_bundle
    return None, None


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


def predict_probs(swing_bundle, whiff_bundle, row: pd.Series) -> dict[str, float]:
    medians = pd.Series(swing_bundle["pitch_medians"])
    frame = apply_pitch_imputation(engineer_features(pd.DataFrame([row])), medians)
    swing_p = float(swing_bundle["model"].predict_proba(frame[MODEL_INPUT_COLS])[:, 1][0])
    whiff_p = float(whiff_bundle["model"].predict_proba(frame[MODEL_INPUT_COLS])[:, 1][0])
    return {
        "swing": swing_p,
        "whiff_if_swing": whiff_p,
        "swing_whiff": swing_p * whiff_p,
        "miss_dist_in": float(frame["miss_dist_in"].iloc[0]),
    }


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
        ),
        yaxis=dict(range=[0.4, 4.55], title="Vertical (ft)"),
        height=520,
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
    balls: int,
    strikes: int,
    runners_on: int,
    guess_family: str,
    guess_vertical: str,
    guess_horizontal: str,
    guess_velocity: str,
) -> None:
    """Show prescription and compare user strategy guesses after Pitch It!"""
    st.markdown("---")
    st.markdown("#### 4 · Results")

    if not has_pitched:
        st.info("Lock in your strategy above, then click **Pitch It!** to reveal the optimal prescription.")
        return

    rx = mock_count_prescription(balls, strikes, runners_on)
    cmp = compare_strategy_guess(
        guess_family=guess_family,
        guess_vertical=guess_vertical,
        guess_horizontal=guess_horizontal,
        guess_velocity=guess_velocity,
        recommendation=rx,
    )

    st.markdown(
        f"""
        <div class="whiff-prescription-banner">
            <div class="whiff-prescription-label">Optimal prescription</div>
            <div class="whiff-prescription-headline">{rx["headline"]}</div>
            <div class="whiff-prescription-meta">
                Count: <b>{rx["count"]}</b> · Velocity intent: <b>{rx["velocity_intent"]}</b> · {rx["confidence"]}
            </div>
            <div class="whiff-prescription-rationale">{rx["rationale"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Your strategy vs. optimal**")
    st.markdown(
        match_row(
            "Pitch Family Match (The Strategy)",
            bool(cmp["family_match"]),
            f"You guessed **{guess_family}** · Optimal: **{rx['pitch_family']}**",
        )
    )
    st.markdown(
        match_row(
            "Location Match (Vertical & Horizontal)",
            bool(cmp["location_match"]),
            "Location drives ~33% of the outcome · "
            f"You guessed **{guess_horizontal} / {guess_vertical}** · "
            f"Optimal: **{rx['horizontal']} / {rx['vertical']}**",
        )
    )
    st.markdown(
        match_row(
            "Velocity Intent Match",
            bool(cmp["velocity_match"]),
            "Speed drives ~19% of whiff execution · "
            f"You guessed **{guess_velocity}** · Optimal: **{rx['velocity_intent']}**",
        )
    )

    score = int(cmp["score"])
    if score == 3:
        st.success(f"Perfect read — {score}/3 strategy elements matched.")
    elif score >= 2:
        st.info(f"Strong plan — {score}/3 strategy elements matched.")
    else:
        st.warning(f"Room to improve — {score}/3 strategy elements matched.")

    st.plotly_chart(build_target_heatmap_placeholder(), use_container_width=True, key="lab_target_heatmap")


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

    swing_bundle, whiff_bundle = load_pitch_lab_models()
    models_ready = swing_bundle is not None and whiff_bundle is not None

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
            <p>Define the situation, set up pitch location, <b>lock your strategy guess</b>, then click
            <b>Pitch It!</b> to see the optimal prescription and how your read compares.</p>
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

    pitch_type = PITCH_FAMILY_DEFAULT_CODE[guess_family]
    speed_tier = VELOCITY_INTENT_TO_TIER[guess_velocity]
    pitch_profile = apply_physics_tiers(
        guess_family,
        release_speed=speed_tier,
        vertical_break="Average",
        spin_rate="Average",
        horizontal_break="Average",
    )
    lab_profile = profile_lookup.get(pitch_type, {})
    if lab_profile:
        for col in PITCH_METRIC_COLS:
            if col in lab_profile and col not in pitch_profile:
                pitch_profile[col] = float(lab_profile[col])

    _render_strategy_feedback(
        has_pitched=bool(st.session_state.get("lab_has_pitched")),
        balls=balls,
        strikes=strikes,
        runners_on=runners_on,
        guess_family=guess_family,
        guess_vertical=guess_vertical if plot_px is not None or not custom_mode else "—",
        guess_horizontal=guess_horizontal if plot_px is not None or not custom_mode else "—",
        guess_velocity=guess_velocity,
    )

    if models_ready and st.session_state.get("lab_has_pitched"):
        with st.expander("Model probabilities (optional)"):
            row = build_pitch_row(
                plate_x, plate_z, sz_bot, sz_top, balls, strikes, runners_on, pitch_type, pitch_profile
            )
            probs = predict_probs(swing_bundle, whiff_bundle, row)
            m1, m2, m3 = st.columns(3)
            m1.metric("P(Swing)", f"{probs['swing'] * 100:.1f}%")
            m2.metric("P(Whiff | Swing)", f"{probs['whiff_if_swing'] * 100:.1f}%")
            m3.metric("P(Swinging Strike)", f"{probs['swing_whiff'] * 100:.1f}%")
            st.caption(f"Thrown to {attack_zone} @ {plate_x:+.2f}/{plate_z:.2f} ft · {guess_family} ({guess_velocity})")


def render_pitch_lab(qualified_df: pd.DataFrame, batter_stand: pd.DataFrame | None = None) -> None:
    """Backward-compatible alias."""
    render_whiff_lab(qualified_df, batter_stand)

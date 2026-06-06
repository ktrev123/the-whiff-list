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
    random_pitch_location,
    zone_description,
)
from src.hitter_rates import hitter_stats_html
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
    "Fastballs": "FF",
    "Breaking Balls": "SL",
    "Off-Speed": "CH",
}

CATEGORY_PITCH_LABELS = {
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
}

PITCH_NAME_TO_TYPE = {
    label: code
    for labels in CATEGORY_PITCH_LABELS.values()
    for label, code in labels.items()
}


def _capture_chart_click_pending() -> None:
    """Remember chart click as pending location — ball moves only on Pitch it."""
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


def _commit_pitch(
    sz_bot: float,
    sz_top: float,
    platoon_bats: str,
) -> None:
    """Apply pending chart click or randomize from placement picks."""
    pending_x = st.session_state.get("lab_pending_px")
    pending_z = st.session_state.get("lab_pending_pz")

    if pending_x is not None and pending_z is not None:
        st.session_state.lab_px = float(pending_x)
        st.session_state.lab_pz = float(pending_z)
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
    st.rerun()


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

    _capture_chart_click_pending()

    st.markdown(
        """
        <div class="methodology-box">
            <h4>How to build a pitch</h4>
            <p>Set up the at-bat below — hitter, pitch type, count, runners, attack zone, and placement.
            The ball <b>does not move</b> until you click <b>Pitch it</b>.</p>
            <ul>
                <li><b>Pick your own spot:</b> click the zone map to mark a location (gold dashed ring),
                then <b>Pitch it</b>.</li>
                <li><b>Random pitch:</b> choose attack zone + placement vertical + placement horizontal,
                then <b>Pitch it</b> for a random spot in that bucket.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hitters = qualified_df.sort_values("player_name")["player_name"].tolist()
    default_idx = hitters.index("Shohei Ohtani") if "Shohei Ohtani" in hitters else 0

    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([2.2, 1.4, 0.8, 0.8, 1.0])
    with ctrl1:
        hitter = st.selectbox("Hitter", hitters, index=default_idx, key="lab_hitter")
    batter_id = int(qualified_df.loc[qualified_df["player_name"] == hitter, "batter"].iloc[0])
    raw_bats = batter_handedness(batter_stand, batter_id)
    platoon_bats = resolve_bats_for_platoon(raw_bats)
    sz_bot, sz_top = hitter_strike_zone(zones_df, profiles, batter_id)

    if raw_bats == "S":
        st.info("**Switch hitter** — choose which side he's batting from for inside/outside on the chart.")
        st.radio(
            "Batting side (matchup)",
            options=["R", "L"],
            format_func=lambda s: "Right-handed" if s == "R" else "Left-handed",
            horizontal=True,
            key="lab_bat_side",
        )
        platoon_bats = resolve_bats_for_platoon(raw_bats)

    with ctrl2:
        pitch_category = st.selectbox(
            "Pitch family",
            list(PITCH_CATEGORY_DEFAULT.keys()),
            key="lab_pitch_category",
        )
    with ctrl3:
        balls = st.selectbox("Balls", [0, 1, 2, 3], index=2, key="lab_balls")
    with ctrl4:
        strikes = st.selectbox("Strikes", [0, 1, 2], index=2, key="lab_strikes")
    with ctrl5:
        runners_on = st.selectbox("Runners on", [0, 1, 2, 3], key="lab_runners")

    pitch_options = CATEGORY_PITCH_LABELS[pitch_category]
    pitch_name = st.selectbox(
        "Specific pitch (league-median physics)",
        list(pitch_options.keys()),
        key="lab_pitch_variant",
    )
    pitch_type = pitch_options[pitch_name]
    pitch_profile = profile_lookup.get(pitch_type, {})
    if not pitch_profile:
        pitch_profile = {col: float(profiles.get(col, 0)) for col in PITCH_METRIC_COLS}

    plate_x = float(st.session_state.lab_px)
    plate_z = float(st.session_state.lab_pz)
    pending_x = st.session_state.lab_pending_px
    pending_z = st.session_state.lab_pending_pz

    inside_label, outside_label = plate_side_labels(platoon_bats)

    st.markdown("**Attack zone**")
    st.radio(
        "Attack zone",
        ZONE_ORDER,
        horizontal=True,
        key="lab_zone_pick",
        label_visibility="collapsed",
    )

    st.markdown("**Placement vertical**")
    st.radio(
        "Placement vertical",
        VERTICAL_ORDER,
        horizontal=True,
        key="lab_vert_pick",
        label_visibility="collapsed",
    )

    st.markdown(
        f"**Placement horizontal** — Inside ({inside_label}) · Middle · Outside ({outside_label})"
    )
    st.radio(
        "Placement horizontal",
        HORIZONTAL_ORDER,
        horizontal=True,
        key="lab_horiz_pick",
        label_visibility="collapsed",
    )

    pitch_ready = st.button("Pitch it", type="primary", key="lab_pitch_it", use_container_width=True)
    if pitch_ready:
        _commit_pitch(sz_bot, sz_top, platoon_bats)

    attack_zone = assign_attack_zone(plate_x, plate_z, sz_top, sz_bot)
    render_hitter_banner(batter_id, hitter, raw_bats, attack_zone, plate_x, plate_z)

    zone_fig = build_location_figure(
        plate_x,
        plate_z,
        sz_bot,
        sz_top,
        bats=platoon_bats,
        attack_zone=attack_zone,
        pending_x=pending_x,
        pending_z=pending_z,
    )
    st.plotly_chart(
        zone_fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="lab_zone_chart",
    )
    if pending_x is not None and pending_z is not None:
        st.caption(
            f"Pending location marked at **{pending_x:+.2f} ft H · {pending_z:.2f} ft V** — click **Pitch it** to throw there."
        )
    else:
        st.caption(
            "Click the zone map to mark your own spot (optional), or use placement picks + **Pitch it** for a random location."
        )

    attack_zone = assign_attack_zone(plate_x, plate_z, sz_top, sz_bot)
    count_label = f"{balls}-{strikes}"
    runners_label = {0: "Bases empty", 1: "Runner on", 2: "Runners on", 3: "Loaded"}.get(runners_on, "")
    st.caption(
        f"**{hitter}** · {pitch_category} ({pitch_name}) · **{count_label}** count · {runners_label} · "
        f"zone sized to hitter height ({sz_bot:.2f}–{sz_top:.2f} ft). "
        f"Physics default: {pitch_profile.get('release_speed', 0):.1f} mph."
    )

    if not models_ready:
        st.info(
            "Model reveal is optional — bundles not found. Run `python notebooks/train_whiff_model.py` "
            "to enable swing / whiff predictions."
        )
        return

    model_note = swing_bundle.get("model_name", "model")
    st.caption(f"Predictions use deployable **{model_note}** models when you reveal below.")

    st.markdown("#### Your guess")
    g1, g2 = st.columns(2)
    guess_swing = g1.radio(f"Will **{hitter}** swing?", ["Yes", "No"], horizontal=True, key="lab_guess_swing")
    swing_no = guess_swing == "No"
    with g2:
        guess_whiff = st.radio(
            "Will it be a whiff?",
            ["Yes", "No"],
            horizontal=True,
            key="lab_guess_whiff",
            disabled=swing_no,
        )
        if swing_no:
            st.caption("Not applicable — you guessed take.")

    reveal = st.button("Reveal model probabilities", type="primary", key="lab_reveal")

    if reveal:
        row = build_pitch_row(
            plate_x, plate_z, sz_bot, sz_top, balls, strikes, runners_on, pitch_type, pitch_profile
        )
        probs = predict_probs(swing_bundle, whiff_bundle, row)

        st.markdown("#### Model says")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("P(Swing)", f"{probs['swing'] * 100:.1f}%")
        m2.metric("P(Whiff | Swing)", f"{probs['whiff_if_swing'] * 100:.1f}%")
        m3.metric("P(Swinging Strike)", f"{probs['swing_whiff'] * 100:.1f}%")
        m4.metric("Attack zone", attack_zone)

        r1, r2 = st.columns(2)
        swing_yes = guess_swing == "Yes"
        whiff_yes = guess_whiff == "Yes" if swing_yes else False
        r1.markdown(
            f"**Swing:** {guess_label(probs['swing'], swing_yes)} — "
            f"model {'expects a swing' if probs['swing'] >= 0.5 else 'expects a take'} "
            f"({probs['swing'] * 100:.0f}%)"
        )
        if swing_yes:
            whiff_yes = guess_whiff == "Yes"
            r2.markdown(
                f"**Whiff:** {guess_label(probs['whiff_if_swing'], whiff_yes)} — "
                f"{'high' if probs['whiff_if_swing'] >= 0.5 else 'low'} miss risk if he swings "
                f"({probs['whiff_if_swing'] * 100:.0f}%)"
            )
        else:
            r2.markdown("**Whiff:** — (you guessed take; whiff only applies on a swing)")

        if probs["swing_whiff"] >= 0.35:
            st.error(f"Ugly pitch profile — {probs['swing_whiff'] * 100:.0f}% swinging-strike probability.")
        elif probs["swing"] < 0.25:
            st.success("Take city — model barely expects a swing.")
        else:
            st.info("Competitive pitch — swing decision is genuinely close.")


def render_pitch_lab(qualified_df: pd.DataFrame, batter_stand: pd.DataFrame | None = None) -> None:
    """Backward-compatible alias."""
    render_whiff_lab(qualified_df, batter_stand)

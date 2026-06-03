"""Interactive pitch simulator: user guesses vs model swing / whiff probabilities."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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

PITCH_NAME_TO_TYPE = {
    "4-Seam Fastball": "FF",
    "Slider": "SL",
    "Changeup": "CH",
    "Curveball": "CU",
    "Sinker": "SI",
    "Cutter": "FC",
    "Sweeper": "SV",
}

LOCATION_PRESETS = {
    "Heart of zone": (0.0, 2.5),
    "Low & away": (1.1, 1.2),
    "High & in": (-0.9, 3.4),
    "Chase below": (0.6, 0.9),
    "Paint the corner": (0.75, 1.35),
}

MLB_HEADSHOT_URL = (
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/{id}/headshot/67/current"
)


def mlb_headshot_url(batter_id: int) -> str:
    return MLB_HEADSHOT_URL.format(id=batter_id)


def batter_handedness(batter_stand: pd.DataFrame | None, batter_id: int) -> str:
    if batter_stand is not None and not batter_stand.empty:
        match = batter_stand.loc[batter_stand["batter"] == batter_id, "bats"]
        if not match.empty:
            return str(match.iloc[0])
    return "R"


def plate_side_labels(bats: str) -> tuple[str, str]:
    """Catcher's view: left = 3B (negative x), right = 1B (positive x)."""
    if bats == "L":
        return "Away", "Inside"
    return "Inside", "Away"


def horizontal_slider_label(bats: str) -> str:
    left_label, right_label = plate_side_labels(bats)
    return f"Horizontal (← {left_label} | {right_label} →)"


def apply_zone_selection(zone_event, plate_x: float, plate_z: float) -> None:
    if not zone_event or not getattr(zone_event, "selection", None):
        return
    points = zone_event.selection.points
    if not points:
        return

    pt = points[-1]
    for candidate in reversed(points):
        if candidate.get("curve_number", 0) == 0:
            pt = candidate
            break

    new_x = round(float(np.clip(float(pt["x"]), -2.0, 2.0)), 2)
    new_z = round(float(np.clip(float(pt["y"]), 0.8, 4.2)), 2)
    if (new_x, new_z) == (round(plate_x, 2), round(plate_z, 2)):
        return
    st.session_state.lab_px = new_x
    st.session_state.lab_pz = new_z
    st.rerun()


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


def build_location_figure(
    plate_x: float,
    plate_z: float,
    sz_bot: float,
    sz_top: float,
    bats: str = "R",
    clickable: bool = True,
) -> go.Figure:
    left_label, right_label = plate_side_labels(bats)
    fig = go.Figure()
    fig.add_shape(
        type="rect",
        x0=-0.708,
        x1=0.708,
        y0=sz_bot,
        y1=sz_top,
        line=dict(color="#f5efe3", width=3),
        fillcolor="rgba(245, 239, 227, 0.06)",
    )
    if clickable:
        xs = np.linspace(-2.0, 2.0, 51)
        zs = np.linspace(0.8, 4.2, 43)
        grid_x, grid_z = np.meshgrid(xs, zs)
        fig.add_trace(
            go.Scatter(
                x=grid_x.ravel(),
                y=grid_z.ravel(),
                mode="markers",
                marker=dict(size=22, color="rgba(255,255,255,0.01)"),
                hoverinfo="skip",
                name="Click to place",
            )
        )
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=plate_x - 0.09,
        x1=plate_x + 0.09,
        y0=plate_z - 0.09,
        y1=plate_z + 0.09,
        fillcolor="#fde725",
        line=dict(color="#0f172a", width=2),
        layer="above",
    )
    label_y = sz_top + 0.22
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
        y=0.62,
        text=f"Bats: <b>{'Left' if bats == 'L' else 'Right'}</b> · Catcher's view",
        showarrow=False,
        font=dict(color="#cbbfa8", size=11),
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
        yaxis=dict(range=[0.5, 4.5], title="Vertical (ft)"),
        height=440,
        margin=dict(t=30, b=20, l=20, r=20),
        showlegend=False,
        dragmode="select",
        clickmode="event+select",
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


def render_pitch_lab(qualified_df: pd.DataFrame, batter_stand: pd.DataFrame | None = None) -> None:
    swing_bundle, whiff_bundle = load_pitch_lab_models()

    if swing_bundle is None or whiff_bundle is None:
        st.info(
            "Pitch Lab models are not bundled for deploy yet. From the project folder run:\n\n"
            "`python notebooks/train_whiff_model.py`\n\n"
            "Then commit `data/model/pitch_lab_*.joblib` and push."
        )
        return

    profiles = load_pitch_lab_profiles()
    zones_df = load_hitter_strike_zones()
    profile_lookup = profiles.get("pitch_types", {})

    if "lab_px" not in st.session_state:
        st.session_state.lab_px = 0.85
        st.session_state.lab_pz = 1.35

    model_note = swing_bundle.get("model_name", "model")
    st.markdown(
        f"""
        <div class="methodology-box">
            <h4>Pitch Lab — build a pitch, make your call</h4>
            <p>Pick a hitter and pitch, set the count, place the ball on the zone (click the chart or use sliders),
            then guess <b>swing</b> and <b>whiff</b> before revealing model probabilities.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Live predictions use deployable **{model_note}** models bundled with the app.")

    hitters = qualified_df.sort_values("player_name")["player_name"].tolist()
    default_idx = hitters.index("Shohei Ohtani") if "Shohei Ohtani" in hitters else 0

    col_setup, col_zone = st.columns([1, 1])

    with col_setup:
        hitter_row = st.columns([3, 1])
        with hitter_row[0]:
            hitter = st.selectbox("Pick your hitter", hitters, index=default_idx, key="lab_hitter")
        batter_id = int(
            qualified_df.loc[qualified_df["player_name"] == hitter, "batter"].iloc[0]
        )
        bats = batter_handedness(batter_stand, batter_id)
        with hitter_row[1]:
            st.image(mlb_headshot_url(batter_id), width=110)
            st.caption(f"Bats **{'L' if bats == 'L' else 'R'}**")

        sz_bot, sz_top = hitter_strike_zone(zones_df, profiles, batter_id)

        pitch_name = st.selectbox("Pick the pitch", list(PITCH_NAME_TO_TYPE.keys()), key="lab_pitch")
        pitch_type = PITCH_NAME_TO_TYPE[pitch_name]
        pitch_profile = profile_lookup.get(pitch_type, {})
        if not pitch_profile:
            pitch_profile = {col: float(profiles.get(col, 0)) for col in PITCH_METRIC_COLS}

        st.caption("Count & leverage")
        c1, c2, c3 = st.columns(3)
        balls = c1.selectbox("Balls", [0, 1, 2, 3], index=2, key="lab_balls")
        strikes = c2.selectbox("Strikes", [0, 1, 2], index=2, key="lab_strikes")
        runners_on = c3.selectbox("Runners on", [0, 1, 2, 3], key="lab_runners")

        st.caption("Quick locations")
        preset_cols = st.columns(len(LOCATION_PRESETS))
        for col, (label, (px, pz)) in zip(preset_cols, LOCATION_PRESETS.items()):
            if col.button(label, key=f"lab_preset_{label}"):
                st.session_state.lab_px = px
                st.session_state.lab_pz = pz
                st.rerun()

        st.caption("Fine-tune location (ft)")
        plate_x = st.slider(
            horizontal_slider_label(bats),
            -2.0,
            2.0,
            float(st.session_state.lab_px),
            0.05,
            key="lab_px",
        )
        plate_z = st.slider(
            "Vertical",
            0.8,
            4.2,
            float(st.session_state.lab_pz),
            0.05,
            key="lab_pz",
        )

    with col_zone:
        zone_fig = build_location_figure(plate_x, plate_z, sz_bot, sz_top, bats=bats)
        zone_event = st.plotly_chart(
            zone_fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="lab_zone_chart",
        )
        apply_zone_selection(zone_event, plate_x, plate_z)
        st.caption(
            f"Click the chart to place the pitch (yellow dot). Zone sized to **{hitter}**'s average height."
        )

    miss_in = miss_distance_inches(plate_x, plate_z, sz_bot, sz_top)
    f1, f2, f3 = st.columns(3)
    f1.metric("Top driver: miss distance", f"{miss_in:.1f} in from zone")
    f2.metric("Top driver: strikes", strikes)
    f3.metric("Top driver: horizontal", f"{plate_x:+.2f} ft")
    st.caption(
        f"Pitch physics default to league-median **{pitch_name}** "
        f"({pitch_profile.get('release_speed', 0):.1f} mph). "
        "Models are league-average — hitter choice sets zone size and context."
    )

    st.markdown("#### Your guess")
    g1, g2 = st.columns(2)
    guess_swing = g1.radio(f"Will **{hitter}** swing?", ["Yes", "No"], horizontal=True, key="lab_guess_swing")
    guess_whiff = g2.radio(
        f"If he swings, will he **whiff**?",
        ["Yes", "No"],
        horizontal=True,
        key="lab_guess_whiff",
    )

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
        m4.metric("Miss distance", f"{probs['miss_dist_in']:.1f} in")

        r1, r2 = st.columns(2)
        swing_yes = guess_swing == "Yes"
        whiff_yes = guess_whiff == "Yes"
        r1.markdown(
            f"**Swing:** {guess_label(probs['swing'], swing_yes)} — "
            f"model {'expects a swing' if probs['swing'] >= 0.5 else 'expects a take'} "
            f"({probs['swing'] * 100:.0f}%)"
        )
        r2.markdown(
            f"**Whiff:** {guess_label(probs['whiff_if_swing'], whiff_yes)} — "
            f"{'high' if probs['whiff_if_swing'] >= 0.5 else 'low'} miss risk if he swings "
            f"({probs['whiff_if_swing'] * 100:.0f}%)"
        )

        if probs["swing_whiff"] >= 0.35:
            st.error(f"Ugly pitch profile — {probs['swing_whiff'] * 100:.0f}% swinging-strike probability.")
        elif probs["swing"] < 0.25:
            st.success("Take city — model barely expects a swing.")
        else:
            st.info("Competitive pitch — swing decision is genuinely close.")

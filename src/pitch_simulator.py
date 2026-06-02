"""Interactive pitch simulator: user guesses vs model swing / whiff probabilities."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.statcast_schema import SEASON_END, SEASON_START
from src.whiff_features import (
    MODEL_INPUT_COLS,
    PITCH_METRIC_COLS,
    apply_pitch_imputation,
    engineer_features,
)

ROOT = Path(__file__).resolve().parents[1]
SWING_MODEL_FILE = ROOT / "data" / "model" / "swing_model.joblib"
WHIFF_MODEL_FILE = ROOT / "data" / "model" / "whiff_model.joblib"

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


@st.cache_resource
def load_model_bundle(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_full_statcast() -> pd.DataFrame:
    path = ROOT / "data" / "statcast_2025.parquet"
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df[(df["game_date"] >= SEASON_START) & (df["game_date"] <= SEASON_END)].copy()


@st.cache_data
def league_pitch_profiles(pitch_data: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        pitch_data.groupby("pitch_type", as_index=False)[PITCH_METRIC_COLS]
        .median(numeric_only=True)
    )
    return grouped


def hitter_strike_zone(pitch_data: pd.DataFrame, batter_id: int) -> tuple[float, float]:
    subset = pitch_data[pitch_data["batter"] == batter_id]
    if subset.empty:
        subset = pitch_data
    return float(subset["sz_bot"].median()), float(subset["sz_top"].median())


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
    clickable: bool = True,
) -> go.Figure:
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
        xs = np.linspace(-2.0, 2.0, 41)
        zs = np.linspace(0.8, 4.2, 35)
        grid_x, grid_z = np.meshgrid(xs, zs)
        fig.add_trace(
            go.Scatter(
                x=grid_x.ravel(),
                y=grid_z.ravel(),
                mode="markers",
                marker=dict(size=10, color="rgba(255,255,255,0.02)"),
                hoverinfo="skip",
                name="Click to place",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[plate_x],
            y=[plate_z],
            mode="markers",
            marker=dict(size=18, color="#fde725", line=dict(color="#0f172a", width=2)),
            name="Your pitch",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-2.5, 2.5], title="Horizontal (ft)", scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0.5, 4.5], title="Vertical (ft)"),
        height=420,
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False,
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


def render_pitch_lab(qualified_df: pd.DataFrame) -> None:
    swing_bundle = load_model_bundle(SWING_MODEL_FILE)
    whiff_bundle = load_model_bundle(WHIFF_MODEL_FILE)

    if swing_bundle is None or whiff_bundle is None:
        st.info("Train models first: `python notebooks/train_whiff_model.py`")
        return

    statcast = load_full_statcast()
    profiles = league_pitch_profiles(statcast)
    profile_lookup = profiles.set_index("pitch_type").to_dict("index")

    if "lab_px" not in st.session_state:
        st.session_state.lab_px = 0.85
        st.session_state.lab_pz = 1.35

    st.markdown(
        """
        <div class="methodology-box">
            <h4>Pitch Lab — build a pitch, make your call</h4>
            <p>Pick a hitter and pitch, set the count, place the ball on the zone (click the chart or use sliders),
            then guess <b>swing</b> and <b>whiff</b> before revealing model probabilities.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hitters = qualified_df.sort_values("player_name")["player_name"].tolist()
    default_idx = hitters.index("Shohei Ohtani") if "Shohei Ohtani" in hitters else 0

    col_setup, col_zone = st.columns([1, 1])

    with col_setup:
        hitter = st.selectbox("Pick your hitter", hitters, index=default_idx, key="lab_hitter")
        batter_id = int(
            qualified_df.loc[qualified_df["player_name"] == hitter, "batter"].iloc[0]
        )
        sz_bot, sz_top = hitter_strike_zone(statcast, batter_id)

        pitch_name = st.selectbox("Pick the pitch", list(PITCH_NAME_TO_TYPE.keys()), key="lab_pitch")
        pitch_type = PITCH_NAME_TO_TYPE[pitch_name]
        pitch_profile = profile_lookup.get(pitch_type, {})
        if not pitch_profile:
            pitch_profile = statcast[PITCH_METRIC_COLS].median(numeric_only=True).to_dict()

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
            "Horizontal (negative = inside to RHH)",
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
        zone_fig = build_location_figure(plate_x, plate_z, sz_bot, sz_top)
        zone_event = st.plotly_chart(
            zone_fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="lab_zone_chart",
        )
        if zone_event and zone_event.selection and zone_event.selection.points:
            pt = zone_event.selection.points[0]
            new_x = round(float(pt["x"]), 2)
            new_z = round(float(pt["y"]), 2)
            if abs(new_x - plate_x) > 0.04 or abs(new_z - plate_z) > 0.04:
                st.session_state.lab_px = new_x
                st.session_state.lab_pz = new_z
                st.rerun()
        st.caption(f"Click the zone or use presets. Sized to **{hitter}**'s average height.")

    miss_in = miss_distance_inches(plate_x, plate_z, sz_bot, sz_top)
    f1, f2, f3 = st.columns(3)
    f1.metric("Top driver: miss distance", f"{miss_in:.1f} in from zone")
    f2.metric("Top driver: strikes", strikes)
    f3.metric("Top driver: horizontal", f"{plate_x:+.2f} ft")
    st.caption(
        f"Pitch physics default to league-median **{pitch_name}** "
        f"({pitch_profile.get('release_speed', 0):.1f} mph). "
        "Models are league-average — hitter choice sets context, not batter-specific coefficients yet."
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

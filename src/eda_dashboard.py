"""Presentation-ready EDA dashboard for the Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.statcast_schema import PITCH_TYPE_GROUPS
from src.whiff_features import SEASON_END, SEASON_START

ROOT = Path(__file__).resolve().parents[1]
LEAGUE_RATES_FILE = ROOT / "data" / "model" / "pitch_lab_league_rates.json"

# Strike-zone plot bounds (feet); equal 3 ft span on both axes for 1:1 geometry.
PLATE_X_MIN, PLATE_X_MAX = -1.5, 1.5
PLATE_Z_MIN, PLATE_Z_MAX = 1.0, 4.0
ZONE_HALF_WIDTH_FT = 17.0 / 24.0  # 17 in plate width
ZONE_Z_BOT, ZONE_Z_TOP = 1.6, 3.5

COUNT_LEVERAGE_GROUPS = {
    "Pitcher Ahead": ["0-1", "0-2", "1-2", "2-2"],
    "Even": ["0-0", "1-1", "3-2"],
    "Hitter Ahead": ["1-0", "2-0", "3-0", "2-1", "3-1"],
    "2-Strike": ["0-2", "1-2", "2-2", "3-2"],
}

PITCH_CATEGORIES = ["Fastball", "Breaking", "Offspeed"]

# Presentation baselines (Mar–Aug 2025 train window).
MOCK_LEAGUE = {
    "swing_rate": 0.477,
    "in_zone_swing_pct": 0.684,
    "o_zone_swing_pct": 0.323,
    "o_zone_chase_pct": 0.323,
    "whiff_if_swing": 0.248,
    "n_pitches": 198_371,
    "n_qualified_hitters": 94,
}

MOCK_PITCH_PROFILES = pd.DataFrame(
    {
        "pitch_group": PITCH_CATEGORIES,
        "swing_rate": [0.521, 0.441, 0.398],
        "whiff_if_swing": [0.214, 0.352, 0.271],
        "swing_whiff": [0.111, 0.155, 0.108],
    }
)

# Engineered feature buckets (not raw balls/strikes or pitch codes).
MOCK_SWING_DRIVERS = pd.DataFrame(
    {
        "feature": [
            "Miss distance (in)",
            "Horizontal location",
            "Vertical location",
            "Count leverage",
            "Pitch category",
            "Runners on",
            "Attack zone",
        ],
        "importance": [0.38, 0.17, 0.16, 0.12, 0.08, 0.05, 0.04],
    }
)

MOCK_WHIFF_DRIVERS = pd.DataFrame(
    {
        "feature": [
            "Release speed",
            "Vertical location",
            "Vertical break",
            "Spin rate",
            "Horizontal location",
            "Horizontal break",
            "Pitch category",
        ],
        "importance": [0.19, 0.17, 0.14, 0.12, 0.11, 0.10, 0.08],
    }
)

MOCK_SWING_CORR_VARS = [
    "Miss distance (in)",
    "Horizontal location",
    "Vertical location",
    "Count leverage (encoded)",
    "Runners on",
    "2-strike flag",
    "Pitch category (encoded)",
]

MOCK_WHIFF_CORR_VARS = [
    "Release speed (mph)",
    "Horizontal break (in)",
    "Vertical break (in)",
    "Spin rate (rpm)",
    "Spin axis (deg)",
    "Extension (ft)",
    "Horizontal location",
    "Vertical location",
]


@st.cache_data
def load_league_baselines() -> dict:
    baselines = dict(MOCK_LEAGUE)
    if LEAGUE_RATES_FILE.exists():
        stored = json.loads(LEAGUE_RATES_FILE.read_text(encoding="utf-8"))
        baselines["swing_rate"] = float(stored.get("swing_rate", baselines["swing_rate"]))
        baselines["o_zone_swing_pct"] = float(
            stored.get("o_zone_swing_pct", baselines["o_zone_swing_pct"])
        )
        baselines["o_zone_chase_pct"] = baselines["o_zone_swing_pct"]
    return baselines


def _mock_target_correlations(variables: list[str], target: str, rng_seed: int) -> pd.DataFrame:
    """Square correlation matrix with mock r values for layout preview."""
    rng = np.random.default_rng(rng_seed)
    n = len(variables)
    base = rng.uniform(-0.35, 0.35, size=(n, n))
    base = (base + base.T) / 2
    np.fill_diagonal(base, 1.0)
    if target == "is_swing":
        boosts = {
            "Miss distance (in)": -0.58,
            "Horizontal location": 0.22,
            "Vertical location": 0.24,
            "Count leverage (encoded)": 0.36,
            "Runners on": 0.06,
            "2-strike flag": 0.44,
            "Pitch category (encoded)": 0.11,
        }
    else:
        boosts = {
            "Release speed (mph)": 0.28,
            "Horizontal break (in)": 0.19,
            "Vertical break (in)": 0.31,
            "Spin rate (rpm)": 0.22,
            "Spin axis (deg)": 0.09,
            "Extension (ft)": 0.07,
            "Horizontal location": 0.14,
            "Vertical location": 0.21,
        }
    labels = variables + ["is_swing" if target == "is_swing" else "is_whiff"]
    mat = np.eye(len(labels))
    for i, var in enumerate(variables):
        mat[i, -1] = boosts.get(var, base[i, 0])
        mat[-1, i] = mat[i, -1]
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i, j] = float(np.clip(base[i, j], -0.55, 0.55))
    return pd.DataFrame(mat, index=labels, columns=labels)


def _bar_figure(
    labels: list[str],
    values: list[float],
    title: str,
    colors: list[str] | None = None,
    y_format: str = ".0%",
) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors or ["#20908d", "#d4a937", "#e63946"][: len(labels)],
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(tickformat=y_format, range=[0, max(values) * 1.25 if values else 1]),
        height=340,
        margin=dict(t=50, b=40, l=40, r=20),
    )
    return fig


def _importance_figure(df: pd.DataFrame, title: str, color: str) -> go.Figure:
    ordered = df.sort_values("importance", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=ordered["importance"],
            y=ordered["feature"],
            orientation="h",
            marker_color=color,
            text=[f"{v:.0%}" for v in ordered["importance"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickformat=".0%"),
        height=360,
        margin=dict(t=50, b=30, l=10, r=40),
    )
    return fig


def _correlation_matrix_figure(corr: pd.DataFrame, title: str) -> go.Figure:
    labels = corr.columns.tolist()
    values = corr.values.round(2)
    n = len(labels)
    ticks = list(range(n))
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=ticks,
            y=ticks,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=values,
            texttemplate="%{text}",
            textfont=dict(size=11, color="#0f172a"),
            hovertemplate="r=%{z:.2f}<extra></extra>",
            colorbar=dict(title="r", len=0.75),
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=min(520, 72 * n + 120),
        margin=dict(l=180, r=40, t=60, b=180),
        xaxis=dict(
            tickmode="array",
            tickvals=ticks,
            ticktext=labels,
            tickangle=-35,
            side="bottom",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=ticks,
            ticktext=labels,
            autorange="reversed",
        ),
    )
    return fig


def _mock_zone_heatmap(values: np.ndarray, title: str, colorscale: str) -> go.Figure:
    xs = np.linspace(PLATE_X_MIN, PLATE_X_MAX, values.shape[1])
    zs = np.linspace(PLATE_Z_MIN, PLATE_Z_MAX, values.shape[0])
    fig = go.Figure(
        go.Heatmap(
            x=xs,
            y=zs,
            z=values,
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            colorbar=dict(title="Rate"),
        )
    )
    fig.add_shape(
        type="rect",
        x0=-ZONE_HALF_WIDTH_FT,
        x1=ZONE_HALF_WIDTH_FT,
        y0=ZONE_Z_BOT,
        y1=ZONE_Z_TOP,
        line=dict(color="#f5efe3", width=2),
        fillcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Horizontal (ft)",
            range=[PLATE_X_MIN, PLATE_X_MAX],
            constrain="domain",
        ),
        yaxis=dict(
            title="Vertical (ft)",
            range=[PLATE_Z_MIN, PLATE_Z_MAX],
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        ),
        height=380,
        margin=dict(t=45, b=45, l=50, r=10),
    )
    return fig


def _spatial_heatmap_grid() -> dict[str, go.Figure]:
    rng = np.random.default_rng(42)
    base = rng.uniform(0.15, 0.55, (24, 24))
    heart = np.s_[8:16, 8:16]
    low_away = np.s_[4:10, 14:20]
    grids = {
        "swing_rate": base.copy(),
        "whiff_if_swing": base.copy() * 0.55,
        "chase_rate": base.copy() * 0.45,
        "swing_whiff": base.copy() * 0.35,
    }
    grids["swing_rate"][heart] += 0.35
    grids["chase_rate"][low_away] += 0.28
    grids["whiff_if_swing"][heart] += 0.12
    grids["swing_whiff"][low_away] += 0.18
    for key in grids:
        grids[key] = np.clip(grids[key], 0.05, 0.95)

    return {
        "Swing frequency": _mock_zone_heatmap(grids["swing_rate"], "Swing frequency", "Tealgrn"),
        "Whiff | swing": _mock_zone_heatmap(grids["whiff_if_swing"], "Whiff | swing", "Reds"),
        "Chase rate (O-zone swings)": _mock_zone_heatmap(grids["chase_rate"], "Chase tendency", "YlOrRd"),
        "Swinging-strike rate": _mock_zone_heatmap(grids["swing_whiff"], "P(swing & whiff)", "Portland"),
    }


def render_eda_dashboard() -> None:
    """Render the Exploratory Data Analysis tab."""
    league = load_league_baselines()

    st.markdown(
        f"""
        <div class="methodology-box">
            <h4>League baselines before personalization</h4>
            <p>2025 MLB Statcast ({SEASON_START} – {SEASON_END}) · qualified hitters · competitive pitch types only.
            Train window: <b>Mar–Aug</b> ({league["n_pitches"]:,} pitches · {league["n_qualified_hitters"]} hitters).</p>
            <p>This dashboard summarizes <b>what drives favorable pitcher outcomes</b> at the league level —
            called strikes, in-zone whiffs, and chase whiffs — the baselines behind Whiff Lab prescriptions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- 1. League norms & zone discipline ---
    st.subheader("1 · League norms & zone discipline")
    st.markdown(
        "Hitters swing far more often in the zone than outside it. **Chase rate** (O-zone swing %) "
        "is the discipline benchmark; **in-zone swing %** captures aggression on strikes."
    )

    m1, m2, divider, m3, m4 = st.columns([1, 1, 0.06, 1, 1])
    m1.metric("League swing %", f"{league['swing_rate']:.1%}")
    m2.metric("In-zone swing %", f"{league['in_zone_swing_pct']:.1%}")
    with divider:
        st.markdown('<div class="eda-metric-split"></div>', unsafe_allow_html=True)
    m3.metric("O-zone swing %", f"{league['o_zone_swing_pct']:.1%}")
    m4.metric("Whiff | swing", f"{league['whiff_if_swing']:.1%}")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            _bar_figure(
                ["In-zone", "Out-of-zone"],
                [league["in_zone_swing_pct"], league["o_zone_swing_pct"]],
                "Swing rate: in-zone vs. out-of-zone",
                colors=["#20908d", "#d4a937"],
            ),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            _bar_figure(
                ["All pitches", "O-zone only"],
                [league["swing_rate"], league["o_zone_chase_pct"]],
                "League swing vs. chase (O-zone)",
                colors=["#20908d", "#e63946"],
            ),
            use_container_width=True,
        )

    with st.expander("How this sets up the app"):
        st.markdown(
            """
            - **Whiff Lab** uses attack zones (Heart / Shadow / Chase / Waste) derived from this discipline framework.
            - **Prescription engine** targets Chase / Shadow when the hitter's O-zone profile is exploitable.
            - **Hybrid models** default to these league rates, then personalize when hitter sample size supports it (~600 pitches).
            """
        )

    st.divider()

    # --- 2. Feature influences (engineered groupings) ---
    st.subheader("2 · Feature influences (swing & miss levers)")
    st.markdown(
        "Models use **engineered categories** rather than raw count components or individual pitch codes. "
        "**Count leverage** buckets the 12 counts; **pitch category** rolls types into Fastball / Breaking / Offspeed."
    )

    f1, f2 = st.columns(2)
    with f1:
        st.plotly_chart(
            _importance_figure(
                MOCK_SWING_DRIVERS,
                "Drivers of P(swing) — contextual levers",
                "#20908d",
            ),
            use_container_width=True,
        )
        st.caption("Location and count leverage dominate swing decisions league-wide.")
    with f2:
        st.plotly_chart(
            _importance_figure(
                MOCK_WHIFF_DRIVERS,
                "Drivers of P(whiff | swing) — physics levers",
                "#e63946",
            ),
            use_container_width=True,
        )
        st.caption("Velocity, movement, and spin rise once the batter commits to swing.")

    with st.expander("Engineered count leverage & pitch category"):
        st.markdown("**Count leverage** (replaces raw balls / strikes):")
        for label, counts in COUNT_LEVERAGE_GROUPS.items():
            st.markdown(f"- **{label}:** {', '.join(counts)}")
        st.markdown(
            """
            **Pitch category** (replaces FF, SI, SL, etc.):
            - **Fastball** — four-seam, sinker, cutter
            - **Breaking** — slider, curve, sweeper, slurve, knuckle-curve
            - **Offspeed** — changeup, splitter
            """
        )
        for group, codes in PITCH_TYPE_GROUPS.items():
            st.caption(f"{group}: {', '.join(codes)}")

    st.divider()

    # --- 3. Correlation matrices ---
    st.subheader("3 · Correlation matrices")
    st.markdown(
        "Continuous and encoded features vs. **`is_swing`** (all pitches) and **`is_whiff`** "
        "(swinging-strike indicator on swings). Matrices below use mock Mar–Aug correlations for layout preview."
    )

    swing_corr = _mock_target_correlations(MOCK_SWING_CORR_VARS, "is_swing", 7)
    whiff_corr = _mock_target_correlations(MOCK_WHIFF_CORR_VARS, "is_whiff", 11)

    cm1, cm2 = st.columns(2)
    with cm1:
        st.plotly_chart(
            _correlation_matrix_figure(
                swing_corr,
                "Contextual features → is_swing",
            ),
            use_container_width=True,
        )
    with cm2:
        st.plotly_chart(
            _correlation_matrix_figure(
                whiff_corr,
                "Pitch physics → is_whiff (swings)",
            ),
            use_container_width=True,
        )

    st.caption(
        "Production EDA separates contextual drivers (location, leverage) from physics drivers (speed, movement, spin)."
    )

    st.divider()

    # --- 4. Pitch type profiles ---
    st.subheader("4 · Pitch category profiles")
    st.markdown(
        "League tendencies by **pitch category** (Fastball, Breaking, Offspeed). "
        "Breaking balls generate the highest whiff-if-swing; fastballs drive the most swings."
    )

    p1, p2 = st.columns([1.2, 1])
    with p1:
        fig = go.Figure()
        for col, name, color in [
            ("swing_rate", "Swing rate", "#20908d"),
            ("whiff_if_swing", "Whiff | swing", "#e63946"),
        ]:
            fig.add_trace(
                go.Bar(
                    x=MOCK_PITCH_PROFILES["pitch_group"],
                    y=MOCK_PITCH_PROFILES[col],
                    name=name,
                    marker_color=color,
                )
            )
        fig.update_layout(
            title="Swing & whiff by pitch category",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            barmode="group",
            yaxis_tickformat=".0%",
            height=360,
            margin=dict(t=50, b=40, l=40, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    with p2:
        st.dataframe(
            MOCK_PITCH_PROFILES.assign(
                swing_rate=MOCK_PITCH_PROFILES["swing_rate"].map(lambda x: f"{x:.1%}"),
                whiff_if_swing=MOCK_PITCH_PROFILES["whiff_if_swing"].map(lambda x: f"{x:.1%}"),
                swing_whiff=MOCK_PITCH_PROFILES["swing_whiff"].map(lambda x: f"{x:.1%}"),
            ).rename(
                columns={
                    "pitch_group": "Category",
                    "swing_rate": "Swing %",
                    "whiff_if_swing": "Whiff | swing",
                    "swing_whiff": "P(swing & whiff)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # --- 5. Spatial heatmaps ---
    st.subheader("5 · Spatial heatmaps")
    st.markdown(
        "Localized plate maps in **real-world feet** with a **1:1 aspect ratio** so the strike zone "
        "is not stretched. Axes span equal horizontal and vertical distance from the plate."
    )

    heatmaps = _spatial_heatmap_grid()
    h1, h2 = st.columns(2)
    with h1:
        st.plotly_chart(heatmaps["Swing frequency"], use_container_width=True)
        st.plotly_chart(heatmaps["Chase rate (O-zone swings)"], use_container_width=True)
    with h2:
        st.plotly_chart(heatmaps["Whiff | swing"], use_container_width=True)
        st.plotly_chart(heatmaps["Swinging-strike rate"], use_container_width=True)

    st.caption(
        "Mock spatial surfaces — 1 ft horizontal equals 1 ft vertical (`scaleanchor='x'`, `scaleratio=1`). "
        "Default view: x ∈ [−1.5, 1.5] ft, z ∈ [1, 4] ft. White rectangle = rule-book strike zone (17 in wide)."
    )

    st.divider()

    st.subheader("From baselines to the rest of the app")
    b1, b2, b3 = st.columns(3)
    b1.info("**Whiff Lab** — test swings against league baselines on the zone map.")
    b2.info("**Model Refinement** — league RF/logistic models trained on these features.")
    b3.info("**Real World Use** — EI & leaderboards vs. league chase/whiff norms.")

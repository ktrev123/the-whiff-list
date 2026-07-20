"""AI Catcher: gamified pitch recommendation dashboard."""

from __future__ import annotations

import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.batter_roster import ROSTER_FILE, load_batter_roster
from src.batter_zone_profiles import (
    DEFAULT_CACHE_PATH as ZONE_CACHE_PATH,
    LEAGUE_ZONE_XWOBAcon,
    batter_zone_lookup,
    load_batter_zone_cache,
)
from src.ev_calculator import load_ev_artifacts
from src.feature_engineering import BASE_STATE_COLUMNS
from src.pitch_recommender import (
    COMPETITIVE_PITCH_TYPES,
    RECOMMENDATION_COLUMNS,
    build_base_state,
    build_count_state,
    generate_strike_zone_grid,
    load_league_average_physics,
    simulate_plate_appearance,
)
from src.statcast_zones import zone_rectangles

st.set_page_config(page_title="AI Catcher: Pitch Optimizer", layout="wide")

MODELS_DIR = ROOT / "models"
MODEL_FILES = (
    "swing_rf_master.joblib",
    "whiff_rf_master.joblib",
    "xwobacon_rf_master.joblib",
)
GITHUB_REPO = "ktrev123/the-whiff-list"
GITHUB_RELEASE_TAG = "v1.0.0"
GITHUB_RELEASE_BASE = (
    f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG}"
)

LEAGUE_SWING_PCT = 47.8
LEAGUE_WHIFF_PCT = 24.5
LEAGUE_O_ZONE_PCT = 31.0
LEAGUE_Z_ZONE_PCT = 68.5
LEAGUE_O_ZONE_WHIFF_PCT = 32.0
LEAGUE_Z_ZONE_WHIFF_PCT = 26.0
LEAGUE_BAT_SPEED = 72.0
LEAGUE_ATTACK_ANGLE = 12.0
LEAGUE_SQUARED_UP_RATE = 28.0

GUESS_PITCH_TYPES = sorted(COMPETITIVE_PITCH_TYPES)

PITCH_CATEGORIES: dict[str, list[tuple[str, str]]] = {
    "Fastballs": [
        ("FF", "Four-Seam Fastball"),
        ("FC", "Cutter"),
        ("SI", "Sinker"),
    ],
    "Horizontal Breaking Balls": [
        ("SL", "Slider"),
        ("ST", "Sweeper"),
        ("SV", "Slurve"),
    ],
    "Vertical Breaking Balls": [
        ("CU", "Curveball"),
        ("KC", "Knuckle Curve"),
    ],
    "Off-Speed": [
        ("CH", "Changeup"),
        ("FS", "Splitter"),
    ],
}

PITCH_LABELS = {
    "FF": "Fastball",
    "SL": "Slider",
    "CH": "Changeup",
    "CU": "Curveball",
    "FC": "Cutter",
    "SI": "Sinker",
    "FS": "Splitter",
    "KC": "Knuckle Curve",
    "ST": "Sweeper",
    "SV": "Slurve",
}

BASE_LABELS = {
    "state_empty": "Bases empty",
    "state_1b": "Runner on 1B",
    "state_2b": "Runner on 2B",
    "state_3b": "Runner on 3B",
    "state_1b_2b": "Runners on 1B & 2B",
    "state_1b_3b": "Runners on 1B & 3B",
    "state_2b_3b": "Runners on 2B & 3B",
    "state_loaded": "Bases loaded",
}

DEFAULT_SZ_TOP = 3.5
DEFAULT_SZ_BOT = 1.5
PLATE_HALF_WIDTH = 17.0 / 24.0
CHART_X_MIN = -1.5
CHART_X_MAX = 1.5
CHART_Z_MIN = 0.5
CHART_Z_MAX = 4.5
CHART_MARGIN = dict(l=8, r=8, t=6, b=32)
CHART_EST_WIDTH_PX = 360
STRIKE_ZONE_COL_RATIOS = [1.2, 2.05, 1.2]
USER_PITCH_MARKER_COLOR = "#ffffff"
USER_PITCH_MARKER_LINE = "#f87171"
# wOBA points from league avg → white; ±span → vibrant blue / red on heatmaps.
XWOBA_GRADIENT_SPAN = 0.12
SWING_GRADIENT_SPAN = 0.12
WHIFF_GRADIENT_SPAN = 0.06
LEAGUE_SWING_RATE = LEAGUE_SWING_PCT / 100.0
LEAGUE_WHIFF_RATE = LEAGUE_WHIFF_PCT / 100.0
LEAGUE_PITCH_WHIFF_RATE = LEAGUE_SWING_RATE * LEAGUE_WHIFF_RATE
ERV_COMPETITIVE_THRESHOLD = 0.010
VIBRANT_BLUE = (37, 99, 235)    # #2563eb
VIBRANT_RED = (220, 38, 38)     # #dc2626
NEUTRAL_WHITE = (255, 255, 255)
LOCATION_TOLERANCE = 0.50  # 6-inch elite command window (feet)
HEADSHOT_WIDTH = 175
BATTER_BOX_HEADSHOT_WIDTH = 170

APP_CSS = """
<style>
    .app-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0.25rem 0 0.35rem 0;
        color: #f8fafc;
        line-height: 1.25;
    }
    .app-subtitle {
        text-align: left;
        color: #94a3b8;
        margin: 0.35rem auto 1.1rem auto;
        font-size: 0.98rem;
        line-height: 1.55;
        max-width: 34rem;
        padding: 0;
    }
    .app-subtitle .step {
        display: flex;
        align-items: flex-start;
        gap: 0.55rem;
        margin: 0.2rem 0;
    }
    .app-subtitle .step-emoji {
        flex: 0 0 1.5rem;
        text-align: center;
        line-height: 1.55;
    }
    .app-subtitle .step-text {
        flex: 1 1 auto;
    }
    .app-subtitle .step-label {
        color: #e2e8f0;
        font-weight: 600;
    }
    .xwoba-heatmap-caption {
        text-align: center;
        margin: 0.35rem 0 0.65rem 0;
    }
    .xwoba-heatmap-caption .xwoba-heatmap-legend {
        margin: 0;
        color: #94a3b8;
        font-size: 0.875rem;
        line-height: 1.45;
    }
    .guessing-heading {
        text-align: center;
        font-size: 1.55rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0.15rem 0 0.65rem 0;
        line-height: 1.35;
    }
    .guessing-situation {
        text-align: center;
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0 0 0.25rem 0;
        line-height: 1.45;
    }
    .guessing-situation strong {
        color: #e2e8f0;
        font-weight: 600;
    }
    .scouting-batter-name {
        text-align: center;
        font-size: 1.65rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0.4rem 0 0.2rem 0;
        line-height: 1.25;
    }
    .scouting-profile-col {
        text-align: center;
        margin: 0 auto;
    }
    .scouting-headshot {
        display: block;
        margin: 0 auto;
        width: 175px;
        max-width: 100%;
        border-radius: 0.35rem;
    }
    .scouting-bats {
        text-align: center;
        font-size: 1.15rem;
        font-weight: 600;
        color: #cbd5e1;
        margin: 0 0 0.35rem 0;
        line-height: 1.3;
    }
    .scouting-report-title {
        text-align: center;
        font-size: 1.35rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 0 0 0.85rem 0;
        line-height: 1.35;
        text-decoration: underline;
        text-underline-offset: 0.28rem;
    }
    .scouting-section-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #94a3b8;
        margin: 0 0 0.35rem 0;
    }
    .xwobacon-box {
        padding: 1rem 1.25rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid rgba(148, 163, 184, 0.35);
        margin: 0.5rem auto 1rem auto;
        max-width: 230px;
    }
    .xwobacon-box .label {
        font-size: 1.25rem;
        font-weight: 600;
        opacity: 0.95;
        margin-bottom: 0.5rem;
    }
    .xwobacon-box .value {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .xwobacon-section-title {
        text-align: center;
        font-size: 1.05rem;
        font-weight: 600;
        margin: 0 0 0.75rem 0;
        color: #f1f5f9;
    }
    .pitch-reveal-section-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 1rem 0;
        color: #f8fafc;
        line-height: 1.25;
    }
    .math-breakdown-title {
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0 0 0.85rem 0;
        color: #f1f5f9;
        line-height: 1.3;
    }
    .strike-zone-preview-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0 0 0.65rem 0;
        color: #f1f5f9;
        line-height: 1.3;
        text-decoration: underline;
        text-underline-offset: 0.28rem;
    }
    .pitch-col-header {
        display: flex;
        align-items: flex-end;
        justify-content: center;
        box-sizing: border-box;
        font-size: 1.05rem;
        font-weight: 600;
        text-align: center;
        margin: 0 0 0.85rem 0;
        padding: 0 0.15rem 0.55rem 0.15rem;
        line-height: 1.25;
        min-height: 3.1rem;
        color: #e2e8f0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.45);
    }
    .pitch-reveal-card {
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 0.5rem;
        padding: 1rem 1.1rem;
        background: rgba(15, 23, 42, 0.45);
        min-height: 11rem;
    }
    .pitch-reveal-card.user {
        border-top: 3px solid #f97316;
    }
    .pitch-reveal-card.ai {
        border-top: 3px solid #22c55e;
    }
    .pitch-reveal-card.ai.competitive {
        border: 1px solid rgba(100, 116, 139, 0.55);
        border-top: 3px solid #64748b;
        background: rgba(15, 23, 42, 0.35);
    }
    .pitch-reveal-card.ai.competitive .card-title,
    .pitch-reveal-card.ai.competitive .pitch-name {
        font-weight: 500;
        color: #94a3b8;
    }
    .pitch-reveal-card.ai.competitive .erv-value {
        font-weight: 600;
        color: #cbd5e1;
    }
    .pitch-reveal-card.ai.high-value {
        border: 2px solid rgba(34, 197, 94, 0.85);
        border-top: 3px solid #4ade80;
        box-shadow:
            0 0 0 1px rgba(34, 197, 94, 0.2),
            0 0 20px rgba(34, 197, 94, 0.16);
        background: rgba(20, 83, 45, 0.18);
    }
    .pitch-reveal-card.ai.high-value .card-title {
        color: #86efac;
    }
    .recommendation-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        margin-bottom: 0.65rem;
        color: #bbf7d0;
        background: rgba(34, 197, 94, 0.22);
        border: 1px solid rgba(74, 222, 128, 0.45);
    }
    .recommendation-status {
        text-align: center;
        padding: 0.9rem 1.1rem;
        border-radius: 0.45rem;
        margin: 0 0 1rem 0;
        line-height: 1.4;
    }
    .recommendation-status .recommendation-headline {
        display: block;
        font-size: 1.2rem;
        font-weight: 700;
        line-height: 1.3;
        margin-bottom: 0.35rem;
    }
    .recommendation-status .recommendation-pitch {
        display: block;
        font-size: 1.45rem;
        font-weight: 700;
        line-height: 1.3;
    }
    .recommendation-status.competitive {
        color: #dcfce7;
        background: rgba(22, 101, 52, 0.35);
        border: 1px solid rgba(74, 222, 128, 0.55);
    }
    .recommendation-status.disagree {
        color: #fecaca;
        background: rgba(127, 29, 29, 0.38);
        border: 1px solid rgba(248, 113, 113, 0.55);
        font-weight: 600;
    }
    .recommendation-status .erv-gap {
        display: block;
        margin-top: 0.45rem;
        font-size: 0.95rem;
        font-weight: 500;
        color: #94a3b8;
    }
    .recommendation-status.competitive .erv-gap {
        color: #86efac;
    }
    .recommendation-status.disagree .erv-gap {
        color: #fca5a5;
    }
    .pitch-feedback {
        text-align: center;
        padding: 0.85rem 1.1rem;
        border-radius: 0.45rem;
        margin: 0 0 1rem 0;
        line-height: 1.45;
    }
    .pitch-feedback ul {
        list-style: disc;
        display: inline-block;
        text-align: left;
        margin: 0 auto;
        padding-left: 1.15rem;
    }
    .pitch-feedback li {
        margin: 0.3rem 0;
        font-size: 0.94rem;
    }
    .pitch-feedback.feedback-success {
        color: #dcfce7;
        background: rgba(22, 101, 52, 0.35);
        border: 1px solid rgba(74, 222, 128, 0.55);
    }
    .pitch-feedback.feedback-warning {
        color: #fef3c7;
        background: rgba(113, 63, 18, 0.42);
        border: 1px solid rgba(251, 191, 36, 0.45);
    }
    .pitch-feedback.feedback-info {
        color: #dbeafe;
        background: rgba(30, 58, 138, 0.38);
        border: 1px solid rgba(96, 165, 250, 0.45);
    }
    #results-strike-zone {
        scroll-margin-top: 1.25rem;
    }
    .results-pitch-summary {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin: 0.35rem 0 0.85rem 0;
    }
    .results-pitch-summary .summary-card {
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 0.45rem;
        padding: 0.75rem 0.85rem;
        background: rgba(15, 23, 42, 0.45);
    }
    .results-pitch-summary .summary-card.user {
        border-top: 3px solid #f97316;
    }
    .results-pitch-summary .summary-card.ai {
        border-top: 3px solid #22c55e;
    }
    .results-pitch-summary .summary-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 0.35rem;
    }
    .results-pitch-summary .summary-type {
        font-size: 1rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 0.2rem;
    }
    .results-pitch-summary .summary-location {
        font-size: 0.92rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .results-pitch-summary .summary-coords {
        display: block;
        margin-top: 0.15rem;
        font-size: 0.76rem;
        font-weight: 400;
        color: #94a3b8;
    }
    .batter-box-side {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        min-height: 100%;
    }
    .batter-box-side img {
        display: block;
        width: 170px;
        max-width: 100%;
        height: auto;
        border-radius: 0.45rem;
        border: 2px solid rgba(148, 163, 184, 0.4);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.35);
    }
    div[data-testid="stColumn"]:has(.batter-box-side) > div[data-testid="stVerticalBlock"] {
        justify-content: center;
        min-height: 100%;
    }
    div[data-testid="stVerticalBlock"]:has(.stPlotlyChart) {
        gap: 0;
    }
    div[data-testid="element-container"]:has(.stPlotlyChart) {
        margin-top: -0.35rem;
        margin-bottom: -0.35rem;
    }
    .pitch-reveal-card .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0 0 0.75rem 0;
        color: #f8fafc;
    }
    .pitch-reveal-card .pitch-name {
        font-size: 1.05rem;
        font-weight: 600;
        margin: 0 0 0.65rem 0;
        color: #e2e8f0;
    }
    .pitch-reveal-card .detail-row {
        font-size: 0.92rem;
        color: #cbd5e1;
        margin: 0.35rem 0;
    }
    .pitch-reveal-card .location-desc {
        font-size: 0.98rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .pitch-reveal-card .location-raw {
        display: block;
        font-size: 0.76rem;
        font-weight: 400;
        color: #94a3b8;
        margin-top: 0.2rem;
    }
    .pitch-reveal-card .erv-row {
        margin-top: 0.75rem;
        padding: 0.65rem 0.75rem;
        border-radius: 0.4rem;
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    .pitch-reveal-card .erv-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #94a3b8;
        margin-bottom: 0.25rem;
    }
    .pitch-reveal-card .erv-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .pitch-reveal-card .erv-smoothed-label {
        font-size: 0.72rem;
        font-weight: 500;
        color: #94a3b8;
    }
    .pitch-reveal-card .erv-raw {
        font-size: 0.76rem;
        color: #94a3b8;
        margin-top: 0.35rem;
    }
    .pitch-reveal-card .physics {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 0.75rem;
    }
</style>
"""

@st.cache_data
def get_batter_roster(_roster_mtime: float) -> dict[str, dict]:
    return load_batter_roster()


def _file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def roster_cache_mtime() -> float:
    return _file_mtime(ROSTER_FILE)


def zone_cache_mtime() -> float:
    return _file_mtime(ZONE_CACHE_PATH)


def batter_profile_from_roster(profile: dict) -> dict[str, float]:
    """Map UI scouting percentages to model rolling-metric decimals."""
    return {
        "swing_pct": profile["swing_pct"] / 100.0,
        "whiff_pct": profile["whiff_pct"] / 100.0,
        "z_swing_pct": profile["z_zone_pct"] / 100.0,
        "o_swing_pct": profile["o_zone_pct"] / 100.0,
        "z_whiff_pct": profile["z_whiff_pct"] / 100.0,
        "o_whiff_pct": profile["o_whiff_pct"] / 100.0,
        "bat_speed": profile.get("bat_speed", LEAGUE_BAT_SPEED),
        "attack_angle": profile.get("attack_angle", LEAGUE_ATTACK_ANGLE),
        "squared_up_rate": profile.get("squared_up_rate", LEAGUE_SQUARED_UP_RATE) / 100.0,
    }


def metric_color_rgb(
    value: float,
    *,
    league: float,
    span: float,
) -> tuple[int, int, int]:
    """Vibrant blue (below avg) → white (avg) → vibrant red (above avg)."""
    delta = float(np.clip((value - league) / span, -1.0, 1.0))
    if delta <= 0.0:
        mix = delta + 1.0
        return tuple(
            int(VIBRANT_BLUE[i] + mix * (NEUTRAL_WHITE[i] - VIBRANT_BLUE[i]))
            for i in range(3)
        )
    mix = delta
    return tuple(
        int(NEUTRAL_WHITE[i] + mix * (VIBRANT_RED[i] - NEUTRAL_WHITE[i]))
        for i in range(3)
    )


def metric_rgba(
    value: float,
    *,
    league: float,
    span: float,
    alpha: float = 0.88,
) -> str:
    r, g, b = metric_color_rgb(value, league=league, span=span)
    return f"rgba({r},{g},{b},{alpha})"


def metric_text_color(value: float, *, league: float, span: float) -> str:
    delta = abs(value - league)
    return "#0f172a" if delta < span * 0.45 else "#f8fafc"


def render_metric_compare_box(
    value: float,
    *,
    caption: str,
    league: float,
    span: float,
    fmt: str,
) -> None:
    bg = metric_rgba(value, league=league, span=span)
    fg = metric_text_color(value, league=league, span=span)
    st.html(
        f'<div class="xwobacon-box" style="background:{bg}; color:{fg};">'
        f'<div class="label">{caption}</div>'
        f'<div class="value">{fmt.format(value)}</div>'
        f"</div>"
    )


def render_pitch_reveal_card(
    title: str,
    row: pd.Series,
    physics_df: pd.DataFrame,
    *,
    variant: str,
    stand: str,
    ai_emphasis: str | None = None,
) -> None:
    pitch_label = PITCH_LABELS.get(row["Type"], row["Type"])
    physics = format_league_pitch_physics(row["Type"], physics_df)
    location_html = format_location_display(
        float(row["plate_x"]),
        float(row["plate_z"]),
        stand=stand,
    )
    emphasis_class = ""
    badge_html = ""
    if variant == "ai" and ai_emphasis == "competitive":
        emphasis_class = " competitive"
    elif variant == "ai" and ai_emphasis == "high_value":
        emphasis_class = " high-value"
        badge_html = (
            '<div class="recommendation-badge">High-Value Opportunity</div>'
        )
    st.html(
        f'<div class="pitch-reveal-card {variant}{emphasis_class}">'
        f"{badge_html}"
        f'<div class="card-title">{title}</div>'
        f'<div class="pitch-name">{pitch_label} ({row["Type"]})</div>'
        f'<div class="detail-row">{location_html}</div>'
        f'<div class="erv-row">'
        f'<div class="erv-label">Expected run value</div>'
        f'<div class="erv-value">{row["ERV"]:+.4f} '
        f'<span class="erv-smoothed-label">smoothed</span></div>'
        f'<div class="detail-row erv-raw">'
        f"Raw ERV: {row['ERV_raw']:+.4f}</div>"
        f"</div>"
        f'<div class="physics">{physics}</div>'
        f"</div>"
    )


def erv_smoothed_delta(user_row: pd.Series, optimal: pd.Series) -> float:
    """User smoothed ERV minus AI optimal smoothed ERV (positive = AI is better)."""
    return float(user_row["ERV"]) - float(optimal["ERV"])


def format_ai_pitch_summary(row: pd.Series, *, stand: str) -> str:
    pitch_label = PITCH_LABELS.get(row["Type"], row["Type"])
    location = describe_plate_location(
        float(row["plate_x"]),
        float(row["plate_z"]),
        stand=stand,
    )
    return f"{pitch_label} ({row['Type']}) - {location}"


def render_results_pitch_summary(
    user_row: pd.Series,
    optimal: pd.Series,
    *,
    stand: str,
) -> None:
    """Compact type + location lines for the results strike zone block."""
    cards: list[str] = []
    for title, row, variant in (
        ("Your pitch", user_row, "user"),
        ("AI Optimal", optimal, "ai"),
    ):
        pitch_label = PITCH_LABELS.get(row["Type"], row["Type"])
        location = describe_plate_location(
            float(row["plate_x"]),
            float(row["plate_z"]),
            stand=stand,
        )
        coords = format_location(row)
        cards.append(
            f'<div class="summary-card {variant}">'
            f'<div class="summary-label">{title}</div>'
            f'<div class="summary-type">{pitch_label} ({row["Type"]})</div>'
            f'<div class="summary-location">{location}'
            f'<span class="summary-coords">{coords}</span>'
            f"</div></div>"
        )
    st.html(f'<div class="results-pitch-summary">{"".join(cards)}</div>')


def render_recommendation_status(
    delta: float,
    optimal: pd.Series,
    *,
    stand: str,
) -> None:
    ai_better = delta >= ERV_COMPETITIVE_THRESHOLD
    summary = format_ai_pitch_summary(optimal, stand=stand)
    gap_html = (
        f'<span class="erv-gap">ERV Gap: {delta:+.4f} runs</span>'
    )
    if ai_better:
        headline = "Your Catcher doesn't agree"
        status_class = "disagree"
    else:
        headline = "Your pitch is competitive."
        status_class = "competitive"
    st.html(
        f'<div class="recommendation-status {status_class}">'
        f'<span class="recommendation-headline">{headline}</span>'
        f'<span class="recommendation-pitch">{summary}</span>'
        f"{gap_html}"
        f"</div>"
    )


def render_you_vs_ai_metric_section(
    title: str,
    user_value: float,
    ai_value: float,
    *,
    league: float,
    span: float,
    fmt: str,
) -> None:
    with st.container(border=True):
        st.html(f'<p class="xwobacon-section-title">{title}</p>')
        box_left, box_right = st.columns(2)
        with box_left:
            render_metric_compare_box(
                user_value,
                caption="Your pitch",
                league=league,
                span=span,
                fmt=fmt,
            )
        with box_right:
            render_metric_compare_box(
                ai_value,
                caption="AI Catcher",
                league=league,
                span=span,
                fmt=fmt,
            )


def xwobacon_color_rgb(
    value: float,
    *,
    league: float = LEAGUE_ZONE_XWOBAcon,
    span: float = XWOBA_GRADIENT_SPAN,
) -> tuple[int, int, int]:
    return metric_color_rgb(value, league=league, span=span)


def xwobacon_rgba(
    value: float,
    *,
    alpha: float = 0.62,
    league: float = LEAGUE_ZONE_XWOBAcon,
    span: float = XWOBA_GRADIENT_SPAN,
) -> str:
    r, g, b = xwobacon_color_rgb(value, league=league, span=span)
    return f"rgba({r},{g},{b},{alpha})"


def xwobacon_text_color(value: float) -> str:
    return metric_text_color(
        value, league=LEAGUE_ZONE_XWOBAcon, span=XWOBA_GRADIENT_SPAN
    )


def render_xwobacon_contact_box(value: float, *, caption: str) -> None:
    """Colored box for model xwOBAcon if contact occurs at this pitch."""
    render_metric_compare_box(
        value,
        caption=caption,
        league=LEAGUE_ZONE_XWOBAcon,
        span=XWOBA_GRADIENT_SPAN,
        fmt="{:.3f}",
    )


def scroll_to_strike_zone_if_requested() -> None:
    """Scroll the main view to the results strike zone once after Throw Pitch!."""
    if not st.session_state.pop("scroll_to_strike_zone", False):
        return
    components.html(
        """
        <script>
            (function () {
                function scrollToStrikeZone() {
                    const doc = window.parent.document;
                    const target = doc.getElementById("results-strike-zone");
                    if (target) {
                        target.scrollIntoView({ behavior: "instant", block: "start" });
                    }
                }
                scrollToStrikeZone();
                setTimeout(scrollToStrikeZone, 120);
            })();
        </script>
        """,
        height=0,
    )


def init_session_state() -> None:
    x_opts, z_opts = grid_axis_options()
    defaults = {
        "game_phase": "setup",
        "optimal_data": None,
        "user_target": None,
        "simulation_df": None,
        "user_pitch_type": GUESS_PITCH_TYPES[0],
        "selected_batter": None,
        "balls": 0,
        "strikes": 0,
        "outs": 0,
        "base_state": "state_empty",
        "p_throws": "R",
        "pick_plate_x": x_opts[len(x_opts) // 2],
        "pick_plate_z": z_opts[len(z_opts) // 2],
        "scroll_to_strike_zone": False,
        "pitch_type_chosen": False,
        "location_chosen": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    sim_df = st.session_state.get("simulation_df")
    if sim_df is not None and not simulation_has_current_schema(sim_df):
        st.session_state.simulation_df = None
        st.session_state.optimal_data = None


def simulation_has_current_schema(sim_df: pd.DataFrame | None) -> bool:
    return (
        sim_df is not None
        and isinstance(sim_df, pd.DataFrame)
        and list(sim_df.columns) == RECOMMENDATION_COLUMNS
    )


def run_ai_simulation(profile: dict) -> pd.DataFrame:
    return simulate_plate_appearance(
        stand=profile["stand"],
        count_state=build_count_state(st.session_state.balls, st.session_state.strikes),
        base_state=build_base_state(st.session_state.base_state),
        batter_profile=batter_profile_from_roster(profile),
        p_throws=st.session_state.p_throws,
        pitch_types=sorted(COMPETITIVE_PITCH_TYPES),
        artifacts=get_ev_artifacts(),
        league_physics=get_league_physics(),
        profile=False,
        batter_id=str(profile["mlb_id"]),
    )


def ensure_current_simulation(profile: dict) -> pd.DataFrame:
    sim_df = st.session_state.simulation_df
    if simulation_has_current_schema(sim_df):
        return sim_df

    sim_df = run_ai_simulation(profile)
    st.session_state.simulation_df = sim_df
    st.session_state.optimal_data = sim_df.iloc[0]
    return sim_df


@st.cache_resource
def ensure_model_artifacts() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    missing = [name for name in MODEL_FILES if not (MODELS_DIR / name).exists()]
    if not missing:
        return MODELS_DIR

    for name in missing:
        dest = MODELS_DIR / name
        url = f"{GITHUB_RELEASE_BASE}/{name}"
        try:
            urllib.request.urlretrieve(url, dest)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise FileNotFoundError(
                f"Could not download {name} from {url}. "
                f"Train models locally (notebooks 05/07/09) or publish release assets. "
                f"Original error: {exc}"
            ) from exc
    return MODELS_DIR


@st.cache_resource
def get_ev_artifacts():
    ensure_model_artifacts()
    return load_ev_artifacts(MODELS_DIR)


@st.cache_resource
def get_league_physics():
    return load_league_average_physics()


@st.cache_data
def get_chart_grid() -> pd.DataFrame:
    grid = generate_strike_zone_grid(sz_top=DEFAULT_SZ_TOP, sz_bot=DEFAULT_SZ_BOT)
    return grid[(grid["plate_z"] >= CHART_Z_MIN) & (grid["plate_z"] <= CHART_Z_MAX)].copy()


def grid_axis_options() -> tuple[list[float], list[float]]:
    grid = get_chart_grid()
    x_opts = [float(v) for v in sorted(grid["plate_x"].unique())]
    z_opts = [float(v) for v in sorted(grid["plate_z"].unique())]
    return x_opts, z_opts


def snap_to_nearest_grid(plate_x: float, plate_z: float) -> tuple[float, float]:
    """Snap a chart click to the nearest simulated plate-location grid point."""
    x_opts, z_opts = grid_axis_options()
    snap_x = min(x_opts, key=lambda value: abs(value - plate_x))
    snap_z = min(z_opts, key=lambda value: abs(value - plate_z))
    return snap_x, snap_z


def reset_pitch_selection() -> None:
    st.session_state.pitch_type_chosen = False
    st.session_state.location_chosen = False


def reset_for_new_batter() -> None:
    """Clear results / locked situation when the batter dropdown changes."""
    st.session_state.game_phase = "setup"
    reset_pitch_selection()
    st.session_state.optimal_data = None
    st.session_state.user_target = None
    st.session_state.simulation_df = None
    st.session_state.scroll_to_strike_zone = False


def pitch_selection_complete() -> bool:
    return bool(st.session_state.pitch_type_chosen and st.session_state.location_chosen)


def apply_chart_location_pick(chart_key: str) -> bool:
    """Update session location from the latest Plotly point selection."""
    chart_state = st.session_state.get(chart_key)
    if chart_state is None:
        return False
    selection = chart_state.selection
    if not selection or not selection.points:
        return False

    point = selection.points[0]
    snap_x, snap_z = snap_to_nearest_grid(float(point["x"]), float(point["y"]))
    if (
        np.isclose(snap_x, st.session_state.pick_plate_x, atol=1e-6)
        and np.isclose(snap_z, st.session_state.pick_plate_z, atol=1e-6)
    ):
        return False

    st.session_state.pick_plate_x = snap_x
    st.session_state.pick_plate_z = snap_z
    st.session_state.location_chosen = True
    sync_user_target_from_sliders()
    return True


def render_location_picker(
    zone_map: dict[int, float],
    *,
    chart_key: str,
    show_target: bool = False,
    batter_profile: dict | None = None,
) -> None:
    """Interactive strike-zone picker: click to place, snap to nearest grid cell."""
    x_opts, z_opts = grid_axis_options()
    if st.session_state.pick_plate_x not in x_opts:
        st.session_state.pick_plate_x = x_opts[len(x_opts) // 2]
    if st.session_state.pick_plate_z not in z_opts:
        st.session_state.pick_plate_z = z_opts[len(z_opts) // 2]

    sync_user_target_from_sliders()
    target = st.session_state.user_target
    show_user_marker = chart_key == "location_picker_setup" or st.session_state.location_chosen

    st.caption(
        "Click anywhere on the chart to place your pitch. "
        "The marker snaps to the nearest available grid location."
    )
    if show_target and pitch_selection_complete():
        st.caption(
            f"Target: **{PITCH_LABELS.get(target['pitch_type'], target['pitch_type'])}** "
            f"({target['pitch_type']}) at "
            f"({target['plate_x']:+.2f} ft, {target['plate_z']:+.2f} ft)"
        )
    render_xwoba_heatmap_caption()
    location_changed = render_strike_zone_chart(
        user_x=target["plate_x"] if show_user_marker else None,
        user_z=target["plate_z"] if show_user_marker else None,
        zone_xwobacon=zone_map,
        interactive=True,
        chart_key=chart_key,
        batter_profile=batter_profile,
    )
    if location_changed:
        st.rerun()


def mlb_headshot_url(mlb_id: int) -> str:
    return (
        "https://img.mlbstatic.com/mlb-photos/image/upload/"
        f"w_360,q_auto:best/v1/people/{mlb_id}/headshot/67/current"
    )


def render_xwoba_heatmap_caption() -> None:
    st.html(
        '<div class="xwoba-heatmap-caption">'
        '<p class="xwoba-heatmap-legend">'
        "xwOBAcon Heatmap (🔴 Above Average, ⚪ Average, 🔵 Below Average)."
        "</p>"
        "</div>"
    )


def stand_label(stand: str) -> str:
    return "Right" if stand == "R" else "Left"


def count_label() -> str:
    return f"{st.session_state.balls}-{st.session_state.strikes}"


def outs_label(outs: int | None = None) -> str:
    value = int(st.session_state.outs if outs is None else outs)
    return "1 out" if value == 1 else f"{value} outs"


def situation_summary_html() -> str:
    """Display-only situation line (outs do not affect ERV / model scoring)."""
    return (
        f'Situation: <strong>{count_label()}</strong>, '
        f'<strong>{BASE_LABELS[st.session_state.base_state]}</strong>, '
        f'<strong>{outs_label()}</strong>, vs. '
        f'<strong>{st.session_state.selected_batter}</strong>'
    )


def randomize_situation() -> None:
    st.session_state.balls = random.randint(0, 3)
    st.session_state.strikes = random.randint(0, 2)
    st.session_state.outs = random.randint(0, 2)
    st.session_state.base_state = random.choice(BASE_STATE_COLUMNS)
    st.session_state.p_throws = random.choice(["R", "L"])


def format_location(row: pd.Series) -> str:
    return f"({row['plate_x']:+.2f} ft, {row['plate_z']:+.2f} ft)"


def describe_plate_location(
    plate_x: float,
    plate_z: float,
    *,
    stand: str,
    sz_top: float = DEFAULT_SZ_TOP,
    sz_bot: float = DEFAULT_SZ_BOT,
) -> str:
    """Plain-language plate location from batter/pitcher perspective."""
    zone_h = sz_top - sz_bot

    if plate_z <= sz_bot + 0.40 * zone_h:
        vertical = "Low"
    elif plate_z >= sz_bot + 0.60 * zone_h:
        vertical = "Up"
    else:
        vertical = "Middle"

    if abs(plate_x) <= 0.32:
        horizontal = "Middle"
    elif stand == "R":
        horizontal = "Away" if plate_x > 0.12 else "In"
    else:
        horizontal = "Away" if plate_x < -0.12 else "In"

    if horizontal == "Middle" and vertical == "Middle":
        return "Middle-Middle"
    if horizontal == "Middle":
        return f"Middle and {vertical}"
    if vertical == "Middle":
        return f"{horizontal} and Middle"
    return f"{vertical} and {horizontal}"


def format_location_display(
    plate_x: float,
    plate_z: float,
    *,
    stand: str,
) -> str:
    """HTML snippet: descriptive label + smaller raw coordinates."""
    desc = describe_plate_location(plate_x, plate_z, stand=stand)
    raw = f"({plate_x:+.2f} ft, {plate_z:+.2f} ft)"
    return (
        f'<span class="location-desc">{desc}</span>'
        f'<span class="location-raw">{raw}</span>'
    )


def spatial_miss_distance(user_x: float, user_z: float, opt_x: float, opt_z: float) -> float:
    """Euclidean distance in feet between user and optimal plate coordinates."""
    return float(np.hypot(user_x - opt_x, user_z - opt_z))


def whiff_if_swing_prob(row: pd.Series) -> float:
    """P(whiff | swing) for a scored pitch row."""
    p_swing = float(row["Swing Prob"])
    if p_swing < 0.01:
        return 0.0
    return float(row["Whiff Prob"]) / p_swing


def _pitch_outcome_probs(user_row: pd.Series) -> tuple[float, float, float]:
    """Return (p_swing, p_take, p_whiff_if_swing) for a scored pitch row."""
    p_swing = float(user_row["Swing Prob"])
    p_take = 1.0 - p_swing
    p_whiff_if_swing = whiff_if_swing_prob(user_row)
    return p_swing, p_take, p_whiff_if_swing


def out_of_zone_bullets(user_row: pd.Series, *, stand: str) -> list[str]:
    """Short bullet lines for pitches outside the zone."""
    if int(user_row["InZ"]) == 1:
        return []

    p_swing, p_take, p_whiff_if_swing = _pitch_outcome_probs(user_row)
    location = describe_plate_location(
        float(user_row["plate_x"]),
        float(user_row["plate_z"]),
        stand=stand,
    )

    if p_swing < 0.22:
        return [
            f"Missed zone: **{location}**",
            f"**{p_take:.0%}** take · **{p_swing:.0%}** chase — playing for the ball",
        ]
    if p_swing >= 0.40 and p_whiff_if_swing >= 0.30:
        return [
            f"Missed zone: **{location}** — viable chase spot",
            f"**{p_swing:.0%}** chase · **{p_whiff_if_swing:.0%}** whiff if they swing",
        ]
    if p_swing >= 0.40:
        return [
            f"Missed zone: **{location}**",
            f"**{p_swing:.0%}** chase · **{p_whiff_if_swing:.0%}** whiff if they swing",
        ]
    return [
        f"Missed zone: **{location}**",
        f"**{p_take:.0%}** take · **{p_swing:.0%}** chase · **{p_whiff_if_swing:.0%}** whiff if swing",
    ]


def _feedback_bullets(*parts: str | list[str]) -> list[str]:
    bullets: list[str] = []
    for part in parts:
        if isinstance(part, list):
            bullets.extend(item.strip() for item in part if item.strip())
        elif part and part.strip():
            bullets.append(part.strip())
    return bullets


def _inline_bold_html(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)


def render_pitch_feedback(bullets: list[str], tone: str) -> None:
    if not bullets:
        return
    tone_class = {
        "success": "feedback-success",
        "warning": "feedback-warning",
        "info": "feedback-info",
    }.get(tone, "feedback-info")
    items = "".join(
        f"<li>{_inline_bold_html(bullet)}</li>" for bullet in bullets
    )
    st.html(f'<div class="pitch-feedback {tone_class}"><ul>{items}</ul></div>')


def evaluate_pitch_guess(
    user_row: pd.Series,
    optimal: pd.Series,
    *,
    erv_delta: float,
    stand: str,
) -> dict[str, object]:
    user_type = str(user_row["Type"])
    user_x = float(user_row["plate_x"])
    user_z = float(user_row["plate_z"])
    zone_bullets = out_of_zone_bullets(user_row, stand=stand)

    opt_type = str(optimal["Type"])
    opt_x = float(optimal["plate_x"])
    opt_z = float(optimal["plate_z"])

    good_type = user_type == opt_type
    distance_ft = spatial_miss_distance(user_x, user_z, opt_x, opt_z)
    good_location = distance_ft <= LOCATION_TOLERANCE
    competitive = erv_delta < ERV_COMPETITIVE_THRESHOLD

    opt_label = PITCH_LABELS.get(opt_type, opt_type)
    ai_summary = format_ai_pitch_summary(optimal, stand=stand)
    ai_location = describe_plate_location(opt_x, opt_z, stand=stand)

    if good_type and good_location:
        bullets = _feedback_bullets(
            zone_bullets,
            "Perfect match — pitch type and location",
        )
        tone = "success"
    elif good_type and not good_location:
        if competitive:
            bullets = _feedback_bullets(
                zone_bullets,
                "Right pitch type — location is competitive",
            )
            tone = "info"
        elif zone_bullets:
            bullets = _feedback_bullets(
                zone_bullets,
                f"AI prefers **{ai_location}**",
            )
            tone = "warning"
        else:
            bullets = _feedback_bullets(
                zone_bullets,
                f"Better location: **{ai_location}**",
            )
            tone = "warning"
    elif not good_type and good_location:
        if competitive:
            bullets = _feedback_bullets(
                zone_bullets,
                f"Strong spot — competitive with **{opt_label} ({opt_type})**",
            )
            tone = "info"
        else:
            bullets = _feedback_bullets(
                zone_bullets,
                f"Try **{ai_summary}** for more ERV",
            )
            tone = "warning"
    elif competitive:
        bullets = _feedback_bullets(
            zone_bullets,
            "Competitive pitch — not the AI optimal",
        )
        tone = "info"
    else:
        bullets = _feedback_bullets(
            zone_bullets,
            f"AI pick: **{ai_summary}**",
        )
        tone = "warning"

    return {
        "bullets": bullets,
        "tone": tone,
    }


def lookup_scenario(
    df: pd.DataFrame,
    pitch_type: str,
    plate_x: float,
    plate_z: float,
) -> pd.Series:
    mask = (
        (df["Type"] == pitch_type)
        & np.isclose(df["plate_x"], plate_x, atol=0.01)
        & np.isclose(df["plate_z"], plate_z, atol=0.01)
    )
    matches = df.loc[mask]
    if matches.empty:
        raise ValueError(
            f"No scenario found for {pitch_type} at ({plate_x}, {plate_z})."
        )
    return matches.iloc[0]


def format_league_pitch_physics(pitch_type: str, physics_df: pd.DataFrame) -> str:
    """League-average velocity and spin for display (does not affect ERV)."""
    row = physics_df.loc[physics_df["pitch_type"] == pitch_type].iloc[0]
    speed = float(row["release_speed"])
    spin = float(row["release_spin_rate"])
    label = PITCH_LABELS.get(pitch_type, pitch_type)
    return f"{label}: {speed:.1f} mph, {spin:.0f} rpm"


@st.cache_data
def get_batter_zone_cache(_cache_mtime: float) -> pd.DataFrame | None:
    if not ZONE_CACHE_PATH.exists():
        return None
    return load_batter_zone_cache(ZONE_CACHE_PATH)


@st.cache_data
def get_batter_zone_heatmap(mlb_id: int, _cache_mtime: float) -> dict[int, float]:
    cache = get_batter_zone_cache(_cache_mtime)
    if cache is None:
        return {zone: LEAGUE_ZONE_XWOBAcon for zone in list(range(1, 10)) + [11, 12, 13, 14]}
    return batter_zone_lookup(mlb_id, cache)


def add_zone_xwobacon_heatmap(fig: go.Figure, zone_xwobacon: dict[int, float]) -> None:
    """13 MLB zone tiles: 3×3 heart plus four outer quadrants (11–14)."""
    for rect in zone_rectangles(
        sz_top=DEFAULT_SZ_TOP,
        sz_bot=DEFAULT_SZ_BOT,
    ):
        value = zone_xwobacon.get(rect.zone, LEAGUE_ZONE_XWOBAcon)
        fig.add_shape(
            type="rect",
            x0=rect.x0,
            x1=rect.x1,
            y0=rect.z0,
            y1=rect.z1,
            line=dict(width=0),
            fillcolor=xwobacon_rgba(value),
            layer="below",
        )


def strike_zone_chart_height(est_width_px: float = CHART_EST_WIDTH_PX) -> int:
    """Match figure height to data aspect ratio so the plot fills its container."""
    plot_width = max(est_width_px - CHART_MARGIN["l"] - CHART_MARGIN["r"], 180)
    z_span = CHART_Z_MAX - CHART_Z_MIN
    x_span = CHART_X_MAX - CHART_X_MIN
    plot_height = plot_width * z_span / x_span
    return int(plot_height + CHART_MARGIN["t"] + CHART_MARGIN["b"])


def build_strike_zone_figure(
    *,
    user_x: float | None = None,
    user_z: float | None = None,
    ai_x: float | None = None,
    ai_z: float | None = None,
    zone_xwobacon: dict[int, float] | None = None,
    show_click_targets: bool = False,
) -> go.Figure:
    fig = go.Figure()

    if zone_xwobacon:
        add_zone_xwobacon_heatmap(fig, zone_xwobacon)

    fig.add_shape(
        type="rect",
        x0=-PLATE_HALF_WIDTH,
        x1=PLATE_HALF_WIDTH,
        y0=DEFAULT_SZ_BOT,
        y1=DEFAULT_SZ_TOP,
        line=dict(color="#e2e8f0", width=2.5),
        fillcolor="rgba(15, 23, 42, 0.0)",
        layer="below",
    )

    if user_x is not None and user_z is not None:
        fig.add_trace(
            go.Scatter(
                x=[user_x],
                y=[user_z],
                mode="markers+text",
                marker=dict(
                    size=18,
                    color=USER_PITCH_MARKER_COLOR,
                    line=dict(width=2.5, color=USER_PITCH_MARKER_LINE),
                ),
                text=["You"],
                textposition="top center",
                textfont=dict(color="#e2e8f0", size=11),
                name="Your pitch",
                hovertemplate="%{x:.2f} ft, %{y:.2f} ft<extra></extra>",
            )
        )

    if ai_x is not None and ai_z is not None:
        fig.add_trace(
            go.Scatter(
                x=[ai_x],
                y=[ai_z],
                mode="markers+text",
                marker=dict(size=16, color="#22c55e", symbol="diamond", line=dict(width=2, color="#ffffff")),
                text=["AI"],
                textposition="bottom center",
                name="AI optimal",
                hovertemplate="%{x:.2f} ft, %{y:.2f} ft<extra></extra>",
            )
        )

    if show_click_targets:
        grid = get_chart_grid()
        fig.add_trace(
            go.Scatter(
                x=grid["plate_x"],
                y=grid["plate_z"],
                mode="markers",
                marker=dict(size=40, color="rgba(0,0,0,0)", line=dict(width=0)),
                hovertemplate="Place pitch: %{x:.2f} ft, %{y:.2f} ft<extra></extra>",
                name="Click to place",
                showlegend=False,
            )
        )

    show_legend = user_x is not None or ai_x is not None
    fig.update_layout(
        xaxis=dict(
            title="Horizontal (ft)",
            range=[CHART_X_MIN, CHART_X_MAX],
            dtick=0.5,
            constrain="domain",
            autorange=False,
        ),
        yaxis=dict(
            title="Vertical (ft)",
            range=[CHART_Z_MIN, CHART_Z_MAX],
            dtick=0.5,
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
            autorange=False,
        ),
        height=strike_zone_chart_height(),
        margin=CHART_MARGIN,
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(15, 23, 42, 0.72)",
            bordercolor="rgba(148, 163, 184, 0.35)",
            borderwidth=1,
            font=dict(size=10, color="#e2e8f0"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.35)",
    )
    return fig


def render_batter_box_image(profile: dict) -> None:
    """Batter headshot beside the plate on their box side (catcher's view)."""
    url = mlb_headshot_url(profile["mlb_id"])
    st.html(
        f'<div class="batter-box-side"><img src="{url}" alt="Batter" '
        f'width="{BATTER_BOX_HEADSHOT_WIDTH}" /></div>'
    )


def render_strike_zone_chart(
    *,
    user_x: float | None = None,
    user_z: float | None = None,
    ai_x: float | None = None,
    ai_z: float | None = None,
    zone_xwobacon: dict[int, float] | None = None,
    interactive: bool = False,
    chart_key: str = "strike_zone_chart",
    batter_profile: dict | None = None,
) -> bool:
    fig = build_strike_zone_figure(
        user_x=user_x,
        user_z=user_z,
        ai_x=ai_x,
        ai_z=ai_z,
        zone_xwobacon=zone_xwobacon,
        show_click_targets=interactive,
    )
    left_pad, center, right_pad = st.columns(
        STRIKE_ZONE_COL_RATIOS, vertical_alignment="center"
    )
    if batter_profile is not None:
        # Catcher's view: LHB box is on the right (+x), RHB box is on the left (-x).
        if batter_profile["stand"] == "L":
            with right_pad:
                render_batter_box_image(batter_profile)
        else:
            with left_pad:
                render_batter_box_image(batter_profile)
    with center:
        if interactive:
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
                on_select="rerun",
                selection_mode="points",
                key=chart_key,
            )
            return apply_chart_location_pick(chart_key)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    return False


def scouting_metric_delta_kwargs(delta: float, *, inverse: bool = False) -> dict:
    """Format league-average delta for scouting st.metric calls."""
    if round(delta, 1) == 0:
        return {
            "delta": "0.0%",
            "delta_color": "off",
            "delta_arrow": "off",
        }
    kwargs: dict = {"delta": f"{delta:+.1f}%"}
    if inverse:
        kwargs["delta_color"] = "inverse"
    return kwargs


def render_scouting_card(batter_name: str, profile: dict) -> None:
    swing_delta = round(profile["swing_pct"] - LEAGUE_SWING_PCT, 1)
    whiff_delta = round(profile["whiff_pct"] - LEAGUE_WHIFF_PCT, 1)
    o_delta = round(profile["o_zone_pct"] - LEAGUE_O_ZONE_PCT, 1)
    z_delta = round(profile["z_zone_pct"] - LEAGUE_Z_ZONE_PCT, 1)
    o_whiff_delta = round(profile["o_whiff_pct"] - LEAGUE_O_ZONE_WHIFF_PCT, 1)
    z_whiff_delta = round(profile["z_whiff_pct"] - LEAGUE_Z_ZONE_WHIFF_PCT, 1)

    img_col, stats_col = st.columns([1, 2], vertical_alignment="center")
    with img_col:
        headshot_url = mlb_headshot_url(profile["mlb_id"])
        st.html(
            '<div class="scouting-profile-col">'
            f'<img class="scouting-headshot" src="{headshot_url}" '
            f'width="{HEADSHOT_WIDTH}" alt="{batter_name}" />'
            f'<p class="scouting-batter-name">{batter_name}</p>'
            f'<p class="scouting-bats">Bats: {stand_label(profile["stand"])}</p>'
            "</div>"
        )
    with stats_col:
        st.html(
            '<p class="scouting-report-title">Scouting Report vs. League Average (2025)</p>'
        )

        with st.container(border=True):
            st.html('<p class="scouting-section-label">Swing Decisions</p>')
            swing_cols = st.columns(3)
            swing_cols[0].metric(
                "Swing%",
                f"{profile['swing_pct']:.1f}%",
                **scouting_metric_delta_kwargs(swing_delta),
            )
            swing_cols[1].metric(
                "O-Zone%",
                f"{profile['o_zone_pct']:.1f}%",
                **scouting_metric_delta_kwargs(o_delta, inverse=True),
            )
            swing_cols[2].metric(
                "Z-Zone%",
                f"{profile['z_zone_pct']:.1f}%",
                **scouting_metric_delta_kwargs(z_delta),
            )

        with st.container(border=True):
            st.html('<p class="scouting-section-label">Whiff Rates</p>')
            whiff_cols = st.columns(3)
            whiff_cols[0].metric(
                "Whiff%",
                f"{profile['whiff_pct']:.1f}%",
                **scouting_metric_delta_kwargs(whiff_delta, inverse=True),
            )
            whiff_cols[1].metric(
                "O-Zone Whiff%",
                f"{profile['o_whiff_pct']:.1f}%",
                **scouting_metric_delta_kwargs(o_whiff_delta, inverse=True),
            )
            whiff_cols[2].metric(
                "Z-Zone Whiff%",
                f"{profile['z_whiff_pct']:.1f}%",
                **scouting_metric_delta_kwargs(z_whiff_delta, inverse=True),
            )


def render_pitch_type_picker() -> None:
    """Four-column pitch picker: category header on top, buttons stacked below."""
    cols = st.columns(4, gap="small")
    for col, (category, pitches) in zip(cols, PITCH_CATEGORIES.items()):
        with col:
            st.html(f'<p class="pitch-col-header">{category}</p>')
            for code, label in pitches:
                selected = st.session_state.user_pitch_type == code
                if st.button(
                    label,
                    use_container_width=True,
                    key=f"pitch_btn_{code}",
                    type="primary" if selected else "secondary",
                ):
                    st.session_state.user_pitch_type = code
                    st.session_state.pitch_type_chosen = True
                    sync_user_target_from_sliders()
                    st.rerun()


def sync_user_target_from_sliders() -> None:
    st.session_state.user_target = {
        "pitch_type": st.session_state.user_pitch_type,
        "plate_x": float(st.session_state.pick_plate_x),
        "plate_z": float(st.session_state.pick_plate_z),
    }


def render_setup_sidebar(profile: dict) -> None:
    if st.sidebar.button("Randomize Situation", use_container_width=True):
        randomize_situation()

    ball_col, strike_col, outs_col = st.sidebar.columns(3)
    ball_col.selectbox("Balls", options=[0, 1, 2, 3], key="balls")
    strike_col.selectbox("Strikes", options=[0, 1, 2], key="strikes")
    outs_col.selectbox("Outs*", options=[0, 1, 2], key="outs")
    st.sidebar.selectbox(
        "Base state*",
        options=list(BASE_LABELS.keys()),
        format_func=lambda key: BASE_LABELS[key],
        key="base_state",
    )
    st.sidebar.selectbox(
        "Pitcher throws",
        options=["R", "L"],
        format_func=lambda hand: "Right (RHP)" if hand == "R" else "Left (LHP)",
        key="p_throws",
    )
    st.sidebar.caption(
        "*Outs and base state do not meaningfully influence swing, whiff, or damage "
        "in this analysis."
    )

    if st.sidebar.button("Lock In Situation & Ask AI", type="primary", use_container_width=True):
        with st.spinner("Running AI Catcher grid search..."):
            sim_df = run_ai_simulation(profile)

        st.session_state.simulation_df = sim_df
        st.session_state.optimal_data = sim_df.iloc[0]
        st.session_state.game_phase = "guessing"
        reset_pitch_selection()
        sync_user_target_from_sliders()
        st.rerun()


def render_guessing_panel(profile: dict) -> None:
    zone_map = get_batter_zone_heatmap(profile["mlb_id"], zone_cache_mtime())

    with st.container(border=True):
        st.html(
            '<p class="guessing-heading">'
            "The AI Catcher has locked in its pitch.<br>"
            "What pitch will you throw and where?"
            "</p>"
        )
        st.html(f'<p class="guessing-situation">{situation_summary_html()}</p>')

    with st.container(border=True):
        st.markdown("**Step 2: Select your pitch type and location**")
        render_pitch_type_picker()

    with st.container(border=True):
        render_location_picker(
            zone_map,
            chart_key="location_picker_guess",
            show_target=True,
            batter_profile=profile,
        )

    throw_disabled = not pitch_selection_complete()
    if st.button("Throw Pitch!", type="primary", disabled=throw_disabled):
        st.session_state.game_phase = "results"
        st.session_state.scroll_to_strike_zone = True
        st.rerun()


def render_results_panel(profile: dict) -> None:
    zone_map = get_batter_zone_heatmap(profile["mlb_id"], zone_cache_mtime())
    sim_df = ensure_current_simulation(profile)
    optimal = st.session_state.optimal_data
    target = st.session_state.user_target
    physics_df = get_league_physics()

    user_row = lookup_scenario(
        sim_df,
        target["pitch_type"],
        target["plate_x"],
        target["plate_z"],
    )
    erv_delta = erv_smoothed_delta(user_row, optimal)
    ai_emphasis = "high_value" if erv_delta >= ERV_COMPETITIVE_THRESHOLD else "competitive"
    guess = evaluate_pitch_guess(
        user_row,
        optimal,
        erv_delta=erv_delta,
        stand=profile["stand"],
    )

    with st.container(border=True):
        st.html('<div id="results-strike-zone"></div>')
        render_xwoba_heatmap_caption()
        render_results_pitch_summary(user_row, optimal, stand=profile["stand"])
        render_strike_zone_chart(
            user_x=target["plate_x"],
            user_z=target["plate_z"],
            ai_x=optimal["plate_x"],
            ai_z=optimal["plate_z"],
            zone_xwobacon=zone_map,
            batter_profile=profile,
        )

    scroll_to_strike_zone_if_requested()

    with st.container(border=True):
        st.html('<p class="pitch-reveal-section-title">Step 3: Pitch Reveal</p>')
        render_recommendation_status(
            erv_delta,
            optimal,
            stand=profile["stand"],
        )
        left, right = st.columns(2, gap="medium")
        with left:
            render_pitch_reveal_card(
                "Your pitch",
                user_row,
                physics_df,
                variant="user",
                stand=profile["stand"],
            )
        with right:
            render_pitch_reveal_card(
                "AI Optimal",
                optimal,
                physics_df,
                variant="ai",
                stand=profile["stand"],
                ai_emphasis=ai_emphasis,
            )

    render_you_vs_ai_metric_section(
        "Swing% — You vs. AI Catcher",
        float(user_row["Swing Prob"]),
        float(optimal["Swing Prob"]),
        league=LEAGUE_SWING_RATE,
        span=SWING_GRADIENT_SPAN,
        fmt="{:.1%}",
    )

    render_you_vs_ai_metric_section(
        "Whiff% (if swing) — You vs. AI Catcher",
        whiff_if_swing_prob(user_row),
        whiff_if_swing_prob(optimal),
        league=LEAGUE_WHIFF_RATE,
        span=WHIFF_GRADIENT_SPAN,
        fmt="{:.1%}",
    )

    render_you_vs_ai_metric_section(
        "xwOBAcon — You vs. AI Catcher",
        float(user_row["xwobacon"]),
        float(optimal["xwobacon"]),
        league=LEAGUE_ZONE_XWOBAcon,
        span=XWOBA_GRADIENT_SPAN,
        fmt="{:.3f}",
    )

    with st.container(border=True):
        render_pitch_feedback(guess["bullets"], str(guess["tone"]))

    table = pd.DataFrame(
        {
            "Pitcher": ["Your pitch", "AI optimal"],
            "Type": [user_row["Type"], optimal["Type"]],
            "Location": [format_location(user_row), format_location(optimal)],
            "Swing Prob": [user_row["Swing Prob"], optimal["Swing Prob"]],
            "Whiff% (if swing)": [
                whiff_if_swing_prob(user_row),
                whiff_if_swing_prob(optimal),
            ],
            "Swing & Whiff Prob": [user_row["Whiff Prob"], optimal["Whiff Prob"]],
            "xwOBAcon": [user_row["xwobacon"], optimal["xwobacon"]],
            "ERV (smoothed)": [user_row["ERV"], optimal["ERV"]],
            "ERV (raw)": [user_row["ERV_raw"], optimal["ERV_raw"]],
        }
    )
    display = table.copy()
    display["Swing Prob"] = display["Swing Prob"].map(lambda v: f"{v:.3f}")
    display["Whiff% (if swing)"] = display["Whiff% (if swing)"].map(lambda v: f"{v:.3f}")
    display["Swing & Whiff Prob"] = display["Swing & Whiff Prob"].map(
        lambda v: f"{v:.3f}"
    )
    display["xwOBAcon"] = display["xwOBAcon"].map(lambda v: f"{v:.3f}")
    display["ERV (smoothed)"] = display["ERV (smoothed)"].map(lambda v: f"{v:+.4f}")
    display["ERV (raw)"] = display["ERV (raw)"].map(lambda v: f"{v:+.4f}")

    with st.container(border=True):
        st.html('<p class="math-breakdown-title">Math Behind the Models</p>')
        st.dataframe(display, use_container_width=True, hide_index=True)

    if st.button("Next Pitch", type="primary"):
        st.session_state.game_phase = "setup"
        reset_pitch_selection()
        st.session_state.optimal_data = None
        st.session_state.user_target = None
        st.session_state.simulation_df = None
        st.rerun()


def main() -> None:
    init_session_state()
    st.markdown(APP_CSS, unsafe_allow_html=True)

    st.html(
        '<p class="app-title">⚾ AI Catcher: Pitch Optimizer ⚾</p>'
        '<div class="app-subtitle">'
        '<div class="step"><span class="step-emoji">🔒</span>'
        '<span class="step-text"><span class="step-label">Step 1:</span> '
        "Lock in the Batter and situation (or select a random one)</span></div>"
        '<div class="step"><span class="step-emoji">🎯</span>'
        '<span class="step-text"><span class="step-label">Step 2:</span> '
        "Select your pitch type and location</span></div>"
        '<div class="step"><span class="step-emoji">&#129302;</span>'
        '<span class="step-text"><span class="step-label">Step 3:</span> '
        "See how your pitch compares to the AI Catcher's optimal choice.</span></div>"
        "</div>"
    )

    try:
        ensure_model_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.title("Step 1: Batter & Situation")
    try:
        batter_roster = get_batter_roster(roster_cache_mtime())
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    batter_names = sorted(batter_roster.keys())
    if (
        st.session_state.selected_batter is None
        or st.session_state.selected_batter not in batter_roster
    ):
        st.session_state.selected_batter = batter_names[0]

    batter_name = st.sidebar.selectbox(
        "Select batter",
        options=batter_names,
        key="selected_batter",
        on_change=reset_for_new_batter,
    )
    profile = batter_roster[batter_name]
    zone_map = get_batter_zone_heatmap(profile["mlb_id"], zone_cache_mtime())

    with st.container(border=True):
        render_scouting_card(batter_name, profile)

    st.sidebar.divider()
    render_setup_sidebar(profile)

    phase = st.session_state.game_phase

    if phase == "setup":
        with st.container(border=True):
            last_name = batter_name.split()[-1]
            st.html('<p class="strike-zone-preview-title">Strike Zone Preview</p>')
            st.write(
                f'Heatmap shows where **{last_name}** does damage on contact (xwOBAcon).'
            )
            render_location_picker(
                zone_map,
                chart_key="location_picker_setup",
                batter_profile=profile,
            )
    elif phase == "guessing":
        render_guessing_panel(profile)
    else:
        render_results_panel(profile)


if __name__ == "__main__":
    main()

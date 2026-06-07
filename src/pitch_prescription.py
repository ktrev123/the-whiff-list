"""Mock swing-and-miss prescription engine for Whiff Lab (no model inference yet)."""

from __future__ import annotations

import plotly.graph_objects as go

PITCH_FAMILIES = ["Fastball", "Breaking", "Offspeed"]
VERTICAL_OPTIONS = ["High", "Middle", "Low"]
HORIZONTAL_OPTIONS = ["Inside", "Middle", "Outside"]
VELOCITY_INTENTS = ["Firm", "Average", "Slow"]

# Optimal strategy by exact count (mock lookup table).
COUNT_PRESCRIPTION: dict[tuple[int, int], dict[str, str]] = {
    (0, 0): {"pitch_family": "Fastball", "vertical": "Middle", "horizontal": "Middle", "velocity_intent": "Average"},
    (0, 1): {"pitch_family": "Fastball", "vertical": "High", "horizontal": "Inside", "velocity_intent": "Firm"},
    (0, 2): {"pitch_family": "Breaking", "vertical": "Low", "horizontal": "Outside", "velocity_intent": "Slow"},
    (1, 0): {"pitch_family": "Fastball", "vertical": "Middle", "horizontal": "Middle", "velocity_intent": "Average"},
    (1, 1): {"pitch_family": "Breaking", "vertical": "Middle", "horizontal": "Outside", "velocity_intent": "Average"},
    (1, 2): {"pitch_family": "Breaking", "vertical": "Low", "horizontal": "Outside", "velocity_intent": "Slow"},
    (2, 0): {"pitch_family": "Fastball", "vertical": "High", "horizontal": "Middle", "velocity_intent": "Firm"},
    (2, 1): {"pitch_family": "Offspeed", "vertical": "Low", "horizontal": "Outside", "velocity_intent": "Slow"},
    (2, 2): {"pitch_family": "Breaking", "vertical": "Low", "horizontal": "Outside", "velocity_intent": "Average"},
    (3, 0): {"pitch_family": "Fastball", "vertical": "Middle", "horizontal": "Middle", "velocity_intent": "Average"},
    (3, 1): {"pitch_family": "Offspeed", "vertical": "Low", "horizontal": "Outside", "velocity_intent": "Slow"},
    (3, 2): {"pitch_family": "Breaking", "vertical": "Low", "horizontal": "Outside", "velocity_intent": "Firm"},
}


def mock_count_prescription(balls: int, strikes: int, runners_on: int = 0) -> dict[str, str]:
    """Return optimal pitch strategy for the exact count (mock rules engine)."""
    base = dict(COUNT_PRESCRIPTION.get((balls, strikes), COUNT_PRESCRIPTION[(0, 0)]))
    count = f"{balls}-{strikes}"

    if runners_on >= 2 and strikes < 2:
        base = {
            "pitch_family": "Fastball",
            "vertical": "High",
            "horizontal": "Middle",
            "velocity_intent": "Firm",
        }
        rationale = f"Runners in scoring position ({count}) — elevated fastball for a swing-and-miss or weak contact."
    elif strikes == 2:
        rationale = (
            f"Two-strike count ({count}) — bury a {base['velocity_intent'].lower()} "
            f"{base['pitch_family'].lower()} {base['horizontal'].lower()} / {base['vertical'].lower()} for chase whiff upside."
        )
    elif balls > strikes:
        rationale = (
            f"Hitter's count ({count}) — {base['pitch_family'].lower()} "
            f"{base['horizontal'].lower()} / {base['vertical'].lower()} to steal a strike or induce weak contact."
        )
    else:
        rationale = (
            f"Pitcher's count ({count}) — attack with {base['pitch_family'].lower()} "
            f"{base['horizontal'].lower()} / {base['vertical'].lower()} at {base['velocity_intent'].lower()} intent."
        )

    headline = (
        f"Recommended Pitch: {base['pitch_family']} — "
        f"{base['horizontal']} / {base['vertical']}"
    )
    return {
        **base,
        "count": count,
        "headline": headline,
        "rationale": rationale,
        "confidence": "Mock · count lookup",
    }


def compare_strategy_guess(
    *,
    guess_family: str,
    guess_vertical: str,
    guess_horizontal: str,
    guess_velocity: str,
    recommendation: dict[str, str],
) -> dict[str, object]:
    """Compare user guesses against the optimal prescription."""
    family_match = guess_family == recommendation["pitch_family"]
    location_match = (
        guess_vertical == recommendation["vertical"]
        and guess_horizontal == recommendation["horizontal"]
    )
    velocity_match = guess_velocity == recommendation["velocity_intent"]
    return {
        "family_match": family_match,
        "location_match": location_match,
        "velocity_match": velocity_match,
        "score": sum([family_match, location_match, velocity_match]),
    }


def match_row(label: str, matched: bool, note: str = "") -> str:
    icon = "✅" if matched else "❌"
    suffix = f' <span class="whiff-match-note">{note}</span>' if note else ""
    return f"{icon} **{label}**{suffix}"


def mock_swing_miss_prescription(
    *,
    balls: int,
    strikes: int,
    runners_on: int,
    attack_zone: str,
    placement_vertical: str,
    placement_horizontal: str,
    pitch_category: str,
    pitch_name: str,
    platoon_bats: str,
    inside_label: str,
    outside_label: str,
) -> dict[str, str]:
    """Legacy rule-based prescription (kept for backward compatibility)."""
    rx = mock_count_prescription(balls, strikes, runners_on)
    horiz_display = (
        outside_label
        if rx["horizontal"] == "Outside"
        else inside_label
        if rx["horizontal"] == "Inside"
        else "Middle"
    )
    return {
        "pitch": PITCH_FAMILY_DISPLAY.get(rx["pitch_family"], rx["pitch_family"]),
        "pitch_family": rx["pitch_family"],
        "horizontal": rx["horizontal"],
        "vertical": rx["vertical"],
        "velocity_intent": rx["velocity_intent"],
        "horizontal_display": horiz_display,
        "attack_zone": attack_zone,
        "headline": rx["headline"],
        "rationale": rx["rationale"],
        "confidence": rx["confidence"],
    }


PITCH_FAMILY_DISPLAY = {
    "Fastball": "4-Seam Fastball",
    "Breaking": "Slider",
    "Offspeed": "Changeup",
}


def build_target_heatmap_placeholder() -> go.Figure:
    """Empty plot area reserved for a future whiff-target heatmap overlay."""
    fig = go.Figure()
    fig.add_shape(
        type="rect",
        x0=-0.708,
        x1=0.708,
        y0=1.5,
        y1=3.5,
        line=dict(color="rgba(245, 239, 227, 0.35)", width=2, dash="dash"),
        fillcolor="rgba(255, 255, 255, 0.02)",
    )
    fig.add_annotation(
        x=0,
        y=2.5,
        text="Whiff target heatmap<br><span style='font-size:11px'>Model overlay coming soon</span>",
        showarrow=False,
        font=dict(color="#cbbfa8", size=14),
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-2.0, 2.0], visible=False),
        yaxis=dict(range=[0.8, 4.2], visible=False),
        height=220,
        margin=dict(t=10, b=10, l=10, r=10),
    )
    return fig

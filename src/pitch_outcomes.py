"""Pitcher-favorable outcome taxonomy and probability decomposition."""

from __future__ import annotations

from typing import Any

import numpy as np

ZONE_HALF_WIDTH_FT = 17.0 / 24.0

OUTCOME_CALLED_STRIKE = "called_strike"
OUTCOME_ZONE_WHIFF = "zone_whiff"
OUTCOME_CHASE_WHIFF = "chase_whiff"
OUTCOME_BALL = "ball"
OUTCOME_CONTACT = "contact"

FAVORABLE_OUTCOMES = {OUTCOME_CALLED_STRIKE, OUTCOME_ZONE_WHIFF, OUTCOME_CHASE_WHIFF}

OUTCOME_LABELS = {
    OUTCOME_CALLED_STRIKE: "Called strike",
    OUTCOME_ZONE_WHIFF: "In-zone whiff",
    OUTCOME_CHASE_WHIFF: "Chase whiff",
    OUTCOME_BALL: "Ball",
    OUTCOME_CONTACT: "Contact / foul",
}


def is_in_zone(plate_x: float, plate_z: float, sz_bot: float, sz_top: float) -> bool:
    """Rule-book strike zone: 17 in wide × hitter height."""
    if abs(plate_x) > ZONE_HALF_WIDTH_FT:
        return False
    return sz_bot <= plate_z <= sz_top


def prob_favorable_decomposed(
    p_swing: float,
    p_whiff_if_swing: float,
    *,
    in_zone: bool,
) -> dict[str, float]:
    """
    Decompose pitcher-favorable probability into three good paths:
    called strike, in-zone whiff, chase whiff.
    """
    p_swing = float(np.clip(p_swing, 0.0, 1.0))
    p_whiff = float(np.clip(p_whiff_if_swing, 0.0, 1.0))
    p_take = 1.0 - p_swing

    if in_zone:
        p_called = p_take
        p_zone_whiff = p_swing * p_whiff
        p_chase_whiff = 0.0
    else:
        p_called = 0.0
        p_zone_whiff = 0.0
        p_chase_whiff = p_swing * p_whiff

    p_favorable = p_called + p_zone_whiff + p_chase_whiff
    p_ball = 0.0 if in_zone else p_take
    p_contact = p_swing * (1.0 - p_whiff)

    return {
        "p_called_strike": p_called,
        "p_zone_whiff": p_zone_whiff,
        "p_chase_whiff": p_chase_whiff,
        "p_favorable": p_favorable,
        "p_ball": p_ball,
        "p_contact": p_contact,
        "p_swing": p_swing,
        "p_whiff_if_swing": p_whiff,
        "in_zone": float(in_zone),
    }


def mock_pitch_probs(
    *,
    balls: int,
    strikes: int,
    in_zone: bool,
    pitch_family: str,
    velocity_intent: str = "Average",
    miss_dist_in: float = 0.0,
    attack_zone: str = "Heart",
    in_zone_swing_pct: float | None = None,
    o_zone_swing_pct: float | None = None,
) -> tuple[float, float]:
    """Location- and hitter-aware swing / whiff estimates when ML bundles are unavailable."""
    league_iz = 0.692
    league_oz = 0.323

    if in_zone:
        p_swing = in_zone_swing_pct if in_zone_swing_pct is not None else league_iz
        if attack_zone == "Heart":
            p_swing *= 1.06
        elif attack_zone == "Shadow":
            p_swing *= 0.90
    else:
        p_swing = o_zone_swing_pct if o_zone_swing_pct is not None else league_oz
        if attack_zone == "Chase":
            p_swing *= 1.05
        elif attack_zone == "Waste":
            p_swing *= 0.50
        chase_penalty = max(0.40, 1.0 - 0.06 * max(0.0, miss_dist_in - 2.0))
        p_swing *= chase_penalty

    if strikes == 2:
        p_swing *= 1.10
    elif balls > strikes:
        p_swing *= 1.05
    elif strikes > balls:
        p_swing *= 0.93

    whiff_base = {"Fastball": 0.22, "Breaking": 0.35, "Offspeed": 0.28}.get(pitch_family, 0.25)
    if velocity_intent == "Firm":
        whiff_base *= 1.08
    elif velocity_intent == "Slow":
        whiff_base *= 0.90

    if in_zone:
        if attack_zone == "Shadow":
            whiff_base *= 1.18
        elif attack_zone == "Heart":
            whiff_base *= 0.86
    else:
        whiff_base *= 1.10 + min(0.12, 0.025 * max(0.0, miss_dist_in - 2.0))

    return float(np.clip(p_swing, 0.05, 0.92)), float(np.clip(whiff_base, 0.08, 0.55))


def simulate_outcome(probs: dict[str, float], rng: np.random.Generator | None = None) -> str:
    """Sample a discrete outcome from decomposed probabilities."""
    rng = rng or np.random.default_rng()
    weights = {
        OUTCOME_CALLED_STRIKE: probs["p_called_strike"],
        OUTCOME_ZONE_WHIFF: probs["p_zone_whiff"],
        OUTCOME_CHASE_WHIFF: probs["p_chase_whiff"],
        OUTCOME_BALL: probs["p_ball"],
        OUTCOME_CONTACT: probs["p_contact"],
    }
    total = sum(weights.values())
    if total <= 0:
        return OUTCOME_BALL
    labels = list(weights.keys())
    values = np.array([weights[k] for k in labels], dtype=float) / total
    return str(rng.choice(labels, p=values))


def outcome_is_favorable(outcome: str) -> bool:
    return outcome in FAVORABLE_OUTCOMES


def format_outcome_probs(probs: dict[str, float]) -> str:
    """Single-line summary of favorable decomposition."""
    return (
        f"Called {probs['p_called_strike']:.0%} · "
        f"Zone whiff {probs['p_zone_whiff']:.0%} · "
        f"Chase whiff {probs['p_chase_whiff']:.0%} · "
        f"**P(favorable) {probs['p_favorable']:.0%}**"
    )


def enrich_probs_dict(probs: dict[str, Any]) -> dict[str, Any]:
    """Add favorable decomposition to a predict_probs() result."""
    in_zone = bool(probs.get("in_zone", False))
    decomposed = prob_favorable_decomposed(
        probs["swing"],
        probs["whiff_if_swing"],
        in_zone=in_zone,
    )
    return {**probs, **decomposed}

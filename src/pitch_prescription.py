"""Pitch recommendation engine — maximize P(favorable) for the situation."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import plotly.graph_objects as go
from src.attack_zones import _fallback_location, assign_attack_zone, boundary_signed_inches
from src.pitch_outcomes import (
    decompose_swing_outcomes,
    format_outcome_probs,
    is_in_zone,
    mock_pitch_probs,
    mock_xwoba_con,
    prob_favorable_decomposed,
)

PITCH_FAMILIES = ["Fastball", "Breaking", "Offspeed"]
VERTICAL_OPTIONS = ["High", "Middle", "Low"]
HORIZONTAL_OPTIONS = ["Inside", "Middle", "Outside"]
VELOCITY_INTENTS = ["Firm", "Average", "Slow"]

PITCH_FAMILY_DISPLAY = {
    "Fastball": "4-Seam Fastball",
    "Breaking": "Slider",
    "Offspeed": "Changeup",
}

LEAGUE_IN_ZONE_SWING = 0.692
LEAGUE_O_ZONE_SWING = 0.323


def _placement_attack_zone(vertical: str, horizontal: str) -> str:
    if vertical == "Middle" and horizontal == "Middle":
        return "Heart"
    if horizontal == "Outside" and vertical == "Low":
        return "Chase"
    if horizontal == "Outside" or vertical in ("High", "Low"):
        return "Shadow"
    return "Heart"


def location_for_placement(
    vertical: str,
    horizontal: str,
    sz_top: float,
    sz_bot: float,
    bats: str,
) -> tuple[float, float]:
    zone = _placement_attack_zone(vertical, horizontal)
    return _fallback_location(zone, vertical, horizontal, sz_top, sz_bot, bats)


def _location_for_placement(
    vertical: str,
    horizontal: str,
    sz_top: float,
    sz_bot: float,
    bats: str,
) -> tuple[float, float]:
    return location_for_placement(vertical, horizontal, sz_top, sz_bot, bats)


def _count_leverage(balls: int, strikes: int) -> str:
    if strikes == 2:
        return "two_strike"
    if balls > strikes:
        return "hitter_ahead"
    if strikes > balls:
        return "pitcher_ahead"
    return "even"


def _strategy_adjustment(
    balls: int,
    strikes: int,
    runners_on: int,
    candidate: dict[str, Any],
    hitter_rates: dict[str, float] | None,
) -> float:
    """Count- and hitter-shaped nudges (no base probability)."""
    rates = hitter_rates or {}
    oz = float(rates.get("o_zone_swing_pct", LEAGUE_O_ZONE_SWING))
    iz = float(rates.get("in_zone_swing_pct", LEAGUE_IN_ZONE_SWING))

    family = candidate["pitch_family"]
    vert = candidate["vertical"]
    horiz = candidate["horizontal"]
    in_zone = candidate["in_zone"]
    zone = candidate.get("attack_zone", "Heart")
    velocity = candidate["velocity_intent"]

    adj = 0.0
    leverage = _count_leverage(balls, strikes)
    chase_delta = oz - LEAGUE_O_ZONE_SWING

    if not in_zone and horiz == "Outside" and vert == "Low" and family in ("Breaking", "Offspeed"):
        adj += 0.06 + max(0.0, chase_delta) * 2.2
    if not in_zone and chase_delta < -0.05:
        adj -= 0.10
    if in_zone and chase_delta < -0.06 and family in ("Breaking", "Offspeed") and vert == "High":
        adj -= 0.10

    if family == "Breaking" and vert == "High":
        adj -= 0.16
    if family == "Breaking" and vert == "High" and horiz == "Inside" and in_zone:
        adj -= 0.08

    if leverage == "even":
        if family == "Fastball" and horiz == "Outside" and vert in ("Middle", "Low"):
            adj += 0.20
        if family == "Fastball" and vert == "Middle" and in_zone:
            adj += 0.08
        if family == "Breaking":
            adj -= 0.12
        if family == "Offspeed" and not in_zone:
            adj -= 0.10

    elif leverage == "pitcher_ahead":
        if family in ("Breaking", "Offspeed") and horiz == "Outside" and vert == "Low" and not in_zone:
            adj += 0.16 + max(0.0, chase_delta) * 1.5
        if family == "Breaking" and zone == "Shadow" and in_zone:
            adj += 0.10
        if family == "Fastball" and vert == "High" and horiz == "Inside" and in_zone:
            adj += 0.08
        if family == "Offspeed" and vert == "Low" and not in_zone and chase_delta > 0.0:
            adj += 0.10
        if family == "Fastball" and horiz == "Outside" and vert == "Middle" and in_zone:
            adj += 0.05

    elif leverage == "hitter_ahead":
        if family == "Fastball" and in_zone and horiz == "Outside":
            adj += 0.24
        if family == "Fastball" and in_zone and vert == "Middle":
            adj += 0.12
        if not in_zone:
            adj -= 0.22
        if family == "Breaking":
            adj -= 0.08

    elif leverage == "two_strike":
        if chase_delta > 0.02:
            if family in ("Breaking", "Offspeed") and not in_zone and horiz == "Outside" and vert == "Low":
                adj += 0.22 + chase_delta * 0.9
            if family == "Breaking" and zone == "Chase":
                adj += 0.14
            if family == "Breaking" and not in_zone and horiz == "Outside":
                adj += 0.08
        elif chase_delta >= -0.02:
            if family == "Breaking" and not in_zone and horiz == "Outside" and vert in ("Low", "Middle"):
                adj += 0.10
            if family == "Fastball" and vert == "High" and in_zone:
                adj += 0.08
        else:
            if family == "Fastball" and vert == "High" and in_zone:
                adj += 0.18
            if family == "Breaking" and horiz == "Outside" and zone == "Shadow" and in_zone:
                adj += 0.10
            if not in_zone:
                adj -= 0.14
        if family == "Fastball" and horiz == "Outside" and in_zone:
            adj += 0.06

    if in_zone and (1.0 - iz) > 0.30:
        adj += 0.05
    if in_zone and iz > 0.74 and family in ("Breaking", "Offspeed"):
        adj += 0.06
    if not in_zone and chase_delta < -0.05:
        adj -= 0.10

    if runners_on >= 2 and strikes < 2 and family == "Fastball" and in_zone:
        adj += 0.08

    if family == "Fastball" and velocity == "Firm" and leverage in ("even", "hitter_ahead"):
        adj += 0.04

    return adj


def _weighted_favorable_score(
    balls: int,
    strikes: int,
    candidate: dict[str, Any],
    hitter_rates: dict[str, float] | None,
) -> float:
    """Re-weight called / zone-whiff / chase paths by count and hitter chase profile."""
    rates = hitter_rates or {}
    chase_delta = float(rates.get("o_zone_swing_pct", LEAGUE_O_ZONE_SWING)) - LEAGUE_O_ZONE_SWING
    leverage = _count_leverage(balls, strikes)

    w_called, w_zone, w_chase = 0.40, 0.35, 0.25

    if leverage == "two_strike":
        w_chase = 0.50 + max(0.0, chase_delta) * 1.2
        w_called = 0.12
        w_zone = max(0.10, 1.0 - w_chase - w_called)
    elif leverage == "hitter_ahead":
        w_called, w_zone, w_chase = 0.52, 0.33, 0.15
    elif leverage == "pitcher_ahead":
        w_chase = 0.38 + max(0.0, chase_delta) * 1.0
        w_called = 0.22
        w_zone = max(0.10, 1.0 - w_chase - w_called)
    elif leverage == "even":
        w_called, w_zone, w_chase = 0.44, 0.36, 0.20

    norm = w_called + w_zone + w_chase
    return (
        w_called * candidate["p_called_strike"]
        + w_zone * candidate["p_zone_whiff"]
        + w_chase * candidate["p_chase_whiff"]
    ) / norm


def _recommendation_score(
    balls: int,
    strikes: int,
    runners_on: int,
    candidate: dict[str, Any],
    hitter_rates: dict[str, float] | None,
) -> float:
    xstrike = float(candidate.get("xstrike", candidate.get("p_favorable", 0.0)))
    xwoba = candidate.get("xwoba_con")
    if xwoba is not None:
        base = xstrike - float(xwoba)
    else:
        base = _weighted_favorable_score(balls, strikes, candidate, hitter_rates)
    return base + _strategy_adjustment(balls, strikes, runners_on, candidate, hitter_rates) * 0.35


def _build_rationale(
    *,
    balls: int,
    strikes: int,
    runners_on: int,
    best: dict[str, Any],
    hitter_rates: dict[str, float] | None,
) -> str:
    count = f"{balls}-{strikes}"
    rates = hitter_rates or {}
    oz = float(rates.get("o_zone_swing_pct", LEAGUE_O_ZONE_SWING))
    iz = float(rates.get("in_zone_swing_pct", LEAGUE_IN_ZONE_SWING))
    chase_note = (
        "aggressive chaser"
        if oz > LEAGUE_O_ZONE_SWING + 0.04
        else "disciplined out of the zone"
        if oz < LEAGUE_O_ZONE_SWING - 0.04
        else "league-average chase profile"
    )
    take_note = (
        "takes in the zone"
        if iz < LEAGUE_IN_ZONE_SWING - 0.04
        else "swings in the zone"
        if iz > LEAGUE_IN_ZONE_SWING + 0.04
        else "league-average in-zone swing rate"
    )
    shape = f"{best['pitch_family']} {best['horizontal'].lower()} / {best['vertical'].lower()}"

    if strikes == 2:
        return (
            f"Two-strike ({count}) — {shape} fits this hitter ({chase_note}). "
            f"P(favorable) {best['p_favorable']:.0%}."
        )
    if balls > strikes:
        return (
            f"Hitter's count ({count}) — need a strike; {shape} attacks the zone while "
            f"this hitter {take_note}. P(favorable) {best['p_favorable']:.0%}."
        )
    if strikes > balls:
        return (
            f"Pitcher's count ({count}) — expand with {shape}; hitter is {chase_note}. "
            f"P(favorable) {best['p_favorable']:.0%}."
        )
    if runners_on >= 2:
        return (
            f"Runners on ({count}) — {shape} for a strike without falling behind; "
            f"hitter {take_note}. P(favorable) {best['p_favorable']:.0%}."
        )
    return (
        f"Even count ({count}) — start with {shape}; league script is fastball to the outer edge. "
        f"Hitter {take_note}. P(favorable) {best['p_favorable']:.0%}."
    )


def _miss_distance_inches(plate_x: float, plate_z: float, sz_top: float, sz_bot: float) -> float:
    signed = boundary_signed_inches(plate_x, plate_z, sz_top, sz_bot)
    return max(0.0, -signed)


def _mock_probs_for_spot(
    *,
    balls: int,
    strikes: int,
    plate_x: float,
    plate_z: float,
    sz_top: float,
    sz_bot: float,
    pitch_family: str,
    velocity_intent: str,
    predict_kwargs: dict[str, Any],
) -> tuple[float, float]:
    in_zone = is_in_zone(plate_x, plate_z, sz_bot, sz_top)
    rates = predict_kwargs.get("hitter_rates") or {}
    return mock_pitch_probs(
        balls=balls,
        strikes=strikes,
        in_zone=in_zone,
        pitch_family=pitch_family,
        velocity_intent=velocity_intent,
        miss_dist_in=_miss_distance_inches(plate_x, plate_z, sz_top, sz_bot),
        attack_zone=assign_attack_zone(plate_x, plate_z, sz_top, sz_bot),
        in_zone_swing_pct=rates.get("in_zone_swing_pct"),
        o_zone_swing_pct=rates.get("o_zone_swing_pct"),
    )


def _score_candidate(
    *,
    balls: int,
    strikes: int,
    pitch_family: str,
    vertical: str,
    horizontal: str,
    velocity_intent: str,
    sz_top: float,
    sz_bot: float,
    bats: str,
    predict_fn: Callable[..., dict[str, float]] | None,
    predict_kwargs: dict[str, Any],
) -> dict[str, Any]:
    px, pz = location_for_placement(vertical, horizontal, sz_top, sz_bot, bats)
    in_zone = is_in_zone(px, pz, sz_bot, sz_top)

    if predict_fn is not None:
        probs = predict_fn(plate_x=px, plate_z=pz, pitch_family=pitch_family, velocity_intent=velocity_intent, **predict_kwargs)
        p_swing = probs["swing"]
        p_whiff = probs["whiff_if_swing"]
        extra = {
            k: probs[k]
            for k in ("p_take", "p_whiff", "p_contact", "xstrike", "xwoba_con")
            if k in probs
        }
    else:
        p_swing, p_whiff = _mock_probs_for_spot(
            balls=balls,
            strikes=strikes,
            plate_x=px,
            plate_z=pz,
            sz_top=sz_top,
            sz_bot=sz_bot,
            pitch_family=pitch_family,
            velocity_intent=velocity_intent,
            predict_kwargs=predict_kwargs,
        )
        extra = {}

    decomposed = prob_favorable_decomposed(p_swing, p_whiff, in_zone=in_zone)
    swing_out = decompose_swing_outcomes(p_swing, p_whiff, in_zone=in_zone)
    if "xstrike" not in extra:
        extra = swing_out
    else:
        extra = {**swing_out, **extra}
    if extra.get("xwoba_con") is None and extra.get("p_contact", 0) > 0:
        extra["xwoba_con"] = mock_xwoba_con(
            pitch_family=pitch_family,
            in_zone=in_zone,
            attack_zone=assign_attack_zone(px, pz, sz_top, sz_bot),
        )
    return {
        "pitch_family": pitch_family,
        "vertical": vertical,
        "horizontal": horizontal,
        "velocity_intent": velocity_intent,
        "plate_x": px,
        "plate_z": pz,
        "attack_zone": assign_attack_zone(px, pz, sz_top, sz_bot),
        "in_zone": in_zone,
        **decomposed,
        **extra,
    }


def recommend_best_pitch(
    *,
    balls: int,
    strikes: int,
    runners_on: int = 0,
    sz_top: float,
    sz_bot: float,
    bats: str = "R",
    predict_fn: Callable[..., dict[str, float]] | None = None,
    predict_kwargs: dict[str, Any] | None = None,
    swing_bundle: dict | None = None,
    whiff_bundle: dict | None = None,
    xwoba_bundle: dict | None = None,
    profile_lookup: dict[str, Any] | None = None,
    use_ml: bool = True,
) -> dict[str, Any]:
    """Search 27 family × location combos; return highest recommendation score."""
    from src.pitch_batch_predict import (
        batch_predict_outcomes_ml,
        batch_predict_outcomes_mock,
        build_recommendation_grid_df,
    )

    predict_kwargs = predict_kwargs or {}
    hitter_rates = predict_kwargs.get("hitter_rates")
    swing_medians = pd.Series(
        (swing_bundle or {}).get("pitch_medians", {})
    )

    grid = build_recommendation_grid_df(
        balls=balls,
        strikes=strikes,
        runners_on=runners_on,
        sz_top=sz_top,
        sz_bot=sz_bot,
        bats=bats,
        profile_lookup=profile_lookup or {},
        swing_medians=swing_medians,
    )

    if use_ml and swing_bundle is not None and whiff_bundle is not None and predict_fn is None:
        scored = batch_predict_outcomes_ml(swing_bundle, whiff_bundle, xwoba_bundle, grid)
    elif predict_fn is not None:
        scored = _score_grid_with_predict_fn(
            grid,
            balls=balls,
            strikes=strikes,
            predict_fn=predict_fn,
            predict_kwargs=predict_kwargs,
        )
    else:
        scored = batch_predict_outcomes_mock(
            grid,
            balls=balls,
            strikes=strikes,
            hitter_rates=hitter_rates,
        )

    scored["recommendation_score"] = scored.apply(
        lambda row: _recommendation_score(
            balls, strikes, runners_on, row.to_dict(), hitter_rates
        ),
        axis=1,
    )
    best = scored.loc[scored["recommendation_score"].idxmax()]
    count = f"{balls}-{strikes}"
    best_dict = best.to_dict()
    rationale = _build_rationale(
        balls=balls,
        strikes=strikes,
        runners_on=runners_on,
        best=best_dict,
        hitter_rates=hitter_rates,
    )
    headline = (
        f"Recommended Pitch: {best_dict['pitch_family']} — "
        f"{best_dict['horizontal']} / {best_dict['vertical']}"
    )

    return {
        **best_dict,
        "count": count,
        "headline": headline,
        "rationale": rationale,
        "confidence": "xStrike ↑ · xwOBAcon ↓ · count + hitter aware",
        "outcome_summary": format_outcome_probs(best_dict),
    }


def _score_grid_with_predict_fn(
    grid: pd.DataFrame,
    *,
    balls: int,
    strikes: int,
    predict_fn: Callable[..., dict[str, float]],
    predict_kwargs: dict[str, Any],
) -> pd.DataFrame:
    """Legacy row-wise path when a custom predict_fn is supplied."""
    rows: list[dict[str, Any]] = []
    for row in grid.itertuples(index=False):
        probs = predict_fn(
            plate_x=row.plate_x,
            plate_z=row.plate_z,
            pitch_family=row.pitch_family,
            velocity_intent=row.velocity_intent,
            **predict_kwargs,
        )
        in_zone = bool(row.in_zone)
        decomposed = prob_favorable_decomposed(
            probs["swing"], probs["whiff_if_swing"], in_zone=in_zone
        )
        swing_out = decompose_swing_outcomes(
            probs["swing"], probs["whiff_if_swing"], in_zone=in_zone
        )
        merged = {**row._asdict(), **decomposed, **swing_out}
        if probs.get("xwoba_con") is not None:
            merged["xwoba_con"] = probs["xwoba_con"]
        rows.append(merged)
    return pd.DataFrame(rows)


def score_thrown_pitch(
    *,
    balls: int,
    strikes: int,
    plate_x: float,
    plate_z: float,
    pitch_family: str,
    velocity_intent: str,
    sz_top: float,
    sz_bot: float,
    predict_fn: Callable[..., dict[str, float]] | None = None,
    predict_kwargs: dict[str, Any] | None = None,
    swing_bundle: dict | None = None,
    whiff_bundle: dict | None = None,
    xwoba_bundle: dict | None = None,
    profile_lookup: dict[str, Any] | None = None,
    use_ml: bool = True,
) -> dict[str, Any]:
    """Score the actual pitch location the user threw."""
    from src.pitch_batch_predict import (
        batch_predict_outcomes_ml,
        batch_predict_outcomes_mock,
        build_thrown_pitch_df,
    )

    predict_kwargs = predict_kwargs or {}
    hitter_rates = predict_kwargs.get("hitter_rates")
    in_zone = is_in_zone(plate_x, plate_z, sz_bot, sz_top)
    swing_medians = pd.Series((swing_bundle or {}).get("pitch_medians", {}))

    if use_ml and swing_bundle is not None and whiff_bundle is not None and predict_fn is None:
        thrown_df = build_thrown_pitch_df(
            balls=balls,
            strikes=strikes,
            runners_on=int(predict_kwargs.get("runners_on", 0)),
            plate_x=plate_x,
            plate_z=plate_z,
            pitch_family=pitch_family,
            velocity_intent=velocity_intent,
            sz_top=sz_top,
            sz_bot=sz_bot,
            profile_lookup=profile_lookup or {},
            swing_medians=swing_medians,
        )
        scored = batch_predict_outcomes_ml(swing_bundle, whiff_bundle, xwoba_bundle, thrown_df)
        return scored.iloc[0].to_dict()

    if predict_fn is not None:
        probs = predict_fn(
            plate_x=plate_x,
            plate_z=plate_z,
            pitch_family=pitch_family,
            velocity_intent=velocity_intent,
            **predict_kwargs,
        )
        return {
            "plate_x": plate_x,
            "plate_z": plate_z,
            "pitch_family": pitch_family,
            "velocity_intent": velocity_intent,
            "attack_zone": assign_attack_zone(plate_x, plate_z, sz_top, sz_bot),
            "in_zone": in_zone,
            **probs,
        }

    thrown_df = build_thrown_pitch_df(
        balls=balls,
        strikes=strikes,
        runners_on=int(predict_kwargs.get("runners_on", 0)),
        plate_x=plate_x,
        plate_z=plate_z,
        pitch_family=pitch_family,
        velocity_intent=velocity_intent,
        sz_top=sz_top,
        sz_bot=sz_bot,
        profile_lookup=profile_lookup or {},
        swing_medians=swing_medians,
    )
    scored = batch_predict_outcomes_mock(
        thrown_df,
        balls=balls,
        strikes=strikes,
        hitter_rates=hitter_rates,
    )
    row = scored.iloc[0].to_dict()
    row["vertical"] = predict_kwargs.get("vertical")
    row["horizontal"] = predict_kwargs.get("horizontal")
    return row


def compare_strategy_guess(
    *,
    guess_family: str,
    guess_vertical: str,
    guess_horizontal: str,
    guess_velocity: str,
    recommendation: dict[str, Any],
) -> dict[str, object]:
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


def mock_count_prescription(balls: int, strikes: int, runners_on: int = 0, **kwargs) -> dict[str, Any]:
    """Backward-compatible wrapper around recommend_best_pitch."""
    return recommend_best_pitch(
        balls=balls,
        strikes=strikes,
        runners_on=runners_on,
        sz_top=kwargs.get("sz_top", 3.5),
        sz_bot=kwargs.get("sz_bot", 1.5),
        bats=kwargs.get("bats", "R"),
    )


def build_target_heatmap_placeholder() -> go.Figure:
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
        text="P(favorable) heatmap<br><span style='font-size:11px'>Coming soon</span>",
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

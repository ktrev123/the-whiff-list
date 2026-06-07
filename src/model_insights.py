"""Layman's-term interpretations and labels for swing / whiff model outputs."""

from __future__ import annotations

import math

FEATURE_LABELS = {
    "plate_x": "Horizontal location (inside/outside)",
    "plate_z": "Vertical location (high/low)",
    "miss_dist_in": "Distance from strike zone (inches)",
    "balls": "Balls in the count",
    "strikes": "Strikes in the count",
    "runners_on": "Runners on base",
    "is_two_strike": "Two-strike count flag",
    "count_state": "Count leverage state",
    "release_speed": "Release velocity (mph)",
    "speed_diff": "Perceived velo minus release speed (mph)",
    "pfx_x": "Horizontal break",
    "pfx_z": "Vertical break",
    "release_spin_rate": "Spin rate (rpm)",
    "spin_axis": "Spin axis (degrees)",
    "release_extension": "Release extension (ft)",
    "pitch_type": "Pitch type",
}


def friendly_feature_name(encoded_name: str) -> str:
    if encoded_name.startswith("num__"):
        raw = encoded_name.replace("num__", "", 1)
        return FEATURE_LABELS.get(raw, raw)
    if encoded_name.startswith("cat__pitch_type_"):
        code = encoded_name.replace("cat__pitch_type_", "", 1)
        return f"Pitch type: {code}"
    if encoded_name.startswith("cat__count_state_"):
        state = encoded_name.replace("cat__count_state_", "", 1)
        return f"Count: {state.replace('_', ' ')}"
    return encoded_name


EXAMPLE_SCENARIOS = [
    {
        "label": "Paint at 2-2, bases empty",
        "pitch_type": "FF",
        "plate_x": 0.35,
        "plate_z": 2.8,
        "balls": 2,
        "strikes": 2,
        "runners_on": 0,
    },
    {
        "label": "1-2 Sweeper chase (12 in. off plate)",
        "pitch_type": "SV",
        "plate_x": 1.2,
        "plate_z": 1.4,
        "balls": 1,
        "strikes": 2,
        "runners_on": 2,
    },
    {
        "label": "3-2 bailout chase with runners on",
        "pitch_type": "SL",
        "plate_x": -0.9,
        "plate_z": 0.9,
        "balls": 3,
        "strikes": 2,
        "runners_on": 3,
    },
    {
        "label": "0-0 heater down the middle",
        "pitch_type": "FF",
        "plate_x": 0.0,
        "plate_z": 2.5,
        "balls": 0,
        "strikes": 0,
        "runners_on": 0,
    },
]


def np_clip(x, lo, hi):
    return max(lo, min(hi, x))


def interpret_roc_auc(auc: float, outcome: str) -> str:
    if auc >= 0.80:
        quality = "strong"
        detail = f"The model reliably separates high-{outcome} pitches from low-{outcome} pitches."
    elif auc >= 0.65:
        quality = "moderate"
        detail = "The model is meaningfully better than guessing, but not perfect."
    else:
        quality = "limited"
        detail = "Treat rankings as directional, not exact."
    return (
        f"**ROC-AUC = {auc:.2f} ({quality}).** "
        f"If you pick one random '{outcome}' pitch and one random non-{outcome} pitch, "
        f"the model scores the {outcome} pitch higher about **{auc * 100:.0f}%** of the time "
        f"(50% = random guessing). {detail}"
    )


def interpret_log_loss(loss: float, base_rate: float, outcome: str) -> str:
    base_rate = float(np_clip(base_rate, 1e-6, 1 - 1e-6))
    baseline = -(base_rate * math.log(base_rate) + (1 - base_rate) * math.log(1 - base_rate))
    delta = baseline - loss
    if delta > 0.05:
        verdict = f"Probability estimates are noticeably sharper than always guessing the average {outcome} rate."
    elif delta > 0.01:
        verdict = "The model improves on a naive average guess, but some predictions are still overconfident."
    else:
        verdict = "Calibration is only slightly better than guessing the same average every time."
    return (
        f"**Log Loss = {loss:.3f}** (lower is better; naive baseline ≈ {baseline:.3f}). {verdict}"
    )


def swing_output_summary() -> str:
    return (
        "**Model A (Swing):** For every pitch, outputs the probability the batter **swings** "
        "based on location, count, leverage, and pitch characteristics (type, velocity, movement, spin). "
        "Answers: *'Will he offer at this pitch?'*"
    )


def whiff_output_summary() -> str:
    return (
        "**Model B (Whiff):** Trained only on pitches the batter **already swung at**. "
        "Outputs the probability that swing ends in a **miss**, using the same location, count, "
        "and pitch-physics inputs. Answers: *'If he swings, will he whiff?'*"
    )


def combined_output_summary() -> str:
    return (
        "**Combined swing-and-whiff risk** = P(swing) × P(whiff | swing). "
        "This is the estimated chance of a swinging strike on that pitch profile."
    )


def xwoba_output_summary() -> str:
    return (
        "**Model C (xwOBAcon):** Trained only on **balls in play** (`hit_into_play`). "
        "Predicts **expected wOBA on contact** (`estimated_woba_using_speedangle`) from location, "
        "count, pitch category, velocity, and movement. Answers: *'If he puts this in play, how much damage?'*"
    )


def three_model_pipeline_summary() -> str:
    return (
        "**Whiff Lab pipeline:** P(take) and P(whiff) from Models A & B; P(contact) = P(swing) × (1 − P(whiff|swing)). "
        "When contact is possible, Model C supplies **xwOBAcon**. **xStrike%** = take + whiff in-zone; whiff only O-zone."
    )


def interpret_r2(r2: float) -> str:
    if r2 >= 0.35:
        quality = "strong"
    elif r2 >= 0.15:
        quality = "moderate"
    else:
        quality = "limited"
    return (
        f"**R² = {r2:.2f} ({quality}).** "
        f"The model explains about **{max(0, r2) * 100:.0f}%** of contact-quality variance on held-out batted balls."
    )


def interpret_mae(mae: float) -> str:
    return (
        f"**MAE = {mae:.3f} wOBA points.** "
        f"Typical contact predictions are within ~{mae * 1000:.0f} points of Statcast xwOBAcon."
    )


def interpret_xwoba_damage(prob: float) -> str:
    if prob >= 0.340:
        return f"{prob:.3f} xwOBAcon — barrel / hot contact risk."
    if prob >= 0.280:
        return f"{prob:.3f} xwOBAcon — league-average contact quality."
    return f"{prob:.3f} xwOBAcon — weak contact profile."


def validation_summary(train_period: str, test_period: str, n_train: int, n_test: int) -> str:
    return (
        f"Both models use the same split: trained on **{train_period}** and tested on "
        f"**{test_period}** (September holdout) — {n_test:,} pitches the models never saw during training."
    )


def interpret_swing_probability_plain(prob: float) -> str:
    pct = prob * 100
    if pct >= 70:
        return f"{pct:.0f}% swing chance — hitter is very likely to offer."
    if pct >= 45:
        return f"{pct:.0f}% swing chance — competitive swing decision territory."
    if pct >= 25:
        return f"{pct:.0f}% swing chance — take is plausible, swing is not automatic."
    return f"{pct:.0f}% swing chance — hitter is likely to let it go."


def interpret_whiff_probability_plain(prob: float) -> str:
    pct = prob * 100
    if pct >= 45:
        return f"{pct:.0f}% whiff-if-swing chance — ugly miss risk once he commits."
    if pct >= 30:
        return f"{pct:.0f}% whiff-if-swing chance — elevated miss risk on contact attempt."
    if pct >= 15:
        return f"{pct:.0f}% whiff-if-swing chance — typical miss rate for a swing."
    return f"{pct:.0f}% whiff-if-swing chance — relatively good bat-to-ball if he swings."

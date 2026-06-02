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
}

EXAMPLE_SCENARIOS = [
    {
        "label": "Paint at 2-2, bases empty",
        "plate_x": 0.35,
        "plate_z": 2.8,
        "balls": 2,
        "strikes": 2,
        "runners_on": 0,
    },
    {
        "label": "1-2 Sweeper chase (12 in. off plate)",
        "plate_x": 1.2,
        "plate_z": 1.4,
        "balls": 1,
        "strikes": 2,
        "runners_on": 2,
    },
    {
        "label": "3-2 bailout chase with runners on",
        "plate_x": -0.9,
        "plate_z": 0.9,
        "balls": 3,
        "strikes": 2,
        "runners_on": 3,
    },
    {
        "label": "0-0 heater down the middle",
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
        "(contact, foul, or whiff) based on location, count, and leverage. "
        "Answers: *'Will he offer at this pitch?'*"
    )


def whiff_output_summary() -> str:
    return (
        "**Model B (Whiff):** Trained only on pitches the batter **already swung at**. "
        "Outputs the probability that swing ends in a **miss** (swinging strike). "
        "Answers: *'If he swings, will he whiff?'*"
    )


def combined_output_summary() -> str:
    return (
        "**Combined swing-and-whiff risk** = P(swing) × P(whiff | swing). "
        "This is the estimated chance of a swinging strike on that pitch profile."
    )


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

"""Whiff Lab input helpers: count leverage labels and mock pitch-physics tiers."""

from __future__ import annotations

PITCH_FAMILIES = ["Fastball", "Breaking", "Offspeed"]

PITCH_FAMILY_DEFAULT_CODE = {
    "Fastball": "FF",
    "Breaking": "SL",
    "Offspeed": "CH",
}

PITCH_FAMILY_DISPLAY_PITCH = {
    "Fastball": "4-Seam Fastball",
    "Breaking": "Slider",
    "Offspeed": "Changeup",
}

# Legacy category strings used by the mock prescription engine.
PITCH_FAMILY_PRESCRIPTION_KEY = {
    "Fastball": "Fastballs",
    "Breaking": "Breaking Balls",
    "Offspeed": "Off-Speed",
}

VELOCITY_INTENTS = ["Firm", "Average", "Slow"]

VERTICAL_GUESS_OPTIONS = ["High", "Middle", "Low"]
HORIZONTAL_GUESS_OPTIONS = ["Inside", "Middle", "Outside"]

PHYSICS_TIERS = ["Firm/High", "Average", "Slow/Low"]

# Map guess velocity labels to physics tier keys.
VELOCITY_INTENT_TO_TIER = {
    "Firm": "Firm/High",
    "Average": "Average",
    "Slow": "Slow/Low",
}

TIER_MULTIPLIERS = {
    "Firm/High": 1.08,
    "Average": 1.0,
    "Slow/Low": 0.92,
}

# League-average baselines by pitch family (mock Mar–Aug 2025 values).
FAMILY_PHYSICS_BASELINE: dict[str, dict[str, float]] = {
    "Fastball": {
        "release_speed": 94.2,
        "pfx_z": 8.6,
        "release_spin_rate": 2280.0,
        "pfx_x": -4.2,
    },
    "Breaking": {
        "release_speed": 84.5,
        "pfx_z": -2.4,
        "release_spin_rate": 2480.0,
        "pfx_x": -8.1,
    },
    "Offspeed": {
        "release_speed": 86.0,
        "pfx_z": 4.2,
        "release_spin_rate": 1820.0,
        "pfx_x": -6.0,
    },
}

COUNT_LEVERAGE_MAP: dict[tuple[int, int], str] = {
    (0, 0): "Even",
    (1, 1): "Even",
    (2, 2): "Even",
    (3, 2): "Even",
    (0, 1): "Pitcher Advantage",
    (0, 2): "Pitcher Advantage",
    (1, 2): "Pitcher Advantage",
    (1, 0): "Hitter Advantage",
    (2, 0): "Hitter Advantage",
    (3, 0): "Hitter Advantage",
    (2, 1): "Hitter Advantage",
    (3, 1): "Hitter Advantage",
}


def count_leverage_label(balls: int, strikes: int) -> str:
    """Map balls-strikes to leverage category for display."""
    return COUNT_LEVERAGE_MAP.get((balls, strikes), "Even")


def format_count_display(balls: int, strikes: int) -> str:
    """Human-readable count + leverage (e.g. '3-1 · Hitter Advantage')."""
    leverage = count_leverage_label(balls, strikes)
    if strikes == 2 and leverage != "Even":
        return f"{balls}-{strikes} · {leverage} · 2-Strike"
    if strikes == 2:
        return f"{balls}-{strikes} · 2-Strike"
    return f"{balls}-{strikes} · {leverage}"


def apply_physics_tiers(
    family: str,
    *,
    release_speed: str,
    vertical_break: str,
    spin_rate: str,
    horizontal_break: str,
) -> dict[str, float]:
    """Return mock physics dict from family baseline + qualitative tier picks."""
    base = FAMILY_PHYSICS_BASELINE.get(family, FAMILY_PHYSICS_BASELINE["Fastball"]).copy()
    tiers = {
        "release_speed": release_speed,
        "pfx_z": vertical_break,
        "release_spin_rate": spin_rate,
        "pfx_x": horizontal_break,
    }
    out: dict[str, float] = {}
    for key, tier in tiers.items():
        mult = TIER_MULTIPLIERS.get(tier, 1.0)
        value = base[key]
        if key in ("pfx_x", "pfx_z"):
            out[key] = float(value * mult) if value >= 0 else float(value / mult)
        else:
            out[key] = float(value * mult)
    return out


def physics_summary_line(profile: dict[str, float]) -> str:
    return (
        f"{profile['release_speed']:.1f} mph · "
        f"V-break {profile['pfx_z']:+.1f} in · "
        f"H-break {profile['pfx_x']:+.1f} in · "
        f"{profile['release_spin_rate']:.0f} rpm"
    )

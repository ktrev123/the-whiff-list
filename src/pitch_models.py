"""Cached Pitch Lab model bundles (swing, whiff, xwOBA)."""

from __future__ import annotations

from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "model"
PITCH_LAB_SWING_FILE = MODEL_DIR / "pitch_lab_swing.joblib"
PITCH_LAB_WHIFF_FILE = MODEL_DIR / "pitch_lab_whiff.joblib"
PITCH_LAB_XWOBA_FILE = MODEL_DIR / "pitch_lab_xwoba.joblib"
SWING_MODEL_FILE = MODEL_DIR / "swing_model.joblib"
WHIFF_MODEL_FILE = MODEL_DIR / "whiff_model.joblib"
XWOBA_MODEL_FILE = MODEL_DIR / "xwoba_model.joblib"


def _patch_model_bundle(bundle: dict | None) -> dict | None:
    if bundle is None:
        return None
    model = bundle.get("model")
    if model is None or not hasattr(model, "named_steps"):
        return bundle
    clf = model.named_steps.get("clf")
    if clf is None and getattr(model, "steps", None):
        clf = model.steps[-1][1]
    if clf is not None and not hasattr(clf, "multi_class"):
        clf.multi_class = "auto"
    return bundle


def _load_bundle(path: Path) -> dict | None:
    if not path.exists():
        return None
    return _patch_model_bundle(joblib.load(path))


@st.cache_resource
def load_pitch_lab_models() -> tuple[dict | None, dict | None, dict | None]:
    """Load swing, whiff, and xwOBA bundles once per Streamlit server process."""
    swing_bundle = None
    whiff_bundle = None
    for swing_path, whiff_path in (
        (PITCH_LAB_SWING_FILE, PITCH_LAB_WHIFF_FILE),
        (SWING_MODEL_FILE, WHIFF_MODEL_FILE),
    ):
        swing_bundle = _load_bundle(swing_path)
        whiff_bundle = _load_bundle(whiff_path)
        if swing_bundle is not None and whiff_bundle is not None:
            break

    xwoba_bundle = None
    for xwoba_path in (PITCH_LAB_XWOBA_FILE, XWOBA_MODEL_FILE):
        xwoba_bundle = _load_bundle(xwoba_path)
        if xwoba_bundle is not None:
            break

    if swing_bundle is None or whiff_bundle is None:
        return None, None, None
    return swing_bundle, whiff_bundle, xwoba_bundle

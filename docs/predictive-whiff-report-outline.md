# Predictive Whiff Modeling: Data Science Report Outline

## Statistical Question

> **Given an incoming pitch's characteristics (velocity, location, spin, movement, pitch type) and situational context (count, runners on), can classification models accurately estimate (A) the probability that a hitter swings and (B) the probability that a swing results in a miss (whiff)?**

---

## I. Executive Summary

### Problem Statement

Season-long metrics such as overall Whiff% summarize what already happened but do not answer pitch-level questions: *Will this hitter swing at a 1–2 Sweeper off the plate with runners on, and if he does, will he miss?*

### Approach

Two linked classifiers on 2025 Statcast data — **Model A (swing)** on all pitches and **Model B (whiff | swing)** on swings only — using location, count, leverage, and pitch-physics features. Results are validated on a September holdout and presented in The Whiff List dashboard and HTML model report.

### Key Results

- [Swing model ROC-AUC / Log Loss — September holdout]
- [Whiff model ROC-AUC / Log Loss — September holdout]
- [Top feature drivers including pitch type, velocity, movement, spin]

---

## II. Data Pipeline & Feature Engineering

### Data Source & Sample

- **Source:** MLB Statcast via `pybaseball` (`notebooks/statcast_pull.py`)
- **Season:** March 27 – September 28, 2025 (regular season only; `game_type = R`)
- **Qualification:** Batters with ≥ 502 AB
- **Unit of analysis:** Individual pitch

### Target Variables

| Model | Label | Y = 1 | Y = 0 | Training rows |
|-------|-------|-------|-------|---------------|
| **(A) Swing** | `is_swing` | Batter swings | Batter takes | All pitches |
| **(B) Whiff** | `is_whiff` | Swinging strike / missed bunt | Contact or foul | Swings only |

Combined swinging-strike risk: **P(swing) × P(whiff | swing)**

### Feature Groups (Model Inputs)

Both models share the same feature matrix (`src/whiff_features.py`):

| Category | Columns | Notes |
|----------|---------|-------|
| **Spatial** | `plate_x`, `plate_z`, `miss_dist_in` | `miss_dist_in` = inches from strike zone boundary (uses `sz_top`, `sz_bot`) |
| **Contextual** | `balls`, `strikes`, `runners_on` | `runners_on` aggregated from `on_1b`, `on_2b`, `on_3b` |
| **Pitch identity** | `pitch_type` | One-hot encoded (FF, SL, SV, CH, CU, etc.) |
| **Velocity** | `release_speed`, `effective_speed` | mph |
| **Movement** | `pfx_x`, `pfx_z` | Horizontal and vertical break |
| **Spin** | `release_spin_rate`, `spin_axis` | rpm and degrees |
| **Deception** | `release_extension` | Release extension (ft) |

Missing pitch-physics values are imputed with **training-set medians** (fit on April–August only, applied to train and test).

### Excluded from Features

- **Outcomes:** `description`, `events`, launch metrics, `woba_value` (target leakage)
- **Player IDs:** `batter`, `pitcher`, `player_name` (reserved for dashboard joins, not model inputs in this version)
- **Deprecated Statcast fields:** `spin_rate_deprecated`, `break_angle_deprecated`, etc.

---

## III. Exploratory Data Analysis

### League Whiff Topography

### Temporal Stability

### Platoon & Pitch-Type Context

---

## IV. Model Development & Evaluation

### Algorithms

### Validation

### Metrics

---

## V. Model Interpretation & Dashboard

### Feature Importance

### Deliverables

---

## VI. Conclusion & Future Work

### Summary

### Future Work

---

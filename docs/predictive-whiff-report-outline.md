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
- **Season:** March 23 – September 27, 2025
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

2D density heatmap of whiff locations relative to the strike zone (Streamlit app).

### Temporal Stability

7-day rolling average of out-of-zone chase volume across the season.

### Platoon & Pitch-Type Context

Chase thermometers by pitch type and batter handedness (Embarrassment Index).

---

## IV. Model Development & Evaluation

### Algorithms

| Model | Candidates | Preprocessing |
|-------|------------|---------------|
| Swing & Whiff | Logistic Regression, Random Forest | StandardScaler on numeric features; OneHotEncoder on `pitch_type` |

Best candidate selected by **ROC-AUC** on the September holdout.

### Validation

| Split | Period |
|-------|--------|
| Train | April – August 2025 |
| Test | September 2025 |

Chronological split (no random shuffle).

### Metrics

| Metric | Interpretation |
|--------|----------------|
| **ROC-AUC** | Ranking quality — separates positive outcomes from negatives |
| **Log Loss** | Probability calibration — lower is better |

---

## V. Model Interpretation & Dashboard

### Feature Importance

Encoded feature importances (including pitch-type dummies) exported to `data/model/model_insights.json` and the HTML report.

### Deliverables

1. **HTML report** — opens automatically after `train_whiff_model.py` (ROC, calibration, heatmaps, scenarios)
2. **Streamlit app** — Predictive Model section with the same diagnostics
3. **Example scenarios** — P(swing), P(whiff|swing), P(swing & whiff) for labeled pitch profiles (e.g. 1–2 Sweeper chase)

---

## VI. Conclusion & Future Work

### Summary

Pitch-level swing and whiff models that include **location, count, leverage, and pitch physics** provide actionable estimates beyond season-long Whiff%.

### Future Work

- Hitter- and pitcher-specific features or embeddings for personalized matchup profiles
- Sequence features (`pitch_number`, prior pitch type) for tunneling effects
- Handedness interactions (`stand`, `p_throws`)

---

## Appendix

- `notebooks/statcast_pull.py` — data ingestion
- `notebooks/train_whiff_model.py` — training + HTML report
- `data/model/model_report.html` — visual output after training

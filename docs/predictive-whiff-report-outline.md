# Predictive Whiff Modeling: Data Science Report

**Project:** The Whiff List  
**Season:** 2025 MLB (Statcast)  
**Author:** [Your Name]  
**Date:** [Submission Date]

---

## Statistical Question

> ***Given an incoming pitch's characteristics (velocity, location, spin) and situational context (count, runners on), can classification models accurately estimate (A) the probability that a hitter swings and (B) the probability that a swing results in a miss (whiff)?***

---

## I. Executive Summary

### Problem Statement

Season-long metrics such as overall Whiff% are easy to quote but hard to act on. They summarize what already happened across thousands of pitches; they do not answer the question a fan in the stands, a broadcaster, or a front-office analyst actually asks before the next pitch is thrown: *Will this hitter swing, and if he does, will he miss?*

Consider a concrete matchup: with runners on and the count 1–2, will Aaron Judge swing at a Sweeper that finishes a foot off the plate? A single season Whiff% cannot express that pitch-by-pitch risk. Stakeholders at every level — from casual viewers following the competition to dugout staff and player-development groups — need estimates tied to **location, pitch type, count, and leverage**, not just a hitter's annual average.

### Approach

This project delivers that granularity through pitch-level classification on 2025 Statcast data and an interactive Streamlit dashboard (**The Whiff List**). The pipeline engineers spatial, contextual, and pitch-physics features; trains swing and whiff probability models with a chronological holdout; and surfaces results in a format usable from a curiosity-driven fan session up to front-office pre-series scouting workflows.

### Key Results

- [ROC-AUC and Log Loss for selected model on September 2025 holdout]
- [Top feature drivers: e.g., miss distance, two-strike counts, movement/spin]
- [Example hitter vulnerability profile from dashboard scatter plot]

---

## II. Data Pipeline & Feature Engineering

### Data Source & Sample

- **Source:** MLB Statcast pitch-level data via `pybaseball`
- **Season window:** March 23 – September 27, 2025
- **Qualification filter:** Batters with ≥ 502 plate appearances (batting-title baseline) to reduce small-sample noise
- **Unit of analysis:** Individual pitch

### Target Variable

Binary classification target:


| Value     | Definition                                                 |
| --------- | ---------------------------------------------------------- |
| **Y = 1** | Swinging strike, swinging strike (blocked), or missed bunt |
| **Y = 0** | All other pitch outcomes                                   |


### Feature Groups

#### Spatial Features


| Column(s)              | Description                                                                    |
| ---------------------- | ------------------------------------------------------------------------------ |
| `plate_x`, `plate_z`   | Horizontal and vertical location at the front of the plate                     |
| `sz_top`, `sz_bot`     | Batter-specific strike zone height                                             |
| Derived: miss distance | Euclidean distance (inches) from the pitch to the nearest strike zone boundary |


#### Contextual Features


| Column(s)                 | Description                                       |
| ------------------------- | ------------------------------------------------- |
| `balls`, `strikes`        | Current count                                     |
| `on_1b`, `on_2b`, `on_3b` | Base-state indicators, aggregated as `runners_on` |


#### Pitch Characteristics

Physical properties of the delivery — distinct from final location and post-pitch outcomes.


| Category  | Column(s)                                                                           | Description                                   |
| --------- | ----------------------------------------------------------------------------------- | --------------------------------------------- |
| Identity  | `pitch_type`                                                                        | Pitch classification (FF, SL, CH, CU, etc.)   |
| Velocity  | `release_speed`, `effective_speed`                                                  | Raw and extension-adjusted perceived velocity |
| Movement  | `pfx_x`, `pfx_z`                                                                    | Horizontal and vertical break                 |
| Spin      | `release_spin_rate`, `spin_axis`                                                    | Spin rate (rpm) and spin direction (degrees)  |
| Deception | `release_extension`, `release_pos_x`, `release_pos_y`, `release_pos_z`, `arm_angle` | Extension, release point, and arm slot        |


#### Matchup Context


| Column(s)           | Description                                                                          |
| ------------------- | ------------------------------------------------------------------------------------ |
| `stand`, `p_throws` | Batter and pitcher handedness; included for interaction with pitch type and movement |


#### Player Identifiers

Retained in the modeling frame and carried through to scored outputs for dashboard integration.


| Column(s)     | Description                                                                                    |
| ------------- | ---------------------------------------------------------------------------------------------- |
| `batter`      | MLBAM batter ID; joins to the qualified-hitter leaderboard and drives per-player scatter plots |
| `pitcher`     | MLBAM pitcher ID; enables pitcher-side aggregation and matchup analysis                        |
| `player_name` | Pitcher display name on each pitch row; used in dashboard tooltips and whiff tables            |


These identifiers support hitter- and pitcher-specific predictions in the app (leaderboard rankings, player breakdowns, and labeled pitch-level scatter points) rather than league-average estimates alone.

### Excluded Columns

The following were withheld from the feature matrix to prevent target leakage or post-swing information:

- **Outcomes:** `description`, `events`, `type`, `launch_speed`, `launch_angle`, `woba_value`, `estimated_ba_using_speedangle`
- **Swing/contact metrics:** `bat_speed`, `swing_length`, `attack_angle`, `attack_direction`, `swing_path_tilt`
- **Game metadata:** `game_pk`, fielder IDs, `home_team`, `away_team`, `umpire`
- **Deprecated fields:** `spin_dir`, `spin_rate_deprecated`, `break_angle_deprecated`, `break_length_deprecated`

---

## III. Exploratory Data Analysis

### League Whiff Topography

A two-dimensional density contour heatmap maps where swing-and-miss events cluster relative to the strike zone. The visualization replaces raw coordinate tables with an intuitive "danger zone" map for non-baseball readers.

### Temporal Stability

A seven-day rolling average of out-of-zone chase volume across the regular season tests whether league-wide chase behavior is stable enough over six months to support temporal generalization in the predictive model.

### Supporting Context

- Platoon and pitch-type breakdowns of whiff severity
- Distribution of miss distance and count-state leverage across qualified hitters

---

## IV. Model Development & Evaluation

### Algorithms

Two binary classifiers were trained and compared:


| Model               | Role                                                         |
| ------------------- | ------------------------------------------------------------ |
| Logistic Regression | Interpretable baseline; standardized feature inputs          |
| Random Forest       | Non-linear decision boundaries; handles feature interactions |


### Validation Design


| Split    | Period              | Rationale                                            |
| -------- | ------------------- | ---------------------------------------------------- |
| Training | April – August 2025 | In-season learning window                            |
| Test     | September 2025      | Forward-looking holdout simulating future prediction |


A random split was not used; the chronological partition preserves temporal ordering and avoids optimistic bias.

### Evaluation Metrics


| Metric       | Interpretation                                                |
| ------------ | ------------------------------------------------------------- |
| **ROC-AUC**  | How well predicted probabilities rank whiffs above non-whiffs |
| **Log Loss** | Penalty for overconfident incorrect probability estimates     |


Both metrics are reported on the September holdout set for each candidate model. The model with the higher ROC-AUC was selected for dashboard integration.

---

## V. Model Interpretation & Dashboard

### Feature Importance

Coefficient magnitudes (logistic regression) and impurity-based importances (random forest) identify the strongest predictors of swing-and-miss probability across spatial, contextual, pitch-characteristic, and player-level inputs.

### The Whiff List Dashboard

The Streamlit application connects the modeling pipeline to an audience-facing interface:

1. **Chase Leaderboard** — ranks qualified hitters by predicted or observed whiff vulnerability
2. **League Heatmap** — league-wide whiff density relative to the strike zone
3. **Player Scatter Plot** — pitch-level risk map for a selected hitter, with color encoding for whiff severity or predicted probability
4. **Seasonal Trends** — rolling volume and severity metrics over the 2025 schedule

---

## VI. Conclusion

### Summary

Pitch-level classification on Statcast data provides a forward-looking estimate of swing-and-miss risk that location-and-count context alone cannot capture. Chronological validation and probability-calibrated metrics (ROC-AUC, Log Loss) demonstrate whether the model generalizes to unseen late-season pitches. The dashboard translates model output and exploratory findings into visuals accessible to readers without Statcast literacy.

### Limitations

- Qualification filter (502 AB) excludes part-time hitters
- Pitch-characteristic coverage varies by pitch tracking quality and missingness
- Rare batter–pitcher matchups may have limited training exposure despite player IDs in the frame

### Future Work

- Target encoding or embeddings for high-cardinality batter/pitcher IDs to reduce sparsity
- Sequence features (`pitch_number`, prior-pitch type) to model tunneling effects
- Calibration plots and threshold analysis for operational decision support

---

## Appendix (Optional)

- Full feature dictionary
- Confusion matrix at selected probability threshold
- Reproducibility notes (`requirements.txt`, data pull scripts, model artifact paths)


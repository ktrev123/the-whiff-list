# Predictive Whiff Modeling: Data Science Report Outline

## Statistical Question

> **Given an incoming pitch's location, ball-metrics, and count, can we train a classification model to accurately estimate the probability of a hitter A.) swinging and B.) missing (whiffing)?**

---

## I. Executive Summary

### The Problem

In baseball analytics, evaluating a hitter's plate discipline traditionally relies on backward-looking volume metrics (like overall Whiff%). Teams need a way to simulate and project an upcoming matchup to determine precisely which incoming pitches a hitter is most likely to wildly chase.

### The Objective

Build a predictive classification model that estimates pitch-level swing-and-miss probabilities while using intuitive data visualizations to make the risk profiles instantly clear to non-baseball users.

---

## II. Data Pipeline & Feature Engineering

### Ingestion & Filter

- Extract pitch-level Statcast data using `pybaseball`
- Filter for the **502 At-Bat batting title qualification** baseline to eliminate small-sample noise

### Target Variable Formulation

Define the binary target variable:

- **Y = 1** — swinging strike / missed bunt
- **Y = 0** — all other pitch outcomes

### Feature Generation

Create predictive inputs from raw Statcast coordinates:

| Category | Features |
|----------|----------|
| **Spatial** | `plate_x`, `plate_z`, raw distance from the customized strike zone boundary |
| **Contextual** | Current count (balls, strikes), base-runner leverage (`runners_on`) |

---

## III. Exploratory Data Analysis (The "Non-Baseball" Context Layer)

### League Whiff Topography

Use a **2D density contour heatmap** to show exactly where the high-risk "danger zones" live relative to the strike zone, bypassing confusing coordinate numbers for the reader.

### Temporal Stability

Analyze the **7-day rolling average of league chase volume** to verify that player eagerness metrics remain stable enough over a six-month season to support a predictive model.

---

## IV. Model Development & Evaluation

### Algorithm Selection

Implement a binary classifier:

- **Logistic Regression** — baseline interpretability
- **XGBoost / Random Forest** — non-linear boundary relationships

### Validation Strategy

Split regular season data **chronologically** (e.g., train on April–August, test on September) to simulate true "future" prediction rather than a random split.

### Metrics

| Metric | Purpose |
|--------|---------|
| **ROC-AUC** | Measure probability sorting — how well the model separates high-risk swings from low-risk takes |
| **Log Loss** | Penalize overconfident wrong predictions |

*Plain-language framing:* how well the model separates an emergency flail from a disciplined take.

---

## V. Model Interpretation & Visual Dashboard Integration

### Feature Importance

Show which factors (e.g., distance from the plate vs. the count) matter most when predicting a miss.

### The Application — "The Whiff List"

Showcase how the live Streamlit dashboard operationalizes the model:

1. **Leaderboard** — ranks hitters by their overall predicted vulnerability
2. **Player Scatter Plot** — acts as a visual "risk map," where color-coded points don't just show historical whiffs, but map out the exact coordinates where the model projects a hitter is entirely defenseless

---

## VI. Conclusion & Future Iterations

### Key Takeaways

Summarize how predictive modeling can anticipate plate-discipline failures before a game even starts.

### Next Steps

Expand model features to include physical pitch characteristics:

- Spin rate
- Vertical / horizontal break vectors

These additions would map out exact player structural blindspots and refine pitch-level whiff probability estimates.

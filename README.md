# The Whiff List

The Whiff List is an interactive Streamlit app built with 2025 MLB Statcast data (`pybaseball`) to explore swing-and-miss behavior and pitch-level whiff risk.

## Statistical question

> Given an incoming pitch's characteristics (velocity, location, spin, movement, pitch type) and situational context (count, runners on), can classification models estimate **(A)** the probability a hitter swings and **(B)** the probability a swing results in a miss?

## What's built

**Dashboard (live)**
- Embarrassment Index leaderboard (502+ AB qualified hitters)
- League whiff heatmap, seasonal trends, platoon chase thermometers
- Predictive Model section: swing and whiff-if-swing models with plain-English metrics and visuals

**Modeling pipeline**
- **Model A (Swing)** — all pitches → P(swing)
- **Model B (Whiff)** — swings only → P(whiff | swing)
- Features: location, count, leverage, and pitch physics (`pitch_type`, velocity, movement, spin, extension)
- September holdout validation; HTML report generated after training

## Tech stack

Python · Streamlit · pandas · pybaseball · Plotly · scikit-learn

## Setup

```bash
pip install -r requirements.txt
```

## Data pipeline

Large Statcast and model artifacts are **not** in git. Regenerate locally (re-pull recommended after schema updates):

```bash
python notebooks/statcast_pull.py
python notebooks/build_whiff_leaderboard.py
python notebooks/train_whiff_model.py
```

The pull keeps **regular season only** (`game_type = R`, Mar 27 – Sep 28 2025), model-relevant columns only, and drops batters that do not resolve in the MLB ID lookup.

Training writes `data/model/model_report.html` (ROC, calibration, heatmaps, scenarios).

## Run the app

```bash
streamlit run app.py
```

## Report outline

Full write-up structure: [docs/predictive-whiff-report-outline.md](docs/predictive-whiff-report-outline.md)

| Section | Status |
|---------|--------|
| Statistical Question | Drafted |
| I. Executive Summary | Drafted |
| II. Data Pipeline & Feature Engineering | Drafted |
| III. Exploratory Data Analysis | Skeleton |
| IV. Model Development & Evaluation | Skeleton |
| V. Model Interpretation & Dashboard | Skeleton |
| VI. Conclusion & Future Work | Skeleton |

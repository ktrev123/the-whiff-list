# The Whiff List

The Whiff List is an interactive Streamlit app built with Statcast pitch-level data from `pybaseball` to identify swing-and-miss tendencies and pitch-level whiff risk for the 2025 MLB season.

## Features

- Embarrassment Index leaderboard for qualified hitters (502+ AB)
- League whiff heatmaps, seasonal trends, and platoon pitch thermometers
- Dual predictive models: **P(swing)** and **P(whiff | swing)** with pitch-physics features
- HTML model report generated after training

## Tech stack

- Python, Streamlit, pandas, pybaseball, Plotly, scikit-learn

## Setup

```bash
pip install -r requirements.txt
```

## Data pipeline

Large Statcast files are **not** in git. Regenerate locally:

```bash
python notebooks/statcast_pull.py
python notebooks/build_whiff_leaderboard.py
python notebooks/train_whiff_model.py
```

Training opens `data/model/model_report.html` with ROC, calibration, and heatmap visuals.

## Run the app

```bash
streamlit run app.py
```

## Report

See [docs/predictive-whiff-report-outline.md](docs/predictive-whiff-report-outline.md) for the data science write-up structure.

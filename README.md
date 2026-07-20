# AI Catcher: Pitch Optimizer

**Live app:** https://ai-catcher-abshire-portfolio.streamlit.app/

Interactive Streamlit dashboard that compares your pitch call against an AI catcher. The app simulates expected run value (ERV) across the strike zone using swing, whiff, and xwOBAcon models trained on 2025 Statcast data.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On first launch, the app downloads three Random Forest models (~260 MB total) from the [v1.0.0 release](https://github.com/ktrev123/the-whiff-list/releases/tag/v1.0.0) if they are not already in `models/`.

## Project layout

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit UI |
| `src/` | Feature engineering, EV models, pitch recommender, batter profiles |
| `data/` | Runtime parquet/CSV caches (qualified batters, zone heatmaps, league averages) |
| `models/` | Trained RF artifacts (`*.joblib`, gitignored — auto-downloaded or retrained) |
| `notebooks/` | EDA, modeling, and evaluation notebooks |
| `scripts/` | Data rebuild and model retraining utilities |
| `tests/` | Unit tests |

## Rebuilding data

Qualified batter roster (≥502 AB in 2025):

```powershell
python scripts\build_batter_roster.py
python scripts\build_batter_zone_profiles.py
```

League-average pitch physics (velocity/spin by pitch type):

```powershell
python scripts\build_league_average_physics.py
```

Full Statcast pull and feature pipeline are documented in `notebooks/00_data_pull.ipynb` and `notebooks/01_eda.ipynb`.

## Retraining models

```powershell
python scripts\retrain_production_models.py
```

See notebooks `05_modeling_swing_rf.ipynb`, `07_modeling_whiff_rf.ipynb`, and `09_modeling_xwobacon_rf.ipynb` for the full training workflow.

## Tests

```powershell
pytest
```

## Deployment notes

- **Live demo:** https://ai-catcher-abshire-portfolio.streamlit.app/ (Streamlit Cloud, repo `ktrev123/the-whiff-list`, entry `app.py`)
- **Streamlit Cloud**: point the app entry to `app.py`. Upload or generate `data/batter_roster.parquet`, `data/batter_zone_xwobacon.parquet`, and `data/players.parquet`; models are fetched from GitHub Releases at runtime.
- **Local**: keep `data/league_average_physics.csv` present (small CSV, tracked in git).
- Large parquet files and model weights are gitignored; publish release assets or rebuild locally before deploying.

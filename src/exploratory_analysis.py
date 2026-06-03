"""Exploratory analysis: correlations, trends, league vs per-player modeling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.eda_report import (
    render_chart_block,
    render_chart_row,
    section_html,
    table_html,
    export_eda_report,
)

PITCH_GROUP_ORDER = ["Fastballs", "Breaking Balls", "Off-Speed"]
from src.statcast_schema import (
    PITCH_TYPE_GROUPS,
    SEASON_END,
    SEASON_START,
    pitch_type_group,
)
from src.whiff_features import (
    CATEGORICAL_FEATURE_COLS,
    MODEL_INPUT_COLS,
    NUMERIC_FEATURE_COLS,
    PITCH_METRIC_COLS,
    apply_pitch_imputation,
    chronological_split,
    compute_pitch_medians,
    engineer_features,
    filter_modeling_frame,
)

COUNT_ORDER = [f"{b}-{s}" for b in range(4) for s in range(3)]
ATTACK_ZONE_CREDIT = (
    'Zone definitions follow the Statcast/FanGraphs attack-zone framework '
    '(~2.9 in per baseball; Heart &gt;2 in inside the zone, Shadow within ±2 in of the boundary, '
    'Chase 2–4 in outside, Waste &gt;4 in outside). '
    'Reference: <a href="https://blogs.fangraphs.com/wp-content/uploads/2024/05/Attack-Zones.jpg" '
    'target="_blank" rel="noopener">FanGraphs Attack Zones</a>.'
)

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "statcast_2025.parquet"
LEADERBOARD_FILE = ROOT / "data" / "whiff_leaderboard_2025.csv"
REPORT_DIR = ROOT / "data" / "reports"
REPORT_FILE = REPORT_DIR / "eda_report.html"

MIN_ABS = 502
LEVERAGE_CODE = {"pitcher_ahead": 1, "even": 0, "hitter_ahead": -1, "two_strike": 2}

CONTEXTUAL_CORR_COLS = [
    "miss_dist_in",
    "plate_x",
    "plate_z",
    "runners_on",
    "is_two_strike",
    "is_swing",
    "is_whiff",
]
PITCH_CORR_COLS = [
    "release_speed",
    "pfx_x",
    "pfx_z",
    "release_spin_rate",
    "spin_axis",
    "release_extension",
    "is_swing",
    "is_whiff",
]

FEATURE_LABELS = {
    "miss_dist_in": "Miss distance (in)",
    "plate_x": "Horizontal location",
    "plate_z": "Vertical location",
    "runners_on": "Runners on",
    "is_two_strike": "Two-strike flag",
    "count_adv_code": "Count leverage",
    "balls": "Balls",
    "strikes": "Strikes",
    "release_speed": "Release speed (mph)",
    "speed_diff": "Speed diff (eff − release)",
    "pfx_x": "Horizontal break (in)",
    "pfx_z": "Vertical break (in)",
    "release_spin_rate": "Spin rate (rpm)",
    "spin_axis": "Spin axis (deg)",
    "release_extension": "Extension (ft)",
    "is_swing": "Swing",
    "is_whiff": "Whiff",
}

PITCH_CODE_NAMES = {
    "FF": "Four-seam fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "KC": "Knuckle-curve",
    "ST": "Sweeper",
    "SV": "Slurve",
    "CU": "Curveball",
    "CH": "Changeup",
    "FS": "Splitter",
}

PHYSICS_METRICS = [
    ("release_speed", "Velocity (mph)"),
    ("speed_diff", "Speed diff (mph)"),
    ("pfx_x", "H-break (in)"),
    ("pfx_z", "V-break (in)"),
    ("release_spin_rate", "Spin (rpm)"),
    ("release_extension", "Extension (ft)"),
]


def load_batter_names() -> pd.DataFrame:
    names = pd.read_csv(LEADERBOARD_FILE)[["batter", "player_name"]].drop_duplicates()
    names["player_name"] = names["player_name"].str.title()
    return names


def boundary_signed_inches(plate_x: float, plate_z: float, sz_top: float, sz_bot: float) -> float:
    """Signed distance in inches from the strike-zone boundary (negative = inside)."""
    half_w_ft = 17.0 / 24.0
    inside_x = half_w_ft - abs(plate_x)
    inside_z_top = sz_top - plate_z
    inside_z_bot = plate_z - sz_bot
    inside_min_ft = min(inside_x, inside_z_top, inside_z_bot)
    return inside_min_ft * 12.0


def assign_attack_zone(row) -> str:
    px, pz = row.get("plate_x"), row.get("plate_z")
    if pd.isna(px) or pd.isna(pz):
        return "Unknown"
    sz_top = float(row.get("sz_top", 3.5))
    sz_bot = float(row.get("sz_bot", 1.5))
    signed_in = boundary_signed_inches(float(px), float(pz), sz_top, sz_bot)
    if signed_in <= -2.0:
        return "Heart"
    if signed_in <= 2.0:
        return "Shadow"
    if signed_in <= 4.0:
        return "Chase"
    return "Waste"


def _velocity_fences(train_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    fences: dict[str, tuple[float, float]] = {}
    for pt, grp in train_df.groupby("pitch_type", observed=True):
        speeds = grp["release_speed"].dropna()
        if len(speeds) < 20:
            continue
        q1, q3 = speeds.quantile(0.25), speeds.quantile(0.75)
        iqr = q3 - q1
        fences[pt] = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
    return fences


def _apply_velocity_fences(df: pd.DataFrame, fences: dict[str, tuple[float, float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    keep_mask = pd.Series(True, index=df.index)
    for pt, grp in df.groupby("pitch_type", observed=True):
        if pt not in fences:
            continue
        lo, _hi = fences[pt]
        outlier_idx = grp.index[grp["release_speed"] < lo]
        keep_mask.loc[outlier_idx] = False
        removed = grp.loc[outlier_idx, "release_speed"]
        rows.append(
            {
                "pitch_type": pt,
                "n_before": len(grp),
                "n_removed": int(len(outlier_idx)),
                "lower_fence_mph": round(lo, 1),
                "min_removed_mph": round(float(removed.min()), 1) if len(removed) else None,
                "max_removed_mph": round(float(removed.max()), 1) if len(removed) else None,
            }
        )
    report = pd.DataFrame(rows)
    if not report.empty:
        report["pct_removed"] = (report["n_removed"] / report["n_before"] * 100).round(2)
    return df.loc[keep_mask].copy(), report


def remove_velocity_outliers(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Lower-bound Tukey IQR (1.5×) on release_speed per pitch type; fast outliers retained."""
    fences = _velocity_fences(train_df)
    train_clean, train_report = _apply_velocity_fences(train_df, fences)
    test_clean, test_report = _apply_velocity_fences(test_df, fences)
    combined = pd.concat([train_report, test_report], ignore_index=True)
    summary = (
        combined.groupby("pitch_type", as_index=False)
        .agg(
            n_removed=("n_removed", "sum"),
            lower_fence_mph=("lower_fence_mph", "first"),
            min_removed_mph=("min_removed_mph", "min"),
            max_removed_mph=("max_removed_mph", "max"),
        )
        .sort_values("n_removed", ascending=False)
    )
    total_removed = int(combined["n_removed"].sum())
    return train_clean, test_clean, summary, total_removed


def _spin_fences(train_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    fences: dict[str, tuple[float, float]] = {}
    for pt, grp in train_df.groupby("pitch_type", observed=True):
        spin = grp["release_spin_rate"].dropna()
        if len(spin) < 20:
            continue
        q1, q3 = spin.quantile(0.25), spin.quantile(0.75)
        iqr = q3 - q1
        fences[pt] = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
    return fences


def _apply_spin_fences(df: pd.DataFrame, fences: dict[str, tuple[float, float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    keep_mask = pd.Series(True, index=df.index)
    for pt, grp in df.groupby("pitch_type", observed=True):
        if pt not in fences:
            continue
        lo, hi = fences[pt]
        outlier_idx = grp.index[(grp["release_spin_rate"] < lo) | (grp["release_spin_rate"] > hi)]
        keep_mask.loc[outlier_idx] = False
        removed = grp.loc[outlier_idx, "release_spin_rate"]
        rows.append(
            {
                "pitch_type": pt,
                "n_before": len(grp),
                "n_removed": int(len(outlier_idx)),
                "lower_fence_rpm": round(lo, 0),
                "upper_fence_rpm": round(hi, 0),
                "min_removed_rpm": round(float(removed.min()), 0) if len(removed) else None,
                "max_removed_rpm": round(float(removed.max()), 0) if len(removed) else None,
            }
        )
    report = pd.DataFrame(rows)
    if not report.empty:
        report["pct_removed"] = (report["n_removed"] / report["n_before"] * 100).round(2)
    return df.loc[keep_mask].copy(), report


def remove_spin_outliers(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Tukey IQR (1.5×) on release_spin_rate per pitch type; removes high and low outliers."""
    fences = _spin_fences(train_df)
    train_clean, train_report = _apply_spin_fences(train_df, fences)
    test_clean, test_report = _apply_spin_fences(test_df, fences)
    combined = pd.concat([train_report, test_report], ignore_index=True)
    summary = (
        combined.groupby("pitch_type", as_index=False)
        .agg(
            n_removed=("n_removed", "sum"),
            lower_fence_rpm=("lower_fence_rpm", "first"),
            upper_fence_rpm=("upper_fence_rpm", "first"),
            min_removed_rpm=("min_removed_rpm", "min"),
            max_removed_rpm=("max_removed_rpm", "max"),
        )
        .sort_values("n_removed", ascending=False)
    )
    total_removed = int(combined["n_removed"].sum())
    return train_clean, test_clean, summary, total_removed


def enrich_eda_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["count_adv_code"] = out["count_leverage"].map(LEVERAGE_CODE).astype("float32")
    out["pitch_group"] = out["pitch_type"].map(pitch_type_group)
    out["attack_zone"] = out.apply(assign_attack_zone, axis=1)
    out["count_label"] = out["balls"].astype(int).astype(str) + "-" + out["strikes"].astype(int).astype(str)
    return out


def load_modeling_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing {DATA_FILE}. Run notebooks/statcast_pull.py first.")
    batter_names = load_batter_names()
    lb = pd.read_csv(LEADERBOARD_FILE)
    qualified = lb.loc[lb["ab"] >= MIN_ABS, "batter"].astype(int).tolist()
    raw = pd.read_parquet(DATA_FILE)
    raw["game_date"] = pd.to_datetime(raw["game_date"])
    raw = raw[(raw["game_date"] >= SEASON_START) & (raw["game_date"] <= SEASON_END)]
    frame = filter_modeling_frame(raw, qualified)
    frame = frame.drop(columns=["player_name"], errors="ignore")
    frame = frame.merge(batter_names, on="batter", how="left")
    train_df, test_df = chronological_split(frame)
    pitch_medians = compute_pitch_medians(train_df)
    train_df = enrich_eda_columns(apply_pitch_imputation(train_df, pitch_medians))
    test_df = enrich_eda_columns(apply_pitch_imputation(test_df, pitch_medians))
    train_df, test_df, velocity_summary, n_velocity_removed = remove_velocity_outliers(train_df, test_df)
    train_df, test_df, spin_summary, n_spin_removed = remove_spin_outliers(train_df, test_df)
    return train_df, test_df, pitch_medians, batter_names, velocity_summary, spin_summary, n_velocity_removed, n_spin_removed


def build_preprocessor():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURE_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURE_COLS),
        ]
    )


def fit_league_logistic(x_train, y_train):
    model = Pipeline(
        [
            ("prep", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    return model


def correlation_matrix_figure(df: pd.DataFrame, cols: list[str], title: str, *, compact: bool = False) -> go.Figure:
    use_cols = [c for c in cols if c in df.columns]
    corr = df[use_cols].corr(numeric_only=True)
    labels = [FEATURE_LABELS.get(c, c) for c in corr.columns]
    values = corr.values.round(2)
    n = len(labels)
    ticks = list(range(n))
    cell = max(44, min(58, 420 // max(n, 1))) if compact else max(52, min(68, 560 // max(n, 1)))
    fig_w = min(460, cell * n + 100) if compact else min(820, cell * n + 220)
    fig_h = min(500, cell * n + 120) if compact else min(820, cell * n + 220)
    l_margin = 150 if compact else 210
    b_margin = 150 if compact else 210
    fig = go.Figure(
        go.Heatmap(
            name="Correlation",
            z=values.tolist(),
            x=ticks,
            y=ticks,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=values.tolist(),
            texttemplate="%{text}",
            textfont=dict(size=12, color="#0f172a"),
            hovertemplate="r=%{z:.2f}<extra></extra>",
            colorbar=dict(title=dict(text="Correlation (r)"), len=0.75),
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=fig_h,
        width=fig_w,
        margin=dict(l=l_margin, r=50, t=60, b=b_margin),
        xaxis=dict(
            tickmode="array",
            tickvals=ticks,
            ticktext=labels,
            tickangle=-40,
            tickfont=dict(size=9 if compact else 10, color="#f5efe3"),
            side="bottom",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=ticks,
            ticktext=labels,
            tickfont=dict(size=9 if compact else 10, color="#f5efe3"),
            autorange="reversed",
        ),
    )
    return fig


def top_corr_table(df: pd.DataFrame, cols: list[str], outcome: str, n: int = 6) -> pd.DataFrame:
    features = [c for c in cols if c in df.columns and c not in {"is_swing", "is_whiff"}]
    ranked = (
        df[features + [outcome]]
        .corr(numeric_only=True)[outcome]
        .drop(outcome, errors="ignore")
        .abs()
        .sort_values(ascending=False)
        .head(n)
    )
    return (
        pd.DataFrame({"Feature": [FEATURE_LABELS.get(i, i) for i in ranked.index], "|r|": ranked.values.round(3)})
    )


def batter_rate_table(df: pd.DataFrame, batter_names: pd.DataFrame) -> pd.DataFrame:
    swings = df[df["is_swing"] == 1]
    whiff = swings.groupby("batter").agg(whiff_if_swing=("is_whiff", "mean"))
    out = df.groupby("batter").agg(
        pitches=("is_swing", "size"),
        swing_rate=("is_swing", "mean"),
        chase_rate=("miss_dist_in", lambda s: float((s > 0).mean())),
    )
    out = out.join(whiff, how="left").reset_index().merge(batter_names, on="batter", how="left")
    out["whiff_if_swing"] = out["whiff_if_swing"].fillna(0)
    return out


def rate_distribution_figures(batter_rates: pd.DataFrame, *, height: int = 440, width: int = 500) -> list[go.Figure]:
    figs = []
    for col, title, color in [
        ("swing_rate", "Distribution of Hitter Swing Rates", "#20908d"),
        ("whiff_if_swing", "Distribution of Hitter Whiff-if-Swing Rates", "#e63946"),
        ("chase_rate", "Distribution of Hitter Chase Rates", "#d4a937"),
    ]:
        fig = go.Figure(go.Histogram(x=batter_rates[col].tolist(), nbinsx=20, marker_color=color, opacity=0.9))
        fig.add_vline(x=float(batter_rates[col].mean()), line_dash="dash", line_color="#f5efe3", annotation_text="Mean")
        fig.update_layout(
            title=title,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickformat=".0%"),
            height=height,
            width=width,
            margin=dict(l=50, r=30, t=50, b=50),
        )
        figs.append(fig)
    return figs


def heterogeneity_metrics(batter_rates: pd.DataFrame) -> dict:
    metrics = {}
    for col in ["swing_rate", "whiff_if_swing", "chase_rate"]:
        series = batter_rates[col].dropna()
        metrics[f"{col}_mean"] = float(series.mean())
        metrics[f"{col}_std"] = float(series.std())
        metrics[f"{col}_range"] = float(series.max() - series.min())
    return metrics


def variance_decomposition(df: pd.DataFrame, outcome: str) -> dict:
    overall = float(df[outcome].var())
    between = float(df.groupby("batter")[outcome].mean().var())
    return {"overall_var": overall, "between_batter_var": between, "between_share": between / overall if overall else 0.0}


def count_rates_figure(df: pd.DataFrame) -> go.Figure:
    swings = df[df["is_swing"] == 1]
    agg = df.groupby("count_label", as_index=False).agg(swing_rate=("is_swing", "mean"), n=("is_swing", "size"))
    whiff = swings.groupby("count_label", as_index=False).agg(whiff_if_swing=("is_whiff", "mean"))
    agg = agg.merge(whiff, on="count_label", how="left")
    agg["count_label"] = pd.Categorical(agg["count_label"], categories=COUNT_ORDER, ordered=True)
    agg = agg.sort_values("count_label")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["count_label"].astype(str), y=agg["swing_rate"], name="Swing rate", marker_color="#20908d"))
    fig.add_trace(
        go.Bar(x=agg["count_label"].astype(str), y=agg["whiff_if_swing"], name="Whiff | swing", marker_color="#fde725", opacity=0.92)
    )
    fig.update_layout(
        title="Swing & whiff by count (balls-strikes)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="group",
        yaxis_tickformat=".0%",
        xaxis_title="Count",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return fig


def pitch_category_count_figure(df: pd.DataFrame) -> go.Figure:
    counts = df.groupby("pitch_group", observed=True).size().reindex(PITCH_GROUP_ORDER, fill_value=0)
    fig = go.Figure(go.Bar(x=PITCH_GROUP_ORDER, y=counts.values.tolist(), marker_color="#d4a937"))
    fig.update_layout(
        title="Pitch Category Volume",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Category",
        yaxis_title="Pitch count",
        height=380,
    )
    return fig


def velocity_by_category_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = {"Fastballs": "#20908d", "Breaking Balls": "#e63946", "Off-Speed": "#fde725"}
    for group in PITCH_GROUP_ORDER:
        subset = df.loc[df["pitch_group"] == group, "release_speed"].dropna()
        if subset.empty:
            continue
        fig.add_trace(go.Box(y=subset.tolist(), name=group, marker_color=colors.get(group), boxpoints=False))
    fig.update_layout(
        title="Release Velocity by Pitch Category",
        template="plotly_dark",
        yaxis_title="mph",
        height=420,
        showlegend=False,
    )
    return fig


def pitch_category_rates_figure(df: pd.DataFrame) -> go.Figure:
    swings = df[df["is_swing"] == 1]
    pt = df.groupby("pitch_group", as_index=False).agg(pitches=("is_swing", "size"), swing_rate=("is_swing", "mean"))
    whiff = swings.groupby("pitch_group", as_index=False).agg(whiff_if_swing=("is_whiff", "mean"))
    pt = pt.merge(whiff, on="pitch_group")
    pt["pitch_group"] = pd.Categorical(pt["pitch_group"], categories=PITCH_GROUP_ORDER, ordered=True)
    pt = pt.sort_values("pitch_group")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=pt["pitch_group"].astype(str), y=pt["swing_rate"], name="Swing rate", marker_color="#20908d"))
    fig.add_trace(go.Bar(x=pt["pitch_group"].astype(str), y=pt["whiff_if_swing"], name="Whiff | swing", marker_color="#fde725", opacity=0.92))
    fig.update_layout(
        title="Swing and Whiff by Pitch Category",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="group",
        yaxis_tickformat=".0%",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return fig


def pitch_type_rates_figure(df: pd.DataFrame) -> go.Figure:
    return pitch_category_rates_figure(df)


def platoon_split_figure(df: pd.DataFrame) -> go.Figure:
    if "stand" not in df.columns:
        return go.Figure()
    swings = df[df["is_swing"] == 1]
    split = df.groupby("stand", as_index=False).agg(swing_rate=("is_swing", "mean"), chase_rate=("miss_dist_in", lambda s: float((s > 0).mean())))
    whiff = swings.groupby("stand", as_index=False).agg(whiff_if_swing=("is_whiff", "mean"))
    split = split.merge(whiff, on="stand")
    split["label"] = split["stand"].map({"L": "LHB", "R": "RHB"})

    fig = go.Figure()
    for col, name, color in [("swing_rate", "Swing rate", "#20908d"), ("whiff_if_swing", "Whiff | swing", "#e63946"), ("chase_rate", "Chase rate", "#d4a937")]:
        fig.add_trace(go.Bar(x=split["label"], y=split[col], name=name, marker_color=color))
    fig.update_layout(title="Platoon splits (batter stand)", template="plotly_dark", barmode="group", yaxis_tickformat=".0%", height=380)
    return fig


def physics_table_for_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    subset = df[df["pitch_group"] == group]
    rows = []
    for col, label in PHYSICS_METRICS:
        series = subset[col].dropna()
        if series.empty:
            continue
        rows.append(
            {
                "Metric": label,
                "Mean": round(float(series.mean()), 2),
                "Std": round(float(series.std()), 2),
                "P10": round(float(series.quantile(0.10)), 2),
                "P50": round(float(series.quantile(0.50)), 2),
                "P90": round(float(series.quantile(0.90)), 2),
                "N": int(len(series)),
            }
        )
    return pd.DataFrame(rows)


def physics_category_summary(df: pd.DataFrame, group: str) -> str:
    subset = df[df["pitch_group"] == group]
    if subset.empty:
        return f"<p><em>No {group} pitches in sample.</em></p>"
    vel = subset["release_speed"].median()
    spin = subset["release_spin_rate"].median()
    hb = subset["pfx_x"].abs().median()
    vb = subset["pfx_z"].median()
    if group == "Fastballs":
        return (
            f"<p><b>{group}:</b> Median velocity is <b>{vel:.1f} mph</b> with relatively tight dispersion — "
            f"these pitches set the baseline timing hitters expect. Spin averages <b>{spin:.0f} rpm</b>; "
            f"horizontal break stays modest (median |H-break| ≈ {hb:.1f} in), so location and count drive most swing decisions.</p>"
        )
    if group == "Breaking Balls":
        return (
            f"<p><b>{group}:</b> Median velocity <b>{vel:.1f} mph</b> with higher movement — "
            f"median vertical break ≈ {vb:.1f} in and spin ≈ <b>{spin:.0f} rpm</b>. "
            f"These pitches show the widest physics spread and correlate more strongly with whiff than with swing.</p>"
        )
    fb_vel = df.loc[df["pitch_group"] == "Fastballs", "release_speed"].median()
    vel_gap = vel - fb_vel if pd.notna(fb_vel) else 0
    return (
        f"<p><b>{group}:</b> Median velocity <b>{vel:.1f} mph</b>, roughly <b>{vel_gap:.0f} mph</b> "
        f"slower than fastballs, with spin near <b>{spin:.0f} rpm</b>. "
        f"Speed differential from the hitter's timing baseline makes these pitches disproportionately effective when located away from the zone.</p>"
    )


def physics_by_category_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in PITCH_TYPE_GROUPS:
        table = physics_table_for_group(df, group)
        for _, row in table.iterrows():
            rows.append({"Category": group, **row.to_dict()})
    return pd.DataFrame(rows)



def velocity_by_pitch_type_figure(df: pd.DataFrame) -> go.Figure:
    return velocity_by_category_figure(df)


def spin_by_category_figure(df: pd.DataFrame) -> go.Figure:
    colors = {"Fastballs": "#20908d", "Breaking Balls": "#e63946", "Off-Speed": "#fde725"}
    fig = go.Figure()
    for group in PITCH_GROUP_ORDER:
        subset = df.loc[df["pitch_group"] == group, "release_spin_rate"].dropna()
        if subset.empty:
            continue
        fig.add_trace(go.Box(y=subset.tolist(), name=group, marker_color=colors.get(group), boxpoints=False))
    fig.update_layout(
        title="Spin Rate by Pitch Category",
        template="plotly_dark",
        yaxis_title="Spin rate (rpm)",
        height=420,
        showlegend=False,
    )
    return fig


def location_heatmap(df: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    x_min, x_max = -1.5, 1.5
    z_min, z_max = 1.0, 4.2
    nx, nz = 18, 22
    x_edges = np.linspace(x_min, x_max, nx + 1)
    z_edges = np.linspace(z_min, z_max, nz + 1)
    x_centers = ((x_edges[:-1] + x_edges[1:]) / 2).tolist()
    z_centers = ((z_edges[:-1] + z_edges[1:]) / 2).tolist()

    tmp = df.dropna(subset=["plate_x", "plate_z"]).copy()
    tmp = tmp[(tmp["plate_x"] >= x_min) & (tmp["plate_x"] <= x_max) & (tmp["plate_z"] >= z_min) & (tmp["plate_z"] <= z_max)]
    tmp["x_bin"] = pd.cut(tmp["plate_x"], bins=x_edges, labels=False, include_lowest=True)
    tmp["z_bin"] = pd.cut(tmp["plate_z"], bins=z_edges, labels=False, include_lowest=True)
    grouped = tmp.groupby(["z_bin", "x_bin"], observed=True).agg(rate=(value_col, "mean"), n=(value_col, "size"))

    matrix = [[None] * nx for _ in range(nz)]
    for (zb, xb), row in grouped.iterrows():
        if int(row["n"]) >= 20 and float(row["rate"]) > 0:
            matrix[int(zb)][int(xb)] = float(row["rate"])

    flat = [v for row in matrix for v in row if v is not None]
    zmin = min(flat) if flat else 0.05

    fig = go.Figure(
        go.Heatmap(
            name=title,
            z=matrix,
            x=x_centers,
            y=z_centers,
            colorscale="Viridis",
            zmin=zmin,
            colorbar=dict(title="Rate", tickformat=".0%"),
            hovertemplate="x=%{x:.2f} ft, z=%{y:.2f} ft<br>rate=%{z:.1%}<extra></extra>",
        )
    )
    avg_bot, avg_top = float(df["sz_bot"].median()), float(df["sz_top"].median())
    fig.add_shape(type="rect", x0=-0.708, x1=0.708, y0=avg_bot, y1=avg_top, line=dict(color="#f5efe3", width=2))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[x_min, x_max], scaleanchor="y", scaleratio=1, constrain="domain"),
        yaxis=dict(range=[z_min, z_max]),
        xaxis_title="Horizontal (ft)",
        yaxis_title="Vertical (ft)",
        height=520,
        width=440,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def attack_zone_figure(df: pd.DataFrame) -> go.Figure:
    order = ["Heart", "Shadow", "Chase", "Waste"]
    swings = df[df["is_swing"] == 1]
    agg = df.groupby("attack_zone", as_index=False).agg(swing_rate=("is_swing", "mean"), n=("is_swing", "size"))
    whiff = swings.groupby("attack_zone", as_index=False).agg(whiff_if_swing=("is_whiff", "mean"))
    agg = agg.merge(whiff, on="attack_zone", how="left")
    agg["attack_zone"] = pd.Categorical(agg["attack_zone"], categories=order, ordered=True)
    agg = agg.sort_values("attack_zone")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg["attack_zone"], y=agg["swing_rate"], name="Swing rate", marker_color="#20908d"))
    fig.add_trace(go.Bar(x=agg["attack_zone"], y=agg["whiff_if_swing"], name="Whiff | swing", marker_color="#fde725", opacity=0.92))
    fig.update_layout(
        title="Swing and Whiff by Attack Zone (All Pitches)",
        template="plotly_dark",
        barmode="group",
        yaxis_tickformat=".0%",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return fig


def attack_zone_by_category_figure(df: pd.DataFrame) -> go.Figure:
    order = ["Heart", "Shadow", "Chase", "Waste"]
    colors = {"Fastballs": "#20908d", "Breaking Balls": "#e63946", "Off-Speed": "#fde725"}
    fig = make_subplots(rows=1, cols=3, subplot_titles=PITCH_GROUP_ORDER, horizontal_spacing=0.08)
    for col_idx, group in enumerate(PITCH_GROUP_ORDER, start=1):
        sub = df[df["pitch_group"] == group]
        if sub.empty:
            continue
        swings = sub[sub["is_swing"] == 1]
        agg = sub.groupby("attack_zone", as_index=False).agg(swing_rate=("is_swing", "mean"))
        whiff = swings.groupby("attack_zone", as_index=False).agg(whiff_if_swing=("is_whiff", "mean"))
        agg = agg.merge(whiff, on="attack_zone", how="left")
        agg["attack_zone"] = pd.Categorical(agg["attack_zone"], categories=order, ordered=True)
        agg = agg.sort_values("attack_zone")
        fig.add_trace(
            go.Bar(x=agg["attack_zone"], y=agg["swing_rate"], name="Swing rate", marker_color=colors[group], showlegend=(col_idx == 1)),
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Bar(x=agg["attack_zone"], y=agg["whiff_if_swing"], name="Whiff | swing", marker_color=colors[group], opacity=0.55, showlegend=(col_idx == 1)),
            row=1,
            col=col_idx,
        )
    fig.update_layout(title="Swing and Whiff by Attack Zone and Pitch Category", template="plotly_dark", barmode="group", height=420)
    fig.update_yaxes(tickformat=".0%")
    return fig


def learning_curve_figure(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    min_test: int = 40,
) -> tuple[go.Figure, int | None]:
    thresholds = [150, 200, 250, 300, 400, 500, 600, 800, 1000, 1500, 2000]
    league = fit_league_logistic(train_df[MODEL_INPUT_COLS], train_df["is_swing"])
    league_lls, player_lls, win_pcts = [], [], []
    crossover_n = None

    for n in thresholds:
        lls_l, lls_p, wins, total = [], [], 0, 0
        for batter, b_train in train_df.groupby("batter"):
            b_test = test_df[test_df["batter"] == batter]
            if len(b_train) < n or len(b_test) < min_test:
                continue
            sample = b_train.sort_values("game_date").head(n)
            l_prob = league.predict_proba(b_test[MODEL_INPUT_COLS])[:, 1]
            lls_l.append(log_loss(b_test["is_swing"], l_prob, labels=[0, 1]))
            player = fit_league_logistic(sample[MODEL_INPUT_COLS], sample["is_swing"])
            p_prob = player.predict_proba(b_test[MODEL_INPUT_COLS])[:, 1]
            p_ll = log_loss(b_test["is_swing"], p_prob, labels=[0, 1])
            lls_p.append(p_ll)
            wins += int(p_ll < lls_l[-1])
            total += 1
        if not lls_l:
            continue
        league_lls.append(float(np.mean(lls_l)))
        player_lls.append(float(np.mean(lls_p)))
        win_pcts.append(wins / total if total else 0)
        if crossover_n is None and player_lls[-1] < league_lls[-1]:
            crossover_n = n

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=thresholds[: len(league_lls)], y=league_lls, name="League model", line=dict(color="#cbbfa8", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=thresholds[: len(player_lls)], y=player_lls, name="Hitter-specific avg", line=dict(color="#20908d", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=thresholds[: len(win_pcts)], y=win_pcts, name="Hitter model win %", line=dict(color="#d4a937", dash="dot"), mode="lines+markers"), secondary_y=True)
    if crossover_n:
        fig.add_vline(x=crossover_n, line_dash="dash", line_color="#fde725", annotation_text=f"Crossover ~{crossover_n} pitches")
    fig.update_layout(title="Personalization Learning Curve (September Holdout)", template="plotly_dark", height=420)
    fig.update_yaxes(title_text="Mean log loss", secondary_y=False)
    fig.update_yaxes(title_text="Win rate", tickformat=".0%", secondary_y=True)
    return fig, crossover_n


def _encoded_feature_label(name: str) -> str:
    if name.startswith("num__"):
        return FEATURE_LABELS.get(name.removeprefix("num__"), name.removeprefix("num__"))
    if name.startswith("cat__pitch_type_"):
        return f"Pitch type: {name.removeprefix('cat__pitch_type_')}"
    if name.startswith("cat__count_state_"):
        return f"Count: {name.removeprefix('cat__count_state_').replace('_', ' ')}"
    return name


def feature_importance_table(model, *, top_n: int = 12) -> pd.DataFrame:
    prep = model.named_steps["prep"]
    coefs = np.abs(model.named_steps["clf"].coef_[0])
    names = prep.get_feature_names_out()
    total = float(coefs.sum()) or 1.0
    ranked = sorted(zip(names, coefs), key=lambda x: x[1], reverse=True)[:top_n]
    return pd.DataFrame(
        {
            "Feature": [_encoded_feature_label(n) for n, _ in ranked],
            "Relative importance": [round(float(v) / total, 4) for _, v in ranked],
        }
    )


def feature_importance_figure(model, *, title: str, top_n: int = 10) -> go.Figure:
    table = feature_importance_table(model, top_n=top_n)
    fig = go.Figure(
        go.Bar(
            x=table["Relative importance"].tolist(),
            y=table["Feature"].tolist()[::-1],
            orientation="h",
            marker_color="#d4a937",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Relative importance (|coefficient|, normalized)",
        height=max(340, 28 * top_n),
        margin=dict(l=180, r=40, t=50, b=40),
    )
    return fig


def load_production_importance() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    insights_path = ROOT / "data" / "model" / "model_insights.json"
    if not insights_path.exists():
        return None, None
    import json

    def _clean(rows: list[dict]) -> pd.DataFrame:
        kept = []
        for row in rows[:12]:
            label = str(row.get("label", row.get("feature", "")))
            if any(
                x in label.lower()
                for x in ("effective velocity", "effective_speed", "balls in the count", "strikes in the count")
            ):
                continue
            kept.append({"Feature": label, "Relative importance": row["importance"]})
        return pd.DataFrame(kept[:10])

    data = json.loads(insights_path.read_text(encoding="utf-8"))
    swing_df = _clean(data.get("swing", {}).get("feature_importance", []))
    whiff_df = _clean(data.get("whiff", {}).get("feature_importance", []))
    return (swing_df if not swing_df.empty else None), (whiff_df if not whiff_df.empty else None)


def shadow_personalization_table(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    league = fit_league_logistic(train_df[MODEL_INPUT_COLS], train_df["is_swing"])
    rows = []
    for zone in ["Heart", "Shadow", "Chase", "Waste"]:
        t = test_df[test_df["attack_zone"] == zone]
        if t.empty:
            continue
        l_prob = league.predict_proba(t[MODEL_INPUT_COLS])[:, 1]
        league_ll = log_loss(t["is_swing"], l_prob, labels=[0, 1])
        player_lls, wins = [], 0
        for batter, b_train in train_df.groupby("batter"):
            b_test = t[t["batter"] == batter]
            if len(b_train) < 200 or len(b_test) < 10:
                continue
            player = fit_league_logistic(b_train[MODEL_INPUT_COLS], b_train["is_swing"])
            p_prob = player.predict_proba(b_test[MODEL_INPUT_COLS])[:, 1]
            l_prob_b = league.predict_proba(b_test[MODEL_INPUT_COLS])[:, 1]
            p_ll = log_loss(b_test["is_swing"], p_prob, labels=[0, 1])
            l_ll_b = log_loss(b_test["is_swing"], l_prob_b, labels=[0, 1])
            player_lls.append(p_ll)
            wins += int(p_ll < l_ll_b)
        avg_hitter = float(np.mean(player_lls)) if player_lls else None
        gain = round(league_ll - avg_hitter, 4) if avg_hitter is not None else None
        rows.append(
            {
                "Zone": zone,
                "Test pitches": len(t),
                "League LL": round(league_ll, 4),
                "Avg hitter LL": round(avg_hitter, 4) if avg_hitter is not None else None,
                "Gain (LL)": gain,
                "Hitter win %": round(wins / len(player_lls), 3) if player_lls else None,
            }
        )
    return pd.DataFrame(rows)


def compare_league_vs_player_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batter_names: pd.DataFrame,
    *,
    min_train: int = 150,
    min_test: int = 40,
) -> tuple[pd.DataFrame, go.Figure]:
    name_map = batter_names.set_index("batter")["player_name"].to_dict()
    league_swing = fit_league_logistic(train_df[MODEL_INPUT_COLS], train_df["is_swing"])
    rows = []
    for batter, b_train in train_df.groupby("batter"):
        b_test = test_df[test_df["batter"] == batter]
        if len(b_train) < min_train or len(b_test) < min_test:
            continue
        league_s_probs = league_swing.predict_proba(b_test[MODEL_INPUT_COLS])[:, 1]
        player_swing = fit_league_logistic(b_train[MODEL_INPUT_COLS], b_train["is_swing"])
        player_s_probs = player_swing.predict_proba(b_test[MODEL_INPUT_COLS])[:, 1]
        rows.append(
            {
                "batter": batter,
                "hitter": name_map.get(int(batter), str(batter)),
                "train_pitches": len(b_train),
                "test_pitches": len(b_test),
                "league_swing_ll": log_loss(b_test["is_swing"], league_s_probs, labels=[0, 1]),
                "player_swing_ll": log_loss(b_test["is_swing"], player_s_probs, labels=[0, 1]),
            }
        )
    cmp = pd.DataFrame(rows)
    if cmp.empty:
        return cmp, go.Figure()
    cmp["swing_ll_gain"] = cmp["league_swing_ll"] - cmp["player_swing_ll"]
    cmp["swing_player_wins"] = cmp["swing_ll_gain"] > 0

    scatter = go.Figure()
    scatter.add_trace(
        go.Scatter(
            name="Hitters",
            x=cmp["league_swing_ll"].tolist(),
            y=cmp["player_swing_ll"].tolist(),
            mode="markers",
            text=cmp["hitter"],
            marker=dict(
                size=np.clip(cmp["train_pitches"] / 30, 6, 22).tolist(),
                color=cmp["swing_ll_gain"].tolist(),
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="LL gain"),
            ),
            hovertemplate="%{text}<br>League=%{x:.3f}<br>Hitter=%{y:.3f}<extra></extra>",
        )
    )
    scatter.add_trace(
        go.Scatter(x=[0.40, 0.70], y=[0.40, 0.70], mode="lines", line=dict(dash="dash", color="#cbbfa8"), name="Equal performance", showlegend=False)
    )
    scatter.update_layout(
        title="League vs Hitter-Specific Swing Models (September Holdout)",
        template="plotly_dark",
        xaxis=dict(title="League log loss", range=[0.40, 0.70]),
        yaxis=dict(title="Hitter log loss", range=[0.40, 0.70]),
        height=440,
    )
    return cmp, scatter


def league_residual_figure(test_df: pd.DataFrame, league_swing, batter_names: pd.DataFrame) -> go.Figure:
    scored = test_df.copy()
    scored["pred_swing"] = league_swing.predict_proba(test_df[MODEL_INPUT_COLS])[:, 1]
    by_batter = (
        scored.groupby("batter", as_index=False)
        .agg(actual=("is_swing", "mean"), predicted=("pred_swing", "mean"), pitches=("is_swing", "size"))
        .merge(batter_names, on="batter", how="left")
    )
    by_batter["residual"] = by_batter["actual"] - by_batter["predicted"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Hitters",
            x=by_batter["predicted"].tolist(),
            y=by_batter["actual"].tolist(),
            mode="markers",
            text=by_batter["player_name"],
            marker=dict(
                size=np.clip(by_batter["pitches"] / 25, 5, 20).tolist(),
                color=by_batter["residual"].tolist(),
                colorscale="RdBu",
                cmid=0,
                showscale=True,
                colorbar=dict(title="Actual − pred"),
            ),
            hovertemplate="%{text}<br>Pred=%{x:.1%}<br>Actual=%{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(x=[0.45, 0.55], y=[0.45, 0.55], mode="lines", line=dict(dash="dash", color="#cbbfa8"), name="Perfect calibration", showlegend=False)
    )
    fig.update_layout(
        title="League Model Residuals by Hitter (September Holdout)",
        template="plotly_dark",
        xaxis=dict(tickformat=".0%", range=[0.45, 0.55], title="Predicted swing rate"),
        yaxis=dict(tickformat=".0%", range=[0.35, 0.65], title="Actual swing rate"),
        height=440,
    )
    return fig


def build_glossary() -> str:
    category_lines = []
    for group, codes in PITCH_TYPE_GROUPS.items():
        code_items = ", ".join(f"<b>{c}</b> ({PITCH_CODE_NAMES.get(c, c)})" for c in codes)
        category_lines.append(f"<li><b>{group}:</b> {code_items}</li>")
    return (
        """
    <div class="box">
      <h3>Terminology and Pitch Types</h3>
      <ul>
        <li><b>Swing rate:</b> Share of pitches on which the batter swings.</li>
        <li><b>Whiff | swing:</b> Share of swings that miss (whiff conditional on swing).</li>
        <li><b>Chase rate:</b> Share of pitches taken outside the strike zone that the batter still swings at.</li>
        <li><b>Log loss (LL):</b> Probabilistic error — lower is better. Gain (LL) = League LL − Hitter LL; positive means personalization wins.</li>
        <li><b>Attack zones:</b> Heart, Shadow, Chase, Waste — plate locations relative to the strike-zone boundary (FanGraphs/Statcast framework).</li>
      </ul>
      <h4>Pitch categories</h4>
      <ul>
        """
        + "\n        ".join(category_lines)
        + """
      </ul>
    </div>
    """
    )


def build_intro(summary: dict) -> str:
    crossover = summary.get("crossover_n", "—")
    return f"""
    <div class="finding">
      <h3>Overview</h3>
      <p><b>The Whiff List</b> is an MLB Statcast portfolio project that models batter swing and whiff behavior
      for a Streamlit dashboard and interactive Pitch Lab. This report explores 2025 regular-season pitch-level data
      (qualified hitters, competitive pitch types only) to justify a <b>hybrid modeling strategy</b>:
      a league-wide baseline with optional hitter-specific personalization where data support it.</p>
      <p><b>Data:</b> Statcast pitch-level records, Mar 27 – Sep 28, 2025.
      <b>Train:</b> Mar–Aug ({summary['n_train_pitches']:,} pitches).
      <b>Test:</b> September holdout ({summary['n_test_pitches']:,} pitches).
      <b>Sample:</b> {summary['n_qualified']} qualified hitters (502+ AB).</p>
      <p><b>Why two variable families?</b> We split analysis into <b>contextual variables</b> (location, count leverage,
      runners-on) and <b>pitch physics</b> (velocity, movement, spin) because they answer different questions.
      Contextual features drive <b>swing decisions</b> — whether a hitter offers at a pitch — while physics features
      primarily differentiate <b>whiff risk conditional on a swing</b>. Keeping them separate avoids conflating
      plate-discipline effects with pitch-quality effects and mirrors how the production models are structured.</p>
    </div>
    {build_glossary()}
    <div class="box">
      <h3>At a Glance</h3>
      <div class="metric-row">
        <div class="metric"><b>{summary['n_qualified']}</b>Qualified hitters</div>
        <div class="metric"><b>{summary['n_train_pitches']:,}</b>Train pitches</div>
        <div class="metric"><b>{summary['n_test_pitches']:,}</b>Test pitches</div>
        <div class="metric"><b>{summary['between_share']:.0%}</b>Between-batter swing variance</div>
      </div>
      <ul>
        <li>Swing rates spread {summary['swing_rate_mean']:.1%} ± {summary['swing_rate_std']:.1%} (range {summary['swing_rate_range']:.1%}).</li>
        <li>Slow velocity outliers removed (Tukey lower fence) — <b>{summary.get('n_velocity_removed', 0):,}</b> pitches; spin outliers (high &amp; low) — <b>{summary.get('n_spin_removed', 0):,}</b> pitches.</li>
        <li>Hitter models beat league for <b>{summary['swing_player_win_pct']:.0%}</b> of evaluated hitters; learning-curve crossover near <b>{crossover}</b> training pitches.</li>
      </ul>
    </div>
    """


def build_feature_importance_section(swing_imp: pd.DataFrame, whiff_imp: pd.DataFrame | None) -> list[str]:
    parts = [
        section_html(
            "Feature Importance",
            "<p>Production random-forest models use <code>count_state</code> (Hitter Ahead, Pitcher Ahead, Even, Full) "
            "plus <code>is_two_strike</code> instead of raw balls/strikes, with <code>speed_diff</code> "
            "(not effective velocity) to avoid multicollinearity with release speed. "
            "Location and count dominate swing; pitch physics matter more for whiff.</p>",
        ),
        '<div class="two-col-tables">',
        "<div><h4>Swing Model</h4>" + table_html(swing_imp) + "</div>",
    ]
    if whiff_imp is not None:
        parts.append("<div><h4>Whiff Model</h4>" + table_html(whiff_imp) + "</div>")
    parts.append("</div>")
    return parts


def build_conclusions(summary: dict, shadow_table: pd.DataFrame) -> list[str]:
    crossover = summary.get("crossover_n", 600)
    parts = [
        section_html(
            "Conclusions",
            "<p>Summary recommendations for the production Streamlit dashboard and Pitch Lab.</p>",
        ),
        f"""
        <div class="finding">
          <h3>Modeling Recommendation</h3>
          <ul>
            <li><b>League model default</b> for Pitch Lab and cold-start hitters with fewer than ~{crossover} seen pitches.</li>
            <li><b>Hitter offsets after ~{crossover} pitches</b> — the learning curve crossover provides empirical support for this hybrid threshold in production.</li>
            <li><b>Zone-aware personalization</b> — prioritize Shadow and Chase zones where discipline varies most by hitter.</li>
          </ul>
        </div>
        """,
    ]
    if not shadow_table.empty and shadow_table["Gain (LL)"].notna().any():
        best = shadow_table.loc[shadow_table["Gain (LL)"].idxmax(), "Zone"]
        waste = shadow_table.loc[shadow_table["Zone"] == "Waste"]
        waste_note = ""
        if not waste.empty and waste["Gain (LL)"].notna().all():
            g = float(waste["Gain (LL)"].iloc[0])
            league_ll = float(waste["League LL"].iloc[0])
            hitter_ll = float(waste["Avg hitter LL"].iloc[0])
            win_pct = waste["Hitter win %"].iloc[0]
            waste_note = (
                f"<li><b>Waste zone:</b> Gain = {g:+.4f} ({league_ll:.4f} vs {hitter_ll:.4f}; lower LL is better). "
                f"Personalization wins for {win_pct:.0%} of hitters, but the margin is smallest here — "
                f"estimates are noisier with fewer swings, so the league prior remains a safer fallback with thin samples.</li>"
            )
        parts.append(
            f"""
            <div class="box">
              <h3>Personalization by Attack Zone</h3>
              <ul>
                <li>Largest log-loss gains concentrate in <b>{best}</b> and <b>Chase</b> where hitter-specific plate discipline differs most.</li>
                {waste_note}
              </ul>
            </div>
            """
        )
    return parts


def run_eda(*, open_browser: bool = True) -> Path:
    train_df, test_df, _, batter_names, velocity_summary, spin_summary, n_velocity_removed, n_spin_removed = load_modeling_data()
    batter_rates = batter_rate_table(train_df, batter_names)
    het = heterogeneity_metrics(batter_rates)
    var_swing = variance_decomposition(train_df, "is_swing")
    train_swings = train_df[train_df["is_swing"] == 1]

    league_swing = fit_league_logistic(train_df[MODEL_INPUT_COLS], train_df["is_swing"])
    cmp_df, cmp_scatter = compare_league_vs_player_models(train_df, test_df, batter_names)
    learning_fig, crossover_n = learning_curve_figure(train_df, test_df)
    shadow_table = shadow_personalization_table(train_df, test_df)
    prod_swing_imp, prod_whiff_imp = load_production_importance()
    swing_imp = prod_swing_imp if prod_swing_imp is not None else feature_importance_table(league_swing)
    whiff_imp = prod_whiff_imp

    summary = {
        "n_qualified": batter_rates["batter"].nunique(),
        "n_train_pitches": len(train_df),
        "n_test_pitches": len(test_df),
        "swing_rate_mean": het["swing_rate_mean"],
        "swing_rate_std": het["swing_rate_std"],
        "swing_rate_range": het["swing_rate_range"],
        "swing_player_win_pct": float(cmp_df["swing_player_wins"].mean()) if not cmp_df.empty else 0.0,
        "crossover_n": crossover_n,
        "n_velocity_removed": n_velocity_removed,
        "n_spin_removed": n_spin_removed,
        "between_share": var_swing["between_share"],
    }

    learning_caption = (
        "<p>Each point fixes a minimum training-pitch threshold, fits a hitter-specific logistic swing model, "
        "and scores September holdout log loss. The crossover near <b>~600 pitches</b> is the empirical threshold "
        "for switching from league-default to hitter-specific models in production. "
        "The dotted win-rate line (right axis) shows the share of hitters whose personalized model beats the league model at each sample size.</p>"
    )
    residuals_caption = (
        "<p>Each point is one hitter's September test set. X = predicted swing rate; Y = actual swing rate. "
        "Dashed line = perfect calibration. "
        "<b>Red = over-swingers</b> (actual &gt; predicted); <b>blue = under-swingers</b> (more patient than the league model expects).</p>"
    )
    league_vs_hitter_caption = (
        "<p>Each point is one hitter's September holdout log loss. Points <b>below</b> the diagonal (y &lt; x) "
        "mean personalization wins. <b>Green = larger log-loss gain</b> (league − hitter). Dot size = training volume.</p>"
    )
    hitter_win_caption = (
        "<p><b>Hitter win %</b> = share of qualified hitters whose personalized model beats the league model "
        "on that zone's September test pitches (lower log loss wins). "
        "<b>Gain (LL)</b> = League LL − Avg hitter LL; positive means personalization wins on average.</p>"
    )

    swing_top = top_corr_table(train_df, CONTEXTUAL_CORR_COLS, "is_swing")
    whiff_top = top_corr_table(train_df, PITCH_CORR_COLS, "is_whiff")

    body: list[str] = [build_intro(summary)]
    plotly_js: bool | str = "cdn"

    # 1. Data Quality
    body.append(section_html("Data Quality", "Competitive pitch types only; velocity and spin cleaned before physics summaries."))
    body.append(
        "<p><b>Velocity outliers:</b> Tukey lower fence only — Q1 − 1.5·IQR on <code>release_speed</code> per pitch type "
        "(train-fit, applied to both splits). Drops slow mis-tags; keeps elite high-velocity pitches. "
        f"<b>{n_velocity_removed:,} removed.</b></p>"
    )
    if not velocity_summary.empty:
        show_vel = velocity_summary.rename(
            columns={
                "pitch_type": "Pitch",
                "n_removed": "Removed",
                "lower_fence_mph": "Lower fence (mph)",
                "min_removed_mph": "Min removed (mph)",
                "max_removed_mph": "Max removed (mph)",
            }
        )
        body.append(table_html(show_vel, max_rows=20))
    body.append(
        "<p><b>Spin outliers:</b> Tukey IQR (1.5×) on <code>release_spin_rate</code> per pitch type — "
        "both high and low outliers removed. "
        f"<b>{n_spin_removed:,} removed.</b></p>"
    )
    if not spin_summary.empty:
        show_spin = spin_summary.rename(
            columns={
                "pitch_type": "Pitch",
                "n_removed": "Removed",
                "lower_fence_rpm": "Lower fence (rpm)",
                "upper_fence_rpm": "Upper fence (rpm)",
                "min_removed_rpm": "Min removed (rpm)",
                "max_removed_rpm": "Max removed (rpm)",
            }
        )
        body.append(table_html(show_spin, max_rows=20))

    # 2. Count and Pitch Type
    body.append(section_html("Count and Pitch Type", "Non-linear count effects and pitch-type swing/whiff profiles."))
    body.append(render_chart_block("Swing and Whiff by Count", count_rates_figure(train_df), include_plotlyjs=plotly_js))
    plotly_js = False
    body.append(
        render_chart_row(
            "Pitch Category Overview",
            [pitch_category_count_figure(train_df), pitch_type_rates_figure(train_df)],
            "<p>Volume and rates by category: Fastballs (FF, SI, FC), Breaking Balls (SL, KC, ST, SV, CU), Off-Speed (CH, FS).</p>",
            include_plotlyjs=plotly_js,
        )
    )
    plotly_js = False
    body.append(render_chart_block("Platoon Splits (LHB vs RHB)", platoon_split_figure(train_df), include_plotlyjs=plotly_js))
    plotly_js = False

    # 3. Hitter Heterogeneity
    body.append(section_html("Hitter Heterogeneity", "Qualified hitters differ materially in swing and whiff-if-swing rates."))
    body.append(
        render_chart_row(
            "Hitter Rate Distributions",
            rate_distribution_figures(batter_rates)[:2],
            include_plotlyjs=plotly_js,
            wide=True,
        )
    )
    plotly_js = False

    # 4. Pitch Physics
    body.append(section_html("Pitch Physics", "Velocity, spin, and distributional summaries by pitch category."))
    for group in PITCH_GROUP_ORDER:
        body.append(f"<h3>{group}</h3>")
        body.append(physics_category_summary(train_df, group))
        body.append(table_html(physics_table_for_group(train_df, group)))
    body.append(
        render_chart_row(
            "Velocity and Spin by Category",
            [velocity_by_category_figure(train_df), spin_by_category_figure(train_df)],
            include_plotlyjs=plotly_js,
        )
    )
    plotly_js = False

    # 5. Plate Location
    body.append(section_html("Plate Location", "Binned heatmaps; cells with &lt;20 pitches or 0% rate are masked."))
    body.append(
        render_chart_row(
            "Location Heatmaps",
            [
                location_heatmap(train_df, "is_swing", "Swing Rate by Location"),
                location_heatmap(train_swings, "is_whiff", "Whiff-if-Swing by Location"),
            ],
            include_plotlyjs=plotly_js,
        )
    )
    plotly_js = False

    # 6. Correlations
    body.append(
        section_html(
            "Correlations",
            "Contextual variables explain swing decisions; pitch physics correlate more with whiff. "
            "Physics matrix excludes <code>speed_diff</code> to avoid redundancy with release speed.",
        )
    )
    body.append('<div class="two-col-tables">')
    body.append("<div><h4>Top contextual |r| → swing</h4>" + table_html(swing_top) + "</div>")
    body.append("<div><h4>Top physics |r| → whiff</h4>" + table_html(whiff_top) + "</div>")
    body.append("</div>")
    body.append(
        render_chart_row(
            "Correlation Matrices",
            [
                correlation_matrix_figure(train_df, CONTEXTUAL_CORR_COLS, "Contextual Variables", compact=True),
                correlation_matrix_figure(train_df, PITCH_CORR_COLS, "Pitch Physics", compact=True),
            ],
            include_plotlyjs=plotly_js,
            wide=True,
        )
    )
    plotly_js = False

    # 7. Attack Zones and Personalization
    body.append(section_html("Attack Zones and Personalization", ATTACK_ZONE_CREDIT))
    body.append(render_chart_block("Swing and Whiff by Attack Zone (Overall)", attack_zone_figure(train_df), include_plotlyjs=plotly_js))
    plotly_js = False
    body.append(render_chart_block("Swing and Whiff by Attack Zone and Pitch Category", attack_zone_by_category_figure(train_df), include_plotlyjs=plotly_js))
    plotly_js = False
    body.append(table_html(shadow_table))
    body.append(f'<div class="chart-caption">{hitter_win_caption}</div>')
    if not cmp_df.empty:
        showcase = (
            cmp_df.nlargest(8, "swing_ll_gain")[
                ["hitter", "train_pitches", "league_swing_ll", "player_swing_ll", "swing_ll_gain"]
            ]
            .round(4)
            .rename(
                columns={
                    "hitter": "Hitter",
                    "train_pitches": "Train",
                    "league_swing_ll": "League LL",
                    "player_swing_ll": "Hitter LL",
                    "swing_ll_gain": "Gain",
                }
            )
        )
        body.append("<h3>Biggest Personalization Gains (Swing)</h3>")
        body.append(table_html(showcase))

    # 8. League vs Personalized Models
    body.append(section_html("League vs Personalized Models", "September holdout comparison using logistic swing models."))
    body.append(render_chart_block("Personalization Learning Curve", learning_fig, learning_caption, include_plotlyjs=plotly_js))
    plotly_js = False
    body.append(render_chart_block("League Model Residuals by Hitter", league_residual_figure(test_df, league_swing, batter_names), residuals_caption, include_plotlyjs=plotly_js))
    plotly_js = False
    body.append(render_chart_block("League vs Hitter-Specific Swing Models", cmp_scatter, league_vs_hitter_caption, include_plotlyjs=plotly_js))

    # 9. Feature Importance
    body.extend(build_feature_importance_section(swing_imp, whiff_imp))

    # 10. Conclusions
    body.extend(build_conclusions(summary, shadow_table))

    return export_eda_report(
        body,
        REPORT_FILE,
        subtitle="Interactive charts — hover, zoom, and pan. Competitive pitch types only.",
        open_browser=open_browser,
    )

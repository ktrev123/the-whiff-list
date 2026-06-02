"""Plotly figures and HTML report export for swing / whiff models."""

from __future__ import annotations

import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

PLOTLY_JS = "cdn"
REPORT_STYLE = """
body { background:#0f172a; color:#f5efe3; font-family:Segoe UI, sans-serif; padding:2rem; max-width:1200px; margin:0 auto; }
h1 { color:#d4a937; }
h2 { color:#d4a937; margin-top:2.5rem; border-bottom:1px solid rgba(245,239,227,0.15); padding-bottom:0.4rem; }
h3 { margin-top:1.5rem; }
.box { background:rgba(255,255,255,0.03); border:1px solid rgba(245,239,227,0.12); border-radius:12px; padding:1rem 1.25rem; margin:1rem 0; }
.metric-row { display:flex; flex-wrap:wrap; gap:1rem; margin:1rem 0; }
.metric { background:#162033; border:1px solid rgba(245,239,227,0.1); border-radius:10px; padding:0.75rem 1rem; min-width:140px; }
.metric b { display:block; font-size:1.4rem; color:#fde725; }
.chart { margin:1rem 0 2rem 0; }
table { width:100%; border-collapse:collapse; margin-top:1rem; font-size:0.9rem; }
th, td { border:1px solid rgba(245,239,227,0.15); padding:0.5rem 0.75rem; text-align:left; }
th { background:#162033; }
"""


def build_roc_figure(block: dict) -> go.Figure:
    roc = block["roc_curve"]
    label = block.get("outcome_label", "outcome")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=roc["fpr"],
            y=roc["tpr"],
            mode="lines",
            name="Model",
            line=dict(color="#fde725", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random guess",
            line=dict(color="#cbbfa8", dash="dash"),
        )
    )
    fig.update_layout(
        title=f"ROC curve — {label}",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="False positive rate",
        yaxis_title=f"{label.title()}s correctly ranked",
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def build_calibration_figure(block: dict) -> go.Figure:
    bins = block["calibration_bins"]
    pred = [b["pred_mean"] for b in bins]
    actual = [b["actual_rate"] for b in bins]
    sizes = [max(8, min(28, b["count"] ** 0.5)) for b in bins]
    label = block.get("outcome_label", "outcome")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pred,
            y=actual,
            mode="markers+lines",
            name="Model bins",
            marker=dict(size=sizes, color="#20908d"),
            line=dict(color="#20908d"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line=dict(color="#cbbfa8", dash="dash"),
        )
    )
    fig.update_layout(
        title=f"Calibration — {label}",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Predicted probability (bin average)",
        yaxis_title=f"Actual {label} rate",
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def build_importance_figure(block: dict) -> go.Figure:
    imp = block["feature_importance"]
    labels = [row["label"] for row in imp][::-1]
    values = [row["importance"] for row in imp][::-1]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color="#d4a937"))
    fig.update_layout(
        title="Feature importance",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Relative importance",
        height=340,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def build_pred_grid_figure(grid_df: pd.DataFrame, prob_col: str, title: str, colorbar_title: str) -> go.Figure:
    pivot = grid_df.pivot_table(index="plate_z", columns="plate_x", values=prob_col, aggfunc="mean")
    fig = go.Figure(
        data=go.Contour(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=[
                [0.0, "rgba(15, 23, 42, 0)"],
                [0.2, "rgba(60, 80, 140, 0.5)"],
                [0.5, "#20908d"],
                [0.8, "#fde725"],
                [1.0, "#e63946"],
            ],
            contours=dict(coloring="fill"),
            colorbar=dict(title=colorbar_title, tickformat=".0%"),
            hovertemplate="x=%{x:.2f}, z=%{y:.2f}<br>prob=%{z:.0%}<extra></extra>",
        )
    )
    avg_bot = float(grid_df["sz_bot"].iloc[0])
    avg_top = float(grid_df["sz_top"].iloc[0])
    fig.add_shape(
        type="rect",
        x0=-0.708,
        x1=0.708,
        y0=avg_bot,
        y1=avg_top,
        line=dict(color="#f5efe3", width=3),
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-2.5, 2.5], title="Horizontal (ft)", scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0.5, 4.5], title="Vertical (ft)"),
        height=440,
    )
    return fig


def build_batter_pred_figure(
    batter_preds: pd.DataFrame,
    name_lookup: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
) -> go.Figure:
    merged = batter_preds.merge(name_lookup, on="batter", how="left")
    merged["player_name"] = merged["player_name"].fillna("Unknown")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged[x_col] * 100,
            y=merged[y_col] * 100,
            mode="markers",
            marker=dict(size=9, color="#20908d", opacity=0.75),
            text=merged["player_name"],
            hovertemplate="<b>%{text}</b><br>Actual: %{x:.1f}%<br>Predicted: %{y:.1f}%<extra></extra>",
        )
    )
    lim = max(float(merged[x_col].max() * 100), float(merged[y_col].max() * 100), 10)
    fig.add_trace(
        go.Scatter(
            x=[0, lim * 1.05],
            y=[0, lim * 1.05],
            mode="lines",
            line=dict(color="#cbbfa8", dash="dash"),
            name="Perfect match",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def _fig_html(fig: go.Figure) -> str:
    return to_html(fig, include_plotlyjs=False, full_html=False)


def _metrics_html(block: dict, rate_label: str) -> str:
    note = f"<p><em>{block['training_note']}</em></p>" if block.get("training_note") else ""
    return f"""
    <div class="box">
      <p>{block['layman']['what_it_outputs']}</p>
      {note}
      <div class="metric-row">
        <div class="metric"><span>ROC-AUC</span><b>{block['roc_auc']:.2f}</b></div>
        <div class="metric"><span>Log Loss</span><b>{block['log_loss']:.3f}</b></div>
        <div class="metric"><span>September {rate_label}</span><b>{block['test_positive_rate'] * 100:.1f}%</b></div>
        <div class="metric"><span>Algorithm</span><b>{block['selected_model'].replace('_', ' ').title()}</b></div>
      </div>
      <p>{block['layman']['roc_auc'].replace('**', '')}</p>
      <p>{block['layman']['log_loss'].replace('**', '')}</p>
    </div>
    """


def _examples_table_html(examples: list[dict]) -> str:
    rows = ""
    for ex in examples:
        rows += f"""
        <tr>
          <td>{ex['label']}</td>
          <td>{ex['count']}</td>
          <td>{ex['runners_on']}</td>
          <td>{ex['swing_prob_pct']}%</td>
          <td>{ex['whiff_if_swing_pct']}%</td>
          <td>{ex['swing_whiff_pct']}%</td>
          <td>{ex['swing_takeaway']}</td>
          <td>{ex['whiff_takeaway']}</td>
        </tr>
        """
    return f"""
    <table>
      <thead>
        <tr>
          <th>Scenario</th><th>Count</th><th>Runners</th>
          <th>P(swing)</th><th>P(whiff|swing)</th><th>P(swing &amp; whiff)</th>
          <th>Swing takeaway</th><th>Whiff takeaway</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    """


def export_training_report(
    insights: dict,
    batter_preds: pd.DataFrame,
    swing_grid: pd.DataFrame,
    whiff_grid: pd.DataFrame,
    name_lookup: pd.DataFrame,
    out_dir: Path,
    open_browser: bool = True,
) -> Path:
    """Write interactive HTML report and individual chart files; optionally open in browser."""
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    swing = insights["swing"]
    whiff = insights["whiff"]
    sections: list[str] = []

    intro = f"""
    <div class="box">
      <p>{insights['layman']['validation'].replace('**', '')}</p>
      <p>{insights['layman']['combined'].replace('**', '')}</p>
    </div>
    """
    sections.append(intro)

    for target_key, block, rate_label in [
        ("swing", swing, "swing rate"),
        ("whiff", whiff, "whiff rate (swings only)"),
    ]:
        sections.append(f"<h2>Model {'A' if target_key == 'swing' else 'B'} — {rate_label.title()}</h2>")
        sections.append(_metrics_html(block, rate_label))

        figs = [
            build_roc_figure(block),
            build_calibration_figure(block),
            build_importance_figure(block),
        ]
        if target_key == "swing":
            figs.append(
                build_pred_grid_figure(
                    swing_grid,
                    "pred_swing_prob",
                    "Swing map (2-2, league-average zone)",
                    "P(swing)",
                )
            )
            figs.append(
                build_batter_pred_figure(
                    batter_preds,
                    name_lookup,
                    "actual_swing_rate",
                    "mean_pred_swing",
                    "Actual swing %",
                    "Predicted swing %",
                    "September: swing predicted vs. actual",
                )
            )
        else:
            figs.append(
                build_pred_grid_figure(
                    whiff_grid,
                    "pred_whiff_prob",
                    "Whiff-if-swing map (2-2, league-average zone)",
                    "P(whiff|swing)",
                )
            )
            figs.append(
                build_batter_pred_figure(
                    batter_preds,
                    name_lookup,
                    "actual_whiff_rate",
                    "mean_pred_whiff",
                    "Actual whiff %",
                    "Predicted whiff-if-swing %",
                    "September: whiff predicted vs. actual",
                )
            )

        for i, fig in enumerate(figs):
            chart_path = charts_dir / f"{target_key}_{i}.html"
            fig.write_html(chart_path, include_plotlyjs=PLOTLY_JS)
            sections.append(f'<div class="chart">{_fig_html(fig)}</div>')

    sections.append("<h2>Combined example pitches</h2>")
    sections.append(
        '<p>P(swing &amp; whiff) = P(swing) × P(whiff | swing) — estimated swinging-strike chance.</p>'
    )
    sections.append(_examples_table_html(insights["example_pitches"]))

    body = "\n".join(sections)
    report_path = out_dir / "model_report.html"
    report_path.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>The Whiff List — Model Report</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>{REPORT_STYLE}</style>
</head>
<body>
  <h1>The Whiff List — Swing &amp; Whiff Model Report</h1>
  <p>Generated after training. Charts are interactive (hover, zoom).</p>
  {body}
</body>
</html>
""",
        encoding="utf-8",
    )

    if open_browser:
        webbrowser.open(report_path.resolve().as_uri())

    return report_path

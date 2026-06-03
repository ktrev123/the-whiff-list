"""HTML report export for exploratory data analysis."""

from __future__ import annotations

import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

from src.model_viz import REPORT_STYLE

PLOTLY_JS_CDN = "https://cdn.plot.ly/plotly-3.0.1.min.js"

REPORT_EXTRA_CSS = """
.section-lead { max-width: 820px; margin: 0 auto 1rem auto; line-height: 1.6; color: #cbbfa8; }
.center-block { display: flex; flex-direction: column; align-items: center; margin: 1.5rem 0 2rem 0; }
.chart-caption {
  max-width: 820px; margin: 0 auto 0.75rem auto;
  font-size: 0.95rem; line-height: 1.55; color: #cbbfa8;
}
.chart-caption a { color: #d4a937; }
.table-wrap { display: flex; justify-content: center; margin: 0.75rem 0 1.5rem 0; }
.compact-table-wrap { max-width: 820px; margin: 0 auto; overflow-x: auto; }
.compact-table-wrap.wide { max-width: 980px; }
table.compact-table {
  width: auto; min-width: 320px; max-width: 100%;
  font-size: 0.88rem; border-collapse: collapse;
}
table.compact-table th, table.compact-table td {
  padding: 0.4rem 0.65rem; border: 1px solid rgba(245,239,227,0.12);
  text-align: right; white-space: nowrap;
}
table.compact-table th:first-child, table.compact-table td:first-child { text-align: left; }
table.compact-table th { background: #162033; }
.two-col-tables {
  display: flex; flex-wrap: wrap; gap: 2rem;
  justify-content: center; align-items: flex-start;
}
.two-col-tables > div { flex: 0 1 auto; }
.chart { margin: 0 auto; max-width: 100%; display: flex; justify-content: center; }
.chart-row {
  display: flex; flex-wrap: wrap; gap: 1.25rem;
  justify-content: center; align-items: flex-start;
  margin: 1rem auto 2rem auto; max-width: 1100px;
}
.chart-row.wide { max-width: 1200px; }
.chart-row .chart-half { flex: 1 1 420px; max-width: 520px; }
.chart-row .chart-half.wide-half { flex: 1 1 480px; max-width: 560px; }
.chart-full { max-width: 860px; margin: 0 auto; overflow-x: auto; }
.finding { border-left: 3px solid #d4a937; padding-left: 1rem; margin: 1rem 0; }
ul { line-height: 1.6; }
"""


def fig_html(fig: go.Figure, *, include_plotlyjs: bool | str = False) -> str:
    return to_html(fig, include_plotlyjs=include_plotlyjs, full_html=False)


def table_html(df, max_rows: int = 25, *, wide: bool = False) -> str:
    if df is None or len(df) == 0:
        return "<p><em>No rows.</em></p>"
    view = df.head(max_rows).copy()
    view = view.fillna("—")
    cls = "compact-table-wrap wide" if wide else "compact-table-wrap"
    return (
        f'<div class="table-wrap"><div class="{cls}">'
        f'{view.to_html(index=False, classes="compact-table", border=0, escape=False, na_rep="—")}'
        f"</div></div>"
    )


def section_html(heading: str, body: str = "") -> str:
    parts = [f"<h2>{heading}</h2>"]
    if body:
        parts.append(f'<div class="section-lead">{body}</div>')
    return "\n".join(parts)


def render_chart_block(
    title: str,
    fig: go.Figure,
    caption: str = "",
    *,
    include_plotlyjs: bool | str = False,
    wrapper_class: str = "center-block",
) -> str:
    parts = [f'<div class="{wrapper_class}">']
    if title:
        parts.append(f"<h3>{title}</h3>")
    if caption:
        parts.append(f'<div class="chart-caption">{caption}</div>')
    parts.append(f'<div class="chart chart-full">{fig_html(fig, include_plotlyjs=include_plotlyjs)}</div>')
    parts.append("</div>")
    return "\n".join(parts)


def render_chart_row(
    title: str,
    figs: list[go.Figure],
    caption: str = "",
    *,
    include_plotlyjs: bool | str = False,
    wide: bool = False,
) -> str:
    row_cls = "chart-row wide" if wide else "chart-row"
    half_cls = "chart-half wide-half" if wide else "chart-half"
    parts = ['<div class="center-block">']
    if title:
        parts.append(f"<h3>{title}</h3>")
    if caption:
        parts.append(f'<div class="chart-caption">{caption}</div>')
    parts.append(f'<div class="{row_cls}">')
    for i, fig in enumerate(figs):
        js = include_plotlyjs if i == 0 else False
        parts.append(f'<div class="{half_cls}">{fig_html(fig, include_plotlyjs=js)}</div>')
    parts.append("</div></div>")
    return "\n".join(parts)


def export_eda_report(
    body_parts: list[str],
    out_path: Path,
    *,
    title: str = "The Whiff List — Exploratory Analysis",
    subtitle: str = "",
    open_browser: bool = True,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    intro = f"<p>{subtitle}</p>" if subtitle else ""
    body = "\n".join(body_parts)
    out_path.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <script src="{PLOTLY_JS_CDN}"></script>
  <style>{REPORT_STYLE}{REPORT_EXTRA_CSS}</style>
</head>
<body>
  <h1>{title}</h1>
  {intro}
  {body}
</body>
</html>
""",
        encoding="utf-8",
    )
    if open_browser:
        webbrowser.open(out_path.resolve().as_uri())
    return out_path

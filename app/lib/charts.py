"""Plotly chart builders for the Gold Lens app.

Every figure is built here so the pages stay declarative. Styling is dark with
a single gold accent, and all figures render on a transparent background so
they sit flush inside Streamlit's containers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lib.data import MONTH_LABELS, NAIVE

GOLD = "#D4AF37"
GOLD_SOFT = "rgba(212, 175, 55, 0.18)"
BLUE = "#4C9BE8"
RED = "#E5534B"
GREEN = "#3FB950"
GREY = "#8B93A7"
GRID = "#232A38"

_BASE_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Inter, system-ui, sans-serif", "size": 13},
    "hovermode": "x unified",
    "margin": {"l": 8, "r": 8, "t": 44, "b": 8},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
}


def _style(fig: go.Figure, *, height: int = 420, title: str | None = None) -> go.Figure:
    """Apply the shared dark layout to ``fig``."""
    fig.update_layout(**_BASE_LAYOUT, height=height, title=title)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False)
    return fig


def price_trend(frame: pd.DataFrame) -> go.Figure:
    """Closing price over the selected range.

    The fill is anchored to the bottom of the visible range rather than to
    ``y=0``: gold has never been anywhere near zero, so a zero baseline throws
    away most of the plot area and flattens the real moves.
    """
    values = frame["close"].astype(float).to_numpy(dtype=float)
    low = float(values.min())
    high = float(values.max())
    pad = (high - low) * 0.08 or max(high * 0.02, 1.0)
    baseline = [low - pad] * len(frame)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=baseline,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["close"],
            mode="lines",
            name="Close",
            line={"color": GOLD, "width": 1.8},
            fill="tonexty",
            fillcolor=GOLD_SOFT,
        )
    )
    fig.update_yaxes(
        title_text="USD / oz",
        tickprefix="$",
        range=[low - pad, high + pad],
    )
    return _style(fig, title="Gold futures close (GC=F)")


def moving_averages(frame: pd.DataFrame) -> go.Figure:
    """Close with the 50- and 200-day averages and their crossover points."""
    close = frame["close"].astype(float)
    ma50 = pd.Series(
        np.asarray(close.rolling(50).mean(), dtype=float), index=frame.index
    )
    ma200 = pd.Series(
        np.asarray(close.rolling(200).mean(), dtype=float), index=frame.index
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["date"], y=close, name="Close", line={"color": GREY, "width": 1.2}
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"], y=ma50, name="MA 50", line={"color": BLUE, "width": 2}
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"], y=ma200, name="MA 200", line={"color": RED, "width": 2}
        )
    )

    spread = (ma50 - ma200).dropna()
    if len(spread) > 1:
        direction = pd.Series(np.sign(spread.to_numpy(dtype=float)), index=spread.index)
        crossed = direction.diff().fillna(0.0) != 0
        crossings = spread.loc[crossed]
        fig.add_trace(
            go.Scatter(
                x=frame.loc[crossings.index, "date"],
                y=close.loc[crossings.index],
                mode="markers",
                name="Cross",
                marker={"color": GOLD, "size": 9, "symbol": "diamond"},
                hovertemplate="%{x|%Y-%m-%d}<extra>MA cross</extra>",
            )
        )

    fig.update_yaxes(title_text="USD / oz", tickprefix="$")
    return _style(fig, title="Moving averages — golden and death crosses")


def monthly_volatility(returns: pd.DataFrame) -> go.Figure:
    """Box plot of daily percentage returns by calendar month."""
    fig = go.Figure()
    for month, label in enumerate(MONTH_LABELS, start=1):
        values = returns.loc[returns["month"] == month, "return_pct"]
        fig.add_trace(
            go.Box(
                y=values,
                name=label,
                marker_color=GOLD,
                line={"width": 1.2},
                boxpoints=False,
                hovertemplate="%{y:.2f}%<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="Daily return (%)", ticksuffix="%")
    return _style(fig, height=420, title="Daily return by month")


def returns_distribution(returns: pd.DataFrame) -> go.Figure:
    """Histogram of daily returns with a fitted normal curve for comparison."""
    values = returns["return_pct"].to_numpy(dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))

    _, edges = np.histogram(values, bins=60)
    bin_width = float(edges[1] - edges[0])
    grid = np.linspace(float(values.min()), float(values.max()), 240)
    pdf = np.exp(-0.5 * ((grid - mean) / std) ** 2) / (std * np.sqrt(2.0 * np.pi))
    scaled = pdf * len(values) * bin_width

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=60,
            name="Observed",
            marker_color=GOLD,
            opacity=0.75,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=scaled,
            mode="lines",
            name=f"Normal fit (σ={std:.2f}%)",
            line={"color": BLUE, "width": 2},
        )
    )
    fig.update_xaxes(title_text="Daily return (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Days")
    return _style(fig, height=420, title="Distribution of daily returns")


def mae_by_model(table: pd.DataFrame) -> go.Figure:
    """Horizontal MAE bars with the naive forecast as the reference line."""
    ordered = table.sort_values("mae", ascending=True)
    colors = [GOLD if name == NAIVE else BLUE for name in ordered["model"]]

    fig = go.Figure(
        go.Bar(
            x=ordered["mae"],
            y=ordered["model"],
            orientation="h",
            marker_color=colors,
            text=[f"${value:,.2f}" for value in ordered["mae"]],
            textposition="outside",
            hovertemplate="%{y}<br>MAE $%{x:.2f}<extra></extra>",
        )
    )
    naive = table.loc[table["model"] == NAIVE, "mae"]
    if not naive.empty:
        fig.add_vline(
            x=float(naive.iloc[0]),
            line={"color": GOLD, "dash": "dot", "width": 1.5},
            annotation_text="naive",
            annotation_position="top",
        )
    fig.update_xaxes(title_text="Mean absolute error (USD)", tickprefix="$")
    fig.update_yaxes(title_text="")
    return _style(fig, height=320, title="MAE by model — lower is better")


def rolling_error(predictions: pd.DataFrame, window: int = 60) -> go.Figure:
    """Rolling mean absolute error over time, per model."""
    fig = go.Figure()
    palette = [GOLD, RED, BLUE]
    for colour, (name, group) in zip(palette, predictions.groupby("model", sort=False)):
        ordered = group.sort_values("date")
        error = (ordered["actual"] - ordered["predicted"]).abs()
        smooth = error.rolling(window, min_periods=window // 3).mean()
        fig.add_trace(
            go.Scatter(
                x=ordered["date"],
                y=smooth,
                mode="lines",
                name=str(name),
                line={"color": colour, "width": 1.8},
            )
        )
    fig.update_yaxes(title_text=f"{window}-bar mean |error| (USD)", tickprefix="$")
    return _style(fig, title=f"Rolling {window}-bar absolute error")


def predicted_vs_actual(
    predictions: pd.DataFrame, model: str, tail: int = 250
) -> go.Figure:
    """Actual price against one model's walk-forward forecast."""
    subset = (
        predictions.loc[predictions["model"] == model]
        .sort_values("date")
        .tail(tail)
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=subset["date"],
            y=subset["actual"],
            mode="lines",
            name="Actual",
            line={"color": GREY, "width": 1.6},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=subset["date"],
            y=subset["predicted"],
            mode="lines",
            name=str(model),
            line={"color": GOLD, "width": 2},
        )
    )
    fig.update_yaxes(title_text="USD / oz", tickprefix="$")
    return _style(fig, title=f"Predicted vs actual — {model}, last {tail} bars")


def forecast_cone(
    history: pd.DataFrame, forecast: pd.DataFrame, tail: int = 180
) -> go.Figure:
    """Recent history with the point forecast and its 80% band."""
    recent = history.sort_values("date").tail(tail)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent["date"],
            y=recent["close"],
            mode="lines",
            name="History",
            line={"color": GREY, "width": 1.6},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["hi80"],
            mode="lines",
            name="80% band",
            line={"color": "rgba(212,175,55,0.35)", "width": 0},
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["lo80"],
            mode="lines",
            name="80% band",
            line={"color": "rgba(212,175,55,0.35)", "width": 0},
            fill="tonexty",
            fillcolor=GOLD_SOFT,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["forecast"],
            mode="lines",
            name="Forecast",
            line={"color": GOLD, "width": 2.4, "dash": "dot"},
        )
    )
    fig.update_yaxes(title_text="USD / oz", tickprefix="$")
    return _style(fig, height=460, title="Forecast with 80% residual band")

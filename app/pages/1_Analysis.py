"""Market analysis: trend, moving averages, seasonality and risk.

This page answers "what does the series actually look like?" before any model
is fitted. Every chart is followed by the one sentence it supports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from lib.charts import (
    monthly_volatility,
    moving_averages,
    price_trend,
    returns_distribution,
)
from lib.data import active_range, filtered_prices, load_monthly_returns


def _annualised_volatility(prices: pd.DataFrame) -> float:
    """Standard deviation of daily log returns, annualised (252 bars)."""
    close = prices["close"].astype(float).to_numpy(dtype=float)
    log_returns = np.diff(np.log(close))
    return float(np.std(log_returns, ddof=1) * np.sqrt(252.0) * 100.0)


def render() -> None:
    """Draw the analysis page."""
    st.title("Market analysis")
    st.caption(
        "**Takeaway:** gold compounded at a healthy rate over this window, but "
        "the daily noise around that trend is roughly twice the size of any "
        "edge a model can find."
    )

    prices = filtered_prices()
    if len(prices) < 30:
        st.warning("Pick a wider date range — fewer than 30 bars is not enough to analyse.")
        return

    close = prices["close"].astype(float).to_numpy()
    first_close = float(close[0])
    last_close = float(close[-1])
    span_years = max(
        (prices["date"].iloc[-1] - prices["date"].iloc[0]).days / 365.25, 1e-9
    )
    total_return = last_close / first_close - 1.0
    cagr = (last_close / first_close) ** (1.0 / span_years) - 1.0
    running_peak = np.maximum.accumulate(close)
    max_drawdown = float(np.min(close / running_peak - 1.0))

    first, second, third, fourth = st.columns(4)
    first.metric("Close", f"${last_close:,.2f}", border=True)
    second.metric("Total return", f"{total_return * 100:+.1f}%", border=True)
    third.metric("CAGR", f"{cagr * 100:+.1f}%", border=True)
    fourth.metric("Max drawdown", f"{max_drawdown * 100:.1f}%", border=True)

    st.plotly_chart(price_trend(prices), width="stretch")
    st.caption(
        "The trend is real but not linear: the 2015–2019 stretch is almost flat, "
        "which is exactly the part a straight-line model cannot represent."
    )

    st.subheader("Momentum")
    st.plotly_chart(moving_averages(prices), width="stretch")

    close_series = prices["close"].astype(float)
    ma50 = pd.Series(
        np.asarray(close_series.rolling(50).mean(), dtype=float), index=prices.index
    )
    ma200 = pd.Series(
        np.asarray(close_series.rolling(200).mean(), dtype=float), index=prices.index
    )
    spread = (ma50 - ma200).dropna()
    direction = pd.Series(np.sign(spread.to_numpy(dtype=float)), index=spread.index)
    cross_count = int((direction.diff().fillna(0.0) != 0).sum())
    st.caption(
        f"{cross_count} MA50/MA200 crossovers in this range. A crossover is a "
        "description of the past 50 and 200 days, not a forecast of the next one."
    )

    st.subheader("Seasonality and risk")
    returns = load_monthly_returns()
    start, end = active_range(prices)
    in_range = returns.loc[
        (returns["date"].dt.date >= start) & (returns["date"].dt.date <= end)
    ]

    left, right = st.columns(2)
    with left:
        st.plotly_chart(monthly_volatility(in_range), width="stretch")
        st.caption(
            "Monthly boxes overlap heavily — the 'best month to buy gold' is a "
            "story the data does not tell."
        )
    with right:
        st.plotly_chart(returns_distribution(in_range), width="stretch")
        st.caption(
            f"Daily volatility is {_annualised_volatility(prices):.1f}% a year, and "
            "the tails are fatter than the fitted bell curve."
        )

    with st.expander("How these numbers are computed"):
        st.markdown(
            "- Returns are log differences of the daily close, in percent.\n"
            "- CAGR uses calendar years between the first and last bar in the range.\n"
            "- Max drawdown is the worst close-to-peak decline inside the range.\n"
            "- The normal curve is fitted with the sample mean and standard "
            "deviation of the same returns, purely as a visual reference."
        )


if __name__ == "__main__":
    render()

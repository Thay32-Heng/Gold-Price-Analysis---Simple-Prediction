"""Forecast page: a point estimate, an honest band, and why it is hard."""

from __future__ import annotations

import streamlit as st

from lib.charts import forecast_cone
from lib.data import (
    DEFAULT_MODELS,
    DRIFT_OLS,
    HORIZON_RANGE,
    NAIVE,
    holdout_gap,
    load_forecast,
    load_prices,
    load_verdict,
)


def render() -> None:
    """Draw the forecast page."""
    st.title("Forecast")
    st.caption(
        "**Takeaway:** for a short horizon the honest forecast is close to "
        "today's price, and the width of the band matters more than the dot."
    )

    prices = load_prices()
    last_close = float(prices["close"].iloc[-1])
    last_date = prices["date"].iloc[-1]

    controls, spacer = st.columns([2, 1])
    with controls:
        horizon = st.slider(
            "Horizon (trading days)",
            min_value=HORIZON_RANGE[0],
            max_value=HORIZON_RANGE[1],
            value=30,
            step=1,
        )
        chosen = st.selectbox(
            "Model",
            options=list(DEFAULT_MODELS),
            index=0,
            help="Naive is the random-walk benchmark. The others are the models "
            "from the comparison page.",
        )
    with spacer:
        st.metric("Last close", f"${last_close:,.2f}", border=True)
        st.caption(f"as of {last_date:%Y-%m-%d}")

    model = chosen if isinstance(chosen, str) else NAIVE
    forecast = load_forecast(model, int(horizon))
    final = forecast.iloc[-1]

    point = float(final["forecast"])
    low = float(final["lo80"])
    high = float(final["hi80"])
    change = float(final["change_pct"])

    left, right = st.columns([1, 2])
    left.metric(
        f"Price in {int(horizon)} trading days",
        f"≈ ${point:,.0f}",
        border=True,
    )
    left.caption(
        f"{model}: {change:+.2f}% against today's ${last_close:,.2f}."
        + (
            " The naive model assumes no drift, so the point estimate is today's price."
            if model == NAIVE
            else ""
        )
    )
    right.metric(
        "80% of outcomes (USD/oz)",
        # No currency symbols inside the value: Streamlit treats text between
        # two ``$`` as LaTeX, which swallowed them and rendered the range in a
        # math font with wide gaps.
        f"{low:,.0f} – {high:,.0f}",
        border=True,
    )
    right.caption(
        "Built from the model's own out-of-sample errors, widened by √time. "
        "Read this as the forecast; the dot on its own is not one."
    )

    st.plotly_chart(forecast_cone(prices, forecast), width="stretch")

    width_pct = (high - low) / point * 100.0
    st.caption(
        f"The band spans **{width_pct:.1f}%** of the point forecast after "
        f"{int(horizon)} trading days. It is built from the model's own "
        "out-of-sample errors, widened by the square root of time."
    )

    if model == DRIFT_OLS:
        st.error(
            "You are looking at the **strawman**. This is the original OLS trend "
            "line: it extrapolates a straight fit through a decade of prices and "
            f"was {holdout_gap()[2]:.1f}× worse than naive on the holdout. The forecast page shows "
            "it so the failure is visible, not so it can be used.",
            icon="⚠️",
        )

    verdict = load_verdict()
    if not verdict.has_edge:
        st.info(
            "**No model beats the naive forecast on this data.** Treat the point "
            "forecast as 'roughly today's price' and read the band as the honest "
            "range. The naive model's band is the cleanest statement of that.",
            icon="🧭",
        )

    with st.expander("Why 1-day gold forecasting is close to impossible"):
        st.markdown(
            "- **It behaves like a random walk.** Daily returns are almost "
            "uncorrelated; the best predictor of tomorrow's close is today's "
            "close, and every model here is judged against exactly that.\n"
            "- **The signal is smaller than the noise.** Daily volatility is "
            "about 1%, while the differences between models are fractions of a "
            "dollar per bar.\n"
            "- **A straight line is the wrong shape.** Gold trended, stalled for "
            "five years, then accelerated — no single slope describes that.\n"
            "- **The band is the honest part.** After 30 days the 80% interval is "
            "several percent wide, which is why a point forecast alone is "
            "misleading.\n"
            "- **This is not investment advice.** The project exists to show how "
            "easily a time-series model can look clever and be useless."
        )
        st.code(
            "from src.model import forecast_future, load_clean_data\n"
            "frame = load_clean_data()\n"
            "band = forecast_future(frame, model='Naive (last price)', horizon=30)",
            language="python",
        )


if __name__ == "__main__":
    render()

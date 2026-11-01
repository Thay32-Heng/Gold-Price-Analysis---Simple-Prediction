"""Gold Lens -- entry point and landing page.

Run from the repository root::

    streamlit run app/Home.py

The module is the shared shell: it configures the page, owns the sidebar date
filter, and registers the navigation. The landing content lives here too, as a
callable page, so the first thing a reader sees is the honest summary.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import streamlit as st

from lib.charts import price_trend
from lib.data import (
    active_range,
    data_freshness,
    holdout_gap,
    load_holdout_table,
    load_prices,
    load_verdict,
    metrics_display,
    refresh_prices,
)

REPO_URL = "https://github.com/Thay32-Heng/Gold-Price-Analysis---Simple-Prediction"

st.set_page_config(
    page_title="Gold Lens — honest gold price analysis",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _sidebar() -> None:
    """Date filter and asset note, shared by every page."""
    prices = load_prices()
    default_start, default_end = active_range(prices)

    with st.sidebar:
        st.markdown("### 🥇 Gold Lens")
        st.caption("Ticker **GC=F** — COMEX gold futures, front month. Daily close, USD/oz.")

        picked = st.date_input(
            "Date range",
            value=(default_start, default_end),
            min_value=default_start,
            max_value=default_end,
            format="YYYY-MM-DD",
            help="Filters the charts and the summary statistics on every page.",
        )

        start, end = _normalise_pick(picked, default_start, default_end)
        st.session_state["date_range"] = (start, end)

        st.divider()
        bars = len(prices)
        as_of, age_days = data_freshness(prices)
        st.caption(
            f"{bars:,} daily bars · {prices['date'].iloc[0]:%b %Y} → {as_of:%b %Y}"
        )
        if age_days <= 5:
            st.success(f"Current — last session {as_of:%d %b %Y}", icon="✅")
        else:
            st.warning(
                f"{age_days} days old — last session {as_of:%d %b %Y}. "
                "Refresh before reading the numbers.",
                icon="⏳",
            )
        if st.button("Refresh from yfinance", width="stretch"):
            with st.spinner("Downloading GC=F…"):
                ok, message = refresh_prices()
            if ok:
                st.success(f"Updated — {message}")
                st.rerun()
            else:
                st.error(f"Refresh failed — {message}")
        st.caption(f"[Source repository]({REPO_URL})")


def _normalise_pick(
    picked: object, fallback_start: date, fallback_end: date
) -> tuple[date, date]:
    """Turn whatever ``st.date_input`` returned into a closed ``(start, end)``.

    Streamlit returns a 1-tuple while the user is still choosing the second
    date, and a bare date if the widget is in single-date mode.
    """
    if isinstance(picked, tuple) and picked:
        start = picked[0]
        end = picked[-1]
        if isinstance(start, date) and isinstance(end, date):
            return start, end
    if isinstance(picked, date):
        return picked, picked
    return fallback_start, fallback_end


def _render_landing() -> None:
    """Landing page: what this is, the verified numbers, the limitations."""
    prices = load_prices()
    start, end = active_range(prices)
    view = prices.loc[
        (prices["date"].dt.date >= start) & (prices["date"].dt.date <= end)
    ]

    if view.empty or len(view) < 2:
        st.title("🥇 Gold Lens")
        st.warning(
            "The selected date range contains no trading days "
            f"({start} → {end}). Widen it in the sidebar — gold futures do not "
            "trade at weekends.",
            icon="⚠️",
        )
        return

    st.title("🥇 Gold Lens")
    st.markdown(
        "**A gold price dashboard that refuses to pretend it can predict tomorrow.** "
        "Every number on these pages is measured out of sample."
    )

    last_close = float(view["close"].iloc[-1])
    prev_close = float(view["close"].iloc[-2]) if len(view) > 1 else last_close
    day_change = last_close - prev_close
    day_pct = (day_change / prev_close * 100.0) if prev_close else 0.0
    log_returns = np.diff(np.log(view["close"].to_numpy(dtype=float)))
    daily_vol = float(np.std(log_returns, ddof=1) * 100.0)
    last_price_date = view["date"].iloc[-1].date()

    # The 30-day figure is measured on the full history so it means the same
    # thing regardless of how the sidebar date filter is set.
    tail = prices["close"].tail(31)
    change_30d = (float(tail.iloc[-1]) / float(tail.iloc[0]) - 1.0) * 100.0
    _, age_days = data_freshness(prices)

    card_price, card_30d, card_vol, card_date = st.columns(4)
    card_price.metric(
        "Gold price",
        f"${last_close:,.2f}",
        delta=f"{day_change:+,.2f} ({day_pct:+.2f}%)",
        border=True,
    )
    card_30d.metric(
        "Last 30 days",
        f"{change_30d:+.1f}%",
        delta=f"{float(tail.iloc[-1]) - float(tail.iloc[0]):+,.0f} USD/oz",
        delta_color="off",
        border=True,
    )
    card_vol.metric("Daily volatility", f"{daily_vol:.2f}%", border=True)
    card_date.metric(
        "Data as of",
        f"{last_price_date:%d %b %Y}",
        delta=f"{age_days} days ago" if age_days else "today",
        delta_color="off",
        border=True,
    )

    if age_days > 5:
        st.warning(
            f"This build is reading a price file that ends {last_price_date:%d %b %Y} "
            f"({age_days} days old). Use **Refresh from yfinance** in the sidebar "
            "before quoting any of these numbers.",
            icon="⏳",
        )

    st.plotly_chart(price_trend(view), width="stretch")

    st.subheader("What this is")
    st.markdown(
        f"- **A decade of gold futures.** {len(prices):,} daily closes for `GC=F`, "
        f"{prices['date'].iloc[0]:%B %Y} to {last_price_date:%B %Y}.\n"
        "- **A model comparison that is actually fair.** An expanding-window "
        "walk-forward, refit monthly, with no shuffling and no peeking at the bar "
        "being predicted.\n"
        "- **A cautionary baseline.** The repository's original OLS straight line "
        "is kept visible, because it is a good example of a model that looks "
        "impressive on a chart and fails completely out of sample."
    )

    st.subheader("The number that matters")
    holdout = load_holdout_table()
    st.dataframe(
        metrics_display(holdout),
        width="stretch",
        hide_index=True,
    )
    naive_mae, ols_mae, ratio = holdout_gap()
    st.error(
        f"On a single chronological holdout the OLS trend line is "
        f"**{ratio:.1f}× worse** than the naive forecast "
        f"(${ols_mae:,.1f} vs ${naive_mae:,.1f} MAE). It is a straight line drawn "
        "through a curve, extrapolated into the future.",
        icon="⚠️",
    )

    verdict = load_verdict()
    if verdict.has_edge:
        st.success(f"**{verdict.headline}** {verdict.detail}", icon="✅")
    else:
        st.warning(f"**{verdict.headline}** {verdict.detail}", icon="🧭")

    st.subheader("Limitations — read these before using any number")
    st.markdown(
        "- **One day ahead, gold is close to a random walk.** The walk-forward "
        "comparison on the *Model comparison* page shows that no model beats "
        "'tomorrow costs what it costs today' by a meaningful margin.\n"
        "- **The 80% band is wide.** It is built from the model's own "
        "out-of-sample errors, not from a promise about the future.\n"
        "- **Futures, not spot.** `GC=F` carries roll effects and is quoted in "
        "USD per troy ounce; it is not the price you would pay at a coin shop.\n"
        "- **Nothing here is investment advice.** The project exists to show how "
        "easy it is to fool yourself with a time-series model."
    )


def main() -> None:
    """Build the shell and hand control to the selected page."""
    _sidebar()
    navigation = st.navigation(
        [
            st.Page(_render_landing, title="Overview", icon="🏠", default=True),
            st.Page("pages/1_Analysis.py", title="Market analysis", icon="📈"),
            st.Page("pages/2_Model_Comparison.py", title="Model comparison", icon="⚖️"),
            st.Page("pages/3_Forecast.py", title="Forecast", icon="🔮"),
        ]
    )
    navigation.run()


if __name__ == "__main__":
    main()

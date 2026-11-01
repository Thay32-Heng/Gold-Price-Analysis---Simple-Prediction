"""Model comparison: the walk-forward table and where the errors come from.

The point of this page is to show the comparison honestly, including the case
where nothing beats the naive forecast.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from lib import md_safe
from lib.charts import mae_by_model, predicted_vs_actual, rolling_error
from lib.data import (
    DEFAULT_MODELS,
    NAIVE,
    holdout_gap,
    load_holdout_table,
    load_predictions,
    load_verdict,
    load_walk_forward_table,
    metrics_display,
)


def render() -> None:
    """Draw the model comparison page."""
    st.title("Model comparison")
    st.caption(
        "**Takeaway:** once you score the models out of sample, the straight-line "
        "OLS forecast is far worse than doing nothing, and no model earns a "
        "meaningful edge over 'tomorrow costs what it costs today'."
    )

    verdict = load_verdict()
    if verdict.has_edge:
        st.success(
            f"**{md_safe(verdict.headline)}**\n\n{md_safe(verdict.detail)}",
            icon="✅",
        )
    else:
        st.warning(
            f"**{md_safe(verdict.headline)}**\n\n{md_safe(verdict.detail)}",
            icon="🧭",
        )

    st.subheader("1. The original model, scored honestly")
    st.markdown(
        "One fit on the first 80% of the series, one score on the last 20%. "
        "This is the comparison the README reports."
    )
    naive_mae, ols_mae, ratio = holdout_gap()
    st.dataframe(metrics_display(load_holdout_table()), width="stretch", hide_index=True)
    st.error(
        f"The OLS line is **{ratio:.1f}× worse** than naive "
        f"(\\${ols_mae:,.1f} vs \\${naive_mae:,.1f} MAE). It was fitted to a decade of "
        "prices and then extrapolated a straight line into 2023–2026, where gold "
        "rose far faster than the fit expected.",
        icon="⚠️",
    )

    st.subheader("2. Expanding-window walk-forward")
    st.markdown(
        "Start with ten years of history, forecast the next month, add that month "
        "to the training set, refit, repeat. No shuffling, and a training row is "
        "only used once its outcome is already known at the refit date."
    )
    table = load_walk_forward_table()
    st.dataframe(metrics_display(table), width="stretch", hide_index=True)
    st.caption(
        "Direction is blank for the naive forecast: it predicts no change, so it "
        "has no direction to be right or wrong about. Bias is the mean signed "
        "error — a large positive bias means the model under-predicts."
    )

    predictions = load_predictions()
    base_rate = float(
        np.mean(predictions["actual"] > predictions["last_close"]) * 100.0
    )
    st.caption(
        f"Gold rose on **{base_rate:.1f}%** of the evaluated bars. Any model that "
        "leans slightly upwards scores about that much directional accuracy "
        "without predicting anything."
    )

    st.plotly_chart(mae_by_model(table), width="stretch")

    st.subheader("3. Where the errors come from")
    st.plotly_chart(rolling_error(predictions), width="stretch")
    st.caption(
        "The OLS line's error grows with every dollar gold moves away from the "
        "fitted trend; the other two track the price because they start from it."
    )

    chosen = st.selectbox(
        "Predicted vs actual for one model",
        options=list(DEFAULT_MODELS),
        index=0,
        help="The last 250 walk-forward bars, one point per bar.",
    )
    model_name = chosen if isinstance(chosen, str) else NAIVE
    st.plotly_chart(predicted_vs_actual(predictions, model_name), width="stretch")

    with st.expander("Method — what stops this from cheating"):
        st.markdown(
            "- **Chronological only.** Rows are consumed in date order; there is "
            "no shuffle anywhere in `src/model.py`.\n"
            "- **No target leakage.** A row enters training only when its forward "
            "return is already realised at the refit date "
            "(`train_end - horizon`).\n"
            "- **No look-ahead features.** Every feature is a backward transform: "
            "lags, rolling means, rolling volatility, day-of-week.\n"
            "- **No tuning on the test window.** Ridge alpha (10.0) and the "
            "boosting parameters are fixed constants, chosen before the "
            "evaluation and never revisited.\n"
            "- **Refit cadence:** every 21 bars (about one trading month); the "
            "evaluation starts after the first 50% of the series.\n"
            "- **The booster is off by default** because it loses to naive on "
            "this data. `python -m src.model` prints its row alongside the three "
            "below; it stays reachable from the API via "
            "`walk_forward_eval(..., include_boosting=True)`."
        )
        st.code(
            "python -m src.model\n"
            "# prints the holdout audit and the walk-forward table",
            language="bash",
        )


if __name__ == "__main__":
    render()

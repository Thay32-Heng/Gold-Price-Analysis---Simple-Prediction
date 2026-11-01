"""Cached data access for the Gold Lens app.

Importing this module does no I/O and no computation: every loader is wrapped
in :func:`streamlit.cache_data` and only runs when a page asks for it.
"""

from __future__ import annotations

import sys
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from lib import PROJECT_ROOT

# ``src`` is a plain directory, not an installed package, so the repository
# root has to be importable before the project modules can be imported.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import DATA_PATH, load_clean_data  # noqa: E402
from src.model import (  # noqa: E402
    DEFAULT_MODELS,
    DRIFT_OLS,
    NAIVE,
    ModelVerdict,
    compare_to_naive,
    forecast_future,
    holdout_split_eval,
    make_features,
    summarise_predictions,
    walk_forward_predictions,
)

__all__ = [
    "DEFAULT_MODELS",
    "DRIFT_OLS",
    "HORIZON_RANGE",
    "MONTH_LABELS",
    "NAIVE",
    "active_range",
    "filtered_prices",
    "holdout_gap",
    "load_features",
    "load_forecast",
    "load_holdout_table",
    "load_monthly_returns",
    "load_predictions",
    "load_prices",
    "load_verdict",
    "load_walk_forward_table",
    "metrics_display",
    "model_sigma",
    "slice_by_date",
]

#: Forecast horizons offered by the UI, in trading days.
HORIZON_RANGE: tuple[int, int] = (1, 60)

MONTH_LABELS: tuple[str, ...] = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

#: Column headers used when a metrics table is shown to a reader.
METRIC_HEADERS: dict[str, str] = {
    "model": "Model",
    "n": "Bars",
    "mae": "MAE ($)",
    "rmse": "RMSE ($)",
    "mape": "MAPE (%)",
    "directional_accuracy": "Direction (%)",
    "bias": "Bias ($)",
}


# --------------------------------------------------------------------------- #
# Raw frames
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def load_prices() -> pd.DataFrame:
    """Return the committed cleaned price history (``data/process``)."""
    return load_clean_data(DATA_PATH)


def data_freshness(prices: pd.DataFrame) -> tuple[date, int]:
    """Return ``(last_session, age_in_days)`` for the loaded history.

    A gold dashboard that quietly shows an eight-month-old close is worse than
    one that admits it, so every page can surface this.
    """
    last = prices["date"].iloc[-1]
    last_date = last.date() if hasattr(last, "date") else last
    return last_date, (date.today() - last_date).days


def refresh_prices() -> tuple[bool, str]:
    """Re-download ``GC=F`` and rewrite the committed CSV.

    Returns ``(ok, message)``. Network failures are reported, never raised, so
    a dead connection cannot take the dashboard down.
    """
    from src.data_cleaning import clean_gold_data

    try:
        frame = clean_gold_data(force_download=True)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        return False, f"{type(exc).__name__}: {exc}"
    st.cache_data.clear()
    return True, f"{len(frame):,} bars, last close ${frame['close'].iloc[-1]:,.2f}"


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    """Return the model feature matrix aligned to :func:`load_prices`."""
    return make_features(load_prices())


@st.cache_data(show_spinner=False)
def load_monthly_returns() -> pd.DataFrame:
    """Daily percentage returns tagged with a calendar month, for the box plot."""
    frame = load_prices()
    returns = frame.loc[:, ["date", "close"]].copy()
    returns["return_pct"] = returns["close"].pct_change() * 100.0
    returns["month"] = returns["date"].dt.month
    returns["month_label"] = returns["month"].map(
        lambda month: MONTH_LABELS[int(month) - 1]
    )
    return returns.dropna(subset=["return_pct"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Model results
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def load_holdout_table() -> pd.DataFrame:
    """The single-split table behind the README's cautionary OLS headline."""
    return holdout_split_eval(load_prices())


@st.cache_data(show_spinner=False)
def holdout_gap() -> tuple[float, float, float]:
    """Return ``(naive_mae, ols_mae, ratio)`` from the single-split holdout."""
    table = load_holdout_table()
    naive = float(table.loc[table["model"] == NAIVE, "mae"].iloc[0])
    ols = float(table.loc[table["model"] == DRIFT_OLS, "mae"].iloc[0])
    return naive, ols, ols / naive


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    """Pooled walk-forward forecasts for every default model."""
    return walk_forward_predictions(load_prices(), models=DEFAULT_MODELS)


@st.cache_data(show_spinner=False)
def load_walk_forward_table() -> pd.DataFrame:
    """Walk-forward MAE / RMSE / MAPE / direction table."""
    return summarise_predictions(load_predictions())


@st.cache_data(show_spinner=False)
def load_verdict() -> ModelVerdict:
    """Plain-English reading of the walk-forward table."""
    return compare_to_naive(load_walk_forward_table())


def model_sigma(model: str) -> float:
    """One-step out-of-sample log-error spread for ``model``.

    Derived from the cached walk-forward predictions, so switching horizons in
    the forecast page never re-runs the evaluation.
    """
    predictions = load_predictions()
    subset = predictions.loc[predictions["model"] == model]
    if subset.empty:
        raise ValueError(f"no walk-forward predictions for {model!r}")
    error = np.log(subset["predicted"].to_numpy(dtype=float)) - np.log(
        subset["actual"].to_numpy(dtype=float)
    )
    return float(np.std(error, ddof=1))


@st.cache_data(show_spinner=False)
def load_forecast(model: str, horizon: int) -> pd.DataFrame:
    """Point forecast plus 80% band for ``horizon`` business days ahead."""
    return forecast_future(
        load_prices(), model=model, horizon=horizon, sigma_1=model_sigma(model)
    )


# --------------------------------------------------------------------------- #
# View helpers
# --------------------------------------------------------------------------- #


def _to_date(value: object) -> date:
    """Coerce a stored widget value to a plain ``date``."""
    return date.fromisoformat(str(value)[:10])


def slice_by_date(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Keep the rows whose date falls inside ``[start, end]``."""
    mask = (frame["date"].dt.date >= start) & (frame["date"].dt.date <= end)
    return frame.loc[mask].reset_index(drop=True)


def active_range(frame: pd.DataFrame) -> tuple[date, date]:
    """The sidebar date range, falling back to the full history."""
    stored = st.session_state.get("date_range")
    if isinstance(stored, tuple) and len(stored) == 2:
        return _to_date(stored[0]), _to_date(stored[1])
    return _to_date(frame["date"].min()), _to_date(frame["date"].max())


def filtered_prices() -> pd.DataFrame:
    """Price history clipped to the sidebar range."""
    frame = load_prices()
    start, end = active_range(frame)
    return slice_by_date(frame, start, end)


def metrics_display(table: pd.DataFrame) -> pd.DataFrame:
    """Rename and round a metrics table for display."""
    display = table.rename(columns=METRIC_HEADERS).copy()
    for column in ("MAE ($)", "RMSE ($)", "Bias ($)"):
        display[column] = display[column].round(2)
    for column in ("MAPE (%)", "Direction (%)"):
        display[column] = display[column].round(1)
    display["Bars"] = display["Bars"].astype(int)
    return display

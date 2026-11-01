"""Leakage-free forecasting utilities for the gold price series.

The repository's original "model" is an OLS straight line fitted to
``days_since_start`` and scored on the same rows it was fitted on. On a
chronological holdout that line is roughly 30x worse than the trivial
"tomorrow costs what it costs today" forecast; :func:`holdout_split_eval`
reproduces those exact numbers so the claim can be audited.

This module keeps that strawman and adds the honest comparison: an
expanding-window walk-forward evaluation in which every forecast is produced
by a model that has never seen the bar it is predicting.

Rules enforced here
-------------------
* **No shuffling.** Rows are always consumed in date order.
* **No leakage.** A training row is only used once its target window has
  closed at the refit date, and every feature at *t* is built from data up to
  and including *t* (there is no ``shift(-1)`` in :func:`make_features`).
* **No tuning on the test window.** Hyper-parameters are fixed constants and
  the walk-forward window is never used to select a model.

Run ``python -m src.model`` for the audit table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge

try:  # ``python -m src.model``
    from .data_cleaning import load_clean_data
except ImportError:  # ``python src/model.py``
    from data_cleaning import load_clean_data

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]

NAIVE: Final = "Naive (last price)"
DRIFT_OLS: Final = "OLS trend (strawman)"
RIDGE: Final = "Ridge (features)"
BOOSTING: Final = "HistGradientBoosting (features)"

#: Models shown in the honest comparison. The booster is excluded by default
#: because it loses to the naive forecast on this data (MAE ~= $21 vs ~$19);
#: pass ``include_boosting=True`` to see it for yourself.
DEFAULT_MODELS: Final = (NAIVE, DRIFT_OLS, RIDGE)
ALL_MODELS: Final = (NAIVE, DRIFT_OLS, RIDGE, BOOSTING)

FEATURE_COLUMNS: Final = (
    "ret_1",
    "ret_lag1",
    "ret_lag2",
    "ret_lag3",
    "ret_lag5",
    "ret_lag10",
    "ma_gap",
    "px_ma50",
    "vol_20",
    "vol_ratio",
    "mom_10",
    "rsi_14",
    "dow_1",
    "dow_2",
    "dow_3",
    "dow_4",
)

METRIC_COLUMNS: Final = (
    "model",
    "n",
    "mae",
    "rmse",
    "mape",
    "directional_accuracy",
    "bias",
)

#: Fixed hyper-parameters. Nothing below is tuned on the evaluation window.
RIDGE_ALPHA: Final = 10.0
BOOSTING_PARAMS: Final = {
    "max_depth": 3,
    "max_iter": 200,
    "learning_rate": 0.05,
    "l2_regularization": 1.0,
    "random_state": 0,
}

#: Two-sided z-score for an 80% interval.
Z80: Final = 1.2815515655446004


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #


def _as_float(values: object) -> NDArray[np.float64]:
    """Coerce any pandas rolling result to a float array."""
    return np.asarray(values, dtype=float)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build predictors that only use information known at the close of *t*.

    Parameters
    ----------
    df:
        Price frame with at least ``date`` and ``close`` columns, sorted
        oldest to newest.

    Returns
    -------
    pandas.DataFrame
        The columns in :data:`FEATURE_COLUMNS`, aligned to ``df.index``.
        Early rows are ``NaN`` because of the rolling warm-up.

    Notes
    -----
    Every column is a *backward* transform: ``shift(k)`` with ``k >= 1``,
    ``rolling(...)`` and ``diff()``. There is deliberately no forward-looking
    operation here.
    """
    close = df["close"].astype(float)
    close_values = close.to_numpy(dtype=float)
    log_ret = pd.Series(np.log(close_values), index=close.index, dtype=float).diff()

    feats = pd.DataFrame(index=df.index)
    feats["ret_1"] = log_ret.to_numpy(dtype=float)
    for lag in (1, 2, 3, 5, 10):
        feats[f"ret_lag{lag}"] = log_ret.shift(lag).to_numpy(dtype=float)

    # pandas' own type stubs are imprecise about rolling results, so the
    # conversions go through np.asarray, which accepts any array-like.
    ma50 = _as_float(close.rolling(50).mean())
    ma200 = _as_float(close.rolling(200).mean())
    feats["ma_gap"] = np.log(ma50 / ma200)
    feats["px_ma50"] = np.log(close_values / ma50)

    vol20 = _as_float(log_ret.rolling(20).std())
    vol60 = _as_float(log_ret.rolling(60).std())
    feats["vol_20"] = vol20
    feats["vol_ratio"] = vol20 / vol60

    feats["mom_10"] = _as_float(log_ret.rolling(10).sum())

    delta = _as_float(close.diff())
    gain = _as_float(pd.Series(delta).clip(lower=0).rolling(14).mean())
    loss = _as_float(pd.Series(-delta).clip(lower=0).rolling(14).mean())
    feats["rsi_14"] = 2.0 * gain / (gain + loss) - 1.0

    weekday = df["date"].dt.day_of_week.to_numpy()
    for day in range(1, 5):
        feats[f"dow_{day}"] = (weekday == day).astype(float)

    return feats.loc[:, list(FEATURE_COLUMNS)].astype(float)


def make_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Forward log return from *t* to *t + horizon*.

    The final ``horizon`` rows are ``NaN`` because their outcome is not yet
    observable. Those rows are never used for training or scoring.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    close = df["close"].astype(float).to_numpy(dtype=float)
    if horizon >= len(close):
        raise ValueError(f"horizon {horizon} is longer than the series")

    forward = np.full(close.shape, np.nan, dtype=float)
    forward[:-horizon] = np.log(close[horizon:] / close[:-horizon])
    return pd.Series(forward, index=df.index, name="target")


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #


def _make_estimator(name: str) -> Any:
    """Return an unfitted scikit-learn regressor for a feature model."""
    if name == RIDGE:
        return Ridge(alpha=RIDGE_ALPHA)
    if name == BOOSTING:
        return HistGradientBoostingRegressor(**BOOSTING_PARAMS)
    raise ValueError(f"{name!r} is not a feature model")


def _predict_block(
    name: str,
    *,
    values: NDArray[np.float64],
    days: NDArray[np.int64],
    matrix: NDArray[np.float64],
    target: NDArray[np.float64],
    train_rows: NDArray[np.int64],
    train_feature_rows: NDArray[np.int64],
    test_rows: NDArray[np.int64],
    horizon: int,
) -> NDArray[np.float64]:
    """Forecast the price at each row of ``test_rows`` using data up to the cut.

    ``test_rows`` are *target* rows: a forecast for row ``t`` is made from the
    close of row ``t - horizon``.
    """
    if name == NAIVE:
        return values[test_rows - horizon]

    if name == DRIFT_OLS:
        model = LinearRegression().fit(
            days[train_rows].reshape(-1, 1), values[train_rows]
        )
        return np.asarray(model.predict(days[test_rows].reshape(-1, 1)), dtype=float)

    if train_feature_rows.size == 0:
        return np.full(test_rows.shape, np.nan)

    # The target for row ``j`` is ``log(close[j + horizon] / close[j])``, so the
    # forecast of ``close[t]`` is produced from the features *and* the price of
    # row ``t - horizon``. Using ``matrix[t]`` here would put the bar being
    # predicted inside the model's input -- a silent look-ahead leak.
    base_rows = test_rows - horizon
    model = _make_estimator(name).fit(
        matrix[train_feature_rows], target[train_feature_rows]
    )
    predicted_log_return = np.asarray(model.predict(matrix[base_rows]), dtype=float)
    return values[base_rows] * np.exp(predicted_log_return)


# --------------------------------------------------------------------------- #
# Walk-forward evaluation
# --------------------------------------------------------------------------- #


def walk_forward_predictions(
    df: pd.DataFrame,
    *,
    horizon: int = 1,
    start_frac: float = 0.5,
    step: int = 21,
    warmup: int = 260,
    models: Sequence[str] = DEFAULT_MODELS,
) -> pd.DataFrame:
    """Expanding-window walk-forward forecasts, one row per bar and model.

    The first ``start_frac`` of the series is used for the initial fit; after
    that the training window grows and every model is refitted every ``step``
    bars. No row is ever predicted by a model that was fitted on it or on
    anything that happened after it.

    Parameters
    ----------
    df:
        Price frame with ``date`` and ``close``, sorted oldest to newest.
    horizon:
        Forecast horizon in trading bars.
    start_frac:
        Fraction of the series consumed before the first out-of-sample
        forecast (0.5 keeps ten years for training).
    step:
        Refit cadence in bars (21 ~= one trading month).
    warmup:
        Minimum number of rows before the first fold, so the 200-day moving
        average is defined.
    models:
        Any subset of :data:`ALL_MODELS`.

    Returns
    -------
    pandas.DataFrame
        Columns ``date``, ``model``, ``actual``, ``last_close``, ``predicted``.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if not 0.0 < start_frac < 1.0:
        raise ValueError(f"start_frac must be in (0, 1), got {start_frac}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    unknown = set(models) - set(ALL_MODELS)
    if unknown:
        raise ValueError(f"unknown model(s): {sorted(unknown)}")

    frame = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(frame["date"])
    values = frame["close"].astype(float).to_numpy()
    days = (dates - dates.iloc[0]).dt.days.to_numpy(dtype=np.int64)

    features = make_features(frame)
    matrix = features.to_numpy(dtype=float)
    target = make_target(frame, horizon).to_numpy(dtype=float)
    usable = (~features.isna().any(axis=1)).to_numpy() & ~np.isnan(target)

    n_rows = len(frame)
    first = max(int(n_rows * start_frac), warmup, horizon + 1)
    if first >= n_rows - horizon:
        raise ValueError(
            f"not enough rows ({n_rows}) for horizon={horizon}, start_frac={start_frac}"
        )

    blocks: list[pd.DataFrame] = []
    for train_end in range(first, n_rows - horizon, step):
        # Only rows whose target window has closed *before* the forecast origin
        # of this fold may be trained on. The origin of the first test bar is
        # ``train_end - horizon``, so the newest legal training row is
        # ``train_end - 2 * horizon``.
        train_rows = np.arange(max(train_end - 2 * horizon + 1, 1))
        test_rows = np.arange(train_end, min(train_end + step, n_rows - horizon))
        if test_rows.size == 0:
            break
        # Every model is scored on the same bars: those whose feature row is
        # usable. Without this the Ridge/HGB series silently starts later than
        # the naive one and the ranking is no longer apples-to-apples.
        test_rows = test_rows[usable[test_rows - horizon]]
        if test_rows.size == 0:
            continue
        train_feature_rows = train_rows[usable[train_rows]]

        for name in models:
            predicted = _predict_block(
                name,
                values=values,
                days=days,
                matrix=matrix,
                target=target,
                train_rows=train_rows,
                train_feature_rows=train_feature_rows,
                test_rows=test_rows,
                horizon=horizon,
            )
            blocks.append(
                pd.DataFrame(
                    {
                        "date": dates.iloc[test_rows].to_numpy(),
                        "model": name,
                        "actual": values[test_rows],
                        "last_close": values[test_rows - horizon],
                        "predicted": predicted,
                    }
                )
            )

    if not blocks:
        raise ValueError("no walk-forward folds were produced; check start_frac")

    predictions = pd.concat(blocks, ignore_index=True)
    predictions = predictions.dropna(subset=["predicted"]).reset_index(drop=True)
    return predictions


def _summarise_block(group: pd.DataFrame) -> dict[str, float]:
    """Compute the error metrics for one model's pooled forecasts."""
    actual = group["actual"].to_numpy(dtype=float)
    predicted = group["predicted"].to_numpy(dtype=float)
    last_close = group["last_close"].to_numpy(dtype=float)
    error = actual - predicted

    predicted_direction = np.sign(predicted - last_close)
    actual_direction = np.sign(actual - last_close)
    if np.all(predicted_direction == 0):
        # A flat forecast has no direction to be right or wrong about.
        directional = float("nan")
    else:
        directional = float(np.mean(predicted_direction == actual_direction) * 100.0)

    return {
        "n": float(len(group)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.mean(np.abs(error) / actual) * 100.0),
        "directional_accuracy": directional,
        "bias": float(np.mean(error)),
    }


def summarise_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Turn long walk-forward predictions into one metrics row per model."""
    rows: list[dict[str, Any]] = []
    for name, group in predictions.groupby("model", sort=False):
        row: dict[str, Any] = {"model": str(name)}
        row.update(_summarise_block(group))
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=list(METRIC_COLUMNS))

    # Naive stays first as the reference; everything else is ranked by MAE.
    table["_naive_first"] = (table["model"] != NAIVE).astype(int)
    table = table.sort_values(["_naive_first", "mae"], kind="stable")
    return (
        table.drop(columns="_naive_first")
        .reset_index(drop=True)
        .loc[:, list(METRIC_COLUMNS)]
    )


def walk_forward_eval(
    df: pd.DataFrame,
    *,
    horizon: int = 1,
    start_frac: float = 0.5,
    step: int = 21,
    warmup: int = 260,
    models: Sequence[str] = DEFAULT_MODELS,
    include_boosting: bool = False,
) -> pd.DataFrame:
    """Walk-forward metric table: MAE, RMSE, MAPE, directional accuracy, bias.

    See :func:`walk_forward_predictions` for the windowing rules.
    """
    chosen = list(models)
    if include_boosting and BOOSTING not in chosen:
        chosen.append(BOOSTING)
    predictions = walk_forward_predictions(
        df,
        horizon=horizon,
        start_frac=start_frac,
        step=step,
        warmup=warmup,
        models=chosen,
    )
    return summarise_predictions(predictions)


def holdout_split_eval(
    df: pd.DataFrame,
    *,
    test_frac: float = 0.2,
    models: Sequence[str] = (NAIVE, DRIFT_OLS),
) -> pd.DataFrame:
    """Single chronological split behind the README's cautionary headline.

    Fit on everything except the last ``test_frac`` of the series, then score
    the holdout once. The absolute errors move whenever the price file is
    refreshed, but the relationship does not: the OLS line lands an order of
    magnitude worse than doing nothing.

    This is the *cautionary* table, not the honest one: a single split on a
    strongly trending series flatters nothing and punishes the extrapolating
    line. Use :func:`walk_forward_eval` for the real comparison.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")

    frame = df.sort_values("date").reset_index(drop=True)
    values = frame["close"].astype(float).to_numpy()
    days = (
        pd.to_datetime(frame["date"]) - pd.to_datetime(frame["date"]).iloc[0]
    ).dt.days.to_numpy(dtype=np.int64)

    split = int(len(frame) * (1.0 - test_frac))
    if split < 2:
        raise ValueError(
            f"test_frac={test_frac} leaves {split} training row(s); "
            "need at least 2 so the naive forecast has a previous close"
        )
    test_rows = np.arange(split, len(frame))
    last_close = values[test_rows - 1]

    blocks: list[pd.DataFrame] = []
    for name in models:
        if name == NAIVE:
            predicted = values[test_rows - 1]
        elif name == DRIFT_OLS:
            model = LinearRegression().fit(
                days[:split].reshape(-1, 1), values[:split]
            )
            predicted = np.asarray(
                model.predict(days[test_rows].reshape(-1, 1)), dtype=float
            )
        else:
            raise ValueError(
                f"holdout_split_eval only supports {NAIVE!r} and {DRIFT_OLS!r}"
            )
        blocks.append(
            pd.DataFrame(
                {
                    "model": name,
                    "actual": values[test_rows],
                    "last_close": last_close,
                    "predicted": predicted,
                }
            )
        )

    return summarise_predictions(pd.concat(blocks, ignore_index=True))


# --------------------------------------------------------------------------- #
# Honest verdict
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelVerdict:
    """Plain-English reading of a walk-forward table.

    Attributes
    ----------
    best_model, best_mae:
        Best non-naive model by MAE, and its MAE.
    naive_mae:
        MAE of the "tomorrow = today" forecast on the same bars.
    improvement_pct:
        Percentage MAE reduction of ``best_model`` over naive. Negative means
        the model is worse than doing nothing.
    has_edge:
        ``True`` only when ``improvement_pct`` clears :data:`EDGE_THRESHOLD`.
    headline, detail:
        Sentences written for a non-specialist reader.
    """

    best_model: str
    best_mae: float
    naive_mae: float
    improvement_pct: float
    has_edge: bool
    headline: str
    detail: str


#: A model must beat naive by this much (percent MAE) before we call it an edge.
EDGE_THRESHOLD: Final = 2.0


def compare_to_naive(
    table: pd.DataFrame, *, threshold_pct: float = EDGE_THRESHOLD
) -> ModelVerdict:
    """Judge a metrics table against the naive forecast, without spin.

    ``threshold_pct`` exists because sub-percent differences on ~1,400 bars are
    noise, not skill: the standard error of daily gold returns swamps them.
    """
    if table.empty:
        raise ValueError("empty metrics table")

    naive_maes = table.loc[table["model"] == NAIVE, "mae"]
    if naive_maes.empty:
        raise ValueError(f"table has no {NAIVE!r} row")
    naive_mae = float(naive_maes.iloc[0])

    challengers = table.loc[table["model"] != NAIVE].sort_values("mae")
    if challengers.empty:
        return ModelVerdict(
            best_model=NAIVE,
            best_mae=naive_mae,
            naive_mae=naive_mae,
            improvement_pct=0.0,
            has_edge=False,
            headline="Only the naive forecast was evaluated.",
            detail="Add another model to make the comparison meaningful.",
        )

    best_model = str(challengers["model"].iloc[0])
    best_mae = float(challengers["mae"].iloc[0])
    improvement = (naive_mae - best_mae) / naive_mae * 100.0
    has_edge = improvement >= threshold_pct

    if has_edge:
        headline = (
            f"{best_model} beats the naive forecast by {improvement:.1f}% MAE "
            f"(${best_mae:,.2f} vs ${naive_mae:,.2f} per bar)."
        )
        detail = (
            "The margin is real but small in absolute terms; treat it as a "
            "slightly better guess, not as a tradable signal."
        )
    else:
        headline = (
            "No model beats the naive 'tomorrow costs what it costs today' "
            "forecast at this horizon."
        )
        detail = (
            f"Best challenger: {best_model} at ${best_mae:,.2f} MAE against "
            f"${naive_mae:,.2f} for naive ({improvement:+.1f}%, inside the noise "
            "band). The OLS trend line is far worse than both."
        )

    return ModelVerdict(
        best_model=best_model,
        best_mae=best_mae,
        naive_mae=naive_mae,
        improvement_pct=improvement,
        has_edge=has_edge,
        headline=headline,
        detail=detail,
    )


# --------------------------------------------------------------------------- #
# Forward forecast
# --------------------------------------------------------------------------- #


def residual_sigma(
    df: pd.DataFrame, model: str, *, horizon: int = 1
) -> float:
    """Standard deviation of a model's walk-forward log errors.

    Used to size the forecast band. Measuring the spread on out-of-sample
    errors rather than in-sample residuals keeps the band honest.
    """
    predictions = walk_forward_predictions(
        df, horizon=horizon, models=(model,)
    )
    log_error = np.log(predictions["predicted"].to_numpy(dtype=float)) - np.log(
        predictions["actual"].to_numpy(dtype=float)
    )
    return float(np.std(log_error, ddof=1))


def forecast_future(
    df: pd.DataFrame,
    model: str = NAIVE,
    horizon: int = 30,
    *,
    sigma_1: float | None = None,
) -> pd.DataFrame:
    """Point forecast plus a residual-based 80% band for the next ``horizon`` bars.

    Parameters
    ----------
    df:
        Full price history, oldest to newest.
    model:
        One of :data:`ALL_MODELS`.
    horizon:
        Number of business days to project.
    sigma_1:
        One-step log-error standard deviation. Pass a cached value to avoid
        re-running the walk-forward; when omitted it is computed here.

    Returns
    -------
    pandas.DataFrame
        One row per future business day with ``date``, ``step``, ``forecast``,
        ``lo80``, ``hi80`` and ``change_pct``.

    Notes
    -----
    The band is the model's own out-of-sample error spread widened by
    ``sqrt(step)`` -- the variance of a random walk grows linearly with time.
    It describes where the price may land, not the uncertainty of the mean.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if model not in ALL_MODELS:
        raise ValueError(f"unknown model: {model!r}")

    frame = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(frame["date"])
    close = frame["close"].astype(float)
    last_date = pd.Timestamp(dates.iloc[-1])
    last_close = float(close.iloc[-1])

    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)
    steps = np.arange(1, horizon + 1, dtype=float)

    if model == NAIVE:
        point_log_return = np.zeros(horizon)
    elif model == DRIFT_OLS:
        days = (dates - dates.iloc[0]).dt.days.to_numpy(dtype=np.int64)
        trend = LinearRegression().fit(days.reshape(-1, 1), close.to_numpy())
        future_days = (future_dates - dates.iloc[0]).days.to_numpy(dtype=np.int64)
        prices = np.asarray(
            trend.predict(future_days.reshape(-1, 1)), dtype=float
        )
        point_log_return = np.log(prices / last_close)
    else:
        features = make_features(frame)
        matrix = features.to_numpy(dtype=float)
        target = make_target(frame, 1).to_numpy(dtype=float)
        usable = (~features.isna().any(axis=1)).to_numpy() & ~np.isnan(target)
        estimator = _make_estimator(model).fit(matrix[usable], target[usable])
        # Predict from the last row that has *features*. The final row's target
        # is NaN by construction, so indexing ``matrix[usable]`` would silently
        # anchor the forecast one bar in the past.
        feature_rows = (~features.isna().any(axis=1)).to_numpy()
        daily = float(
            np.asarray(estimator.predict(matrix[feature_rows][-1:]), dtype=float)[0]
        )
        point_log_return = daily * steps

    if sigma_1 is None:
        sigma_1 = residual_sigma(frame, model)
    band = Z80 * sigma_1 * np.sqrt(steps)

    forecast = last_close * np.exp(point_log_return)
    return pd.DataFrame(
        {
            "date": future_dates.to_numpy(),
            "step": steps.astype(int),
            "forecast": forecast,
            "lo80": last_close * np.exp(point_log_return - band),
            "hi80": last_close * np.exp(point_log_return + band),
            "change_pct": (forecast / last_close - 1.0) * 100.0,
        }
    )


# --------------------------------------------------------------------------- #
# Standalone audit
# --------------------------------------------------------------------------- #


def _print_table(table: pd.DataFrame) -> None:
    """Render a metrics table as fixed-width text for the CLI audit."""
    header = (
        f"{'model':<34}{'n':>6}{'MAE':>11}{'RMSE':>11}"
        f"{'MAPE':>9}{'dir.acc':>10}{'bias':>10}"
    )
    print(header)
    print("-" * len(header))
    for record in table.to_dict(orient="records"):
        directional = float(record["directional_accuracy"])
        accuracy = "     n/a" if np.isnan(directional) else f"{directional:7.1f}%"
        print(
            f"{str(record['model']):<34}{int(record['n']):>6}"
            f"{float(record['mae']):>11.2f}{float(record['rmse']):>11.2f}"
            f"{float(record['mape']):>8.2f}%{accuracy:>10}"
            f"{float(record['bias']):>10.2f}"
        )


def main() -> None:
    """Print the verified holdout audit and the walk-forward comparison."""
    frame = load_clean_data()

    print(f"rows: {len(frame)}   {frame['date'].min():%Y-%m-%d} -> "
          f"{frame['date'].max():%Y-%m-%d}   last close ${frame['close'].iloc[-1]:,.2f}")
    print()

    holdout = holdout_split_eval(frame)
    print("1) Verified chronological holdout (last 20%, one fit, no refits)")
    _print_table(holdout)
    naive_mae = float(holdout.loc[holdout["model"] == NAIVE, "mae"].iloc[0])
    ols_mae = float(holdout.loc[holdout["model"] == DRIFT_OLS, "mae"].iloc[0])
    print(f"   -> OLS is {ols_mae / naive_mae:.1f}x worse than naive "
          f"(${ols_mae:,.1f} vs ${naive_mae:,.1f})")
    print()

    walk_forward = walk_forward_eval(frame, include_boosting=True)
    print("2) Expanding-window walk-forward (refit every 21 bars, no shuffle)")
    _print_table(walk_forward)
    print()

    verdict = compare_to_naive(walk_forward)
    print(f"verdict: {verdict.headline}")
    print(f"         {verdict.detail}")


if __name__ == "__main__":
    main()

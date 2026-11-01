"""Headless smoke check for the Gold Lens app.

Runs three layers of verification:

1. every data loader returns a frame without raising;
2. every page module imports cleanly and exposes ``render``;
3. Streamlit's own ``AppTest`` renders each page and reports no exception.

Exit code 0 means all three passed. Usage::

    .venv/bin/python scripts/smoke_check.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"

for entry in (str(ROOT), str(APP_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

PAGES = (
    "pages/1_Analysis.py",
    "pages/2_Model_Comparison.py",
    "pages/3_Forecast.py",
)


def _load_module(name: str, path: Path) -> ModuleType:
    """Import a module from a file path (page filenames start with a digit)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _check_data_layer() -> list[str]:
    """Exercise every cached loader the pages depend on."""
    from lib.data import (
        DEFAULT_MODELS,
        holdout_gap,
        load_forecast,
        load_holdout_table,
        load_monthly_returns,
        load_predictions,
        load_prices,
        load_verdict,
        load_walk_forward_table,
        metrics_display,
        model_sigma,
    )

    errors: list[str] = []
    prices = load_prices()
    if len(prices) < 1000:
        errors.append(f"load_prices returned {len(prices)} rows")

    monthly = load_monthly_returns()
    if monthly.empty:
        errors.append("load_monthly_returns returned an empty frame")

    holdout = load_holdout_table()
    if len(holdout) != 2:
        errors.append(f"load_holdout_table has {len(holdout)} rows, expected 2")

    naive_mae, ols_mae, ratio = holdout_gap()
    # Structural checks only: the absolute numbers move every time the price
    # file is refreshed, but the relationship must not.
    if naive_mae <= 0 or not math.isfinite(naive_mae):
        errors.append(f"naive holdout MAE is {naive_mae!r}")
    if ols_mae <= naive_mae:
        errors.append(
            f"OLS holdout MAE {ols_mae:.2f} is not worse than naive {naive_mae:.2f}; "
            "the cautionary baseline has stopped being cautionary"
        )
    if not 5.0 < ratio < 200.0:
        errors.append(f"OLS/naive ratio is {ratio:.2f}, outside the expected range")

    table = load_walk_forward_table()
    if len(table) != len(DEFAULT_MODELS):
        errors.append(f"walk-forward table has {len(table)} rows")

    verdict = load_verdict()
    if verdict.has_edge:
        errors.append("verdict claims an edge that the evaluation does not support")

    predictions = load_predictions()
    if predictions.empty:
        errors.append("load_predictions returned an empty frame")

    for model in DEFAULT_MODELS:
        sigma = model_sigma(model)
        if not sigma > 0:
            errors.append(f"model_sigma({model!r}) = {sigma}")
        forecast = load_forecast(model, 30)
        if len(forecast) != 30:
            errors.append(f"load_forecast({model!r}, 30) returned {len(forecast)} rows")
        if not bool((forecast["lo80"] <= forecast["forecast"]).all()):
            errors.append(f"forecast band is inverted for {model!r}")

    display = metrics_display(table)
    if "MAE ($)" not in display.columns:
        errors.append("metrics_display did not rename the columns")

    return errors


def _check_page_imports() -> list[str]:
    """Import each page module and confirm it exposes ``render``."""
    errors: list[str] = []
    for relative in PAGES:
        path = APP_DIR / relative
        if not path.is_file():
            errors.append(f"missing page: {relative}")
            continue
        module = _load_module(f"smoke_{path.stem}", path)
        if not callable(getattr(module, "render", None)):
            errors.append(f"{relative} has no callable render()")
    return errors


def _check_date_filter() -> list[str]:
    """The sidebar range must actually change what the landing page reports."""
    from datetime import date

    from streamlit.testing.v1 import AppTest

    prices = load_prices_for_filter()
    start, end = date(2024, 1, 1), date(2024, 12, 31)
    expected = float(
        prices.loc[prices["date"].dt.date <= end, "close"].iloc[-1]
    )

    app = AppTest.from_file(str(APP_DIR / "Home.py"), default_timeout=120)
    app.run()
    app.date_input[0].set_value((start, end))
    app.run()

    errors: list[str] = []
    if app.exception:
        errors.append(f"date filter raised: {app.exception[0].message}")
        return errors

    reported = next(
        (metric.value for metric in app.metric if metric.label == "Gold price"), None
    )
    if reported is None:
        errors.append("date filter: 'Last close' metric disappeared")
    elif reported != f"${expected:,.2f}":
        errors.append(
            f"date filter: last close is {reported}, expected ${expected:,.2f}"
        )
    return errors


def load_prices_for_filter():
    """Load the committed prices without touching Streamlit state."""
    from src.data_cleaning import load_clean_data

    return load_clean_data()


def _check_app_test() -> tuple[list[str], float]:
    """Render every page with Streamlit's AppTest and collect exceptions."""
    from streamlit.testing.v1 import AppTest

    errors: list[str] = []
    started = time.perf_counter()

    app = AppTest.from_file(str(APP_DIR / "Home.py"), default_timeout=120)
    app.run()
    if app.exception:
        errors.append(f"Home.py raised: {app.exception[0].message}")
    elapsed = time.perf_counter() - started

    for relative in PAGES:
        page = AppTest.from_file(str(APP_DIR / "Home.py"), default_timeout=120)
        page.run()
        page.switch_page(relative)
        page.run()
        if page.exception:
            errors.append(f"{relative} raised: {page.exception[0].message}")
        if not page.title:
            errors.append(f"{relative} rendered no title")

    return errors, elapsed


def _quiet_streamlit() -> None:
    """Silence cache-miss warnings: this script calls cached loaders on purpose."""
    from streamlit import config, logger as streamlit_logger

    config.set_option("logger.level", "error")
    streamlit_logger.set_log_level("error")


def main() -> int:
    """Run all checks and report a single pass/fail line per layer."""
    _quiet_streamlit()
    failures: list[str] = []

    data_errors = _check_data_layer()
    print(f"data layer      : {'ok' if not data_errors else 'FAIL'}")
    failures.extend(data_errors)

    page_errors = _check_page_imports()
    print(f"page imports    : {'ok' if not page_errors else 'FAIL'}")
    failures.extend(page_errors)

    app_errors, elapsed = _check_app_test()
    print(f"app render      : {'ok' if not app_errors else 'FAIL'} ({elapsed:.2f}s to first paint)")
    failures.extend(app_errors)

    filter_errors = _check_date_filter()
    print(f"date filter     : {'ok' if not filter_errors else 'FAIL'}")
    failures.extend(filter_errors)

    if failures:
        print("\nfailures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nsmoke check: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

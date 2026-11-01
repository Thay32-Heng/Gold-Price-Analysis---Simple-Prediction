# Build plan — Gold Price Analysis (model + Streamlit app)

Repo: `/home/romeo/Projects/Gold-Price-Analysis---Simple-Prediction`
Python: use `.venv/bin/python` (streamlit 1.66.1, sklearn 1.9.1, plotly 7.1.0, statsmodels, yfinance)
**Never run `git commit` or `git push`.** Leave the working tree dirty for Thay to review.

## Part 1 — fix what is broken

| # | File | Bug | Fix |
|---|---|---|---|
| 1 | `notebooks/exploration.ipynb` cell 1 | reads `data/Process` (capital P); folder is `data/process` | use `data/process` |
| 2 | `notebooks/exploration.ipynb` cell 4 | `month_order` never defined → NameError | define `month_order` (calendar order) |
| 3 | `notebooks/exploration.ipynb` cell 3 | saves `graph2_customer_spending.png` | save `graph2_moving_avg.png` |
| 4 | `src/data_cleaning.py` | `skiprows=[1, 2]` is fragile; `data/raw` gitignored so it re-downloads | keep behaviour, make the yfinance MultiIndex flattening explicit and idempotent |
| 5 | `README.md` | says `data/processed/gold_cleaned.csv`; real path is `data/process/cleaned_gold_price_data.csv` | correct the path everywhere |
| 6 | `requirements.txt` | unpinned, missing streamlit/plotly | pin working versions |

Data facts (verified): 2795 rows, 2015-01-02 → 2026-02-13, columns `date,close,high,low,open,volume`, no nulls, close $1050.8 → $5318.4.

## Part 2 — the model (this is the point)

The current "model" is `LinearRegression` on `days_since_start` → `close`, fit and scored on the same rows.
Verified holdout result (last 20%, 559 days):

| Predictor | MAE | MAPE |
|---|---|---|
| OLS straight line | $885.7 | 25.8% |
| Naive (tomorrow = today) | $28.8 | 0.9% |

So OLS is **30.8× worse than doing nothing**. Keep OLS in the app *as the cautionary baseline*, do not delete it.

New `src/model.py` must provide:

1. `make_features(df)` — log returns, lagged returns (1,2,3,5,10), MA50/MA200 gap, rolling volatility (20d), RSI-ish or momentum (10d), day-of-week.
2. `make_target(df, horizon=1)` — forward log return over `horizon` days.
3. `walk_forward_eval(...)` — expanding-window walk-forward, refit every `step` bars, **never shuffle, never leak**.
4. Models to compare:
   - `NaiveLastPrice` (predict yesterday's close)
   - `DriftOLS` (the existing time-index OLS — the strawman)
   - `Ridge` on the feature set
   - `HistGradientBoosting` on the feature set (optional, only if it helps)
5. Metrics per model: MAE, RMSE, MAPE, and directional accuracy. Return a tidy DataFrame.
6. `forecast_future(df, model, horizon)` → point forecast + a residual-based 80% band.

Honesty rule: **do not tune anything on the test window, and do not hide a model that loses to naive.** The app's job is to show the comparison, not to fake an edge. If nothing beats naive at 1 day, say so in the UI.

## Part 3 — the Streamlit app

```
app/
  Home.py                 # landing: what this is, verified numbers, repo link, limitations
  lib/
    __init__.py
    data.py               # cached loaders, no side effects at import
    charts.py             # plotly builders
  pages/
    1_Analysis.py         # EDA: price, MA50/MA200, monthly volatility box, returns dist
    2_Model_Comparison.py # the walk-forward table + equity/error curves
    3_Forecast.py         # horizon slider, point forecast + band, "why this is hard" note
.streamlit/config.toml    # dark theme, gold accent
```

Requirements:
- `st.navigation` + `st.Page` (streamlit 1.66 API), not the old `pages/` auto-discovery alone.
- `st.cache_data` on every data load; the app must start in under 2 s.
- All charts plotly, no matplotlib in the app.
- Every page has a one-line takeaway in plain English.
- Sidebar: date-range slider that actually filters, and a ticker/asset note (`GC=F`).
- No hardcoded absolute paths — resolve from the repo root.
- Works with `streamlit run app/Home.py` from the repo root.

## Acceptance checks (must all pass before saying it is done)

1. `.venv/bin/python -m compileall app src` → clean.
2. `pyright app src` → no errors.
3. `streamlit run app/Home.py --server.headless true --server.port 8601` starts, and `curl -s localhost:8601/_stcore/health` returns `ok`.
4. A smoke script that imports each page module and calls its data functions returns without exception.
5. The walk-forward table in `src/model.py` reproduces the naive-vs-OLS gap above (within rounding) when run standalone.
6. No `print()` debug spam in the app; no notebook left broken (re-run it headless if practical).

## README

Rewrite the top of `README.md` with: what the app does, how to run it (`streamlit run app/Home.py`), the honest model comparison table, and the limitation that 1-day gold forecasting is near-random.

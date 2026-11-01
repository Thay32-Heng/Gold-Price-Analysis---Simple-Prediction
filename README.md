# Gold Price Analysis & Prediction

An analysis of COMEX gold futures (`GC=F`) that is honest about what a
time-series model can and cannot do. It ships a Streamlit app, a leakage-free
walk-forward evaluation, and a cautionary baseline: the original "prediction"
in this repository is a straight line that is **35.3× worse than doing
nothing**.

## What the app does

- **Market analysis** — trend, MA50/MA200 crossovers, monthly seasonality,
  return distribution, all filtered by a date range.
- **Model comparison** — the same models scored two ways: a single
  chronological holdout, and an expanding-window walk-forward with no
  shuffling and no leakage.
- **Forecast** — a point estimate and a residual-based 80% band for 1–60
  trading days ahead, with the reason the band is wide written next to it.

Every page states its takeaway in one sentence, including the uncomfortable
one: at a one-day horizon nothing here beats "tomorrow costs what it costs
today".

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app/Home.py
```

Then open <http://localhost:8501>. The app reads the committed CSV at
`data/process/cleaned_gold_price_data.csv`; it never downloads anything at
start-up.

Reproduce the numbers without the UI:

```bash
python -m src.model          # holdout audit + walk-forward table
python scripts/smoke_check.py # data, page imports, app render, date filter
```

## The honest model comparison

**Single chronological holdout** (fit on the first 80%, score the last 20% —
559 bars):

| Predictor | MAE | MAPE |
|---|---|---|
| Naive (tomorrow = today) | $40.16 | 1.05% |
| OLS straight line on `days_since_start` | $1,418.54 | 36.11% |

The OLS line is **35.3× worse** than the naive forecast. It is kept in the app
as a cautionary baseline, not as a model.

**Expanding-window walk-forward** (refit every 21 bars, 1,397 evaluated bars,
2020-07-27 → 2026-02-12):

| Model | MAE | RMSE | MAPE | Direction |
|---|---|---|---|---|
| Naive (last price) | $23.78 | $41.07 | 0.83% | — |
| Ridge on lagged returns, MA gap, volatility | $23.74 | $41.04 | 0.83% | 52.9% |
| HistGradientBoosting (same features) | $24.34 | $42.40 | 0.85% | 50.6% |
| OLS trend (strawman) | $467.15 | $741.14 | 13.49% | 47.3% |

Ridge "wins" by $0.04 per bar, which is noise. Its 52.9% directional accuracy
sits *below* the share of up-days in the window (53.9%), so it is not evidence
of skill either. **Verdict: no model beats naive at a 1-day horizon.** The
boosting model is excluded from the default comparison because it loses.

## Project structure

```
data/
  process/cleaned_gold_price_data.csv   # committed, 2,975 rows, 2015-01-02 → 2026-10-30
  create_data.py                        # raw download helper
notebooks/exploration.ipynb             # the original EDA and the OLS plot
src/
  data_cleaning.py                      # download + normalise yfinance layouts
  model.py                              # features, walk-forward, metrics, forecast
app/
  Home.py                               # entry point, sidebar shell, landing page
  lib/data.py                           # cached loaders (no I/O at import)
  lib/charts.py                         # plotly builders
  pages/1_Analysis.py
  pages/2_Model_Comparison.py
  pages/3_Forecast.py
images/                                 # figures referenced below
scripts/smoke_check.py                  # headless verification
.streamlit/config.toml                  # dark theme, gold accent
```

## Rebuilding the data

`data/raw/` is git-ignored. To regenerate the cleaned file from yfinance:

```bash
python -m src.data_cleaning
```

The loader normalises every yfinance layout it has produced over the years
(flat header, `MultiIndex` header, and the two metadata rows `Ticker`/`Date`),
and it is idempotent — running it twice changes nothing.

## Visualisations & insights

### 1. Historical Price Trend

![Historical Trend](images/graph1_trend.png)

- **Long-term Growth**: Identifies the overall upward trajectory of gold, confirming its status as a reliable safe-haven asset for long-term investors.

- **Market Events**: Highlights significant price spikes and drops that correlate with major global economic shifts and geopolitical events.

- **Insight:** Analyzes the long-term growth of gold as a safe-haven asset over the past decade.

### 2. Moving Average Analysis (MA50 vs MA200)

![Moving Average](images/graph2_moving_avg.png)

- **Market Momentum**: Detects "Golden Cross" (buy) and "Death Cross" (sell) signals when short-term trends cross long-term averages.

- **Trend Stability**: Evaluates whether the market is in a Bullish or Bearish state based on price positioning relative to the 50-day and 200-day averages.

- **Insight:** Identifies "Golden Cross" and "Death Cross" signals to determine market momentum.

### 3. Monthly Volatility Analysis

![Monthly Volatility](images/graph3_monthly_volatility_box.png)

- **Seasonal Patterns**: Pinpoints specific months that historically experience the highest price swings, helping investors time their entries.

- **Price Ranges**: Compares median prices across different months to identify seasonal buying opportunities or overvalued periods.

- **Insight:** Uses Box Plots to detect which months typically exhibit the highest price fluctuations.

### 4. Daily Returns Distribution

![Daily Returns](images/graph4_returns_dist.png)

- **Risk Assessment**: Analyzes the frequency of daily percentage changes to measure the volatility risk associated with holding gold.

- **Statistical Balance:**: Determines if price movements follow a normal distribution (Bell Curve), which indicates predictable market behavior.

- **Insight:** Visualizes the risk profile of gold. A normal distribution suggests stable daily price changes.

### 5. Actual vs. Predicted Price

![Prediction](images/graph5_regression.png)

- **Model Accuracy**: Evaluates how closely the Linear Regression model tracks actual historical movements to validate its predictive power.

- **Future Trajectory**: Uses the regression line ($y = mx + b$) to project the general direction of gold prices based on historical data.

- **Insight:** Evaluates the Linear Regression model performance in tracking actual price movements.
- **Model Equation:** $y = mx + b$

> This chart is the reason the rest of the project exists. The regression line
> looks reasonable in-sample because it is drawn through the same points it was
> fitted on. Scored on data it has never seen, it is 35.3× worse than
> predicting "tomorrow costs what it costs today".

## Limitations

- **One-day gold forecasting is close to random.** The walk-forward
  comparison is the evidence, not an opinion.
- **`GC=F` is a futures contract.** It carries roll effects and is quoted in
  USD per troy ounce; it is not the price you would pay at a coin shop.
- **The 80% band is wide by construction.** It comes from the model's own
  out-of-sample errors, widened by the square root of time.
- **Nothing here is investment advice.** The project exists to demonstrate how
  easily a time-series model can look clever and be useless.

"""Prove the walk-forward harness catches look-ahead leakage.

The bug this guards against is subtle and was present in this repository once:
the target for row ``j`` is ``log(close[j + horizon] / close[j])``, so a forecast
for ``close[t]`` has to be built from the feature row at ``t - horizon``. Using
the feature row at ``t`` puts the bar being predicted inside the model's input.

On gold the difference is invisible, because daily returns are close to
uncorrelated and the buggy forecast collapses onto the naive one. That is
exactly why the bug survived: the data could not expose it. So this script
runs the same harness on synthetic AR(1) series, where the leak is measurable,
and compares the fixed predictor against a faithful copy of the old one.

Run from the repository root::

    python scripts/leakage_check.py

The gate is deliberately about the *shape* of the result rather than one seed's
draw: over ``SEEDS`` shocks, the leak must be worth nothing on an uncorrelated
series and must be clearly worth something on an autocorrelated one. A single
seed cannot carry that claim -- the phi=0 effect alone swings by ~1.3 points
between seeds -- so the reported numbers are means with a standard error.

Exit code is 0 when the probe behaves, and 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.model as model  # noqa: E402

#: Return autocorrelation used for the probe. 0.0 is a random walk in levels;
#: 0.9 is strongly autocorrelated and is where a leak shows up loudly.
PHIS: tuple[float, ...] = (0.0, 0.5, 0.9)
ROWS = 1_200
SEEDS: tuple[int, ...] = tuple(range(1, 9))

#: The real predictor, captured before ``_predict_block`` is monkeypatched.
_ORIGINAL_BLOCK = model._predict_block  # noqa: SLF001

MODEL_NAME = "Ridge (features)"


def make_series(phi: float, seed: int, rows: int = ROWS) -> pd.DataFrame:
    """An AR(1) log-return series, as a price frame."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, 0.01, rows)
    returns = np.zeros(rows)
    for i in range(1, rows):
        returns[i] = phi * returns[i - 1] + shocks[i]
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {"date": pd.bdate_range("2015-01-01", periods=rows), "close": close}
    )


def _leaky_block(
    name: str,
    *,
    values: np.ndarray,
    days: np.ndarray,
    matrix: np.ndarray,
    target: np.ndarray,
    train_rows: np.ndarray,
    train_feature_rows: np.ndarray,
    test_rows: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """The pre-fix predictor: features from row ``t``, scale from row ``t - h``.

    Kept as a deliberate copy of the old code so the comparison is honest. The
    naive and OLS branches are delegated, since they never had the bug.
    """
    if name in (model.NAIVE, model.DRIFT_OLS):
        return _ORIGINAL_BLOCK(
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
    if train_feature_rows.size == 0:
        return np.full(test_rows.shape, np.nan)
    estimator = model._make_estimator(name).fit(  # noqa: SLF001
        matrix[train_feature_rows], target[train_feature_rows]
    )
    predicted_log_return = np.asarray(estimator.predict(matrix[test_rows]), dtype=float)
    return values[test_rows - horizon] * np.exp(predicted_log_return)


def skill_vs_naive(predictions: pd.DataFrame) -> tuple[float, float]:
    """Return ``(mae, improvement_pct)`` for the probe model against naive."""
    rows = predictions[predictions["model"] == MODEL_NAME]
    if rows.empty:
        raise LookupError(
            f"no rows for {MODEL_NAME!r} in the walk-forward output; the model "
            "label changed and the probe would compare nothing against nothing"
        )
    naive_mae = float(np.abs(rows["actual"] - rows["last_close"]).mean())
    mae = float(np.abs(rows["actual"] - rows["predicted"]).mean())
    if not np.isfinite(naive_mae) or not np.isfinite(mae) or naive_mae <= 0:
        raise ValueError(f"non-finite error terms: naive={naive_mae!r} mae={mae!r}")
    return mae, (1.0 - mae / naive_mae) * 100.0


def leak_effect(frame: pd.DataFrame) -> tuple[float, float, float]:
    """Return ``(fixed_skill, leaky_skill, effect)`` in percentage points."""
    try:
        model._predict_block = _ORIGINAL_BLOCK  # noqa: SLF001
        _, fixed_skill = skill_vs_naive(model.walk_forward_predictions(frame))

        model._predict_block = _leaky_block  # type: ignore[assignment]  # noqa: SLF001
        _, leaky_skill = skill_vs_naive(model.walk_forward_predictions(frame))
    finally:
        # A raise mid-probe must not leave the module patched: every later
        # caller in this process would silently run the leaky predictor.
        model._predict_block = _ORIGINAL_BLOCK  # noqa: SLF001
    return fixed_skill, leaky_skill, leaky_skill - fixed_skill


def main() -> int:
    """Run the probe and report the table."""
    print(
        f"{'phi':>5}  {'fixed skill':>12}  {'leaky skill':>12}  "
        f"{'leak effect':>12}  {'se':>6}"
    )
    print("-" * 58)

    effects: dict[float, float] = {}
    errors: dict[float, float] = {}
    for phi in PHIS:
        per_seed = [leak_effect(make_series(phi, seed)) for seed in SEEDS]
        paired = np.asarray(per_seed, dtype=float)
        if not np.all(np.isfinite(paired)):
            print(f"FAIL: phi={phi} produced a non-finite leak effect.")
            return 1
        fixed_skill = float(paired[:, 0].mean())
        leaky_skill = float(paired[:, 1].mean())
        effects[phi] = float(paired[:, 2].mean())
        errors[phi] = float(paired[:, 2].std(ddof=1) / np.sqrt(len(SEEDS)))
        if not np.isfinite(errors[phi]):
            # With one seed std(ddof=1) is NaN, and every comparison against a
            # NaN limit is False -- the gate would pass by not existing.
            print(
                f"FAIL: the standard error at phi={phi} is not finite; "
                f"the gate needs at least 2 seeds, got {len(SEEDS)}."
            )
            return 1
        print(
            f"{phi:5.1f}  {fixed_skill:+11.2f}%  {leaky_skill:+11.2f}%  "
            f"{effects[phi]:+11.2f}pts  {errors[phi]:6.2f}"
        )

    print()
    print(f"Means over {len(SEEDS)} seeds; se = standard error of the mean.")
    print(
        "A correct harness cannot beat naive on an uncorrelated series, and the "
        "leak must be worth nothing there."
    )

    # phi=0: the leak must be indistinguishable from zero. The band is a 99%
    # two-sided one (t(0.995, df = SEEDS-1) = 3.5 at n=8), plus a small
    # absolute cushion for rounding. It is worth being explicit about what
    # that buys: with ~0.4 points of standard error the gate would still miss
    # a leak smaller than about 1.7 points, so this proves "no big leak", not
    # "exactly zero".
    zero_limit = 3.5 * errors[0.0] + 0.25
    if abs(effects[0.0]) > zero_limit:
        print(
            f"FAIL: the leak moved phi=0 by {effects[0.0]:+.2f} points "
            f"(limit {zero_limit:.2f}); a leak-free harness should sit at zero."
        )
        return 1

    # phi=0.9: the probe must still be able to see a leak. Without this, a
    # broken probe would report "no leak" forever and always pass.
    if effects[0.9] < 2.0:
        print(
            f"FAIL: the probe cannot detect a known leak "
            f"(phi=0.9 moved only {effects[0.9]:+.2f} points), so a passing "
            "result would be meaningless."
        )
        return 1

    print("OK: no effect where there is nothing to leak, clear effect where there is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

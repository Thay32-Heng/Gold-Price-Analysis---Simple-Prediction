"""Download and clean the gold futures series (``GC=F``).

``yfinance`` returns a different frame shape depending on its version and on
how many tickers were requested: sometimes a flat
``Open/High/Low/Close/Volume`` header, sometimes a ``MultiIndex`` with the
field on level 0 and the ticker on level 1. When that frame is written to CSV,
two extra metadata rows (``Ticker`` and ``Date``) end up beneath the header.

:func:`load_clean_data` normalises every one of those layouts to the same six
columns, and it is idempotent: feeding it an already-clean file returns an
identical frame.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RAW_PATH: Path = PROJECT_ROOT / "data" / "raw" / "gold_price_data.csv"
DATA_PATH: Path = PROJECT_ROOT / "data" / "process" / "cleaned_gold_price_data.csv"

TICKER = "GC=F"
START_DATE = "2015-01-01"
#: ``None`` means "up to the most recent completed session", so the committed
#: CSV cannot silently go stale the way a hard-coded date does.
END_DATE: str | None = None

#: Canonical output order.
COLUMNS: tuple[str, ...] = ("date", "close", "high", "low", "open", "volume")

#: Column labels that carry the date rather than a price field.
_DATE_LABELS: tuple[str, ...] = ("date", "price", "datetime", "index", "unnamed:_0")

#: Values yfinance puts in the two metadata rows beneath the CSV header.
_METADATA_MARKERS: tuple[str, ...] = ("ticker", "date")


def _normalise_label(label: object) -> str:
    """Lower-case and snake-case a column label, including MultiIndex tuples."""
    if isinstance(label, tuple):
        label = label[0]
    return str(label).strip().lower().replace(" ", "_")


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse a yfinance ``MultiIndex`` header to its field level.

    A ``(field, ticker)`` MultiIndex becomes plain ``field`` labels. Duplicate
    labels (one per ticker) keep the first occurrence, which is correct for a
    single-ticker download.
    """
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame

    flattened = frame.copy()
    flattened.columns = [_normalise_label(column) for column in frame.columns]
    duplicated = flattened.columns.duplicated(keep="first")
    if duplicated.any():
        LOGGER.warning(
            "Dropping %d duplicate column(s) from a multi-ticker download",
            int(duplicated.sum()),
        )
        flattened = flattened.loc[:, ~duplicated]
    return flattened


def _drop_metadata_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove the ``Ticker`` / ``Date`` rows yfinance writes under the header.

    Detection is by content rather than row position, so an already-clean file
    (which has no metadata rows) passes through untouched.
    """
    if frame.empty:
        return frame

    is_metadata = np.zeros(len(frame), dtype=bool)
    for column in frame.columns:
        lowered = frame[column].astype(str).str.strip().str.lower()
        is_metadata |= lowered.isin(_METADATA_MARKERS).to_numpy(dtype=bool)
    if not is_metadata.any():
        return frame
    return frame.loc[~is_metadata].reset_index(drop=True)


def _normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce any supported raw layout into the canonical six-column frame."""
    if frame.empty:
        raise ValueError("the raw gold price frame is empty")

    cleaned = _flatten_columns(frame)
    cleaned.columns = [_normalise_label(column) for column in cleaned.columns]
    cleaned = _drop_metadata_rows(cleaned)

    if not any(column in _DATE_LABELS for column in cleaned.columns):
        # In-memory yfinance frames carry the date on the index instead.
        cleaned = cleaned.reset_index()
        cleaned.columns = [_normalise_label(column) for column in cleaned.columns]

    date_column = next(
        (column for column in cleaned.columns if column in _DATE_LABELS), None
    )
    if date_column is None:
        raise ValueError(f"no date column found: {list(cleaned.columns)}")
    if date_column != "date":
        cleaned = cleaned.rename(columns={date_column: "date"})

    missing = [column for column in COLUMNS if column not in cleaned.columns]
    if missing:
        raise ValueError(f"missing column(s): {missing}")

    cleaned = cleaned.loc[:, list(COLUMNS)].copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    for column in COLUMNS[1:]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    return (
        cleaned.dropna()
        .drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_clean_data(path: Path | None = None) -> pd.DataFrame:
    """Load the cleaned gold price CSV as a typed, date-sorted frame.

    Parameters
    ----------
    path:
        CSV to read. Defaults to :data:`DATA_PATH`.

    Returns
    -------
    pandas.DataFrame
        Columns ``date`` (datetime64), ``close``, ``high``, ``low``, ``open``
        (float64) and ``volume``, sorted oldest to newest with a clean
        ``RangeIndex``.

    Raises
    ------
    FileNotFoundError
        If the file is missing -- run ``python -m src.data_cleaning`` first.
    """
    csv_path = DATA_PATH if path is None else Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found; run `python -m src.data_cleaning` first"
        )
    return _normalise_frame(pd.read_csv(csv_path))


def _download_raw(raw_path: Path) -> pd.DataFrame:
    """Download ``GC=F`` and keep a copy of the raw frame on disk."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s from %s", TICKER, START_DATE)
    raw = yf.download(
        TICKER, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no rows for {TICKER}")
    raw.to_csv(raw_path)
    return raw


def clean_gold_data(
    *,
    raw_path: Path = RAW_PATH,
    output_path: Path = DATA_PATH,
    force_download: bool = False,
) -> pd.DataFrame:
    """Rebuild ``data/process/cleaned_gold_price_data.csv``.

    The raw download is git-ignored and is only fetched when it is missing (or
    when ``force_download`` is set). Cleaning goes through the same
    normalisation as :func:`load_clean_data`, so the raw layout never matters.
    """
    if force_download or not raw_path.exists():
        raw = _download_raw(raw_path)
        frame = _normalise_frame(raw)
    else:
        frame = _normalise_frame(pd.read_csv(raw_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    LOGGER.info("Cleaned %d rows -> %s", len(frame), output_path)
    return frame


def main() -> None:
    """CLI entry point: ``python -m src.data_cleaning``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    frame = clean_gold_data()
    print(
        f"{len(frame)} rows | {frame['date'].min():%Y-%m-%d} -> "
        f"{frame['date'].max():%Y-%m-%d} | close ${frame['close'].iloc[0]:,.2f} -> "
        f"${frame['close'].iloc[-1]:,.2f}"
    )


if __name__ == "__main__":
    main()

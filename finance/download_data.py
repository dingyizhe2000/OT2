"""Download finance data for the OT bull/bear pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DATE_END, DATE_START, RAW_DIR, TICKERS

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "yfinance is required. Install with: pip install yfinance"
    ) from exc


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _download_one_ticker(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted close for one ticker."""
    end_plus_one = (pd.Timestamp(end) + timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(
        ticker,
        start=start,
        end=end_plus_one,
        auto_adjust=False,
        progress=False,
        interval="1d",
    )
    if df.empty:
        raise ValueError(f"No data downloaded for ticker={ticker}.")

    if "Adj Close" in df.columns:
        series = df["Adj Close"].copy()
    elif "Close" in df.columns:
        series = df["Close"].copy()
    else:
        raise ValueError(f"Missing Adj Close/Close columns for ticker={ticker}.")

    series.name = ticker
    return series


def download_prices(start: str = DATE_START, end: str = DATE_END) -> pd.DataFrame:
    """Download all configured tickers and return a Date x Ticker price matrix."""
    frames = []
    for ticker in tqdm(TICKERS, desc="Downloading tickers", unit="ticker"):
        s = _download_one_ticker(ticker=ticker, start=start, end=end)
        frames.append(s)

    prices = pd.concat(frames, axis=1).sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices[TICKERS]
    prices = prices.dropna(how="all")
    return prices


def save_raw_artifacts(prices: pd.DataFrame) -> None:
    """Save raw price and return files."""
    close_path = RAW_DIR / "adj_close_prices.csv"
    returns_path = RAW_DIR / "log_returns.csv"
    metadata_path = RAW_DIR / "metadata.json"

    prices.to_csv(close_path, index_label="date")
    log_returns = np.log(prices / prices.shift(1)).dropna(how="all")
    log_returns.to_csv(returns_path, index_label="date")

    metadata = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date_start": DATE_START,
        "date_end": DATE_END,
        "tickers": TICKERS,
        "raw_files": [
            str(Path("data_finance") / "raw" / "adj_close_prices.csv"),
            str(Path("data_finance") / "raw" / "log_returns.csv"),
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))


def main() -> None:
    ensure_dirs()
    prices = download_prices()
    save_raw_artifacts(prices=prices)
    print(f"Saved raw files to: {RAW_DIR}")


if __name__ == "__main__":
    main()

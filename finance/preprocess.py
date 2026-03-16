"""Preprocess finance data using scaled raw log returns."""

from __future__ import annotations

import json
from typing import Tuple

import numpy as np
import pandas as pd

from config import (
    MARKET_TICKER,
    PROCESSED_CSV_DIR,
    PROCESSED_NPY_DIR,
    RAW_DIR,
    RETURN_SCALE,
    SPLITS,
    TICKERS,
)


def ensure_dirs() -> None:
    PROCESSED_CSV_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_NPY_DIR.mkdir(parents=True, exist_ok=True)


def clear_processed_outputs() -> None:
    """Remove all previous processed outputs."""
    for p in PROCESSED_CSV_DIR.glob("*"):
        if p.is_file():
            p.unlink()
    for p in PROCESSED_NPY_DIR.glob("*"):
        if p.is_file():
            p.unlink()


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    price_path = RAW_DIR / "adj_close_prices.csv"
    return_path = RAW_DIR / "log_returns.csv"
    if not price_path.exists() or not return_path.exists():
        raise FileNotFoundError(
            "Missing raw files. Run: python finance/download_data.py"
        )

    prices = pd.read_csv(price_path, parse_dates=["date"], index_col="date")
    log_returns = pd.read_csv(return_path, parse_dates=["date"], index_col="date")

    prices = prices[TICKERS].sort_index().apply(pd.to_numeric, errors="coerce")
    log_returns = log_returns[TICKERS].sort_index().apply(pd.to_numeric, errors="coerce")

    # Align to first date where all assets have valid returns.
    first_valid_dates = [log_returns[c].first_valid_index() for c in TICKERS]
    if any(dt is None for dt in first_valid_dates):
        missing_assets = [t for t, dt in zip(TICKERS, first_valid_dates) if dt is None]
        raise ValueError(f"No valid return history for assets: {missing_assets}")
    common_start = max(first_valid_dates)
    prices = prices.loc[common_start:].copy()
    log_returns = log_returns.loc[common_start:].copy()
    return prices, log_returns


def classify_market_state(market_price: pd.Series) -> pd.Series:
    """Classify each date as bull or bear using MA200 and 60-day return."""
    ma200 = market_price.rolling(window=200, min_periods=1).mean()
    ret60 = market_price.pct_change(60).fillna(0.0)
    bull = (market_price >= ma200) & (ret60 >= 0.0)
    state = pd.Series(np.where(bull, "bull", "bear"), index=market_price.index)
    state.name = "state"
    return state


def save_processed_outputs(log_returns: pd.DataFrame, state: pd.Series) -> None:
    scaled_returns = (log_returns * RETURN_SCALE).dropna(how="any")
    if scaled_returns.empty:
        raise ValueError("Scaled return matrix is empty.")

    full_df = scaled_returns.copy()
    aligned_state = state.reindex(full_df.index).fillna("bear")
    full_df.insert(0, "state", aligned_state.values)
    full_df.to_csv(PROCESSED_CSV_DIR / "full_scaled_log_returns.csv", index_label="date")

    (PROCESSED_NPY_DIR / "feature_columns.json").write_text(
        json.dumps({"columns": TICKERS}, indent=2)
    )

    train_start, train_end = SPLITS["train"]
    train_df = full_df.loc[train_start:train_end].copy()
    train_bull_df = train_df.loc[train_df["state"] == "bull"].copy()
    train_bear_df = train_df.loc[train_df["state"] == "bear"].copy()

    eval_specs = {
        "eval_2021_bull": ("2021-01-01", "2021-12-31", "bull"),
        "eval_2022_bear": ("2022-01-01", "2022-12-31", "bear"),
        "eval_2024_bull": ("2024-01-01", "2024-12-31", "bull"),
        "eval_2025_bear": ("2025-01-01", "2025-12-31", "bear"),
    }

    train_bull_df.to_csv(PROCESSED_CSV_DIR / "train_bull.csv", index_label="date")
    train_bear_df.to_csv(PROCESSED_CSV_DIR / "train_bear.csv", index_label="date")
    np.save(
        PROCESSED_NPY_DIR / "train_bull.npy",
        train_bull_df[TICKERS].to_numpy(dtype=np.float32),
    )
    np.save(
        PROCESSED_NPY_DIR / "train_bear.npy",
        train_bear_df[TICKERS].to_numpy(dtype=np.float32),
    )

    eval_counts = {}
    for name, (start, end, regime) in eval_specs.items():
        subset = full_df.loc[start:end].copy()
        subset = subset.loc[subset["state"] == regime].copy()
        subset.to_csv(PROCESSED_CSV_DIR / f"{name}.csv", index_label="date")
        np.save(
            PROCESSED_NPY_DIR / f"{name}.npy",
            subset[TICKERS].to_numpy(dtype=np.float32),
        )
        eval_counts[name] = int(subset.shape[0])

    summary = {
        "splits": SPLITS,
        "tickers": TICKERS,
        "market_ticker_for_state": MARKET_TICKER,
        "state_rule": "bull if close>=MA200 and 60d return>=0 else bear",
        "preprocess": {
            "method": "scaled_raw_log_returns",
            "return_scale": RETURN_SCALE,
        },
        "npy_feature_order": TICKERS,
        "row_count_full": int(full_df.shape[0]),
        "row_count_train": {
            "train_total": int(train_df.shape[0]),
            "train_bull": int(train_bull_df.shape[0]),
            "train_bear": int(train_bear_df.shape[0]),
        },
        "evaluation_sets_row_count": eval_counts,
        "export_policy": {
            "train": ["train_bull", "train_bear"],
            "eval": list(eval_specs.keys()),
        },
    }
    (PROCESSED_CSV_DIR / "processing_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


def main() -> None:
    ensure_dirs()
    clear_processed_outputs()
    prices, log_returns = load_raw_data()
    state = classify_market_state(prices[MARKET_TICKER])
    save_processed_outputs(log_returns=log_returns, state=state)
    print(f"Saved processed CSV files to: {PROCESSED_CSV_DIR}")
    print(f"Saved processed NPY files to: {PROCESSED_NPY_DIR}")


if __name__ == "__main__":
    main()

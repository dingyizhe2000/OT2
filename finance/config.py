"""Configuration for the finance data pipeline."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT_DIR / "data_finance" / "raw"
PROCESSED_CSV_DIR = ROOT_DIR / "data_finance" / "processed" / "csv"
PROCESSED_NPY_DIR = ROOT_DIR / "data_finance" / "processed" / "npy"

TICKERS = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "XLF",
    "XLE",
    "XLU",
    "TLT",
    "IEF",
    "LQD",
    "VNQ",
]

MARKET_TICKER = "SPY"
DATE_START = "2004-01-01"
DATE_END = "2025-12-31"

SPLITS = {
    "train": ("2004-01-01", "2020-12-31"),
}

RETURN_SCALE = 100.0

"""Run the end-to-end finance data pipeline."""

from __future__ import annotations

import argparse

from download_data import main as download_main
from preprocess import main as preprocess_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and preprocess finance data for OT bull/bear experiments."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step and reuse existing raw files.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_download:
        print("[1/2] Downloading raw data...")
        download_main()
    else:
        print("[1/2] Skipped download step.")

    if not args.skip_preprocess:
        print("[2/2] Processing data...")
        preprocess_main()
    else:
        print("[2/2] Skipped preprocessing step.")

    print("Pipeline finished.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf


def download_symbol(
    symbol: str,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    outdir: str = "data",
) -> Tuple[str, pd.DataFrame]:
    """Download historical data for ``symbol`` and save it as a CSV.

    Returns the path to the written CSV file and the resulting DataFrame.
    """
    os.makedirs(outdir, exist_ok=True)

    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        end_date = end or "today"
        raise SystemExit(
            f"No data found for symbol {symbol} between {start} and {end_date}"
        )

    df = df.rename(columns=str.title).reset_index()
    use_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in use_cols if col not in df.columns]
    if missing_cols:
        cols = ", ".join(missing_cols)
        raise SystemExit(f"Missing expected columns in download: {cols}")

    output_path = os.path.join(outdir, f"{symbol}.csv")
    df[use_cols].to_csv(output_path, index=False)
    return output_path, df[use_cols]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download historical market data for a single symbol.",
    )
    parser.add_argument("--symbol", type=str, required=True, help="Ticker symbol to download (e.g. AAPL).")
    parser.add_argument(
        "--start",
        type=str,
        default="2010-01-01",
        help="Inclusive start date for the download (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Exclusive end date for the download (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data",
        help="Directory where the CSV file will be written.",
    )
    args = parser.parse_args()

    output_path, df = download_symbol(args.symbol, start=args.start, end=args.end, outdir=args.outdir)
    print(f"Wrote {output_path} rows={len(df)}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import pandas as pd


def clean_csv(path_in: str, path_out: str | None = None) -> None:
    # Read raw lines so we can scrub quirky second-line tickers like ",Aapl,...".
    with open(path_in, "r", newline="") as handle:
        lines = handle.readlines()
    if len(lines) >= 2 and lines[1].strip().lower().startswith(",aapl"):
        lines = [lines[0]] + lines[2:]
        with open(path_in, "w", newline="") as handle:
            handle.writelines(lines)

    # Parse with pandas and coerce types.
    df = pd.read_csv(path_in)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.sort_values("Date")

    output = path_out or path_in
    df.to_csv(output, index=False)
    print(f"Cleaned CSV written to {output} rows={len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="path_in", required=True)
    parser.add_argument("--out", dest="path_out", default=None)
    arguments = parser.parse_args()
    clean_csv(arguments.path_in, arguments.path_out)
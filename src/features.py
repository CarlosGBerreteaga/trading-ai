from __future__ import annotations

import argparse
import os
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from ta.momentum import rsi
from ta.trend import ema_indicator, macd, macd_signal, sma_indicator
from ta.volatility import average_true_range

try:
    from .utils import ensure_ts_index
except ImportError:  # pragma: no cover
    from utils import ensure_ts_index


def _safe_read_csv(path: str) -> pd.DataFrame:
    """Read a CSV while scrubbing the stray second-line ticker if present."""
    with open(path, "r", newline="") as handle:
        lines = handle.readlines()
    if len(lines) >= 2 and lines[1].strip().startswith(","):
        lines = [lines[0]] + lines[2:]
        with open(path, "w", newline="") as handle:
            handle.writelines(lines)
    return pd.read_csv(path, parse_dates=["Date"])


def add_returns(
    df: pd.DataFrame,
    col: str = "Close",
    horizons: Iterable[int] = (1, 5, 10, 20, 63, 126, 252),
) -> pd.DataFrame:
    result = df.copy()
    for horizon in horizons:
        result[f"ret_{horizon}"] = result[col].pct_change(horizon)
    return result


def _long_horizon_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    volume = df["Volume"]

    daily_ret = close.pct_change()
    df["volatility_21"] = daily_ret.rolling(21).std()
    df["volatility_63"] = daily_ret.rolling(63).std()

    df["sma50"] = sma_indicator(close, window=50)
    df["sma200"] = sma_indicator(close, window=200)
    df["ema50"] = ema_indicator(close, window=50)
    df["ema200"] = ema_indicator(close, window=200)

    df["sma_ratio_50_200"] = (df["sma50"] / df["sma200"]).replace([np.inf, -np.inf], np.nan) - 1.0
    df["ema_ratio_50_200"] = (df["ema50"] / df["ema200"]).replace([np.inf, -np.inf], np.nan) - 1.0

    rolling_max_252 = close.rolling(252).max()
    rolling_min_252 = close.rolling(252).min()
    df["price_over_52w_high"] = (close / rolling_max_252) - 1.0
    df["price_over_52w_low"] = (close / rolling_min_252) - 1.0

    vol_mean_21 = volume.rolling(21).mean()
    vol_std_21 = volume.rolling(21).std()
    vol_mean_63 = volume.rolling(63).mean()
    vol_std_63 = volume.rolling(63).std()
    df["volume_zscore_21"] = (volume - vol_mean_21) / vol_std_21
    df["volume_zscore_63"] = (volume - vol_mean_63) / vol_std_63

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_ts_index(df, "Date")

    df = add_returns(df, "Close", horizons=(1, 5, 10, 20, 63, 126, 252))

    df["rsi14"] = rsi(df["Close"], window=14)
    df["macd"] = macd(df["Close"], window_slow=26, window_fast=12)
    df["macd_signal"] = macd_signal(df["Close"], window_slow=26, window_fast=12)
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["sma10"] = sma_indicator(df["Close"], window=10)
    df["sma20"] = sma_indicator(df["Close"], window=20)
    df["ema10"] = ema_indicator(df["Close"], window=10)
    df["ema20"] = ema_indicator(df["Close"], window=20)
    df["sma_diff"] = (df["sma10"] - df["sma20"]) / df["Close"]
    df["ema_diff"] = (df["ema10"] - df["ema20"]) / df["Close"]

    df["atr14"] = average_true_range(df["High"], df["Low"], df["Close"], window=14)
    df["atr_pct"] = df["atr14"] / df["Close"]

    df = _long_horizon_features(df)

    idx = df.index
    df["dow"] = idx.dayofweek
    df["dom"] = idx.day
    df["moy"] = idx.month

    df["y"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return df.dropna()


def generate_features(
    symbol: str,
    indir: str = "data",
    outdir: str = "data",
) -> Tuple[str, pd.DataFrame]:
    """Create feature parquet for ``symbol``.

    Returns the output path and the resulting feature DataFrame.
    """
    in_csv = os.path.join(indir, f"{symbol}.csv")
    if not os.path.exists(in_csv):
        raise SystemExit(f"Missing {in_csv}. Run data_download first.")

    raw = _safe_read_csv(in_csv)
    features = build_features(raw)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{symbol}_features.parquet")
    features.to_parquet(out_path, index=True)
    return out_path, features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--indir", type=str, default="data")
    parser.add_argument("--outdir", type=str, default="data")
    args = parser.parse_args()

    out_path, features = generate_features(args.symbol, indir=args.indir, outdir=args.outdir)
    print(f"Wrote {out_path} rows={len(features)} cols={len(features.columns)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests

try:
    from .pipeline import run_pipeline
except ImportError:  # pragma: no cover
    from pipeline import run_pipeline  # type: ignore

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _sanitize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    # Yahoo! Finance uses '-' in place of '.' (e.g., BRK.B -> BRK-B)
    return symbol.replace(".", "-")


def fetch_sp500_tickers(source: str = WIKI_URL) -> List[str]:
    resp = requests.get(source, headers={"User-Agent": "Mozilla/5.0 (compatible; trading-ai/1.0)"}, timeout=10)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)
    if not tables:
        raise RuntimeError("No tables found when scraping S&P 500 constituents.")
    df = tables[0]
    if "Symbol" not in df.columns:
        raise RuntimeError("Unexpected table format; 'Symbol' column not found.")
    return [_sanitize_symbol(sym) for sym in df["Symbol"].tolist()]


def load_tickers(ticker_file: Path | None, limit: int | None) -> List[str]:
    if ticker_file and ticker_file.exists():
        raw = ticker_file.read_text()
        had_backtick_newlines = "`n" in raw
        if had_backtick_newlines:
            raw = raw.replace("`n", "\n")
        tickers = [
            _sanitize_symbol(line)
            for line in raw.splitlines()
            if line.strip()
        ]
        if had_backtick_newlines:
            ticker_file.write_text("\n".join(tickers), encoding="utf-8")
    else:
        tickers = fetch_sp500_tickers()
        if ticker_file:
            ticker_file.write_text("\n".join(tickers), encoding="utf-8")
    if limit is not None and limit > 0:
        tickers = tickers[:limit]
    return tickers


def evaluate_tickers(
    symbols: Iterable[str],
    years: int,
    prob_threshold: float,
    min_hold_days: int,
    data_dir: str,
    models_dir: str,
    min_trading_days: int,
) -> pd.DataFrame:
    records: List[dict] = []
    for idx, symbol in enumerate(symbols, start=1):
        print(f"[{idx}] Evaluating {symbol}...", flush=True)
        try:
            summary = run_pipeline(
                symbol=symbol,
                years=years,
                prob_threshold=prob_threshold,
                min_hold_days=min_hold_days,
                data_dir=data_dir,
                models_dir=models_dir,
                ntfy_topic=None,
            )
        except BaseException as exc:  # pragma: no cover - runtime feedback only
            if isinstance(exc, KeyboardInterrupt):
                raise
            print(f"    ! Skipping {symbol}: {exc}", file=sys.stderr)
            records.append(
                {
                    "symbol": symbol,
                    "status": f"error: {exc}",
                }
            )
            continue

        trading_days = int(summary.get("trading_days") or 0)
        if min_trading_days > 0 and trading_days < min_trading_days:
            records.append(
                {
                    "symbol": symbol,
                    "status": f"filtered_short_window (<{min_trading_days})",
                    "trading_days": trading_days,
                }
            )
            continue

        stats = summary["stats"]
        latest_signal = summary.get("latest_signal") or {}
        records.append(
            {
                "symbol": symbol,
                "status": "ok",
                "equity": float(summary["total_return"] + 1.0),
                "total_return_pct": summary["total_return"] * 100.0,
                "CAGR": stats.get("CAGR"),
                "Sharpe": stats.get("Sharpe"),
                "MaxDD": stats.get("MaxDD"),
                "Vol": stats.get("Vol"),
                "backtest_csv": summary.get("backtest_csv"),
                "alerts_csv": summary.get("alerts_csv"),
                "trading_days": trading_days,
                "latest_action": str(latest_signal.get("action") or "").upper(),
                "latest_action_date": latest_signal.get("date"),
                "latest_probability": latest_signal.get("probability"),
                "latest_signal": latest_signal.get("signal"),
                "previous_signal": latest_signal.get("previous_signal"),
            }
        )
    df = pd.DataFrame(records)
    df = df.sort_values(by=["status", "CAGR", "Sharpe"], ascending=[True, False, False])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape S&P 500 tickers, run the long-hold pipeline, and rank recommendations.",
    )
    parser.add_argument("--years", type=int, default=20, help="Historical window length for downloads.")
    parser.add_argument("--prob-threshold", type=float, default=0.55, help="Probability threshold for bullish signal.")
    parser.add_argument("--min-hold-days", type=int, default=252, help="Minimum holding period in trading days.")
    parser.add_argument("--top", type=int, default=10, help="Number of top tickers to display.")
    parser.add_argument("--max-tickers", type=int, default=25, help="Maximum tickers to evaluate (avoid 500 at once).")
    parser.add_argument("--data", default="data", help="Data directory for CSV/parquet outputs.")
    parser.add_argument("--models", default="models", help="Model directory.")
    parser.add_argument(
        "--min-trading-days",
        type=int,
        default=504,
        help="Skip recommendations whose backtest covers fewer trading days than this threshold (default: 504).",
    )
    parser.add_argument(
        "--ticker-cache",
        type=Path,
        default=Path("sp500_tickers.txt"),
        help="Optional cache file for ticker list (created if absent).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_recommendations.csv"),
        help="Where to save the aggregated results.",
    )
    args = parser.parse_args()

    tickers = load_tickers(args.ticker_cache, args.max_tickers)
    print(f"Loaded {len(tickers)} tickers to evaluate.")

    results = evaluate_tickers(
        symbols=tickers,
        years=args.years,
        prob_threshold=args.prob_threshold,
        min_hold_days=args.min_hold_days,
        data_dir=args.data,
        models_dir=args.models,
        min_trading_days=args.min_trading_days,
    )

    results.to_csv(args.output, index=False)
    print(f"Saved full results to {args.output}")

    ok_results = results[results["status"] == "ok"].copy()
    if ok_results.empty:
        print("No successful backtests to recommend.")
        return

    ok_results = ok_results.sort_values(by=["CAGR", "Sharpe"], ascending=False)
    top_k = ok_results.head(args.top)

    print("\nTop recommendations:")
    print(
        top_k[
            [
                "symbol",
                "equity",
                "total_return_pct",
                "CAGR",
                "Sharpe",
                "MaxDD",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:0.4f}")
    )


if __name__ == "__main__":
    main()





from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    from .portfolio import DEFAULT_PORTFOLIO_PATH, PortfolioState
    from .recommend import evaluate_tickers, load_tickers
except ImportError:  # pragma: no cover
    from portfolio import DEFAULT_PORTFOLIO_PATH, PortfolioState  # type: ignore[no-redef]
    from recommend import evaluate_tickers, load_tickers  # type: ignore[no-redef]

TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate portfolio updates based on the latest recommendation sweep.",
    )
    parser.add_argument("--years", type=int, default=20, help="Historical lookback window for each pipeline run.")
    parser.add_argument("--prob-threshold", type=float, default=0.55, help="Probability threshold for buy signals.")
    parser.add_argument("--min-hold-days", type=int, default=252, help="Minimum holding period enforced in backtests.")
    parser.add_argument("--max-tickers", type=int, default=200, help="Maximum tickers to evaluate from the S&P universe.")
    parser.add_argument("--top", type=int, default=25, help="Number of ideas to keep in the portfolio.")
    parser.add_argument("--data", type=Path, default=Path("data"), help="Directory for downloaded data.")
    parser.add_argument("--models", type=Path, default=Path("models"), help="Directory for trained models.")
    parser.add_argument(
        "--ticker-cache",
        type=Path,
        default=Path("sp500_tickers.txt"),
        help="Optional cache of S&P 500 constituents.",
    )
    parser.add_argument(
        "--portfolio",
        type=Path,
        default=DEFAULT_PORTFOLIO_PATH,
        help="Portfolio JSON tracking owned tickers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview trades without modifying the portfolio file.",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=Path("data") / "auto_trades",
        help="Directory where trade summaries should be written.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep current positions even if they fall outside the top set (buys are additive).",
    )
    return parser.parse_args()


def summarise_expected_returns(df: pd.DataFrame) -> Dict[str, float]:
    numeric_cols = ["CAGR", "Sharpe", "MaxDD", "total_return_pct"]
    summary: Dict[str, float] = {}
    for col in numeric_cols:
        if col in df.columns and not df[col].dropna().empty:
            summary[f"{col}_mean"] = float(df[col].dropna().mean())
            summary[f"{col}_median"] = float(df[col].dropna().median())
    return summary


def build_trades(
    current: List[str],
    recommended: List[str],
    keep_existing: bool,
) -> Tuple[List[str], List[str], List[str]]:
    current_set = {sym.upper() for sym in current}
    recommended_set = {sym.upper() for sym in recommended}
    if keep_existing:
        new_set = current_set | recommended_set
    else:
        new_set = recommended_set

    buys = sorted(new_set - current_set)
    holds = sorted(new_set & current_set)
    sells = sorted(current_set - new_set)
    return buys, holds, sells


def export_summary(
    export_dir: Path,
    trades: Dict[str, List[str]],
    top_df: pd.DataFrame,
    portfolio_path: Path,
) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime(TIMESTAMP_FMT)
    out_path = export_dir / f"auto_trades_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "top": top_df.to_dict(orient="records"),
        "trades": trades,
        "portfolio": str(portfolio_path),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    portfolio = PortfolioState.load(args.portfolio)
    current_holdings = portfolio.list_owned()

    tickers = load_tickers(args.ticker_cache, args.max_tickers)
    results = evaluate_tickers(
        symbols=tickers,
        years=args.years,
        prob_threshold=args.prob_threshold,
        min_hold_days=args.min_hold_days,
        data_dir=str(args.data),
        models_dir=str(args.models),
    )

    successful = results[results["status"] == "ok"].copy()
    if successful.empty:
        print("No successful backtests; portfolio unchanged.")
        return

    successful = successful.sort_values(by=["CAGR", "Sharpe"], ascending=[False, False])
    top_df = successful.head(args.top).reset_index(drop=True)
    recommended_symbols = top_df["symbol"].tolist()

    buys, holds, sells = build_trades(current_holdings, recommended_symbols, args.keep_existing)
    trades = {"buy": buys, "hold": holds, "sell": sells}

    expected = summarise_expected_returns(top_df)

    print("=== Auto Trade Summary ===")
    print(f"Existing holdings ({len(current_holdings)}): {', '.join(current_holdings) or '(none)'}")
    print(f"Recommended top {len(recommended_symbols)}: {', '.join(recommended_symbols)}")
    print(f"Buys ({len(buys)}): {', '.join(buys) or '(none)'}")
    print(f"Holds ({len(holds)}): {', '.join(holds) or '(none)'}")
    print(f"Sells ({len(sells)}): {', '.join(sells) or '(none)'}")
    if expected:
        print("\nExpected performance (historical backtest aggregates):")
        for key, value in expected.items():
            print(f"  {key}: {value:.4f}")

    summary_path = export_summary(args.export, trades, top_df, args.portfolio)
    print(f"\nSaved trade summary to {summary_path}")

    if args.dry_run:
        print("Dry run enabled; portfolio.json not modified.")
        return

    if args.keep_existing:
        portfolio.add(buys)
    else:
        portfolio.set_owned(holds + buys)
    portfolio.save()
    print(f"Updated portfolio file: {portfolio.path}")


if __name__ == "__main__":
    main()

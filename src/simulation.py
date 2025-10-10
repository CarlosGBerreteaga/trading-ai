from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:  # pragma: no cover
    from .portfolio import DEFAULT_PORTFOLIO_PATH
    from .recommend import evaluate_tickers, load_tickers
except ImportError:  # pragma: no cover
    from portfolio import DEFAULT_PORTFOLIO_PATH  # type: ignore[no-redef]
    from recommend import evaluate_tickers, load_tickers  # type: ignore[no-redef]


TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: float
    profit: float

    @property
    def return_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price * 100.0

    @property
    def hold_days(self) -> int:
        return max((self.exit_date - self.entry_date).days, 0)


def _read_csv(path: Path, index_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if index_col not in df.columns:
        raise FileNotFoundError(f"Expected column '{index_col}' in {path}")
    df[index_col] = pd.to_datetime(df[index_col])
    df = df.set_index(index_col).sort_index()
    return df


def simulate_symbol(symbol: str, capital: float, data_dir: Path) -> Dict[str, object]:
    alerts_path = data_dir / f"{symbol}_alerts.csv"
    backtest_path = data_dir / f"{symbol}_backtest.csv"

    if not alerts_path.exists() or not backtest_path.exists():
        raise FileNotFoundError(f"Missing backtest outputs for {symbol}. Run pipeline first.")

    alerts = _read_csv(alerts_path, "Date")
    backtest = _read_csv(backtest_path, "Date")

    base_capital = float(capital)
    cash = float(capital)
    shares = 0.0
    entry_price: Optional[float] = None
    entry_date: Optional[pd.Timestamp] = None
    trades: List[Trade] = []

    for idx, row in alerts.iterrows():
        action = str(row["action"]).upper()
        price = float(row["close"])
        if action == "BUY" and shares == 0.0:
            if price <= 0:
                continue
            shares = cash / price
            entry_price = price
            entry_date = idx
            cash = 0.0
        elif action == "SELL" and shares > 0.0 and entry_price is not None and entry_date is not None:
            proceeds = shares * price
            profit = proceeds - shares * entry_price
            cash = proceeds
            trades.append(
                Trade(
                    symbol=symbol,
                    entry_date=entry_date.to_pydatetime(),
                    exit_date=idx.to_pydatetime(),
                    entry_price=entry_price,
                    exit_price=price,
                    shares=shares,
                    profit=profit,
                )
            )
            shares = 0.0
            entry_price = None
            entry_date = None

    last_price = None
    price_history_path = data_dir / f"{symbol}.csv"
    if price_history_path.exists():
        prices = _read_csv(price_history_path, "Date")
        close_candidates = [col for col in prices.columns if col.lower() == "close"]
        close_col = close_candidates[0] if close_candidates else prices.columns[0]
        last_price = float(prices[close_col].ffill().iloc[-1])
    if last_price is None:
        last_price = float(alerts["close"].iloc[-1]) if not alerts.empty else 0.0

    final_value = cash + shares * last_price

    if shares > 0.0 and entry_price is not None and entry_date is not None:
        # Close the final open position at the latest available close.
        profit = final_value - shares * entry_price
        trades.append(
            Trade(
                symbol=symbol,
                entry_date=entry_date.to_pydatetime(),
                exit_date=backtest.index[-1].to_pydatetime(),
                entry_price=entry_price,
                exit_price=last_price,
                shares=shares,
                profit=profit,
            )
        )
        shares = 0.0
        cash = final_value

    equity_curve = backtest["equity"].astype(float) * base_capital
    if len(equity_curve) > 600:
        step = max(1, len(equity_curve) // 600)
        equity_curve = equity_curve.iloc[::step]

    trade_dicts = [
        {
            "symbol": t.symbol,
            "entry_date": t.entry_date.isoformat(),
            "exit_date": t.exit_date.isoformat(),
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "shares": t.shares,
            "profit": t.profit,
            "return_pct": t.return_pct,
            "hold_days": t.hold_days,
        }
        for t in trades
    ]

    return {
        "symbol": symbol,
        "initial_capital": capital,
        "final_capital": final_value,
        "equity_curve": [
            {"date": ts.isoformat(), "value": float(val)} for ts, val in equity_curve.items()
        ],
        "trades": trade_dicts,
    }


def run_simulation(
    *,
    initial_capital: float = 1000.0,
    years: int = 20,
    prob_threshold: float = 0.55,
    min_hold_days: int = 252,
    max_tickers: int = 200,
    top: int = 5,
    min_trading_days: int = 0,
    data_dir: Path = Path("data"),
    models_dir: Path = Path("models"),
    ticker_cache: Path = Path("sp500_tickers.txt"),
) -> Dict[str, object]:
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    if years <= 0:
        raise ValueError("years must be positive.")
    if max_tickers <= 0:
        raise ValueError("max_tickers must be positive.")
    if top <= 0:
        raise ValueError("top must be positive.")

    tickers = load_tickers(ticker_cache, max_tickers)
    results = evaluate_tickers(
        symbols=tickers,
        years=years,
        prob_threshold=prob_threshold,
        min_hold_days=min_hold_days,
        data_dir=str(data_dir),
        models_dir=str(models_dir),
        min_trading_days=min_trading_days,
    )

    if results.empty:
        raise RuntimeError("No tickers were evaluated successfully.")

    success = results[results["status"] == "ok"].copy()
    if success.empty:
        raise RuntimeError("All backtests failed; cannot simulate.")

    success["final_value_est"] = initial_capital * (success["total_return_pct"] / 100.0 + 1.0)
    best_row = success.sort_values(by=["final_value_est", "CAGR", "Sharpe"], ascending=[False, False, False]).iloc[0]
    best_symbol = str(best_row["symbol"])

    sim = simulate_symbol(best_symbol, initial_capital, Path(data_dir))
    final_value = float(sim["final_capital"])
    all_trades: List[Dict[str, object]] = list(sim["trades"])

    per_symbol_summary: List[Dict[str, object]] = [
        {
            "symbol": best_symbol,
            "initial": initial_capital,
            "final": final_value,
            "return_pct": (final_value / initial_capital - 1.0) * 100.0,
            "CAGR": float(best_row.get("CAGR")) if pd.notna(best_row.get("CAGR")) else None,
            "Sharpe": float(best_row.get("Sharpe")) if pd.notna(best_row.get("Sharpe")) else None,
            "MaxDD": float(best_row.get("MaxDD")) if pd.notna(best_row.get("MaxDD")) else None,
            "alerts_csv": best_row.get("alerts_csv"),
            "backtest_csv": best_row.get("backtest_csv"),
        }
    ]

    trade_df = pd.DataFrame(all_trades)
    if not trade_df.empty:
        trade_df["profit"] = trade_df["profit"].astype(float)
        trade_df["hold_days"] = trade_df["hold_days"].astype(float)
        total_profit = trade_df["profit"].sum()
        winning = trade_df[trade_df["profit"] > 0]
        losing = trade_df[trade_df["profit"] <= 0]
        trade_summary = {
            "count": int(len(trade_df)),
            "wins": int(len(winning)),
            "losses": int(len(losing)),
            "win_rate": float(len(winning) / len(trade_df)) * 100.0,
            "avg_win": float(winning["profit"].mean()) if not winning.empty else 0.0,
            "avg_loss": float(losing["profit"].mean()) if not losing.empty else 0.0,
            "total_profit": float(total_profit),
            "avg_hold_days": float(trade_df["hold_days"].mean()),
            "median_hold_days": float(trade_df["hold_days"].median()),
            "max_hold_days": float(trade_df["hold_days"].max()),
        }
    else:
        trade_summary = {
            "count": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_profit": 0.0,
            "avg_hold_days": 0.0,
            "median_hold_days": 0.0,
            "max_hold_days": 0.0,
        }

    return {
        "initial_capital": initial_capital,
        "final_capital": final_value,
        "return_pct": (final_value / initial_capital - 1.0) * 100.0,
        "selected_symbol": best_symbol,
        "per_symbol": per_symbol_summary,
        "trade_summary": trade_summary,
        "trades": all_trades,
        "leaderboard": [best_row.to_dict()],
    }


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate capital allocation across top ML-driven trades.")
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--prob-threshold", type=float, default=0.55)
    parser.add_argument("--min-hold-days", type=int, default=252)
    parser.add_argument("--max-tickers", type=int, default=200)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--min-trading-days", type=int, default=0,
                        help="Discard strategies whose backtest spans fewer trading days than this threshold (default 0).")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--ticker-cache", type=Path, default=Path("sp500_tickers.txt"))
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON file to store the simulation summary.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    summary = run_simulation(
        initial_capital=args.initial_capital,
        years=args.years,
        prob_threshold=args.prob_threshold,
        min_hold_days=args.min_hold_days,
        max_tickers=args.max_tickers,
        top=args.top,
        data_dir=args.data,
        models_dir=args.models,
        ticker_cache=args.ticker_cache,
        min_trading_days=args.min_trading_days,
    )
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from typing import Dict, Optional

import pandas as pd

try:
    from .backtest import run_backtest
    from .data_download import download_symbol
    from .features import generate_features
    from .train import train_model
except ImportError:  # pragma: no cover
    from backtest import run_backtest
    from data_download import download_symbol
    from features import generate_features
    from train import train_model


def _compute_date_range(years: int) -> Dict[str, str]:
    today = pd.Timestamp.today().normalize()
    start = today - pd.DateOffset(years=years)
    return {"start": str(start.date()), "end": str(today.date())}


def run_pipeline(
    symbol: str = "SPY",
    years: int = 5,
    data_dir: str = "data",
    models_dir: str = "models",
    prob_threshold: float = 0.55,
    min_hold_days: int = 21,
    hedge_symbol: Optional[str] = None,
    hedge_weight: float = 1.0,
    notify_phone: Optional[str] = None,
    notify_from: Optional[str] = None,
    notify_limit: int = 5,
    twilio_sid: Optional[str] = None,
    twilio_token: Optional[str] = None,
    ntfy_topic: Optional[str] = None,
    ntfy_limit: int = 5,
    ntfy_server: str = "https://ntfy.sh",
    ntfy_token: Optional[str] = None,
    ntfy_title: str = "Trading Alert",
    ntfy_priority: Optional[str] = None,
) -> Dict[str, object]:
    if years <= 0:
        raise ValueError("years must be positive")

    date_range = _compute_date_range(years)
    start, end = date_range["start"], date_range["end"]

    print(f"Downloading {symbol} from {start} to {end}...")
    download_symbol(symbol, start=start, end=end, outdir=data_dir)
    if hedge_symbol:
        print(f"Downloading {hedge_symbol} from {start} to {end}...")
        download_symbol(hedge_symbol, start=start, end=end, outdir=data_dir)

    print(f"Building features for {symbol}...")
    generate_features(symbol, indir=data_dir, outdir=data_dir)

    print(f"Training model for {symbol}...")
    train_result = train_model(symbol, data_dir=data_dir, model_out=models_dir)

    print(f"Running backtest for {symbol} from {start} to {end}...")
    backtest_result = run_backtest(
        symbol,
        data_dir=data_dir,
        model_path=train_result["model_path"],
        prob_threshold=prob_threshold,
        start=start,
        end=end,
        min_hold_days=min_hold_days,
        hedge_symbol=hedge_symbol,
        hedge_weight=hedge_weight,
        notify_phone=notify_phone,
        notify_from=notify_from,
        notify_limit=notify_limit,
        twilio_sid=twilio_sid,
        twilio_token=twilio_token,
        ntfy_topic=ntfy_topic,
        ntfy_limit=ntfy_limit,
        ntfy_server=ntfy_server,
        ntfy_token=ntfy_token,
        ntfy_title=ntfy_title,
        ntfy_priority=ntfy_priority,
    )

    equity = backtest_result["equity"]
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0

    summary = {
        "total_return": total_return,
        "stats": backtest_result["stats"],
        "splits": train_result["splits"],
        "train_metrics": train_result["metrics"],
        "date_range": date_range,
        "model_path": train_result["model_path"],
        "backtest_csv": backtest_result["csv_path"]
,        "alerts_csv": backtest_result["alerts_path"],
        "alert_sids": backtest_result["alert_sids"],
        "notification_error": backtest_result["notification_error"],
        "ntfy_status": backtest_result["ntfy_status"],
        "ntfy_error": backtest_result["ntfy_error"],
    }

    print("\n=== Pipeline Summary ===")
    print(f"Data window: {start} to {end}")
    print(f"Model path: {summary['model_path']}")
    print(f"Backtest CSV: {summary['backtest_csv']}")
    print(f"Alerts CSV: {summary['alerts_csv']}")
    if notify_phone:
        if summary["alert_sids"]:
            print(f"SMS alerts sent: {len(summary['alert_sids'])}")
        if summary["notification_error"]:
            print(f"SMS warning: {summary['notification_error']}")
    if ntfy_topic:
        if summary["ntfy_status"]:
            print(f"ntfy alerts posted: {len(summary['ntfy_status'])}")
        if summary["ntfy_error"]:
            print(f"ntfy warning: {summary['ntfy_error']}")
    print(f"Total return: {summary['total_return']*100:.2f}%")
    print("Stats:")
    for key, value in summary["stats"].items():
        print(f"  {key}: {value:.4f}")

    alerts_preview = backtest_result["alerts"].head()
    if len(alerts_preview):
        print("\nNext alerts:")
        for idx, row in alerts_preview.iterrows():
            date_str = pd.Timestamp(idx).date()
            print(
                f"  {date_str}: {row['action']} @ {row['close']:.2f} (prob={row['proba']:.3f})"
            )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: download, feature engineering, training, and backtest.",
    )
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--data", default="data")
    parser.add_argument("--models", default="models")
    parser.add_argument("--prob-threshold", type=float, default=0.55)
    parser.add_argument("--min-hold-days", type=int, default=21)
    parser.add_argument("--hedge-symbol", type=str, default=None)
    parser.add_argument("--hedge-weight", type=float, default=1.0)
    parser.add_argument("--notify-phone", type=str, default=None)
    parser.add_argument("--notify-from", type=str, default=None)
    parser.add_argument("--notify-limit", type=int, default=5)
    parser.add_argument("--twilio-sid", type=str, default=None)
    parser.add_argument("--twilio-token", type=str, default=None)
    parser.add_argument("--ntfy-topic", type=str, default=None)
    parser.add_argument("--ntfy-limit", type=int, default=5)
    parser.add_argument("--ntfy-server", type=str, default="https://ntfy.sh")
    parser.add_argument("--ntfy-token", type=str, default=None)
    parser.add_argument("--ntfy-title", type=str, default="Trading Alert")
    parser.add_argument("--ntfy-priority", type=str, default=None)
    args = parser.parse_args()

    run_pipeline(
        symbol=args.symbol,
        years=args.years,
        data_dir=args.data,
        models_dir=args.models,
        prob_threshold=args.prob_threshold,
        min_hold_days=args.min_hold_days,
        hedge_symbol=args.hedge_symbol,
        hedge_weight=args.hedge_weight,
        notify_phone=args.notify_phone,
        notify_from=args.notify_from,
        notify_limit=args.notify_limit,
        twilio_sid=args.twilio_sid,
        twilio_token=args.twilio_token,
        ntfy_topic=args.ntfy_topic,
        ntfy_limit=args.ntfy_limit,
        ntfy_server=args.ntfy_server,
        ntfy_token=args.ntfy_token,
        ntfy_title=args.ntfy_title,
        ntfy_priority=args.ntfy_priority,
    )


if __name__ == "__main__":
    main()

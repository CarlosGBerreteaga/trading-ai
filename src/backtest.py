from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

try:
    from .utils import ensure_ts_index
except ImportError:  # pragma: no cover
    from utils import ensure_ts_index

try:
    from .notify import NotificationError, send_ntfy_messages, send_twilio_messages
except ImportError:  # pragma: no cover
    try:
        from notify import NotificationError, send_ntfy_messages, send_twilio_messages
    except ImportError:  # pragma: no cover
        NotificationError = RuntimeError  # type: ignore[assignment]
        send_twilio_messages = None  # type: ignore[assignment]
        send_ntfy_messages = None  # type: ignore[assignment]


def _load_asset_returns(symbol: str, data_dir: str, index: pd.DatetimeIndex) -> pd.Series:
    path = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Missing data file for hedge asset: {path}")
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.reindex(index).ffill()
    close = df["Close"].fillna(method="ffill")
    return close.pct_change().fillna(0.0)


def equity_stats(equity_curve: pd.Series) -> Dict[str, float]:
    eq = equity_curve.dropna()
    ret = eq.pct_change().dropna()
    if len(ret) == 0:
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0, "Vol": 0.0}
    ann = 252
    cagr = eq.iloc[-1] ** (ann / len(eq)) - 1
    vol = ret.std() * np.sqrt(ann)
    sharpe = (ret.mean() * ann) / (vol if vol > 0 else np.nan)
    mdd = (eq / eq.cummax() - 1.0).min()
    return {
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "MaxDD": float(mdd),
        "Vol": float(vol),
    }


def _enforce_min_hold(signal: pd.Series, min_hold_days: int) -> pd.Series:
    if min_hold_days <= 1 or signal.empty:
        return signal.astype(int)

    values = signal.astype(int).to_numpy(copy=True)
    result = values.copy()
    current_state = result[0]
    days_in_state = 1

    for idx in range(1, len(result)):
        desired = values[idx]
        if desired == current_state:
            days_in_state += 1
        else:
            if days_in_state >= min_hold_days:
                current_state = desired
                days_in_state = 1
                result[idx] = current_state
            else:
                result[idx] = current_state
                days_in_state += 1

    return pd.Series(result, index=signal.index, name=signal.name)


def run_backtest(
    symbol: str,
    data_dir: str = "data",
    model_path: Optional[str] = None,
    prob_threshold: float = 0.55,
    slippage_bps: float = 3.0,
    cost_bps: float = 3.0,
    start: Optional[str] = None,
    end: Optional[str] = None,
    *,
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
    feat_path = os.path.join(data_dir, f"{symbol}_features.parquet")
    df = pd.read_parquet(feat_path)
    df = ensure_ts_index(df)

    model_path = model_path or os.path.join("models", f"{symbol}_gbdt.pkl")
    pack = joblib.load(model_path)
    clf, features = pack["model"], pack["features"]

    if start:
        df = df.loc[pd.to_datetime(start):]
    if end:
        df = df.loc[:pd.to_datetime(end)]

    proba = clf.predict_proba(df[features])[:, 1]
    df = df.copy()
    df["proba"] = proba

    raw_signal = (df["proba"] > prob_threshold).astype(int)
    signal = _enforce_min_hold(raw_signal, min_hold_days)

    position = signal.shift(1).fillna(0)
    primary_ret = df["ret_1"].fillna(0)

    if hedge_symbol:
        hedge_ret = _load_asset_returns(hedge_symbol, data_dir, df.index) * float(hedge_weight)
    else:
        hedge_ret = pd.Series(0.0, index=df.index)

    gross = position * primary_ret + (1 - position) * hedge_ret

    turns = signal.diff().abs().fillna(0.0)
    per_side = (slippage_bps + cost_bps) / 10000.0
    fees = turns * (2 * per_side)
    net = gross - fees

    equity = (1 + net).cumprod()
    stats = equity_stats(equity)

    out = pd.DataFrame(
        {
            "equity": equity,
            "net_ret": net,
            "signal": signal,
            "proba": df["proba"],
            "primary_ret": primary_ret,
            "hedge_ret": hedge_ret,
            "position": position,
        }
    )
    out_path = os.path.join(data_dir, f"{symbol}_backtest.csv")
    out.to_csv(out_path, index=True)

    prev_signal = signal.shift(1).fillna(0)
    entries = (signal == 1) & (prev_signal == 0)
    exits = (signal == 0) & (prev_signal == 1)

    alerts = pd.DataFrame(
        {
            "action": np.where(entries, "BUY", np.where(exits, "SELL", None)),
            "close": df["Close"],
            "proba": df["proba"],
            "signal": signal,
            "net_ret": net,
        }
    )
    alerts = alerts.dropna(subset=["action"])
    alerts_path = os.path.join(data_dir, f"{symbol}_alerts.csv")
    alerts.to_csv(alerts_path, index=True)

    alert_sids: List[str] = []
    ntfy_status: List[int] = []
    notification_error: Optional[str] = None
    ntfy_error: Optional[str] = None

    if notify_phone:
        if send_twilio_messages is None:
            notification_error = "Notification module unavailable; install twilio package."
        else:
            try:
                alert_sids = send_twilio_messages(
                    symbol,
                    alerts.iterrows(),
                    notify_phone,
                    from_number=notify_from,
                    account_sid=twilio_sid,
                    auth_token=twilio_token,
                    limit=notify_limit,
                )
            except NotificationError as exc:  # pragma: no cover - runtime feedback only
                notification_error = str(exc)

    if ntfy_topic:
        if send_ntfy_messages is None:
            ntfy_error = "Notification module unavailable; install requests package."
        else:
            try:
                ntfy_status = send_ntfy_messages(
                    symbol,
                    alerts.iterrows(),
                    ntfy_topic,
                    limit=ntfy_limit,
                    server=ntfy_server,
                    token=ntfy_token,
                    title=ntfy_title,
                    priority=ntfy_priority,
                )
            except NotificationError as exc:  # pragma: no cover - runtime feedback only
                ntfy_error = str(exc)

    return {
        "csv_path": out_path,
        "alerts_path": alerts_path,
        "stats": stats,
        "equity": equity,
        "frame": out,
        "alerts": alerts,
        "alert_sids": alert_sids,
        "notification_error": notification_error,
        "ntfy_status": ntfy_status,
        "ntfy_error": ntfy_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--prob-threshold", type=float, default=0.55)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--cost-bps", type=float, default=3.0)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
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

    result = run_backtest(
        args.symbol,
        data_dir=args.data,
        model_path=args.model_path,
        prob_threshold=args.prob_threshold,
        slippage_bps=args.slippage_bps,
        cost_bps=args.cost_bps,
        start=args.start,
        end=args.end,
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
    print(f"Backtest stats: {result['stats']}")
    print(f"Wrote {result['csv_path']}")
    print(f"Alerts CSV: {result['alerts_path']}")
    if result["alert_sids"]:
        print(f"Sent {len(result['alert_sids'])} SMS alerts")
    if result["notification_error"]:
        print(f"SMS warning: {result['notification_error']}")
    if result["ntfy_status"]:
        print(f"Posted {len(result['ntfy_status'])} ntfy alerts")
    if result["ntfy_error"]:
        print(f"ntfy warning: {result['ntfy_error']}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

try:
    from .utils import ensure_ts_index
except ImportError:  # pragma: no cover
    from utils import ensure_ts_index

FEATURES = [
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_63",
    "ret_126",
    "ret_252",
    "rsi14",
    "macd",
    "macd_signal",
    "macd_hist",
    "sma10",
    "sma20",
    "sma50",
    "sma200",
    "ema10",
    "ema20",
    "ema50",
    "ema200",
    "sma_diff",
    "ema_diff",
    "sma_ratio_50_200",
    "ema_ratio_50_200",
    "atr_pct",
    "volatility_21",
    "volatility_63",
    "price_over_52w_high",
    "price_over_52w_low",
    "volume_zscore_21",
    "volume_zscore_63",
    "dow",
    "dom",
    "moy",
]


def ts_split(
    df: pd.DataFrame,
    train_start: str,
    valid_start: str,
    test_start: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = ensure_ts_index(df)

    train_start_ts = pd.to_datetime(train_start)
    valid_start_ts = pd.to_datetime(valid_start)
    test_start_ts = pd.to_datetime(test_start)

    train_end = valid_start_ts - pd.Timedelta(days=1)
    valid_end = test_start_ts - pd.Timedelta(days=1)

    train = df.loc[train_start_ts:train_end]
    valid = df.loc[valid_start_ts:valid_end]
    test = df.loc[test_start_ts:]
    if len(train) == 0 or len(valid) == 0 or len(test) == 0:
        raise SystemExit("One of the splits is empty; adjust your dates.")
    return train, valid, test


def _default_split_dates(df: pd.DataFrame) -> Tuple[str, str, str]:
    df = ensure_ts_index(df)
    start = df.index.min()
    end = df.index.max()
    if start == end:
        raise SystemExit("Not enough data to create train/valid/test splits.")

    span = end - start
    train_cut = start + span * 0.6
    valid_cut = start + span * 0.8

    train_start = str(start.date())
    valid_start = str((train_cut + pd.Timedelta(days=1)).date())
    test_start = str((valid_cut + pd.Timedelta(days=1)).date())
    return train_start, valid_start, test_start


def train_model(
    symbol: str,
    data_dir: str = "data",
    model_out: str = "models",
    train_start: Optional[str] = None,
    valid_start: Optional[str] = None,
    test_start: Optional[str] = None,
) -> Dict[str, object]:
    path = os.path.join(data_dir, f"{symbol}_features.parquet")
    df = pd.read_parquet(path)

    auto_train, auto_valid, auto_test = _default_split_dates(df)
    train_start = train_start or auto_train
    valid_start = valid_start or auto_valid
    test_start = test_start or auto_test

    train_df, valid_df, test_df = ts_split(df, train_start, valid_start, test_start)
    X_tr, y_tr = train_df[FEATURES], train_df["y"]
    X_va, y_va = valid_df[FEATURES], valid_df["y"]
    X_te, y_te = test_df[FEATURES], test_df["y"]

    clf = GradientBoostingClassifier()
    clf.fit(X_tr, y_tr)

    metrics: Dict[str, Dict[str, float]] = {}

    def eval_split(name: str, Xs: pd.DataFrame, ys: pd.Series) -> None:
        proba = clf.predict_proba(Xs)[:, 1]
        pred = (proba > 0.5).astype(int)
        auc = roc_auc_score(ys, proba)
        report_dict = classification_report(ys, pred, output_dict=True)
        report_text = classification_report(ys, pred, digits=3)

        metrics[name] = {
            "auc": float(auc),
            "accuracy": float(report_dict["accuracy"]),
            "precision": float(report_dict["weighted avg"]["precision"]),
            "recall": float(report_dict["weighted avg"]["recall"]),
            "f1": float(report_dict["weighted avg"]["f1-score"]),
        }

        print(f"\n== {name} ==")
        print(f"AUC: {auc:.3f}")
        print(report_text)

    eval_split("TRAIN", X_tr, y_tr)
    eval_split("VALID", X_va, y_va)
    eval_split("TEST", X_te, y_te)

    os.makedirs(model_out, exist_ok=True)
    out_path = os.path.join(model_out, f"{symbol}_gbdt.pkl")
    payload = {
        "model": clf,
        "features": FEATURES,
        "splits": {
            "train_start": train_start,
            "valid_start": valid_start,
            "test_start": test_start,
        },
        "metrics": metrics,
    }
    joblib.dump(payload, out_path)

    return {
        "model_path": out_path,
        "metrics": metrics,
        "splits": payload["splits"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--valid-start", default=None)
    parser.add_argument("--test-start", default=None)
    parser.add_argument("--model-out", default="models")
    args = parser.parse_args()

    result = train_model(
        args.symbol,
        data_dir=args.data,
        model_out=args.model_out,
        train_start=args.train_start,
        valid_start=args.valid_start,
        test_start=args.test_start,
    )
    print(f"Saved model to {result['model_path']}")


if __name__ == "__main__":
    main()

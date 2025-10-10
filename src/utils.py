from __future__ import annotations

from typing import Hashable

import pandas as pd


def ensure_ts_index(df: pd.DataFrame, date_col: Hashable | None = None) -> pd.DataFrame:
    """Return a DataFrame indexed by a timestamp column.

    If ``date_col`` is provided, coerce that column to datetime, drop invalid rows,
    and use it as the sorted, de-duplicated index. Otherwise prefer an existing
    DatetimeIndex or a ``Date`` column.
    """
    if date_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            result = df.copy()
            if getattr(result.index, "tz", None) is not None:
                result.index = result.index.tz_localize(None)
            result = result.sort_index()
            result = result[~result.index.duplicated(keep="last")]
            return result
        if "Date" in df.columns:
            date_col = "Date"
        else:
            raise ValueError("No datetime column or index found; specify date_col")

    if date_col not in df.columns:
        raise ValueError(f"DataFrame missing required column: {date_col}")

    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    result = result.dropna(subset=[date_col])
    if result.empty:
        raise ValueError("No valid timestamps after coercion")

    result = result.sort_values(date_col)
    result = result.drop_duplicates(subset=[date_col], keep="last")
    result = result.set_index(date_col)
    if getattr(result.index, "tz", None) is not None:
        result.index = result.index.tz_localize(None)

    return result
